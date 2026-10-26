#!/usr/bin/env python3
"""Local face and pose detection shared by the frame detector and the pipeline.

MediaPipe removed the legacy ``mediapipe.solutions`` API in the 0.10.x series.
The pinned ``mediapipe==0.10.33`` only ships the Tasks API, which needs model
files that are not bundled with the wheel.  OpenCV 5 additionally dropped the
Haar cascade classifier.  Without this module both face and pose protection
silently returned "nothing found".

This module provides one lazily initialised detector pair:

* MediaPipe Tasks ``FaceDetector`` / ``PoseLandmarker`` with automatic model
  download into ``_models/`` next to the source files.
* Fallback to the legacy ``mediapipe.solutions`` API when an older MediaPipe
  is installed.
* Fallback to the OpenCV Haar cascade for faces when OpenCV still provides it.

The result objects mimic the legacy ``solutions`` structures (``detections``
with ``location_data.relative_bounding_box`` and ``pose_landmarks.landmark``)
so existing call sites keep working unchanged.
"""
from __future__ import annotations

import os
import threading

# MediaPipe/TensorFlow Lite write C++ info lines ("Created TensorFlow Lite
# XNNPACK delegate", "Feedback manager requires ...") straight to the console.
# They are harmless; only warnings of level ERROR and above stay visible.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import urllib.request
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mp = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_models")

FACE_MODEL_FILENAME = "blaze_face_short_range.tflite"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

_LOCK = threading.Lock()
_FACE_DETECTOR: Optional["FaceDetectorAdapter"] = None
_POSE_DETECTOR: Optional["PoseDetectorAdapter"] = None
_FACE_INIT_ATTEMPTED = False
_POSE_INIT_ATTEMPTED = False
_LAST_INIT_ERRORS: List[str] = []


# ---------------------------------------------------------------------------
# Legacy-compatible result structures
# ---------------------------------------------------------------------------
@dataclass
class RelativeBoundingBox:
    xmin: float
    ymin: float
    width: float
    height: float


@dataclass
class LocationData:
    relative_bounding_box: RelativeBoundingBox


@dataclass
class FaceDetection:
    location_data: LocationData
    score: List[float] = field(default_factory=list)


@dataclass
class FaceResult:
    detections: List[FaceDetection] = field(default_factory=list)


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 1.0


@dataclass
class LandmarkList:
    landmark: List[Landmark] = field(default_factory=list)


@dataclass
class PoseResult:
    pose_landmarks: Optional[LandmarkList] = None


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
def ensure_model_file(filename: str, url: str, timeout: float = 60.0) -> Optional[str]:
    """Return the local path of a MediaPipe model, downloading it once.

    Downloads are atomic (temporary file plus rename) so a cancelled download
    never leaves a truncated model behind.  Returns ``None`` when the file is
    unavailable and cannot be fetched.
    """
    path = os.path.join(MODEL_DIR, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 1024:
        return path
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        tmp = f"{path}.part.{os.getpid()}"
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
        if os.path.getsize(tmp) <= 1024:
            os.remove(tmp)
            return None
        os.replace(tmp, path)
        return path
    except Exception as exc:  # network or filesystem problems are non-fatal
        _LAST_INIT_ERRORS.append(f"model download failed for {filename}: {exc}")
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return None


def last_init_errors() -> List[str]:
    """Return human-readable reasons why a detector could not be initialised."""
    return list(_LAST_INIT_ERRORS)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
def _to_rgb_uint8(rgb: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(rgb)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


class FaceDetectorAdapter:
    """Face detector with a ``process(rgb)`` method returning ``FaceResult``."""

    def __init__(self, backend: str, impl: Any) -> None:
        self.backend = backend
        self._impl = impl
        self._lock = threading.Lock()

    def process(self, rgb: Any) -> FaceResult:
        arr = _to_rgb_uint8(rgb)
        if arr is None:
            return FaceResult()
        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            return FaceResult()
        try:
            with self._lock:
                if self.backend == "mediapipe_tasks":
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
                    result = self._impl.detect(image)
                    detections = []
                    for det in getattr(result, "detections", None) or []:
                        box = det.bounding_box
                        score = float(det.categories[0].score) if det.categories else 0.0
                        detections.append(FaceDetection(
                            LocationData(RelativeBoundingBox(
                                box.origin_x / float(w), box.origin_y / float(h),
                                box.width / float(w), box.height / float(h),
                            )),
                            [score],
                        ))
                    return FaceResult(detections)
                if self.backend == "mediapipe_solutions":
                    result = self._impl.process(arr)
                    detections = []
                    for det in getattr(result, "detections", None) or []:
                        box = det.location_data.relative_bounding_box
                        score = float(det.score[0]) if det.score else 0.0
                        detections.append(FaceDetection(
                            LocationData(RelativeBoundingBox(box.xmin, box.ymin, box.width, box.height)),
                            [score],
                        ))
                    return FaceResult(detections)
                if self.backend == "haar":
                    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                    boxes = self._impl.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28))
                    detections = [
                        FaceDetection(
                            LocationData(RelativeBoundingBox(x / float(w), y / float(h), bw / float(w), bh / float(h))),
                            [0.5],
                        )
                        for x, y, bw, bh in boxes
                    ]
                    return FaceResult(detections)
        except Exception as exc:  # detector failures must never break a run
            _LAST_INIT_ERRORS.append(f"face detection failed ({self.backend}): {exc}")
        return FaceResult()


class PoseDetectorAdapter:
    """Pose detector with a ``process(rgb)`` method returning ``PoseResult``."""

    def __init__(self, backend: str, impl: Any) -> None:
        self.backend = backend
        self._impl = impl
        self._lock = threading.Lock()

    def process(self, rgb: Any) -> PoseResult:
        arr = _to_rgb_uint8(rgb)
        if arr is None:
            return PoseResult()
        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            return PoseResult()
        try:
            with self._lock:
                if self.backend == "mediapipe_tasks":
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
                    result = self._impl.detect(image)
                    poses = getattr(result, "pose_landmarks", None) or []
                    if not poses:
                        return PoseResult()
                    landmarks = [
                        Landmark(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0) or 0.0),
                                 float(getattr(lm, "visibility", 0.0) or 0.0),
                                 float(getattr(lm, "presence", 1.0) or 0.0))
                        for lm in poses[0]
                    ]
                    return PoseResult(LandmarkList(landmarks))
                if self.backend == "mediapipe_solutions":
                    result = self._impl.process(arr)
                    raw = getattr(result, "pose_landmarks", None)
                    if raw is None:
                        return PoseResult()
                    landmarks = [
                        Landmark(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0) or 0.0),
                                 float(getattr(lm, "visibility", 0.0) or 0.0))
                        for lm in raw.landmark
                    ]
                    return PoseResult(LandmarkList(landmarks))
        except Exception as exc:
            _LAST_INIT_ERRORS.append(f"pose detection failed ({self.backend}): {exc}")
        return PoseResult()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def _create_face_detector(min_confidence: float) -> Optional[FaceDetectorAdapter]:
    if mp is not None:
        try:
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore

            model_path = ensure_model_file(FACE_MODEL_FILENAME, FACE_MODEL_URL)
            if model_path:
                options = vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    min_detection_confidence=float(min_confidence),
                )
                return FaceDetectorAdapter("mediapipe_tasks", vision.FaceDetector.create_from_options(options))
        except Exception as exc:
            _LAST_INIT_ERRORS.append(f"mediapipe tasks face detector unavailable: {exc}")
        try:
            solutions = getattr(mp, "solutions", None)
            if solutions is not None:
                impl = solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=float(min_confidence)
                )
                return FaceDetectorAdapter("mediapipe_solutions", impl)
        except Exception as exc:
            _LAST_INIT_ERRORS.append(f"mediapipe solutions face detector unavailable: {exc}")
    if cv2 is not None and hasattr(cv2, "CascadeClassifier"):
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                return FaceDetectorAdapter("haar", cascade)
        except Exception as exc:
            _LAST_INIT_ERRORS.append(f"haar cascade unavailable: {exc}")
    return None


def _create_pose_detector(min_confidence: float, model_complexity: int) -> Optional[PoseDetectorAdapter]:
    if mp is None:
        return None
    try:
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        model_path = ensure_model_file(POSE_MODEL_FILENAME, POSE_MODEL_URL)
        if model_path:
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=float(min_confidence),
            )
            return PoseDetectorAdapter("mediapipe_tasks", vision.PoseLandmarker.create_from_options(options))
    except Exception as exc:
        _LAST_INIT_ERRORS.append(f"mediapipe tasks pose landmarker unavailable: {exc}")
    try:
        solutions = getattr(mp, "solutions", None)
        if solutions is not None:
            impl = solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=int(model_complexity),
                enable_segmentation=False,
                min_detection_confidence=float(min_confidence),
            )
            return PoseDetectorAdapter("mediapipe_solutions", impl)
    except Exception as exc:
        _LAST_INIT_ERRORS.append(f"mediapipe solutions pose unavailable: {exc}")
    return None


def get_face_detector(min_confidence: float = 0.5) -> Optional[FaceDetectorAdapter]:
    """Return the shared face detector or ``None`` when no backend works."""
    global _FACE_DETECTOR, _FACE_INIT_ATTEMPTED
    if _FACE_INIT_ATTEMPTED:
        return _FACE_DETECTOR
    with _LOCK:
        if not _FACE_INIT_ATTEMPTED:
            _FACE_INIT_ATTEMPTED = True
            _FACE_DETECTOR = _create_face_detector(min_confidence)
    return _FACE_DETECTOR


def get_pose_detector(min_confidence: float = 0.4, model_complexity: int = 0) -> Optional[PoseDetectorAdapter]:
    """Return the shared pose detector or ``None`` when no backend works."""
    global _POSE_DETECTOR, _POSE_INIT_ATTEMPTED
    if _POSE_INIT_ATTEMPTED:
        return _POSE_DETECTOR
    with _LOCK:
        if not _POSE_INIT_ATTEMPTED:
            _POSE_INIT_ATTEMPTED = True
            _POSE_DETECTOR = _create_pose_detector(min_confidence, model_complexity)
    return _POSE_DETECTOR


def detect_face_boxes(rgb: Any, min_confidence: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Return absolute ``(x, y, w, h)`` face boxes for an RGB array."""
    detector = get_face_detector(min_confidence)
    arr = _to_rgb_uint8(rgb)
    if detector is None or arr is None:
        return []
    h, w = arr.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    for det in detector.process(arr).detections:
        box = det.location_data.relative_bounding_box
        x = max(0, min(w - 1, int(round(box.xmin * w))))
        y = max(0, min(h - 1, int(round(box.ymin * h))))
        bw = max(1, min(w - x, int(round(box.width * w))))
        bh = max(1, min(h - y, int(round(box.height * h))))
        boxes.append((x, y, bw, bh))
    return boxes


def detect_pose_points(rgb: Any, min_visibility: float = 0.35) -> List[Tuple[float, float, float, int]]:
    """Return visible pose landmarks as ``(x, y, confidence, landmark_index)``.

    The confidence is the smaller of MediaPipe's visibility and presence so
    that limbs extrapolated outside the photo (common in tall screenshots) are
    not treated as visible body parts.
    """
    detector = get_pose_detector()
    arr = _to_rgb_uint8(rgb)
    if detector is None or arr is None:
        return []
    h, w = arr.shape[:2]
    result = detector.process(arr)
    if result.pose_landmarks is None:
        return []
    points: List[Tuple[float, float, float, int]] = []
    for index, lm in enumerate(result.pose_landmarks.landmark):
        confidence = min(float(lm.visibility), float(lm.presence))
        if confidence < min_visibility:
            continue
        if -0.05 <= lm.x <= 1.05 and -0.05 <= lm.y <= 1.05:
            points.append((lm.x * w, lm.y * h, confidence, index))
    return points


def backend_summary() -> str:
    """Return a short description of the active detector backends."""
    face = _FACE_DETECTOR.backend if _FACE_DETECTOR is not None else "none"
    pose = _POSE_DETECTOR.backend if _POSE_DETECTOR is not None else "none"
    return f"face={face}, pose={pose}"
