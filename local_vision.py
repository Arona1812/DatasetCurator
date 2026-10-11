"""MediaPipe Tasks adapters for the curator's normalized face/pose contract."""
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import os
import tempfile

import numpy as np
import requests

MODELS = {
    "face": ("blaze_face_short_range.tflite", "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"),
    "pose": ("pose_landmarker_lite.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
}


def ensure_model(kind, cache_dir):
    """Download public weights once; never send image data to the model host."""
    name, url = MODELS[kind]
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.is_file() and target.stat().st_size > 1024:
        return str(target)
    fd, temporary = tempfile.mkstemp(prefix=name, suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as output, requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 32 * 1024 * 1024:
                    raise ValueError("Unexpected local vision model size")
                output.write(chunk)
        if size < 1024:
            raise ValueError("Incomplete local vision model")
        os.replace(temporary, target)
        return str(target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TasksAdapter:
    def __init__(self, kind, cache_dir):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.kind = kind
        self.lock = Lock()
        base = BaseOptions(model_asset_path=ensure_model(kind, cache_dir))
        if kind == "face":
            self.detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=base, running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.5))
        else:
            self.detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=base, running_mode=vision.RunningMode.IMAGE, num_poses=1,
                min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5))

    def process(self, rgb):
        data = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width = data.shape[:2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=data)
        with self.lock:
            result = self.detector.detect(image)
        if self.kind == "pose":
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks) if landmarks else None)
        detections = []
        for item in result.detections:
            box = item.bounding_box
            relative = SimpleNamespace(xmin=box.origin_x / width, ymin=box.origin_y / height,
                                       width=box.width / width, height=box.height / height)
            detections.append(SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=relative),
                                              score=[max((category.score for category in item.categories), default=0.0)]))
        return SimpleNamespace(detections=detections)

    def close(self):
        with self.lock:
            self.detector.close()
