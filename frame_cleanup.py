#!/usr/bin/env python3
"""Local, reversible social-media frame and UI-border cleanup.

The detector is intentionally local-only. It analyses a bounded preview, scores
all four sides independently, creates one or more crop candidates, and persists
both positive and negative decisions. Full-resolution source files are never
modified. User decisions are stored separately from detector cache entries.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    from scipy.ndimage import uniform_filter1d  # type: ignore
except Exception:  # pragma: no cover - small fallback for minimal installs
    def uniform_filter1d(values, size=3, mode="nearest"):
        arr = np.asarray(values, dtype=np.float32)
        size = max(1, int(size))
        if size <= 1:
            return arr
        pad = size // 2
        padded = np.pad(arr, (pad, size - pad - 1), mode="edge")
        kernel = np.ones(size, dtype=np.float32) / float(size)
        return np.convolve(padded, kernel, mode="valid")

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

FRAME_DETECTOR_SCHEMA_VERSION = "smart-frame-v1"
FRAME_USER_DECISION_SCHEMA_VERSION = "v1"
FRAME_DECISION_FILENAME = "_frame_cleanup_decisions.json"
FRAME_CACHE_SUBDIR = "frame_cleanup"
FRAME_CROP_SUBDIR = "crops"
FRAME_PREVIEW_SUBDIR = "review_previews"
PIL_HARD_IMAGE_PIXEL_LIMIT = 350_000_000
DEFAULT_ANALYSIS_MAX_SIDE = 1024
DEFAULT_HIGH_CONFIDENCE = 0.76
DEFAULT_MEDIUM_CONFIDENCE = 0.58
DEFAULT_MIN_BORDER_PX = 24
DEFAULT_MIN_CONTENT_PX = 400

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass(frozen=True)
class DetectorSettings:
    analysis_max_side: int = DEFAULT_ANALYSIS_MAX_SIDE
    min_border_px: int = DEFAULT_MIN_BORDER_PX
    min_content_px: int = DEFAULT_MIN_CONTENT_PX
    high_confidence: float = DEFAULT_HIGH_CONFIDENCE
    medium_confidence: float = DEFAULT_MEDIUM_CONFIDENCE
    advanced_types: bool = True

    def fingerprint(self) -> str:
        payload = {
            "schema": FRAME_DETECTOR_SCHEMA_VERSION,
            "analysis_max_side": int(self.analysis_max_side),
            "min_border_px": int(self.min_border_px),
            "min_content_px": int(self.min_content_px),
            "high_confidence": round(float(self.high_confidence), 4),
            "medium_confidence": round(float(self.medium_confidence), 4),
            "advanced_types": bool(self.advanced_types),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def file_sha1(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(path: str) -> Tuple[int, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            width, height = image.size
            pixels = int(width) * int(height)
            if pixels > PIL_HARD_IMAGE_PIXEL_LIMIT:
                raise ValueError(f"image_too_large_{width}x{height}_{pixels}px")
            try:
                orientation = int(image.getexif().get(274, 1) or 1)
            except Exception:
                orientation = 1
            if orientation in (5, 6, 7, 8):
                return int(height), int(width)
            return int(width), int(height)


def load_bounded_rgb(path: str, max_side: int) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            width, height = image.size
            if int(width) * int(height) > PIL_HARD_IMAGE_PIXEL_LIMIT:
                raise ValueError(f"image_too_large_{width}x{height}")
            try:
                image.draft("RGB", (int(max_side) * 2, int(max_side) * 2))
            except Exception:
                pass
            prepared = ImageOps.exif_transpose(image)
            prepared.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
            return prepared.convert("RGB").copy()


def _cache_root(cache_dir: str) -> str:
    root = os.path.join(cache_dir, FRAME_CACHE_SUBDIR)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, FRAME_CROP_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(root, FRAME_PREVIEW_SUBDIR), exist_ok=True)
    return root


def _decision_cache_path(cache_dir: str, source_hash: str, settings: DetectorSettings) -> str:
    return os.path.join(_cache_root(cache_dir), f"{source_hash}_{settings.fingerprint()}_decision.json")


def crop_cache_path(cache_dir: str, source_hash: str, bbox: Iterable[int]) -> str:
    bbox_key = "_".join(str(int(v)) for v in bbox)
    suffix = hashlib.sha1(bbox_key.encode("ascii", errors="ignore")).hexdigest()[:10]
    return os.path.join(_cache_root(cache_dir), FRAME_CROP_SUBDIR, f"{source_hash}_{suffix}.jpg")


def user_decision_path(output_root: str) -> str:
    return os.path.join(output_root, FRAME_DECISION_FILENAME)


def load_user_decisions(output_root: str) -> Dict[str, Any]:
    path = user_decision_path(output_root)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("decisions"), dict):
            return payload
    except Exception:
        pass
    return {
        "schema_version": FRAME_USER_DECISION_SCHEMA_VERSION,
        "updated_at": "",
        "decisions": {},
    }


def save_user_decision(
    output_root: str,
    source_hash: str,
    source_path: str,
    decision: str,
    bbox: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    payload = load_user_decisions(output_root)
    normalized = str(decision or "auto").strip().lower()
    if normalized not in {"auto", "accept", "keep_original", "manual"}:
        raise ValueError(f"unsupported frame decision: {decision}")
    record = {
        "source_hash": source_hash,
        "source_path": os.path.abspath(source_path),
        "decision": normalized,
        "bbox": [int(v) for v in bbox] if bbox is not None else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if normalized == "auto":
        payload["decisions"].pop(source_hash, None)
    else:
        payload["decisions"][source_hash] = record
    payload["schema_version"] = FRAME_USER_DECISION_SCHEMA_VERSION
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(user_decision_path(output_root), payload)
    return record


def reset_user_decisions(output_root: str) -> int:
    path = user_decision_path(output_root)
    count = len(load_user_decisions(output_root).get("decisions", {}) or {})
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return count


def reset_detector_cache(cache_dir: str) -> int:
    root = os.path.join(cache_dir, FRAME_CACHE_SUBDIR)
    count = 0
    if os.path.isdir(root):
        for _, _, files in os.walk(root):
            count += len(files)
        shutil.rmtree(root, ignore_errors=True)
    return count


def _candidate_side_metrics(
    img: np.ndarray,
    side: str,
    min_border: int,
    min_content: int,
    advanced: bool,
) -> Optional[Dict[str, Any]]:
    h, w = img.shape[:2]
    gray = img.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge = np.zeros((h, w), np.float32)
    edge[:, :-1] += gx
    edge[:-1, :] += gy
    edge = (edge > 18).astype(np.float32)

    vertical = side in ("left", "right")
    dim = w if vertical else h
    max_border = min(int(dim * 0.34), dim - min_content)
    if max_border <= min_border:
        return None

    if vertical:
        start = int(h * 0.08)
        end = max(start + 1, int(h * 0.92))
        seam_profile = np.mean(gx[start:end], axis=0)
        line_std = np.std(gray, axis=0)
        line_edge = np.mean(edge, axis=0)
        line_mean = np.mean(gray, axis=0)
        dark_share = np.mean(gray < 42, axis=0)
        bright_share = np.mean(gray > 213, axis=0)
    else:
        start = int(w * 0.08)
        end = max(start + 1, int(w * 0.92))
        seam_profile = np.mean(gy[:, start:end], axis=1)
        line_std = np.std(gray, axis=1)
        line_edge = np.mean(edge, axis=1)
        line_mean = np.mean(gray, axis=1)
        dark_share = np.mean(gray < 42, axis=1)
        bright_share = np.mean(gray > 213, axis=1)

    seam_profile = uniform_filter1d(seam_profile.astype(np.float32), 3, mode="nearest")
    if side in ("right", "bottom"):
        candidate_indexes = np.arange(dim - min_border - 1, dim - max_border - 2, -1)
    else:
        candidate_indexes = np.arange(min_border - 1, max_border)
    if len(candidate_indexes) == 0:
        return None

    strengths = seam_profile[candidate_indexes]
    top_n = min(24, len(candidate_indexes))
    strong_indexes = candidate_indexes[np.argpartition(strengths, -top_n)[-top_n:]]

    texture_profile = uniform_filter1d(line_edge.astype(np.float32), 5, mode="nearest")
    texture_delta = np.abs(np.diff(texture_profile))
    delta_indexes = np.clip(candidate_indexes, 0, len(texture_delta) - 1)
    delta_values = texture_delta[delta_indexes]
    delta_n = min(12, len(candidate_indexes))
    transition_indexes = candidate_indexes[np.argpartition(delta_values, -delta_n)[-delta_n:]]

    picks = {int(v) for v in list(strong_indexes) + list(transition_indexes)}
    for frac in (0.02, 0.035, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30):
        distance = int(round(dim * frac))
        index = distance - 1 if side in ("left", "top") else dim - distance - 1
        if min_border <= distance <= max_border:
            picks.add(int(index))

    best: Optional[Dict[str, Any]] = None
    for index in picks:
        distance = index + 1 if side in ("left", "top") else dim - index - 1
        if distance < min_border or distance > max_border:
            continue
        probe = max(8, min(56, distance))
        if side in ("left", "top"):
            outer_slice = slice(0, distance)
            inner_slice = slice(distance, min(dim, distance + probe))
        else:
            outer_slice = slice(dim - distance, dim)
            inner_slice = slice(max(0, dim - distance - probe), dim - distance)
        if inner_slice.start >= inner_slice.stop:
            continue

        seam = float(seam_profile[index])
        boundary_values = gx[:, index] if vertical else gy[index, :]
        continuity = float(np.mean(boundary_values > 14))
        outer_edge = float(np.mean(line_edge[outer_slice]))
        inner_edge = float(np.mean(line_edge[inner_slice]))
        texture_gain = float(np.clip((inner_edge - outer_edge) / 0.12, 0.0, 1.0))
        flat_share = float(np.mean(line_std[outer_slice] < 18))
        dominant_extreme = max(
            float(np.mean(dark_share[outer_slice])),
            float(np.mean(bright_share[outer_slice])),
        )
        mean_stability = max(0.0, 1.0 - min(1.0, float(np.std(line_mean[outer_slice])) / 28.0))
        # A stable mean alone is not evidence for a frame: random/noisy photo
        # content can have nearly identical column means. Only use mean
        # stability when the outer band itself has little edge activity.
        low_edge_factor = float(np.clip(1.0 - outer_edge / 0.22, 0.0, 1.0))
        flatness = max(flat_share, dominant_extreme, mean_stability * low_edge_factor * 0.85)
        seam_score = float(np.clip((seam - 4.0) / 28.0, 0.0, 1.0))
        continuity_score = float(np.clip((continuity - 0.12) / 0.62, 0.0, 1.0))
        solid_score = 0.42 * seam_score + 0.28 * continuity_score + 0.22 * flatness + 0.08 * texture_gain
        textured_score = 0.48 * seam_score + 0.30 * continuity_score + 0.22 * texture_gain
        kind = "solid_bar" if flatness >= 0.62 else "textured_canvas"
        score = solid_score if kind == "solid_bar" else textured_score
        fraction = distance / float(dim)
        if fraction < 0.018:
            score *= 0.80
        if fraction > 0.27:
            score *= 0.88
        if not advanced and kind != "solid_bar":
            score *= 0.30
        record = {
            "side": side,
            "distance": int(distance),
            "score": round(float(score), 5),
            "kind": kind,
            "seam": round(seam, 4),
            "continuity": round(continuity, 4),
            "flatness": round(flatness, 4),
            "texture_gain": round(texture_gain, 4),
            "outer_edge": round(outer_edge, 4),
            "inner_edge": round(inner_edge, 4),
        }
        if best is None or float(record["score"]) > float(best["score"]):
            best = record
    return best


def _detect_face_boxes(preview: Image.Image) -> List[Tuple[int, int, int, int]]:
    if cv2 is None:
        return []
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return []
        rgb = np.asarray(preview.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(28, 28))
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]
    except Exception:
        return []


def _bbox_preserves_faces(bbox: List[int], faces: List[Tuple[int, int, int, int]]) -> Tuple[bool, List[str]]:
    x1, y1, x2, y2 = bbox
    flags: List[str] = []
    for x, y, width, height in faces:
        fx1, fy1, fx2, fy2 = x, y, x + width, y + height
        intersection = max(0, min(x2, fx2) - max(x1, fx1)) * max(0, min(y2, fy2) - max(y1, fy1))
        area = max(1, width * height)
        if intersection / float(area) < 0.92:
            flags.append("candidate_may_cut_detected_face")
    return not flags, flags


def _build_candidate(
    accepted: Dict[str, Dict[str, Any]],
    preview_size: Tuple[int, int],
    original_size: Tuple[int, int],
    faces: List[Tuple[int, int, int, int]],
    min_content_px: int,
) -> Optional[Dict[str, Any]]:
    preview_w, preview_h = preview_size
    orig_w, orig_h = original_size
    left = int((accepted.get("left") or {}).get("distance", 0))
    right = int((accepted.get("right") or {}).get("distance", 0))
    top = int((accepted.get("top") or {}).get("distance", 0))
    bottom = int((accepted.get("bottom") or {}).get("distance", 0))
    preview_bbox = [left, top, preview_w - right, preview_h - bottom]
    if preview_bbox[2] <= preview_bbox[0] or preview_bbox[3] <= preview_bbox[1]:
        return None
    scale_x = orig_w / float(preview_w)
    scale_y = orig_h / float(preview_h)
    bbox = [
        max(0, int(round(preview_bbox[0] * scale_x))),
        max(0, int(round(preview_bbox[1] * scale_y))),
        min(orig_w, int(round(preview_bbox[2] * scale_x))),
        min(orig_h, int(round(preview_bbox[3] * scale_y))),
    ]
    if bbox[2] - bbox[0] < min_content_px or bbox[3] - bbox[1] < min_content_px:
        return None
    retained_area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(max(1, orig_w * orig_h))
    if retained_area < 0.42:
        return None
    face_safe, safety_flags = _bbox_preserves_faces(preview_bbox, faces)
    scores = [float(v.get("score", 0)) for v in accepted.values()]
    confidence = sum(scores) / max(1, len(scores))
    if len(accepted) >= 2:
        confidence += 0.055
    if len(accepted) >= 3:
        confidence += 0.035
    if any(v.get("kind") == "textured_canvas" for v in accepted.values()):
        confidence -= 0.025
    if not face_safe:
        confidence -= 0.30
    single_side = len(accepted) == 1
    if single_side:
        only = next(iter(accepted.values()))
        if only.get("kind") != "solid_bar":
            confidence -= 0.20
        if str(only.get("side")) in {"left", "right"}:
            confidence -= 0.04
        # A lone edge can be a wall, sky line, dark curtain or subject edge.
        # Keep it reviewable but never auto-apply without a user decision.
        confidence = min(confidence, 0.735)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    signals = []
    for side, values in accepted.items():
        signals.append(f"{side}:{values.get('kind')}:{float(values.get('score', 0)):.2f}")
    signals.extend(safety_flags)
    return {
        "bbox": bbox,
        "preview_bbox": preview_bbox,
        "sides": sorted(accepted.keys()),
        "side_details": accepted,
        "confidence": round(confidence, 5),
        "retained_area_ratio": round(retained_area, 5),
        "signals": signals,
        "face_safe": bool(face_safe),
    }


def _analyse_preview(
    preview: Image.Image,
    original_size: Tuple[int, int],
    settings: DetectorSettings,
) -> Dict[str, Any]:
    img = np.asarray(preview.convert("RGB"), dtype=np.float32)
    h, w = img.shape[:2]
    scale_x = original_size[0] / float(w)
    scale_y = original_size[1] / float(h)
    min_border_x = max(4, int(round(settings.min_border_px / max(scale_x, 1e-6))))
    min_border_y = max(4, int(round(settings.min_border_px / max(scale_y, 1e-6))))
    min_content_x = max(100, int(round(settings.min_content_px / max(scale_x, 1e-6))))
    min_content_y = max(100, int(round(settings.min_content_px / max(scale_y, 1e-6))))

    side_candidates = {
        "left": _candidate_side_metrics(img, "left", min_border_x, min_content_x, settings.advanced_types),
        "right": _candidate_side_metrics(img, "right", min_border_x, min_content_x, settings.advanced_types),
        "top": _candidate_side_metrics(img, "top", min_border_y, min_content_y, settings.advanced_types),
        "bottom": _candidate_side_metrics(img, "bottom", min_border_y, min_content_y, settings.advanced_types),
    }

    accepted: Dict[str, Dict[str, Any]] = {}
    # Independent solid bars. Top and bottom are deliberately independent of side borders.
    for side, candidate in side_candidates.items():
        if not candidate:
            continue
        score = float(candidate.get("score", 0))
        kind = str(candidate.get("kind", ""))
        threshold = 0.66 if side in {"top", "bottom"} else 0.72
        if kind == "solid_bar" and score >= threshold:
            accepted[side] = dict(candidate)

    # Pair evidence makes weaker but coherent canvas borders usable.
    for first, second in (("left", "right"), ("top", "bottom")):
        a, b = side_candidates.get(first), side_candidates.get(second)
        if not a or not b:
            continue
        if float(a.get("score", 0)) < 0.47 or float(b.get("score", 0)) < 0.47:
            continue
        def _pair_usable(value: Dict[str, Any]) -> bool:
            if value.get("kind") == "solid_bar":
                return True
            return float(value.get("outer_edge", 1)) <= 0.18 and float(value.get("texture_gain", 0)) >= 0.20
        if not (_pair_usable(a) and _pair_usable(b)):
            continue
        ratio = min(int(a["distance"]), int(b["distance"])) / float(max(1, max(int(a["distance"]), int(b["distance"]))))
        boost = 0.08 if ratio >= 0.55 else 0.03
        aa, bb = dict(a), dict(b)
        aa["score"] = min(1.0, float(aa["score"]) + boost)
        bb["score"] = min(1.0, float(bb["score"]) + boost)
        accepted[first], accepted[second] = aa, bb

    # Three or four mutually supporting edges often indicate a framed canvas.
    viable = [
        (side, value)
        for side, value in side_candidates.items()
        if value
        and float(value.get("score", 0)) >= 0.44
        and (
            value.get("kind") == "solid_bar"
            or float(value.get("outer_edge", 1)) <= 0.18
        )
    ]
    if settings.advanced_types and len(viable) >= 3:
        for side, value in viable:
            promoted = dict(value)
            promoted["score"] = min(1.0, float(promoted["score"]) + 0.08)
            accepted[side] = promoted

    # A blurred social-media canvas can have four geometrically coherent
    # boundaries with only a weak seam. Promote this pattern to review rather
    # than missing it completely; it still cannot become a high-confidence
    # auto-crop from this rule alone.
    if settings.advanced_types and all(side_candidates.get(side) for side in ("left", "right", "top", "bottom")):
        left_c = side_candidates["left"]
        right_c = side_candidates["right"]
        top_c = side_candidates["top"]
        bottom_c = side_candidates["bottom"]
        lr_ratio = min(int(left_c["distance"]), int(right_c["distance"])) / float(max(1, max(int(left_c["distance"]), int(right_c["distance"]))))
        tb_ratio = min(int(top_c["distance"]), int(bottom_c["distance"])) / float(max(1, max(int(top_c["distance"]), int(bottom_c["distance"]))))
        coherent_blurred_canvas = (
            lr_ratio >= 0.62
            and tb_ratio >= 0.62
            and all(float(v.get("outer_edge", 1)) <= 0.08 for v in (left_c, right_c, top_c, bottom_c))
            and all(float(v.get("continuity", 0)) >= 0.28 for v in (left_c, right_c, top_c, bottom_c))
            and all(float(v.get("seam", 0)) >= 4.8 for v in (left_c, right_c, top_c, bottom_c))
        )
        if coherent_blurred_canvas:
            for side, value in side_candidates.items():
                promoted = dict(value)
                promoted["kind"] = "blurred_canvas"
                promoted["score"] = max(0.56, min(0.64, float(promoted.get("score", 0)) + 0.10))
                accepted[side] = promoted

    # Textured/gradient top and bottom bars can stand alone if the seam is strong.
    if settings.advanced_types:
        for side in ("top", "bottom"):
            value = side_candidates.get(side)
            if not value:
                continue
            if float(value.get("score", 0)) >= 0.59 and (
                float(value.get("flatness", 0)) >= 0.52
                or (float(value.get("texture_gain", 0)) >= 0.35 and float(value.get("continuity", 0)) >= 0.50)
            ):
                accepted[side] = dict(value)

    # Never accept a lone textured vertical edge; that is usually a wall/door/tree.
    for side in list(accepted.keys()):
        value = accepted[side]
        if value.get("kind") == "textured_canvas" and side in {"left", "right"}:
            opposite = "right" if side == "left" else "left"
            if opposite not in accepted and not ({"top", "bottom"} & set(accepted.keys())):
                del accepted[side]

    faces = _detect_face_boxes(preview)
    candidate_sets: List[Dict[str, Dict[str, Any]]] = []
    if accepted:
        candidate_sets.append(dict(accepted))
    if "left" in accepted and "right" in accepted:
        candidate_sets.append({"left": accepted["left"], "right": accepted["right"]})
    if "top" in accepted and "bottom" in accepted:
        candidate_sets.append({"top": accepted["top"], "bottom": accepted["bottom"]})
    for side, value in accepted.items():
        if value.get("kind") == "solid_bar" and float(value.get("score", 0)) >= 0.78:
            candidate_sets.append({side: value})

    candidates: List[Dict[str, Any]] = []
    seen = set()
    for accepted_set in candidate_sets:
        built = _build_candidate(accepted_set, (w, h), original_size, faces, settings.min_content_px)
        if not built:
            continue
        key = tuple(built["bbox"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(built)
    candidates.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)

    best = candidates[0] if candidates else None
    confidence = float(best.get("confidence", 0)) if best else 0.0
    if not best:
        level = "low"
        recommendation = "keep_original"
    elif confidence >= settings.high_confidence and bool(best.get("face_safe", True)):
        level = "high"
        recommendation = "auto_accept"
    elif confidence >= settings.medium_confidence:
        level = "medium"
        recommendation = "review"
    else:
        level = "low"
        recommendation = "keep_original"

    return {
        "detected": bool(best and confidence >= settings.medium_confidence),
        "confidence": round(confidence, 5),
        "confidence_level": level,
        "recommendation": recommendation,
        "candidate_bbox": list(best["bbox"]) if best else None,
        "candidate_sides": list(best.get("sides", [])) if best else [],
        "signals": list(best.get("signals", [])) if best else [],
        "candidates": candidates[:5],
        "side_candidates": side_candidates,
        "preview_size": [w, h],
        "face_boxes_detected": len(faces),
    }


def analyze_frame_cleanup(
    image_path: str,
    cache_dir: str,
    source_hash: Optional[str] = None,
    settings: Optional[DetectorSettings] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    settings = settings or DetectorSettings()
    source_hash = source_hash or file_sha1(image_path)
    cache_path = _decision_cache_path(cache_dir, source_hash, settings)
    if use_cache and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if (
                isinstance(cached, dict)
                and cached.get("schema_version") == FRAME_DETECTOR_SCHEMA_VERSION
                and cached.get("source_hash") == source_hash
                and cached.get("settings_fingerprint") == settings.fingerprint()
            ):
                cached["cache_hit"] = True
                return cached
        except Exception:
            pass

    original_size = image_dimensions(image_path)
    result: Dict[str, Any]
    try:
        if min(original_size) < 400:
            result = {
                "detected": False,
                "confidence": 0.0,
                "confidence_level": "low",
                "recommendation": "keep_original",
                "candidate_bbox": None,
                "candidate_sides": [],
                "signals": ["source_too_small"],
                "candidates": [],
                "side_candidates": {},
                "preview_size": list(original_size),
                "face_boxes_detected": 0,
            }
        else:
            preview = load_bounded_rgb(image_path, settings.analysis_max_side)
            result = _analyse_preview(preview, original_size, settings)
    except Exception as exc:
        result = {
            "detected": False,
            "confidence": 0.0,
            "confidence_level": "low",
            "recommendation": "keep_original",
            "candidate_bbox": None,
            "candidate_sides": [],
            "signals": [f"detector_error:{type(exc).__name__}:{exc}"],
            "candidates": [],
            "side_candidates": {},
            "preview_size": [],
            "face_boxes_detected": 0,
        }

    result.update({
        "schema_version": FRAME_DETECTOR_SCHEMA_VERSION,
        "settings_fingerprint": settings.fingerprint(),
        "source_hash": source_hash,
        "source_path": os.path.abspath(image_path),
        "source_filename": os.path.basename(image_path),
        "original_size": list(original_size),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_hit": False,
    })
    if use_cache:
        atomic_write_json(cache_path, result)
    return result


def materialize_crop(
    image_path: str,
    bbox: Iterable[int],
    cache_dir: str,
    source_hash: Optional[str] = None,
) -> str:
    source_hash = source_hash or file_sha1(image_path)
    normalized_bbox = [int(v) for v in bbox]
    destination = crop_cache_path(cache_dir, source_hash, normalized_bbox)
    if os.path.isfile(destination):
        return destination
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(image_path) as opened:
            if int(opened.size[0]) * int(opened.size[1]) > PIL_HARD_IMAGE_PIXEL_LIMIT:
                raise ValueError("image exceeds hard pixel limit")
            source = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = source.size
            x1, y1, x2, y2 = normalized_bbox
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            source.crop((x1, y1, x2, y2)).save(destination, "JPEG", quality=96, subsampling=0)
    return destination


def resolve_frame_cleanup(
    image_path: str,
    source_hash: str,
    cache_dir: str,
    output_root: str,
    mode: str = "auto_high_review_medium",
    settings: Optional[DetectorSettings] = None,
) -> Dict[str, Any]:
    analysis = analyze_frame_cleanup(image_path, cache_dir, source_hash, settings=settings, use_cache=True)
    user_payload = load_user_decisions(output_root)
    user = (user_payload.get("decisions", {}) or {}).get(source_hash) or {}
    user_decision = str(user.get("decision", "auto") or "auto")
    bbox = None
    applied_by = "none"

    if user_decision in {"accept", "manual"}:
        bbox = user.get("bbox") or analysis.get("candidate_bbox")
        applied_by = f"user_{user_decision}"
    elif user_decision == "keep_original":
        applied_by = "user_keep_original"
    else:
        mode = str(mode or "auto_high_review_medium")
        if mode == "auto_high_review_medium" and analysis.get("confidence_level") == "high":
            bbox = analysis.get("candidate_bbox")
            applied_by = "auto_high"
        elif mode == "auto_high_keep_medium" and analysis.get("confidence_level") in {"high", "medium"}:
            bbox = analysis.get("candidate_bbox")
            applied_by = "auto_high_or_medium"
        elif mode == "suggest_only":
            applied_by = "suggest_only"

    effective_path = image_path
    if isinstance(bbox, list) and len(bbox) == 4:
        effective_path = materialize_crop(image_path, bbox, cache_dir, source_hash)

    return {
        "source_original_path": os.path.abspath(image_path),
        "effective_image_path": effective_path,
        "frame_crop_path": effective_path if effective_path != image_path else "",
        "frame_cleanup_applied": effective_path != image_path,
        "frame_cleanup_applied_by": applied_by,
        "frame_cleanup_user_decision": user_decision,
        "frame_cleanup_bbox": [int(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None,
        "frame_cleanup_analysis": analysis,
    }


def build_review_preview(
    image_path: str,
    bbox: Optional[Iterable[int]],
    cache_dir: str,
    source_hash: Optional[str] = None,
    max_panel: Tuple[int, int] = (620, 720),
) -> str:
    source_hash = source_hash or file_sha1(image_path)
    bbox_list = [int(v) for v in bbox] if bbox is not None else None
    key = "none" if bbox_list is None else "_".join(str(v) for v in bbox_list)
    suffix = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    path = os.path.join(_cache_root(cache_dir), FRAME_PREVIEW_SUBDIR, f"{source_hash}_{suffix}.jpg")
    if os.path.isfile(path):
        return path

    original = load_bounded_rgb(image_path, max(max_panel))
    if bbox_list:
        orig_w, orig_h = image_dimensions(image_path)
        prev_w, prev_h = original.size
        scale_x = prev_w / float(orig_w)
        scale_y = prev_h / float(orig_h)
        pb = [
            int(round(bbox_list[0] * scale_x)),
            int(round(bbox_list[1] * scale_y)),
            int(round(bbox_list[2] * scale_x)),
            int(round(bbox_list[3] * scale_y)),
        ]
        cropped = original.crop(tuple(pb))
    else:
        cropped = original.copy()

    def panel(img: Image.Image, label: str) -> Image.Image:
        canvas = Image.new("RGB", max_panel, "#202020")
        work = img.copy()
        work.thumbnail((max_panel[0] - 20, max_panel[1] - 50), Image.Resampling.LANCZOS)
        x = (max_panel[0] - work.width) // 2
        y = 40 + (max_panel[1] - 50 - work.height) // 2
        canvas.paste(work, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), label, fill="white")
        return canvas

    left = panel(original, "ORIGINAL")
    right = panel(cropped, "VORSCHLAG / CROP")
    combined = Image.new("RGB", (max_panel[0] * 2, max_panel[1]), "#111111")
    combined.paste(left, (0, 0))
    combined.paste(right, (max_panel[0], 0))
    combined.save(path, "JPEG", quality=90)
    return path


def scan_source_images(input_folder: str, output_root: Optional[str] = None) -> List[str]:
    if not input_folder or not os.path.isdir(input_folder):
        return []
    output_abs = os.path.normcase(os.path.abspath(output_root)) if output_root else ""
    paths: List[str] = []
    for name in os.listdir(input_folder):
        path = os.path.join(input_folder, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        if output_abs and os.path.normcase(os.path.abspath(path)).startswith(output_abs + os.sep):
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def decision_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(str(record.get("confidence_level", "low")) for record in records)
    return {
        "high": int(counter.get("high", 0)),
        "medium": int(counter.get("medium", 0)),
        "low": int(counter.get("low", 0)),
        "total": int(sum(counter.values())),
    }
