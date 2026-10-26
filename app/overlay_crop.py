"""Crop suggestion that removes stickers, date stamps and other overlays.

The audit marks images with a training-toxic overlay. A small locator request
returns where the overlays sit; this module finds the largest crop that cuts
every overlay away, keeps the person completely and stays a usable photo.
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Sequence

Box = Sequence[float]  # [x1, y1, x2, y2] in pixels


def _intersects(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _contains(outer: Box, inner: Box) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def overlay_free_crop(
    width: int,
    height: int,
    overlays: Sequence[Box],
    keep_box: Box,
    min_area_fraction: float = 0.55,
    min_side: int = 512,
    aspect_range: Sequence[float] = (0.45, 2.2),
    margin_fraction: float = 0.01,
) -> Optional[List[int]]:
    """Return ``[x, y, w, h]`` of the largest overlay-free crop, or None.

    Every overlay is removed by moving one image edge past it. The crop must
    contain ``keep_box`` (the person), keep at least ``min_area_fraction`` of
    the image, a short side of ``min_side`` pixels and a normal aspect ratio.
    """
    if width <= 0 or height <= 0 or not overlays:
        return None
    margin = max(2.0, margin_fraction * max(width, height))
    boxes = [
        [max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3])]
        for b in overlays[:4]
        if b[2] > b[0] and b[3] > b[1]
    ]
    if not boxes:
        return None
    best: Optional[List[float]] = None
    best_area = 0.0
    for sides in itertools.product(("top", "bottom", "left", "right"), repeat=len(boxes)):
        left, top, right, bottom = 0.0, 0.0, float(width), float(height)
        for side, box in zip(sides, boxes):
            if side == "top":
                top = max(top, box[3] + margin)
            elif side == "bottom":
                bottom = min(bottom, box[1] - margin)
            elif side == "left":
                left = max(left, box[2] + margin)
            else:
                right = min(right, box[0] - margin)
        crop = [left, top, right, bottom]
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0 or min(w, h) < min_side:
            continue
        if not aspect_range[0] <= w / h <= aspect_range[1]:
            continue
        if w * h < min_area_fraction * width * height:
            continue
        if not _contains(crop, keep_box) or any(_intersects(crop, box) for box in boxes):
            continue
        if w * h > best_area:
            best, best_area = crop, w * h
    if best is None:
        return None
    x1, y1 = int(round(best[0])), int(round(best[1]))
    return [x1, y1, int(best[2]) - x1, int(best[3]) - y1]


def keep_box_for_person(width: int, height: int, face_bbox: Optional[Sequence[float]],
                        pose_bbox: Optional[Sequence[float]] = None) -> Optional[List[float]]:
    """Area that must stay in the crop: the face with headroom, plus the body when known.

    Both inputs use ``[x, y, w, h]``. Returns ``[x1, y1, x2, y2]`` or None without a face.
    """
    if not face_bbox or len(face_bbox) != 4:
        return None
    fx, fy, fw, fh = [float(v) for v in face_bbox]
    box = [fx - 0.3 * fw, fy - 0.45 * fh, fx + 1.3 * fw, fy + 1.25 * fh]
    if pose_bbox and len(pose_bbox) == 4:
        px, py, pw, ph = [float(v) for v in pose_bbox]
        box = [min(box[0], px), min(box[1], py), max(box[2], px + pw), max(box[3], py + ph)]
    return [max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]), min(float(height), box[3])]


def overlay_boxes_from_response(regions: Sequence[dict], width: int, height: int) -> List[List[float]]:
    """Convert locator boxes on a 0-1000 grid into pixel boxes; drops invalid entries."""
    boxes: List[List[float]] = []
    for region in regions or []:
        try:
            x1, y1, x2, y2 = (float(region[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            continue
        boxes.append([x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0])
    return boxes
