#!/usr/bin/env python3
"""Synthetic test images for the frame detector.

Every image is generated procedurally: a textured "photo" (noise, gradients,
circles) placed on app chrome or social-media canvases.  No photographs and no
personal data are involved.  Each builder returns the PIL image plus the bbox
``[x1, y1, x2, y2]`` that a correct detector should propose first (``None`` for
"keep original").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Case = Tuple[Image.Image, Optional[List[int]], str]


def _photo(width: int, height: int, seed: int, saturation: float = 1.0) -> Image.Image:
    """Return a textured, photo-like RGB image."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    # Low-frequency structure plus fine noise so that every row and column
    # carries variation like a real photograph.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    structure = (
        0.5 * np.sin(xx / 23.0) + 0.5 * np.cos(yy / 31.0)
        + 0.35 * np.sin((xx + yy) / 57.0)
    )
    gray = 128.0 + 48.0 * structure + 14.0 * base
    r = np.clip(gray + 40.0 * saturation * np.sin(xx / 41.0), 0, 255)
    g = np.clip(gray + 30.0 * saturation * np.cos(yy / 37.0), 0, 255)
    b = np.clip(gray - 35.0 * saturation * np.sin(yy / 29.0), 0, 255)
    img = Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    for index in range(6):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        radius = int(rng.integers(max(8, width // 16), max(12, width // 5)))
        color = tuple(int(v) for v in rng.integers(30, 225, size=3))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    return img


def _chrome_text(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, color=(235, 235, 235)) -> None:
    """Draw a text-like row of short bars (icons/labels) on chrome."""
    cursor = x
    step = max(6, height)
    while cursor < x + width:
        bar = min(step * 3, x + width - cursor)
        draw.rectangle([cursor, y, cursor + bar, y + height], fill=color)
        cursor += bar + step * 2


def story_with_bars(seed: int = 1) -> Case:
    """Full-width photo between black status/message bars (Instagram story)."""
    width, height = 1080, 2400
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    _chrome_text(draw, 40, 24, 300, 22)
    top, bottom = 150, 2050
    canvas.paste(_photo(width, bottom - top, seed), (0, top))
    draw.rounded_rectangle([24, 2210, 820, 2300], radius=40, outline=(120, 120, 120), width=3)
    _chrome_text(draw, 60, 2240, 220, 24, color=(160, 160, 160))
    return canvas, [0, top, width, bottom], "story_bars"


def wide_top_border(seed: int = 2) -> Case:
    """Photo sitting below a dark bar taller than a third of the image."""
    width, height = 720, 1280
    canvas = Image.new("RGB", (width, height), (51, 51, 51))
    top, bottom = 470, 1000
    canvas.paste(_photo(width, bottom - top, seed), (0, top))
    return canvas, [0, top, width, bottom], "wide_border"


def nested_colour_in_monochrome(seed: int = 3) -> Case:
    """Colour photo pasted over a larger monochrome duplicate inside a story."""
    width, height = 1080, 2400
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    outer_top, outer_bottom = 150, 2200
    mono = _photo(width, outer_bottom - outer_top, seed, saturation=0.0).convert("L").convert("RGB")
    canvas.paste(mono, (0, outer_top))
    inner = _photo(620, 880, seed + 10, saturation=1.6)
    ix, iy = 230, 720
    canvas.paste(inner, (ix, iy))
    return canvas, [ix, iy, ix + inner.width, iy + inner.height], "nested_inner_image"


def polaroid_in_chat(seed: int = 4) -> Case:
    """Photo inside a white frame on a dark chat background."""
    width, height = 1080, 2400
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 100], fill=(0, 0, 0))
    _chrome_text(draw, 120, 40, 360, 26)
    frame = [136, 440, 944, 1920]
    draw.rectangle(frame, fill=(245, 245, 245))
    photo = _photo(700, 1260, seed)
    px, py = 190, 500
    canvas.paste(photo, (px, py))
    _chrome_text(draw, 300, 2080, 480, 30, color=(230, 230, 230))
    return canvas, [px, py, px + photo.width, py + photo.height], "polaroid"


def dark_photo_on_black(seed: int = 5) -> Case:
    """Dark photo with flat near-black margins on pure black app chrome."""
    width, height = 1080, 2400
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    _chrome_text(draw, 60, 40, 420, 30)
    top, bottom = 720, 1800
    photo_box = Image.new("RGB", (width, bottom - top), (31, 31, 31))
    subject = _photo(640, bottom - top, seed)
    dark = Image.eval(subject, lambda v: int(v * 0.55))
    photo_box.paste(dark, (220, 0))
    canvas.paste(photo_box, (0, top))
    return canvas, [220, top, 860, bottom], "dark_on_black"


def gradient_canvas_inner(seed: int = 6) -> Case:
    """Photo centred on a vertical gradient story canvas."""
    width, height = 1080, 2145
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    r = (180 - 60 * yy) * np.ones((height, width), np.float32)
    g = (150 - 50 * yy) * np.ones((height, width), np.float32)
    b = (130 - 40 * yy) * np.ones((height, width), np.float32)
    canvas = Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8), "RGB")
    photo = _photo(758, 1411, seed)
    px, py = 161, 411
    canvas.paste(photo, (px, py))
    return canvas, [px, py, px + photo.width, py + photo.height], "gradient_canvas"


def rotated_card(seed: int = 7) -> Case:
    """Rotated illustration on a gradient: must never be auto-applied."""
    width, height = 640, 1136
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    r = (250 - 40 * yy) * np.ones((height, width), np.float32)
    g = (205 - 70 * yy) * np.ones((height, width), np.float32)
    b = (180 - 60 * yy) * np.ones((height, width), np.float32)
    canvas = Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8), "RGB")
    card = _photo(430, 720, seed).rotate(9, resample=Image.Resampling.BICUBIC, expand=True)
    mask = Image.new("L", (430, 720), 255).rotate(9, resample=Image.Resampling.BICUBIC, expand=True)
    canvas.paste(card, (80, 150), mask)
    return canvas, None, "rotated_card"


def plain_photo(seed: int = 8) -> Case:
    """An ordinary photo without any frame: keep original."""
    photo = _photo(1200, 1600, seed)
    # A natural vignette: the lower part darkens smoothly but stays textured.
    arr = np.asarray(photo).astype(np.float32)
    fade = np.ones((1600, 1), dtype=np.float32)
    fade[1100:, 0] = np.linspace(1.0, 0.25, 500, dtype=np.float32)
    arr *= fade[:, :, None]
    photo = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return photo, None, "plain_photo"


def blurred_canvas_inner(seed: int = 9) -> Case:
    """Sharp photo on a blurred, uniform fill (shared post inside a story)."""
    width, height = 1080, 2400
    canvas = Image.new("RGB", (width, height), (117, 117, 112))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 130], fill=(0, 0, 0))
    draw.rectangle([0, 2205, width, height], fill=(0, 0, 0))
    photo = _photo(864, 1536, seed)
    px, py = 108, 323
    canvas.paste(photo, (px, py))
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.6))
    canvas.paste(photo, (px, py))
    return canvas, [px, py, px + photo.width, py + photo.height], "blurred_canvas"


ALL_CASES = {
    "story_with_bars": story_with_bars,
    "wide_top_border": wide_top_border,
    "nested_colour_in_monochrome": nested_colour_in_monochrome,
    "polaroid_in_chat": polaroid_in_chat,
    "dark_photo_on_black": dark_photo_on_black,
    "gradient_canvas_inner": gradient_canvas_inner,
    "rotated_card": rotated_card,
    "plain_photo": plain_photo,
    "blurred_canvas_inner": blurred_canvas_inner,
}


def build_all() -> Dict[str, Case]:
    """Return every synthetic case keyed by name."""
    return {name: builder() for name, builder in ALL_CASES.items()}


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import os
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "synthetic_frames_out"
    os.makedirs(target, exist_ok=True)
    for name, (image, expected, kind) in build_all().items():
        image.save(os.path.join(target, f"{name}.jpg"), "JPEG", quality=92)
        print(name, kind, expected)
