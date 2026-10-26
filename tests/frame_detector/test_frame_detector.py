#!/usr/bin/env python3
"""Regression tests for the local frame detector on synthetic images.

Run with ``python -m pytest tests/frame_detector -q`` from the repository
root.  The images are generated on the fly (see ``synthetic_frames.py``); no
photographs are stored in the repository.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import frame_cleanup as fc  # noqa: E402
import synthetic_frames  # noqa: E402

CASES = synthetic_frames.build_all()


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / float(union) if union else 0.0


def _analyse(tmp_path, name: str):
    image, expected, kind = CASES[name]
    path = tmp_path / f"{name}.jpg"
    image.save(path, "JPEG", quality=92)
    result = fc.analyze_frame_cleanup(str(path), str(tmp_path / "cache"), use_cache=False)
    return result, expected, kind


@pytest.mark.parametrize(
    "name",
    [
        "story_with_bars",
        "wide_top_border",
        "nested_colour_in_monochrome",
        "polaroid_in_chat",
        "dark_photo_on_black",
        "gradient_canvas_inner",
        "blurred_canvas_inner",
    ],
)
def test_erster_kandidat_trifft_das_foto(tmp_path, name):
    result, expected, _kind = _analyse(tmp_path, name)
    assert result["candidates"], f"{name}: keine Kandidaten"
    best = result["candidates"][0]["bbox"]
    assert _iou(best, expected) >= 0.90, f"{name}: bbox {best} statt {expected}"


def test_innenbild_steht_vor_dem_aeusseren_rahmen(tmp_path):
    result, expected, _kind = _analyse(tmp_path, "nested_colour_in_monochrome")
    best = result["candidates"][0]
    assert best["crop_type"] == "nested_inner_image"
    assert _iou(best["bbox"], expected) >= 0.90
    # Der äußere Story-Crop bleibt als weitere Option sichtbar.
    outer = [c for c in result["candidates"][1:] if c["bbox"][1] < expected[1]]
    assert outer, "äußerer Rahmen-Crop fehlt als Alternative"
    assert result["recommendation"] == "review"


def test_breiter_rand_wird_gefunden(tmp_path):
    result, expected, _kind = _analyse(tmp_path, "wide_top_border")
    assert result["confidence_level"] in {"high", "medium"}
    assert _iou(result["candidate_bbox"], expected) >= 0.90


def test_gedrehte_karte_wird_nicht_automatisch_beschnitten(tmp_path):
    result, _expected, _kind = _analyse(tmp_path, "rotated_card")
    assert result["recommendation"] != "auto_accept"


def test_normales_foto_bleibt_unveraendert(tmp_path):
    result, _expected, _kind = _analyse(tmp_path, "plain_photo")
    assert result["recommendation"] == "keep_original"
    assert not result["candidates"] or result["confidence"] < fc.DEFAULT_MEDIUM_CONFIDENCE


def test_schwarzer_rahmen_ohne_naht_wird_erkannt(tmp_path):
    result, expected, _kind = _analyse(tmp_path, "dark_photo_on_black")
    best = result["candidates"][0]["bbox"]
    assert _iou(best, expected) >= 0.90
    assert result["recommendation"] in {"review", "auto_accept"}


def test_ergebnis_behaelt_die_schnittstelle(tmp_path):
    result, _expected, _kind = _analyse(tmp_path, "story_with_bars")
    for key in (
        "detected", "confidence", "confidence_level", "recommendation", "candidate_bbox",
        "candidate_sides", "signals", "candidates", "layout_class", "crop_type",
        "layout_flags", "schema_version", "source_hash", "original_size",
    ):
        assert key in result, key
    for candidate in result["candidates"]:
        assert len(candidate["bbox"]) == 4
        assert "crop_type" in candidate
        assert "subject_safe" in candidate
