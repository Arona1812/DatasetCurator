"""Offline regressions for the face-sharpness check: python -m unittest test_face_blur -v."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import dataset_curator_v2 as curator


class FaceBlurDecision(unittest.TestCase):
    def test_sharp_rating_from_audit_overrules_low_local_value(self):
        self.assertEqual(curator.face_blur_decision(2.5, 35.0, 75.0), "note")

    def test_value_above_threshold_needs_no_action(self):
        self.assertIsNone(curator.face_blur_decision(40.0, 35.0, 30.0))

    def test_blurry_rating_and_very_low_value_rejects(self):
        self.assertEqual(curator.face_blur_decision(5.0, 35.0, 40.0), "reject")

    def test_blurry_rating_and_moderately_low_value_goes_to_review(self):
        self.assertEqual(curator.face_blur_decision(25.0, 35.0, 40.0), "review")

    def test_approximate_model_bbox_never_hard_rejects(self):
        self.assertEqual(curator.face_blur_decision(5.0, 35.0, 40.0, used_local_bbox=False), "review")

    def test_missing_measurement_is_ignored(self):
        self.assertIsNone(curator.face_blur_decision(-1.0, 35.0, 40.0))


class FaceBlurMeasurement(unittest.TestCase):
    def test_small_faces_are_not_upscaled_before_measuring(self):
        rng = np.random.default_rng(3)
        pixels = rng.integers(0, 255, size=(1200, 900), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "sharp.png")
            Image.fromarray(pixels, "L").save(path)
            small_face = curator.local_blur_variance_in_face(path, [400, 300, 90, 110])
            large_face = curator.local_blur_variance_in_face(path, [100, 100, 600, 700])
        # Identical sharp texture: a small face must not look several times
        # blurrier than a large one only because it is small.
        self.assertGreater(small_face, 0.5 * large_face)


if __name__ == "__main__":
    unittest.main()
