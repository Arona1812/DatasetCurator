"""Overlay crop: geometry and the audited replacement, offline only:
python -m unittest tests.test_overlay_crop -v."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import dataset_curator_v2 as curator
from overlay_crop import keep_box_for_person, overlay_boxes_from_response, overlay_free_crop


class Geometry(unittest.TestCase):
    def test_sticker_oben_links_wird_weggeschnitten(self):
        # 1080x1290 selfie, music sticker in the upper left, face in the middle.
        crop = overlay_free_crop(1080, 1290, [[60, 0, 300, 110]], keep_box_for_person(1080, 1290, [480, 330, 380, 420]))
        x, y, w, h = crop
        self.assertTrue(y >= 110 or x >= 300)
        self.assertGreaterEqual(w * h, 0.55 * 1080 * 1290)
        self.assertTrue(x <= 480 - 0.3 * 380 and x + w >= 480 + 1.3 * 380)

    def test_kein_zuschnitt_wenn_das_overlay_auf_der_person_liegt(self):
        self.assertIsNone(overlay_free_crop(1000, 1000, [[400, 400, 600, 600]], [300, 300, 700, 700]))

    def test_zu_kleiner_rest_wird_abgelehnt(self):
        self.assertIsNone(overlay_free_crop(1000, 1000, [[0, 0, 1000, 600]], [400, 700, 600, 900]))

    def test_raster_wird_in_pixel_umgerechnet_und_unsinn_verworfen(self):
        boxes = overlay_boxes_from_response([{"x1": 0, "y1": 0, "x2": 500, "y2": 100}, {"x1": 900, "y1": 5, "x2": 10, "y2": 9}], 2000, 1000)
        self.assertEqual(boxes, [[0.0, 0.0, 1000.0, 100.0]])


class ReplacementRow(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.image = str(Path(folder.name) / "sticker.jpg")
        Image.new("RGB", (1080, 1290), (120, 110, 100)).save(self.image)
        self.row = {"original_filename": "sticker.jpg", "original_path": self.image, "quality_total": 75.0,
                    "main_face_bbox": [480, 330, 380, 420], "watermark_or_overlay": True}

    def run_crop(self, crop_audit):
        patches = [
            mock.patch.object(curator, "locate_overlay_regions", return_value=[{"x1": 50, "y1": 0, "x2": 280, "y2": 90}]),
            mock.patch.object(curator, "openai_audit_image", return_value=crop_audit),
            mock.patch.object(curator, "load_cached_audit", return_value=None),
            mock.patch.object(curator, "normalize_audit_scores", side_effect=lambda a: a),
            mock.patch.object(curator, "save_cached_audit"),
            mock.patch.object(curator, "local_subject_metrics", return_value={"width": 1080, "height": 1180}),
            mock.patch.object(curator, "apply_local_score_adjustments", side_effect=lambda r: r),
            mock.patch.object(curator, "local_status_override", return_value=("keep", [])),
            mock.patch.object(curator, "USE_CLIP_DUPLICATE_SCORING", False),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return curator.evaluate_overlay_crop(self.row, self.image, "hash", "[1/1]")

    def test_zuschnitt_ohne_overlay_ersetzt_das_original(self):
        crop = self.run_crop({"quality_total": 74.0, "watermark_or_overlay": False, "suggested_status": "keep", "shot_type": "headshot"})
        self.assertEqual(crop["crop_variant"], "overlay_removed")
        self.assertEqual(crop["crop_of"], "sticker.jpg")
        self.assertTrue(crop["is_rescue_crop"])
        self.assertEqual(crop["original_path"], self.image)
        self.assertGreater(crop["rescue_crop_bbox"][1], 0)

    def test_zuschnitt_mit_restlichem_overlay_wird_verworfen(self):
        self.assertIsNone(self.run_crop({"quality_total": 80.0, "watermark_or_overlay": True, "suggested_status": "keep"}))

    def test_deutlich_schlechterer_zuschnitt_wird_verworfen(self):
        self.assertIsNone(self.run_crop({"quality_total": 60.0, "watermark_or_overlay": False, "suggested_status": "keep"}))


if __name__ == "__main__":
    unittest.main()
