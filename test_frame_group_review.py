"""Offline checks for crop recovery and explicitly selected template batches."""
import json
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy

from PIL import Image
import numpy as np

import frame_cleanup as frames
from frame_group_review import build_group_plan, validate_group_selection
from photo_regions import expand_photo_region


def record(key, size=(200, 300), decision="auto"):
    w, h = size
    return {"source_hash": key, "original_size": list(size), "user_decision": decision,
            "layout_class": "uniform_canvas", "layout_flags": [], "confidence_level": "medium",
            "candidate_bbox": [round(w*.1), round(h*.1), round(w*.9), round(h*.9)],
            "display_bbox": [round(w*.1), round(h*.08), round(w*.9), round(h*.92)]}


class GroupPlans(unittest.TestCase):
    def setUp(self):
        self.records = [record("template", decision="manual"), record("a"), record("b", (400, 600))]

    def test_template_scales_and_does_not_apply_a_decision(self):
        before = deepcopy(self.records)
        plan = build_group_plan(self.records, "template")
        self.assertEqual(plan["targets"][1]["bbox"], [40, 48, 360, 552])
        self.assertEqual(self.records, before)
        self.assertEqual(len(validate_group_selection(self.records, plan, ["b"])), 1)

    def test_confirmed_and_different_layouts_are_excluded(self):
        self.records.extend([record("keep", decision="keep_original"), record("different", (200, 400))])
        self.records[1]["candidate_bbox"][0] = 60
        self.records[2]["layout_flags"] = ["nested_app_screenshot"]
        self.assertFalse(build_group_plan(self.records, "template")["targets"])
        self.assertFalse(build_group_plan(self.records, "a"))

    def test_stale_template_and_foreign_selection_rejected(self):
        plan = build_group_plan(self.records, "template")
        with self.assertRaises(ValueError):
            validate_group_selection(self.records, plan, ["unpreviewed"])
        self.records[0]["display_bbox"][1] += 1
        with self.assertRaises(ValueError):
            validate_group_selection(self.records, plan, ["a"])


class PhotoRecovery(unittest.TestCase):
    def test_weak_internal_boundary_expands_to_complete_photo(self):
        pixels = np.full((700, 500, 3), 240, dtype=np.uint8)
        pixels[100:620, 70:430] = np.random.default_rng(4).integers(40, 45, (520, 360, 3), dtype=np.uint8)
        result = expand_photo_region(Image.fromarray(pixels), [70, 100, 430, 480])
        self.assertIsNotNone(result)
        self.assertLessEqual(abs(result["bbox"][3] - 620), 2)
        self.assertTrue(all(a <= b for a, b in zip(result["bbox"][:2], [70, 100])))
        self.assertTrue(all(a >= b for a, b in zip(result["bbox"][2:], [430, 480])))

    def test_complete_photo_and_absent_proposal_are_preserved(self):
        pixels = np.full((700, 500, 3), 240, dtype=np.uint8)
        pixels[100:620, 70:430] = np.random.default_rng(4).integers(40, 190, (520, 360, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        self.assertIsNone(expand_photo_region(image, [70, 100, 430, 620]))
        self.assertIsNone(expand_photo_region(image, None))

    def test_uncertain_boundaries_cannot_be_automatically_cropped(self):
        analysis = {"confidence_level": "high", "crop_type": "uniform_canvas",
                    "candidate_bbox": [10, 10, 90, 90], "layout_flags": ["photo_boundary_uncertain"]}
        for mode in ("auto_high_review_medium", "auto_high_keep_medium"):
            self.assertIsNone(frames.automatic_crop_bbox(analysis, mode, ["uniform_canvas"]))


class GroupPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataset_curator_ui
        cls.ui = dataset_curator_ui

    def create_state(self, directory):
        records = []
        for i in range(3):
            path = Path(directory) / f"image{i}.png"
            Image.new("RGB", (200, 300), (40+i*50, 30, 80)).save(path)
            item = record(frames.file_sha1(str(path)), decision="manual" if i == 0 else "auto")
            item.update(source_path=str(path), filename=path.name)
            records.append(item)
        state = {"records": records, "output_root": directory, "cache_dir": directory,
                 "frame_mode": "suggest_only", "auto_accept_types": [],
                 "last_confirmed_crop_hash": records[0]["source_hash"]}
        frames.save_user_decision(directory, records[0]["source_hash"], records[0]["source_path"], "manual", records[0]["display_bbox"])
        return state

    def test_preview_uses_last_confirmed_after_auto_advance_then_saves_only_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.create_state(directory)
            profile = Path(directory) / "_subject_profile.json"
            profile.write_text('{}', encoding="utf-8")
            preview = self.ui.preview_frame_group_ui(state, state["records"][1]["source_hash"])
            self.assertEqual(len(preview[1]), 2)
            selected = state["records"][2]["source_hash"]
            result = self.ui.apply_frame_group_ui(state, preview[0], [selected], "review", "1")
            saved = frames.load_user_decisions(directory)["decisions"]
            self.assertIn(selected, saved)
            self.assertNotIn(state["records"][1]["source_hash"], saved)
            self.assertEqual(result[0]["records"][2]["user_decision"], "manual")
            self.assertEqual(json.loads(profile.read_text())["audit_stale_images"][0]["source_hash"], selected)
            self.assertEqual(state["records"][2]["user_decision"], "auto")

    def test_changed_source_prevents_all_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.create_state(directory)
            plan = build_group_plan(state["records"], state["records"][0]["source_hash"])
            Image.new("RGB", (200, 300), "red").save(state["records"][2]["source_path"])
            result = self.ui.apply_frame_group_ui(state, plan, [t["source_hash"] for t in plan["targets"]], "review", "1")
            self.assertEqual(len(frames.load_user_decisions(directory)["decisions"]), 1)
            self.assertTrue(all(r["user_decision"] == "auto" for r in result[0]["records"][1:]))


if __name__ == "__main__":
    unittest.main()
