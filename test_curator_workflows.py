"""Offline regressions: run with python -m unittest test_curator_workflows -v."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

import frame_cleanup as frames
from openai_models import model_preset, prepare_response_payload
from vision_frames import attach_photo_proposal, propose_photo_region, refine_to_edges, validate_proposal


def analysis(level="high", crop_type="uniform_canvas"):
    return {"confidence_level": level, "candidate_bbox": [10, 10, 90, 90],
            "crop_type": crop_type, "layout_flags": [], "recommendation": "auto_accept"}


class ModelRequests(unittest.TestCase):
    def test_sol_maps_none_to_low_and_removes_sampling(self):
        payload = {"_reasoning_effort": "none", "temperature": .1, "top_p": .9,
                   "top_logprobs": 2, "input": [{"role": "user", "content": "test"}]}
        result = prepare_response_payload("gpt-6.1-sol", payload)
        self.assertEqual(result["reasoning"], {"effort": "low"})
        self.assertNotIn("temperature", result)
        self.assertNotIn("top_p", result)
        self.assertIn("temperature", payload)

    def test_luna_preserves_none(self):
        result = prepare_response_payload("gpt-6-luna", {"_reasoning_effort": "none", "temperature": .1})
        self.assertEqual(result["reasoning"]["effort"], "none")
        self.assertEqual(result["temperature"], .1)

    def test_chatgpt_alias_has_no_reasoning(self):
        result = prepare_response_payload("chat-latest", {"_reasoning_effort": "high", "temperature": .1})
        self.assertNotIn("reasoning", result)
        self.assertNotIn("temperature", result)
        self.assertEqual(model_preset("chatgpt")[0], "chat-latest")

    def test_max_remains_supported_on_gpt6(self):
        self.assertEqual(prepare_response_payload("gpt-6-astra", {"_reasoning_effort": "max"})["reasoning"]["effort"], "max")


class FramePolicy(unittest.TestCase):
    def test_automation_and_queue_agree(self):
        for mode in ("suggest_only", "auto_high_review_medium", "auto_high_keep_medium"):
            for level in ("high", "medium"):
                for allowed in ([], ["uniform_canvas"]):
                    item = analysis(level)
                    self.assertEqual(frames.frame_needs_review(item, mode, allowed), frames.automatic_crop_bbox(item, mode, allowed) is None)

    def test_decisions_leave_queue(self):
        for decision in ("accept", "manual", "keep_original"):
            self.assertFalse(frames.frame_needs_review(analysis(), "suggest_only", [], decision))

    def test_nested_layout_never_automatic(self):
        item = analysis()
        item["layout_flags"] = ["nested_inner_image_detected"]
        self.assertIsNone(frames.automatic_crop_bbox(item, "auto_high_keep_medium", ["uniform_canvas"]))
        self.assertTrue(frames.frame_needs_review(item, "auto_high_keep_medium", ["uniform_canvas"]))

    def test_semantic_alternative_blocks_automatic_crop(self):
        item = analysis()
        attach_photo_proposal(item, {"decision": "crop", "bbox": [20, 20, 80, 80]})
        self.assertIsNone(frames.automatic_crop_bbox(item, "auto_high_review_medium", ["uniform_canvas", "semantic_photo"]))

    def test_cached_semantic_review_reaches_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source.png"
            Image.new("RGB", (100, 100), "blue").save(source)
            digest = frames.file_sha1(str(source))
            from curator_core import atomic_write_json
            atomic_write_json(str(Path(directory)/"vision_frames"/f"{digest}_current.json"),
                              {"source_hash": digest, "prompt_version": "photo-region-v1", "decision": "manual_review", "bbox": [0, 0, 100, 100]})
            with patch.object(frames, "analyze_frame_cleanup", return_value=analysis()):
                result = frames.resolve_frame_cleanup(str(source), digest, directory, directory, "auto_high_review_medium")
            self.assertTrue(result["frame_cleanup_review_pending"])
            self.assertFalse(result["frame_cleanup_applied"])

    def test_original_unchanged_and_user_crop_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "blue").save(source)
            before = source.read_bytes()
            digest = frames.file_sha1(str(source))
            frames.save_user_decision(directory, digest, str(source), "manual", [5, 5, 95, 95])
            with patch.object(frames, "analyze_frame_cleanup", return_value=analysis()):
                result = frames.resolve_frame_cleanup(str(source), digest, directory, directory, "suggest_only")
            self.assertEqual(result["frame_cleanup_bbox"], [5, 5, 95, 95])
            self.assertFalse(result["frame_cleanup_review_pending"])
            self.assertEqual(source.read_bytes(), before)
            with Image.open(result["effective_image_path"]) as crop:
                self.assertEqual(crop.size, (90, 90))


class SemanticRegions(unittest.TestCase):
    def test_invalid_and_destructive_boxes_rejected(self):
        for bbox in ([500, 0, 400, 1000], [-1, 0, 900, 1000], [0, 0, 50, 50], [True, 0, 1000, 1000]):
            with self.assertRaises(ValueError):
                validate_proposal({"decision": "crop", "bbox": bbox}, (1000, 1000))

    def test_full_image_becomes_keep(self):
        result = validate_proposal({"decision": "crop", "bbox": [0, 0, 1000, 1000]}, (1200, 800))
        self.assertEqual(result["decision"], "keep_original")

    def test_keep_original_does_not_create_a_crop(self):
        item = analysis("low")
        item["recommendation"] = "keep_original"
        attach_photo_proposal(item, {"decision": "keep_original", "bbox": [0, 0, 100, 100]})
        self.assertFalse(item.get("candidates"))
        self.assertFalse(frames.frame_needs_review(item, "auto_high_review_medium", []))

    def test_edge_refinement(self):
        image = Image.new("RGB", (500, 500), "white")
        image.paste(Image.new("RGB", (400, 400), "black"), (50, 50))
        self.assertEqual(refine_to_edges(image, [53, 48, 448, 453], image.size), [50, 50, 450, 450])

    def test_cache_prevents_repeated_paid_request(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "image.png"
            Image.new("RGB", (500, 500), "blue").save(source)
            post = Mock(return_value=Mock(status_code=200, json=lambda: {
                "status": "completed", "usage": {"total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps({"decision": "crop", "bbox": [100, 100, 900, 900], "reason": "External border"})}]}],
            }))
            first = propose_photo_region(str(source), "testhash", directory, "test-key", post=post)
            second = propose_photo_region(str(source), "testhash", directory, "", post=post)
            self.assertEqual(first["bbox"], [50, 50, 450, 450])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(post.call_count, 1)

    def test_incomplete_response_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"image.png"
            Image.new("RGB", (500, 500), "blue").save(source)
            post = Mock(return_value=Mock(status_code=200, json=lambda: {"status": "incomplete"}))
            with self.assertRaises(ValueError):
                propose_photo_region(str(source), "testhash", directory, "test-key", post=post)
            self.assertFalse(list(Path(directory).glob("vision_frames/*.json")))


class UIWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataset_curator_ui
        cls.ui = dataset_curator_ui

    def test_undecided_original_can_be_explicitly_selected(self):
        record = {"user_decision": "auto", "source_path": "missing", "original_size": [100, 100], "candidates": []}
        _, radio = self.ui._frame_option_gallery_update(record)
        self.assertIsNone(radio["value"])
        record["user_decision"] = "keep_original"
        _, confirmed = self.ui._frame_option_gallery_update(record)
        self.assertEqual(confirmed["value"], "original")

    def test_save_and_next_does_not_skip_pending_image(self):
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for i in range(3):
                source = Path(directory) / f"image{i}.png"
                Image.new("RGB", (100, 100), (i*60, 30, 90)).save(source)
                records.append({**analysis("medium"), "source_hash": str(i), "source_path": str(source),
                                "filename": source.name, "original_size": [100, 100], "user_decision": "auto", "candidates": []})
            state = {"records": records, "output_root": directory, "cache_dir": directory, "frame_mode": "suggest_only", "auto_accept_types": []}
            result = self.ui.save_frame_and_navigate_ui(state, "review", "1", "0", "original", True)
            self.assertEqual(len(result), 16)
            self.assertEqual(result[4], "1")
            self.assertEqual(len(self.ui._frame_filter_records(result[0], "review")), 2)
            last = self.ui.save_frame_and_navigate_ui(result[0], "review", "1", "1", "original", True)
            done = self.ui.save_frame_and_navigate_ui(last[0], "review", "1", "2", "original", True)
            self.assertEqual(done[4], "")
            self.assertFalse(self.ui._frame_filter_records(done[0], "review"))

    def test_local_project_does_not_require_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(self.ui, "SETTINGS_PATH", str(Path(directory)/"settings.json")):
            status, _, ready = self.ui.initialize_workspace_ui("example", directory, "", True, "auto_high_review_medium", ["uniform_canvas"], False, True)
            self.assertTrue(ready, status)

    def test_two_corner_crop_maps_to_original_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"image.png"
            Image.new("RGB", (2400, 1200), "blue").save(source)
            state = {"records": [{"source_hash": "test", "source_path": str(source), "original_size": [2400, 1200]}]}
            first = self.ui.select_crop_corner_ui(state, "test", None, Mock(index=(100, 50)))
            second = self.ui.select_crop_corner_ui(state, "test", first[1], Mock(index=(1000, 500)))
            self.assertEqual(list(second[2:6]), [200, 100, 2000, 1000])
            self.assertIsNone(second[1])

    def test_build_preserves_full_option_set(self):
        with patch.object(self.ui, "load_settings", return_value=dict(self.ui.DEFAULTS)):
            app = self.ui.build_ui()
        self.assertGreater(len(app.config["components"]), 450)
        self.assertGreater(len(app.config["dependencies"]), 50)


if __name__ == "__main__":
    unittest.main()
