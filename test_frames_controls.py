"""Offline persistence and UI checks for Frames actions and economical vision."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import frame_cleanup as frames
from vision_frames import propose_photo_region


class FramesControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataset_curator_ui
        cls.ui = dataset_curator_ui

    def state(self, directory):
        records = []
        for index in range(3):
            source = Path(directory) / f"sample{index}.png"
            Image.new("RGB", (200, 300), (index*70, 20, 90)).save(source)
            box = [20, 30, 180, 270]
            records.append({"source_hash": frames.file_sha1(str(source)), "source_path": str(source),
                            "filename": source.name, "original_size": [200, 300], "user_decision": "auto",
                            "candidate_bbox": box, "display_bbox": box, "confidence_level": "high",
                            "crop_type": "uniform_canvas", "layout_class": "uniform_canvas", "layout_flags": [],
                            "candidates": [{"bbox": box, "crop_type": "uniform_canvas", "confidence": .95}]})
        records[1]["layout_flags"] = ["photo_boundary_uncertain"]
        return {"records": records, "output_root": directory, "cache_dir": directory,
                "frame_mode": "suggest_only", "auto_accept_types": ["uniform_canvas"]}

    def test_exclusion_persists_filters_before_audit_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(directory)
            source = state["records"][0]
            before = Path(source["source_path"]).read_bytes()
            frames.save_user_decision(directory, source["source_hash"], source["source_path"], "exclude")
            paths = [r["source_path"] for r in state["records"]]
            survivors, rejects = frames.filter_excluded_sources(paths, directory)
            self.assertNotIn(source["source_path"], survivors)
            self.assertFalse(rejects[0]["selected"])
            self.assertEqual(rejects[0]["final_status"], "reject")
            self.assertEqual(Path(source["source_path"]).read_bytes(), before)
            frames.save_user_decision(directory, source["source_hash"], source["source_path"], "auto")
            self.assertEqual(frames.filter_excluded_sources(paths, directory), (paths, []))

    def test_excluded_image_is_never_cropped_by_automation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(directory)
            source = state["records"][0]
            frames.save_user_decision(directory, source["source_hash"], source["source_path"], "exclude")
            with patch.object(frames, "analyze_frame_cleanup", return_value=source), patch.object(frames, "materialize_crop") as crop:
                result = frames.resolve_frame_cleanup(source["source_path"], source["source_hash"], directory, directory, "auto_high_keep_medium")
            self.assertTrue(result["dataset_excluded"])
            self.assertFalse(result["frame_cleanup_review_pending"])
            crop.assert_not_called()

    def test_exclusion_advances_and_can_be_reinstated_without_deleting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(directory)
            profile = Path(directory) / '_subject_profile.json'
            profile.write_text('{}')
            selected = state["records"][0]["source_hash"]
            result = self.ui.exclude_frame_and_navigate_ui(state, "review", "1", selected, True)
            updated = result[0]
            self.assertEqual(result[4], state["records"][1]["source_hash"])
            self.assertEqual(len(self.ui._frame_filter_records(updated, "excluded")), 1)
            self.assertEqual(len(self.ui._frame_filter_records(updated, "detected")), 2)
            self.assertEqual(self.ui._frame_current_option_key(updated["records"][0], updated), "original")
            self.assertTrue(json.loads(profile.read_text())["profile_rebuild_required"])
            restored = self.ui.set_frame_review_decision_ui(updated, "all", "1", selected, "keep_original", None, 0, 0, 0, 0)[0]
            self.assertFalse(self.ui._frame_filter_records(restored, "excluded"))

    def test_automation_updates_workspace_and_respects_manual_and_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(directory)
            state["records"][2]["user_decision"] = "keep_original"
            settings = Path(directory)/'settings.json'
            profile = Path(directory)/'_subject_profile.json'
            profile.write_text('{}')
            with patch.object(self.ui, "SETTINGS_PATH", str(settings)), patch.object(self.ui, "load_settings", return_value={}):
                updated, summary, message = self.ui.set_frame_automation_ui(state, "auto_high_review_medium", ["uniform_canvas"], "review")
            self.assertEqual(len(self.ui._frame_filter_records(updated, "review")), 1)
            self.assertEqual(json.loads((Path(directory)/'_project_workspace.json').read_text())["frame"]["mode"], "auto_high_review_medium")
            self.assertEqual(json.loads(settings.read_text())["c_frame_cleanup_mode"], "auto_high_review_medium")
            self.assertEqual(json.loads(profile.read_text())["audit_stale_images"][0]["source_hash"], state["records"][0]["source_hash"])
            _, radio = self.ui._frame_option_gallery_update(updated["records"][0], updated)
            self.assertEqual(radio["value"], "candidate:0")
            _, uncertain = self.ui._frame_option_gallery_update(updated["records"][1], updated)
            self.assertIsNone(uncertain["value"])
            self.assertEqual(updated["records"][2]["user_decision"], "keep_original")

    def test_luna_default_uses_no_reasoning_and_separate_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.jpg'
            Image.new('RGB', (500, 500), 'blue').save(source)
            response = Mock(status_code=200, json=lambda: {
                "status": "completed", "output": [{"content": [{"type": "output_text", "text": json.dumps({
                    "decision": "keep_original", "bbox": [0, 0, 1000, 1000], "reason": "Full photo"})}]}]})
            post = Mock(return_value=response)
            propose_photo_region(str(source), 'sourcehash', directory, 'fake-test-key', post=post)
            payload = post.call_args.kwargs['json']
            self.assertEqual(payload['model'], 'gpt-6-luna')
            self.assertEqual(payload['reasoning']['effort'], 'none')
            self.assertEqual(payload['max_output_tokens'], 1024)
            propose_photo_region(str(source), 'sourcehash', directory, 'fake-test-key', model='gpt-6.1-sol', post=post)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args.kwargs['json']['reasoning']['effort'], 'low')

    def test_empty_automatic_type_selection_does_not_enable_silent_noop_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(directory)
            result = self.ui.set_frame_automation_ui(state, 'auto_high_review_medium', [], 'review')
            self.assertEqual(result[0]['frame_mode'], 'suggest_only')
            self.assertFalse((Path(directory)/'_project_workspace.json').exists())


if __name__ == '__main__':
    unittest.main()
