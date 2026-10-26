"""Skipping the frame module after preflight: python -m unittest tests.test_frame_skip -v."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dataset_curator_ui as ui


class FrameSkip(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.input = Path(folder.name)
        self.output = Path(ui.output_root_for(str(self.input), "probe"))
        self.output.mkdir(parents=True)
        self.workspace_path = self.output / "_project_workspace.json"
        self.workspace_path.write_text(json.dumps({
            "preflight": {"completed": True},
            "frame": {"enabled": True, "mode": "auto_high_review_medium"},
        }), encoding="utf-8")

    def start(self):
        captured = {}

        def fake_run(_script, _config_path, config, _train_dir):
            captured["config"] = config
            yield "started", [], 0, "running", ""

        values = {"input_folder": str(self.input), "trigger_word": "probe"}
        with mock.patch.object(ui, "_validate_run_values", return_value=(values, None)), \
                mock.patch.object(ui, "_workspace_preflight_is_current", return_value=True), \
                mock.patch.object(ui, "build_run_config_from_ui_values", return_value={"ENABLE_IG_FRAME_CROP": True}), \
                mock.patch.object(ui, "run_script", side_effect=fake_run):
            updates = list(ui.start_curator())
        return updates, captured.get("config")

    def test_ohne_analyse_und_ohne_ueberspringen_bleibt_das_audit_gesperrt(self):
        updates, config = self.start()
        self.assertIsNone(config)
        self.assertIn(ui.tr("Rahmen überspringen", "Skip frames"), updates[0][0])
        self.assertTrue(updates[0][3].startswith("❌"))

    def test_uebersprungene_rahmen_starten_das_audit_mit_originalbildern(self):
        status, _summary = ui.skip_frame_module_ui("probe", str(self.input))
        self.assertTrue(status.startswith("⏭"))
        saved = json.loads(self.workspace_path.read_text(encoding="utf-8"))
        self.assertFalse(saved["frame"]["enabled"])
        self.assertEqual(saved["frame"]["mode"], "auto_high_review_medium")
        _updates, config = self.start()
        self.assertFalse(config["ENABLE_IG_FRAME_CROP"])

    def test_ueberspringen_ohne_projekt_meldet_fehler(self):
        status, _summary = ui.skip_frame_module_ui("", str(self.input))
        self.assertTrue(status.startswith("❌"))


if __name__ == "__main__":
    unittest.main()
