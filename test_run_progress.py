"""Offline regressions for the live progress bar: python -m unittest test_run_progress -v."""
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import dataset_curator_ui as ui


class ProgressParsing(unittest.TestCase):
    def test_bildzeile_zaehlt_als_fortschritt(self):
        self.assertEqual(ui.parse_progress("[12/80] image.jpg"), (12, 80))

    def test_textdetails_zaehlen_als_fortschritt(self):
        self.assertEqual(ui.parse_progress("Text details ready: 27/92 | a.jpg"), (27, 92))

    def test_erstbeschreibungen_zaehlen_als_fortschritt(self):
        self.assertEqual(ui.parse_progress("Initial caption ready: 3/72 | a.jpg"), (3, 72))

    def test_unterphasen_zaehlen_nicht(self):
        self.assertIsNone(ui.parse_progress("   START [12/80] local_subject_metrics a.jpg"))
        self.assertIsNone(ui.parse_progress("Text details ready. Refresh the preview."))


class RunProgress(unittest.TestCase):
    def test_balken_folgt_der_caption_stufe(self):
        script = textwrap.dedent("""
            print("[2/2] b.jpg")
            for i in range(1, 5):
                print(f"Text details ready: {i}/4 | x.jpg")
        """)
        with tempfile.TemporaryDirectory() as folder:
            script_path = Path(folder) / "fake_stage.py"
            script_path.write_text(script, encoding="utf-8")
            with mock.patch.object(ui, "VENV_PYTHON", sys.executable), \
                    mock.patch.object(ui, "SCRIPT_DIR", folder), \
                    mock.patch.object(ui, "ACTIVE_RUN_STATE_PATH", str(Path(folder) / "state.json")):
                updates = list(ui.run_script(str(script_path), str(Path(folder) / "cfg.json"), {}))
        progress = [u[2] for u in updates]
        self.assertIn(0.5, progress)
        self.assertIn(1.0, progress)
        self.assertTrue(any("2/4" in str(u[3]) for u in updates))


if __name__ == "__main__":
    unittest.main()
