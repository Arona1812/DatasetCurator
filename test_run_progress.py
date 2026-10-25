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

    def test_balken_bildet_alle_schritte_des_laufs_ab(self):
        script = textwrap.dedent("""
            print("[STAGE 1/2] Audit")
            print("[1/2] a.jpg")
            print("[2/2] b.jpg")
            print("[STAGE 2/2] Captions & export")
            print("Export ready: 1/2 | x_001")
            print("Export ready: 2/2 | x_002")
        """)
        with tempfile.TemporaryDirectory() as folder:
            script_path = Path(folder) / "fake_run.py"
            script_path.write_text(script, encoding="utf-8")
            with mock.patch.object(ui, "VENV_PYTHON", sys.executable), \
                    mock.patch.object(ui, "SCRIPT_DIR", folder), \
                    mock.patch.object(ui, "ACTIVE_RUN_STATE_PATH", str(Path(folder) / "state.json")):
                updates = list(ui.run_script(str(script_path), str(Path(folder) / "cfg.json"), {}))
        progress = [round(u[2], 2) for u in updates]
        for expected in (0.25, 0.5, 0.75, 1.0):
            self.assertIn(expected, progress)
        self.assertLessEqual(progress.index(0.5), progress.index(0.75))
        self.assertTrue(any(ui.tr("Schritt 2/2", "Step 2/2") in str(u[3]) for u in updates))


class InsightFaceWarning(unittest.TestCase):
    def test_veraltete_insightface_warnung_wird_ausgeblendet(self):
        import warnings

        import dataset_curator_v2  # noqa: F401  (installs the filter)

        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("`estimate` is deprecated since version 0.26 and will be removed", FutureWarning)
        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
