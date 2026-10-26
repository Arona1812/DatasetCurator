"""Luna caption writer on a real offline export, with simulated responses only:
python -m unittest test_caption_writer -v."""
import json
import unittest
from unittest.mock import patch

import test_frames_export as fixtures
from caption_grammar import caption_opening


def response(caption):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps({"caption": caption})}]}]}


class CaptionWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FramesExport.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.FramesExport.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.FramesExport(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.backend = self.fixture.backend
        self.stage = self.fixture.stage()
        self.row = self.stage["all_rows"][0]
        self.profile = self.stage["subject_profile"]

    def good_caption(self):
        facts = self.backend.collect_caption_facts(self.row, {}, self.profile)
        required = " ".join(f"{e['phrase']}." for e in facts["identity"])
        return " ".join(f"{caption_opening(facts)}. {required} The subject stands upright near a plain wall, "
                        "framed from the head to the waist in soft even light.".split())

    def writer_mode(self, **extra):
        return patch.multiple(self.backend, CAPTION_COMPOSITION_MODE="luna_writer", API_KEY="test-only", **extra)

    def test_writer_sieht_bild_und_regeln_und_speichert_das_ergebnis(self):
        calls = []
        good = self.good_caption()

        def fake(model, payload, phase_label=""):
            calls.append(payload)
            return response(good)

        with self.writer_mode(ENABLE_CACHE=True), patch.object(self.backend, "responses_api_call", side_effect=fake):
            first = self.backend.build_caption(self.row, {}, self.profile)
            again = self.backend.build_caption(self.row, {}, self.profile)
            stored = self.backend.cached_written_caption(self.row, {}, self.profile)
        self.assertEqual(first, good)
        self.assertEqual(again, good)
        self.assertEqual(stored, good)
        self.assertEqual(len(calls), 1)
        content = calls[0]["input"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["input_text", "input_image"])
        self.assertIn("must_include", json.loads(content[0]["text"]))

    def test_regelverstoss_bekommt_einen_reparaturversuch(self):
        answers = iter([response("A short caption without the trigger."), response(self.good_caption())])
        with self.writer_mode(), patch.object(self.backend, "responses_api_call", side_effect=lambda *a, **k: next(answers)):
            caption = self.backend.build_caption(self.row, {}, self.profile)
        self.assertEqual(caption, self.good_caption())
        self.assertEqual(self.row["caption_source"], "written_repair")

    def test_dauerhafter_regelverstoss_nutzt_die_lokale_caption(self):
        with self.writer_mode(), patch.object(self.backend, "responses_api_call", return_value=response("Wrong.")):
            caption = self.backend.build_caption(self.row, {}, self.profile)
        self.assertEqual(caption, self.backend.build_local_caption(self.row, {}, self.profile))
        self.assertEqual(self.row["caption_source"], "local_fallback")

    def test_lokaler_modus_fragt_kein_modell(self):
        with patch.object(self.backend, "responses_api_call", side_effect=AssertionError("No model call")):
            caption = self.backend.build_caption(self.row, {}, self.profile)
        self.assertTrue(caption.startswith("A "))
        self.assertNotIn(", ,", caption)

    def test_export_ohne_api_schluessel_bricht_vor_dem_austausch_ab(self):
        ready = self.fixture.output / "01_train_ready"
        ready.mkdir(exist_ok=True)
        sentinel = ready / "preserved.txt"
        sentinel.write_text("old export", encoding="utf-8")
        with patch.multiple(self.backend, CAPTION_COMPOSITION_MODE="luna_writer", API_KEY=""):
            with self.assertRaisesRegex(ValueError, "API key"):
                self.fixture.export()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "old export")


if __name__ == "__main__":
    unittest.main()
