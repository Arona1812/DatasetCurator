"""Separate first descriptions, profile edits and real offline export/preview."""
import json
import tempfile
import unittest
import inspect
from unittest.mock import patch

import initial_captions as initial
import test_frames_export as fixtures
from curator_core import normalize_run_config_payload


class InitialCaptionCache(unittest.TestCase):
    def test_nur_luna_ist_anbieter_alte_lokale_werte_wechseln_zu_luna(self):
        self.assertEqual(normalize_run_config_payload({"INITIAL_CAPTION_ENGINE": "gpt-6-luna"})["INITIAL_CAPTION_ENGINE"], "gpt-6-luna")
        for legacy in ["florence2", "joycaption"]:
            data = normalize_run_config_payload({"INITIAL_CAPTION_ENGINE": legacy, "INITIAL_CAPTION_FLORENCE_TASK": "<CAPTION>"})
            self.assertEqual(data["INITIAL_CAPTION_ENGINE"], "gpt-6-luna")
            self.assertNotIn("INITIAL_CAPTION_FLORENCE_TASK", data)
        for invalid in ["qwen", "gpt-6.1-sol", "unknown"]:
            with self.assertRaises(ValueError):
                normalize_run_config_payload({"INITIAL_CAPTION_ENGINE": invalid})

    def test_gespeicherte_luna_beschreibungen_bleiben_gueltig(self):
        # Settings of earlier releases for Luna; a changed value would invalidate every stored description.
        earlier = {"version": "initial-caption-v1", "engine": "gpt-6-luna", "revision": "", "prompt": initial.PROMPT,
                   "max_new_tokens": 4096, "do_sample": False, "seed": 20261012, "quantization": "api",
                   "image_preparation": "exif-rgb-full-resolution-v1"}
        self.assertEqual(initial.settings(), earlier)

    def test_geaenderte_bytes_und_abgeschnittene_texte_werden_nicht_genutzt(self):
        with tempfile.TemporaryDirectory() as directory:
            config = initial.settings()
            raw = {"status": "ok", "caption": "A person standing outdoors.", "raw_output": "raw", "hit_token_limit": False}
            record = initial.store_caption(directory, "first-image", config, raw)
            self.assertEqual(initial.cached_caption(directory, "first-image", config), record)
            self.assertIsNone(initial.cached_caption(directory, "different-image", config))
            for bad in [dict(raw, caption=""), dict(raw, hit_token_limit=True), dict(raw, status="error")]:
                with self.assertRaises(ValueError): initial.store_caption(directory, "first-image", config, bad)


class CaptionSettings(unittest.TestCase):
    def test_ui_eingaben_und_einstellungen_passen_zusammen(self):
        import dataset_curator_ui as ui
        with patch.object(ui, "load_settings", return_value=dict(ui.DEFAULTS)):
            app = ui.build_ui()
        start = next(block for block in app.fns.values() if getattr(block.fn, "__name__", "") == "start_curator")
        values = tuple(component.value for component in start.inputs)
        self.assertEqual(len(values), len(ui.CURATOR_INPUT_NAMES))
        config = ui.build_run_config_from_ui_values(values)
        self.assertEqual(config["INITIAL_CAPTION_ENGINE"], "gpt-6-luna")
        self.assertNotIn("INITIAL_CAPTION_FLORENCE_TASK", config)
        self.assertEqual(config["CAPTION_COMPOSITION_MODE"], "local")
        self.assertFalse(config["USE_KREA_AI_CAPTIONING"])
        save = next(block for block in app.fns.values() if getattr(block.fn, "__name__", "") == "save_settings_fn")
        arguments = inspect.signature(ui.save_settings_fn).bind(*(component.value for component in save.inputs)).arguments
        self.assertNotIn("c_initial_caption_engine", arguments)
        self.assertEqual(arguments["c_caption_composition_mode"], "local")


class TwoStageExports(unittest.TestCase):
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

    def prepared_stage(self):
        self.fixture.stage()
        path = self.fixture.output / "_caption_stage.json"
        stage = json.loads(path.read_text(encoding="utf-8"))
        for row in stage["all_rows"]:
            row["initial_caption"] = initial.store_caption(self.fixture.output / "_cache" / "initial_captions",
                initial.image_sha256(self.backend.row_effective_image_path(row)), initial.settings("gpt-6-luna"),
                {"status": "ok", "caption": "Unfiltered raw description with brown eyes, a red wool jacket, standing by a window in soft daylight.", "hit_token_limit": False})
            row["clothing_description"] = "a red jacket"
        stage["selected_sorted"] = stage["all_rows"]
        path.write_text(json.dumps(stage), encoding="utf-8")
        return stage

    def test_cached_first_caption_survives_profile_target_and_trigger_changes(self):
        stage = self.prepared_stage()
        rows = stage["all_rows"]
        original = rows[0]["initial_caption"]
        with patch.object(self.backend, "INITIAL_CAPTION_ENGINE", "gpt-6-luna"), \
             patch.object(self.backend, "responses_api_call", side_effect=AssertionError("No new recognition")), \
             patch.object(self.backend, "TRIGGER_WORD", "changed"), \
             patch.object(self.backend, "TRAINING_TARGET", "flux2_dev"):
            self.backend.prepare_initial_captions(rows)
        self.assertEqual(rows[0]["initial_caption"], original)

    def test_export_preserves_raw_sidecar_and_composes_without_image_calls(self):
        stage = self.prepared_stage()
        with patch.object(self.backend, "INITIAL_CAPTION_ENGINE", "gpt-6-luna"), \
             patch.object(self.backend, "responses_api_call", side_effect=AssertionError("No new model calls")):
            self.fixture.export()
        folder = self.fixture.output / "01_train_ready"
        sidecar = json.loads((folder / "offline_001.caption.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["initial_caption"], stage["all_rows"][0]["initial_caption"])
        self.assertEqual(sidecar["final_caption"], (folder / "offline_001.txt").read_text(encoding="utf-8"))
        self.assertNotIn("Unfiltered", sidecar["final_caption"])
        for path, before in self.fixture.originals.items(): self.assertEqual(path.read_bytes(), before)

    def test_source_change_blocks_continue_before_replacing_exports(self):
        self.prepared_stage()
        ready = self.fixture.output / "01_train_ready"
        ready.mkdir(exist_ok=True)
        sentinel = ready / "preserved.txt"
        sentinel.write_text("old export", encoding="utf-8")
        self.fixture.sources[0].write_bytes(self.fixture.sources[1].read_bytes())
        with self.assertRaises(ValueError): self.fixture.export()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "old export")

    def test_unsaved_profile_preview_matches_export_and_does_not_write(self):
        import dataset_curator_ui as ui
        stage = self.prepared_stage()
        before = (self.fixture.output / "_subject_profile.json").read_bytes()
        values = ["woman", "", "blue", "", "", "", "blonde", "", "", False, "", False, "", []]
        policy = ["include_clothing", "include_gender_class"]
        preview = ui.preview_profile_caption("offline", str(self.fixture.root), json.dumps(stage["subject_profile"]),
            stage["all_rows"][0]["original_filename"], "ernie", policy, "canonical_deviations", "local", False, *values)
        self.assertTrue(preview[2], preview[3])
        self.assertIn("red jacket", preview[2])
        self.assertEqual(preview[0].size, (200, 300))
        self.assertEqual((self.fixture.output / "_subject_profile.json").read_bytes(), before)
        profile = ui.apply_profile_editor_values(stage["subject_profile"], *values)
        with patch.object(self.backend, "CAPTION_POLICY", {key: key in policy for key in ui.CAPTION_FIELD_CHOICES}):
            self.assertEqual(preview[2], self.backend.build_caption(stage["all_rows"][0], {}, profile))
