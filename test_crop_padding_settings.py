"""Offline regressions for the headshot padding setting: python -m unittest test_crop_padding_settings -v."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dataset_curator_ui as ui


class CropPaddingSettings(unittest.TestCase):
    def _load(self, saved):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "_ui_settings.json"
            if saved is not None:
                path.write_text(json.dumps(saved), encoding="utf-8")
            with mock.patch.object(ui, "SETTINGS_PATH", str(path)), \
                    mock.patch.object(ui, "LANG_OVERRIDE_PATH", str(Path(folder) / "_lang.json")):
                return ui.load_settings()

    def test_neue_installation_nutzt_die_empfohlene_polsterung(self):
        self.assertEqual(self._load(None)["c_crop_pad"], 0.6)

    def test_alter_standardwert_wird_auf_die_empfehlung_umgestellt(self):
        self.assertEqual(self._load({"c_crop_pad": 1.5})["c_crop_pad"], 0.6)

    def test_eigene_werte_aus_alten_dateien_bleiben_erhalten(self):
        self.assertEqual(self._load({"c_crop_pad": 0.8})["c_crop_pad"], 0.8)

    def test_bewusst_gespeicherte_eins_komma_fuenf_bleibt_erhalten(self):
        saved = {"c_crop_pad": 1.5, "c_crop_pad_revision": ui.CROP_PAD_REVISION}
        self.assertEqual(self._load(saved)["c_crop_pad"], 1.5)



class ModelPresetSaving(unittest.TestCase):
    def test_modelle_uebernehmen_speichert_sie_dauerhaft(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "_ui_settings.json"
            path.write_text(json.dumps({"c_model": "chat-latest", "c_trigger": "Nina"}), encoding="utf-8")
            with mock.patch.object(ui, "SETTINGS_PATH", str(path)), \
                    mock.patch.object(ui, "LANG_OVERRIDE_PATH", str(Path(folder) / "_lang.json")):
                result = ui.apply_and_save_model_preset("economical")
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(result), len(ui.MODEL_PRESET_SETTING_KEYS) + 1)
        self.assertEqual(saved["c_model"], result[0])
        self.assertEqual(saved["c_trigger"], "Nina")
        self.assertTrue(result[-1].startswith("💾"))


if __name__ == "__main__":
    unittest.main()
