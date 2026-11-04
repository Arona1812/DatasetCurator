"""Offline regressions for saved slider values outside their range: python -m unittest tests.test_slider_bounds -v."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import dataset_curator_ui as ui


class SliderBounds(unittest.TestCase):
    def test_zu_kleiner_wert_wird_auf_das_minimum_gesetzt(self):
        with gr.Blocks() as blocks:
            slider = gr.Slider(label="Target dataset size", minimum=5, maximum=200, step=1, value=3)
        self.assertEqual(ui.clamp_slider_values(blocks), ["Target dataset size"])
        self.assertEqual(slider.value, 5)
        self.assertEqual(slider.preprocess(slider.value), 5)

    def test_zu_grosser_wert_wird_auf_das_maximum_gesetzt(self):
        with gr.Blocks() as blocks:
            slider = gr.Slider(label="Blur", minimum=5, maximum=200, step=5, value=500)
        ui.clamp_slider_values(blocks)
        self.assertEqual(slider.value, 200)

    def test_gueltige_werte_bleiben_unveraendert(self):
        with gr.Blocks() as blocks:
            slider = gr.Slider(label="Blur", minimum=5, maximum=200, step=5, value=25)
        self.assertEqual(ui.clamp_slider_values(blocks), [])
        self.assertEqual(slider.value, 25)

    def test_gespeicherte_datensatzgroesse_drei_startet_mit_fuenf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "_ui_settings.json"
            path.write_text(json.dumps({"c_target": 3}), encoding="utf-8")
            with mock.patch.object(ui, "SETTINGS_PATH", str(path)), \
                    mock.patch.object(ui, "LANG_OVERRIDE_PATH", str(Path(folder) / "_lang.json")):
                blocks = ui.build_ui()
        targets = [b for b in blocks.blocks.values() if isinstance(b, gr.Slider) and b.minimum == 5 and b.maximum == 200 and b.step == 1]
        self.assertEqual([b.value for b in targets], [5])


if __name__ == "__main__":
    unittest.main()
