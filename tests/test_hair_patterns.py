"""Multi-colored hair: detection, profile canon and captions: python -m unittest tests.test_hair_patterns -v."""
import unittest
from unittest import mock

import dataset_curator_ui as ui
import dataset_curator_v2 as curator

POLICY = {"include_hair_when_variable": True, "include_hair_always": False}


def detect(text):
    row = {"hair_description": text}
    return curator.canonical_hair_color(row), curator.canonical_hair_color_modifier(row), curator.canonical_hair_accent_color(row)


class Detection(unittest.TestCase):
    def test_weisse_straehnen_vorn_wie_rogue(self):
        self.assertEqual(detect("long black hair with white front streaks"), ("black", "front_streaks", "white"))

    def test_money_piece_veraendert_die_grundfarbe_nicht(self):
        self.assertEqual(detect("dark brown hair with platinum money piece"), ("dark_brown", "front_streaks", "platinum"))

    def test_zweigeteilte_haare(self):
        base, pattern, accent = detect("split-dyed hair, half blonde and half black")
        self.assertEqual(pattern, "split_dye")
        self.assertEqual({base, accent}, {"blonde", "black"})

    def test_lose_straehnen_ohne_farbe_sind_nur_frisur(self):
        self.assertEqual(detect("blonde hair, loose strands framing her face")[1], "")

    def test_highlights_bleiben_wie_bisher(self):
        self.assertEqual(detect("brown hair with blonde highlights"), ("brown", "blonde_highlights", ""))


class Captions(unittest.TestCase):
    def state(self, text, pattern="none", accent=""):
        profile = {"canonical_features": {"hair_color": "black", "hair_pattern": pattern, "hair_accent_color": accent}}
        traits = {"hair_color_base": curator.canonical_hair_color({"hair_description": text}), "hair_form": ""}
        with mock.patch.object(curator, "VARIABLE_FEATURE_CAPTION_MODE", "canonical_deviations"):
            return curator.get_hair_feature_state({"hair_description": text}, profile, traits, {}, POLICY, "krea2_character")

    def test_kanonische_straehnen_gehoeren_zum_trigger(self):
        state = self.state("long black hair with white front streaks", "front_streaks", "white")
        self.assertFalse(state["must_caption"])

    def test_fehlende_kanonische_straehnen_werden_genannt(self):
        state = self.state("long black hair", "front_streaks", "white")
        self.assertEqual(state["phrase"], "black hair without white front streaks")

    def test_straehnen_als_abweichung_werden_genannt(self):
        self.assertEqual(self.state("long black hair with white front streaks")["phrase"], "black hair with white front streaks")

    def test_zweigeteilte_haare_als_abweichung(self):
        self.assertIn("split-dyed", self.state("half black, half blonde hair")["phrase"])


class ProfileCanon(unittest.TestCase):
    def traits(self, texts):
        rows = {}
        for index, text in enumerate(texts):
            row = {"hair_description": text}
            rows[str(index)] = {"hair_color_base": curator.canonical_hair_color(row),
                                "hair_color_modifier": curator.canonical_hair_color_modifier(row),
                                "hair_color_accent": curator.canonical_hair_accent_color(row)}
        return {"per_image_traits": rows}

    def test_muster_in_der_mehrheit_wird_kanon(self):
        profile = curator.attach_profile_variability_policies(
            self.traits(["black hair with white front streaks"] * 3 + ["black hair"]), [])
        self.assertEqual(profile["canonical_features"]["hair_pattern"], "front_streaks")
        self.assertEqual(profile["canonical_features"]["hair_accent_color"], "white")

    def test_seltenes_muster_ist_kein_kanon(self):
        profile = curator.attach_profile_variability_policies(
            self.traits(["black hair with white front streaks"] + ["black hair"] * 3), [])
        self.assertEqual(profile["canonical_features"]["hair_pattern"], "none")

    def test_editor_wert_bleibt_bestaetigt(self):
        profile = ui.apply_profile_editor_values({}, "woman", "", "", "", "", "", "black", "", "", False, "", False, "", [],
                                                 "split_dye", "blonde")
        self.assertEqual(profile["canonical_features"]["hair_pattern"], "split_dye")
        self.assertEqual(profile["user_confirmed_identity_overrides"]["canonical_features"]["hair_accent_color"], "blonde")


if __name__ == "__main__":
    unittest.main()
