"""Natural sentences for training captions: python -m unittest tests.test_caption_grammar -v."""
import unittest

import caption_grammar as grammar
import caption_writer as writer
from curator_core import normalize_run_config_payload


class PoseSentences(unittest.TestCase):
    def satz(self, pose, expression="", gaze=""):
        return " ".join(grammar.pose_sentences("She", pose, expression, gaze))

    def test_finites_verb_bekommt_kein_is(self):
        self.assertEqual(self.satz("stands on a balcony", "neutral, composed", "faces the camera"),
                         "She stands on a balcony, with a neutral and composed expression, facing the camera.")

    def test_kein_doppelter_artikel_beim_ausdruck(self):
        self.assertNotIn("a a", self.satz("seated in a car", "a relaxed expression and a small smile"))

    def test_ausdruck_als_satz_wird_eigener_satz(self):
        self.assertIn("She smiles broadly with her teeth showing.",
                      self.satz("standing by a wall", "smiles broadly with her teeth showing"))

    def test_blick_ohne_verb_bekommt_looking(self):
        self.assertIn("looking toward the lens", self.satz("seated close to the camera", "neutral", "directed toward the lens"))
        self.assertIn("looking toward the phone", self.satz("", "neutral", "toward the phone"))

    def test_their_wird_zum_pronomen(self):
        self.assertTrue(self.satz("Their head is slightly tilted").startswith("Her head is slightly tilted."))

    def test_doppelter_blick_entfaellt(self):
        self.assertEqual(self.satz("facing the camera with her head tilted", "", "facing the camera"),
                         "She is facing the camera with her head tilted.")


    def test_ausdruck_mit_lippen_wird_sauber(self):
        self.assertIn("with a pensive expression and pursed lips", self.satz("seated in a chair", "pensive, lips pursed"))


class IdentityAndScene(unittest.TestCase):
    def test_merkmale_stehen_in_einem_ganzen_satz_statt_als_tag_liste(self):
        sentence = grammar.identity_sentence("She", [
            {"kind": "piercing", "phrase": "nose ring"}, {"kind": "piercing", "phrase": "ear gauge"},
            {"kind": "makeup", "phrase": "dramatic eyeliner"}, {"kind": "accessory", "phrase": "necklace"}])
        self.assertEqual(sentence, "She has a nose ring and an ear gauge, and wears dramatic eyeliner and a necklace.")

    def test_satzanfaenge_werden_gross_geschrieben(self):
        self.assertEqual(grammar.descriptive_sentence("soft light. the background is dim", "She"),
                         "Soft light. The background is dim.")

    def test_reiner_farbhintergrund_wird_zum_satz(self):
        self.assertEqual(grammar.descriptive_sentence("Plain and light gray", "She", background=True),
                         "The background is plain and light gray.")


class WriterRules(unittest.TestCase):
    def facts(self, **omitted):
        return {"trigger": "Wiktoria", "caption_profile": "krea2_character", "training_target": "krea2",
                "photo_type": "portrait photo", "visual_style": "", "gender_class": "", "anchor": [],
                "pronoun": "She", "identity": [{"kind": "makeup", "phrase": "dramatic eyeliner"}],
                "scene": {"clothing": "black top"},
                "scene_rules": {"expression": True, "gaze": True, "lighting": True, "background": False},
                "omitted": {"piercings": False, "makeup": False, "tattoos": True, **omitted}, "feature_states": {}}

    def test_regelkonforme_caption_hat_keine_fehler(self):
        caption = ("Wiktoria, a portrait photo. She wears dramatic eyeliner and a black top, standing close "
                   "to the camera with a calm expression in soft light.")
        self.assertEqual(writer.rule_errors(caption, self.facts()), [])

    def test_fehlende_pflichtphrase_und_falscher_anfang_werden_gemeldet(self):
        errors = writer.rule_errors("A portrait of Wiktoria in a black top, standing in soft daylight near a window today.", self.facts())
        self.assertTrue(any("start with" in e for e in errors))
        self.assertTrue(any("dramatic eyeliner" in e for e in errors))

    def test_ausgeschlossene_piercings_werden_gemeldet(self):
        caption = ("Wiktoria, a portrait photo. She wears dramatic eyeliner and a nose ring, standing close to "
                   "the camera in soft light.")
        self.assertIn("piercings must be omitted by the caption rules", writer.rule_errors(caption, self.facts(piercings=True)))

    def test_anfrage_nennt_pflichtphrasen_und_verbietet_abgeschaltete_szenenteile(self):
        request = writer.writer_request(self.facts(), "Krea 2 Raw", "A woman in a black top.")
        self.assertEqual(request["payload"]["must_include"], ["dramatic eyeliner"])
        self.assertIn("Do not describe: background and surroundings.", request["instructions"])
        self.assertIn("Start with exactly this text: Wiktoria, a portrait photo", request["instructions"])

    def test_alter_textdetail_modus_wird_zum_writer(self):
        self.assertEqual(normalize_run_config_payload({"CAPTION_COMPOSITION_MODE": "luna_text"})["CAPTION_COMPOSITION_MODE"],
                         "luna_writer")
        with self.assertRaises(ValueError):
            normalize_run_config_payload({"CAPTION_COMPOSITION_MODE": "invented"})



class HairRule(unittest.TestCase):
    def test_haar_in_der_pose_ist_erlaubt_haarbeschreibung_nicht(self):
        import dataset_curator_v2 as curator
        self.assertFalse(curator._describes_hair("Wiktoria, a photo. One hand is raised near her hair."))
        self.assertTrue(curator._describes_hair("Her top is mostly obscured by long hair."))
        self.assertTrue(curator._describes_hair("Her red wavy hair falls over her shoulders."))


if __name__ == "__main__":
    unittest.main()
