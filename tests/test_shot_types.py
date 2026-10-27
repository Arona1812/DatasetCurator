"""Offline regressions for consistent shot types: python -m unittest tests.test_shot_types -v."""
import unittest

from shot_types import apply_shot_type, reconcile_shot_type


def row(shot, subtype="", face_height=None, **extra):
    data = {"shot_type": shot, "frame_subtype": subtype, "face_visible": face_height is not None}
    if face_height is not None:
        data["face_bbox_ai"] = [0.3, 0.1, face_height * 0.8, face_height]
    data.update(extra)
    return data


class ShotTypeReconciliation(unittest.TestCase):
    def test_selfie_ab_brust_mit_grossem_gesicht_ist_headshot(self):
        self.assertEqual(reconcile_shot_type(row("medium", "selfie", 0.42)), "headshot")

    def test_close_up_ist_immer_headshot(self):
        self.assertEqual(reconcile_shot_type(row("medium", "close_up")), "headshot")

    def test_dreiviertel_aufnahme_ist_medium(self):
        self.assertEqual(reconcile_shot_type(row("full_body", "three_quarter_body", 0.15)), "medium")

    def test_dreiviertel_ohne_gesicht_ist_medium(self):
        self.assertEqual(reconcile_shot_type(row("full_body", "three_quarter_body")), "medium")

    def test_full_body_mit_grossem_gesicht_ist_medium(self):
        self.assertEqual(reconcile_shot_type(row("full_body", "mirror_selfie", 0.22)), "medium")

    def test_echter_ganzkoerper_bleibt_full_body(self):
        self.assertEqual(reconcile_shot_type(row("full_body", "full_body", 0.09)), "full_body")

    def test_huefte_aufwaerts_bleibt_medium(self):
        self.assertEqual(reconcile_shot_type(row("medium", "portrait", 0.27)), "medium")

    def test_ohne_gesichtsmass_bleibt_der_auditwert(self):
        self.assertEqual(reconcile_shot_type(row("medium", "selfie")), "medium")

    def test_lokale_gesichtsbox_dient_als_ersatz(self):
        data = row("medium", "portrait", main_face_bbox=[100, 50, 300, 400], height=1000)
        data["face_visible"] = True
        self.assertEqual(reconcile_shot_type(data), "headshot")

    def test_auditwert_bleibt_erhalten_und_zweiter_aufruf_aendert_nichts(self):
        data = row("medium", "selfie", 0.42)
        apply_shot_type(data, from_audit=True)
        apply_shot_type(data)
        self.assertEqual(data["shot_type"], "headshot")
        self.assertEqual(data["audit_shot_type"], "medium")
        self.assertEqual(data["status_notes"], ["shot_type_medium_to_headshot"])

    def test_neues_audit_ersetzt_die_alte_korrektur(self):
        data = row("medium", "selfie", 0.42)
        apply_shot_type(data, from_audit=True)
        data.update({"shot_type": "full_body", "frame_subtype": "full_body", "face_bbox_ai": [0.4, 0.1, 0.07, 0.09]})
        apply_shot_type(data, from_audit=True)
        self.assertEqual(data["shot_type"], "full_body")
        self.assertNotIn("audit_shot_type", data)


if __name__ == "__main__":
    unittest.main()
