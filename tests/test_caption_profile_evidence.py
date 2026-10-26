"""Initial descriptions may corroborate audits but cannot become identity facts."""
import copy
import unittest
from unittest.mock import patch

from caption_profile_evidence import compare_caption, visible_claims
import test_frames_export as fixtures


class CaptionClaimParsing(unittest.TestCase):
    def test_positive_specific_claims_without_absence_or_generic_jewelry(self):
        claims = visible_claims(
            "A woman with long blonde hair and blue eyes is wearing round glasses. "
            "A nose stud and a tattoo are visible. A ring is on her finger.")
        self.assertEqual(claims["hair_color"], "blonde")
        self.assertEqual(claims["eye_color"], "blue_family")
        self.assertEqual(claims["glasses"], "present")
        self.assertEqual(claims["nose_piercing"], "present")
        self.assertEqual(claims["tattoo"], "present")
        self.assertNotIn("ear_jewelry", claims)
        self.assertNotIn("glasses", visible_claims("She has no glasses and wears a blue dress."))
        self.assertNotIn("eye_color", visible_claims("She has blue eyeshadow."))

    def test_ambiguous_hair_and_multiple_faces_are_not_supporting_evidence(self):
        self.assertNotIn("hair_color", visible_claims("One person has blonde hair, another has brown hair."))
        row = {"initial_caption": {"caption": "A woman with blonde hair and blue eyes.",
                                   "settings": {"engine": "joycaption"}}, "face_count_local": 2}
        evidence = compare_caption(row, audit_hair="blonde hair", audit_eye="blue", eye_reliable=True)
        self.assertEqual(evidence["skipped"], "multiple_faces")
        self.assertEqual(evidence["agreements"], {})


class BackendProfileCrosscheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FramesExport.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.FramesExport.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.FramesExport(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.backend = self.fixture.backend

    def row(self, engine="florence2", caption=None):
        return {"original_filename": "example.jpg", "file_hash": "abc",
                "base_status": "keep", "quality_total": 90, "shot_type": "headshot",
                "face_visible": True, "main_face_ratio": 0.12, "occlusion_type": "none",
                "hair_description": "long blonde hair", "eye_color": "blue",
                "eye_appearance": "natural_eyes", "has_glasses_now": False,
                "initial_caption": {"caption": caption or
                    "A woman with long blonde hair and green eyes is wearing round glasses. "
                    "A nose stud is visible.", "settings": {"engine": engine}}}

    def test_profile_receives_only_audited_agreements_and_records_conflicts(self):
        row = self.row()
        before = copy.deepcopy(row)
        evidence = self.backend.caption_profile_evidence_for_row(row)
        self.assertEqual(evidence["agreements"], {"hair_color": "blonde"})
        self.assertEqual(evidence["conflicts"]["eye_color"],
                         {"caption": "green", "audit": "blue_family"})
        self.assertEqual(evidence["conflicts"]["glasses"],
                         {"caption": "present", "audit": "absent"})
        self.assertEqual(evidence["unverified"]["nose_piercing"], "present")
        payload = self.backend._profile_sample_payload([row])[0]
        self.assertEqual(payload["initial_caption_agreements"], {"hair_color": "blonde"})
        self.assertNotIn("green", str(payload["initial_caption_agreements"]))
        self.assertEqual(row, before)
        profile = self.backend.attach_initial_caption_evidence_summary({}, [row])
        self.assertEqual(profile["initial_caption_evidence_summary"]["conflicts"],
                         {"eye_color": 1, "glasses": 1})
        self.assertEqual(profile["initial_caption_evidence_summary"]["providers"],
                         {"florence2": 1})

    def test_profile_cache_tracks_initial_caption_and_provider_without_target(self):
        row = self.row(caption="A woman with long blonde hair.")
        first = self.backend.profile_input_hash([row])
        row["initial_caption"]["settings"]["engine"] = "joycaption"
        self.assertNotEqual(first, self.backend.profile_input_hash([row]))
        row["initial_caption"]["settings"]["engine"] = "florence2"
        row["initial_caption"]["caption"] = "A woman with dark brown hair."
        self.assertNotEqual(first, self.backend.profile_input_hash([row]))
        row["initial_caption"]["caption"] = "A woman with long blonde hair."
        with patch.object(self.backend, "TRAINING_TARGET", "ideogram4"):
            self.assertEqual(first, self.backend.profile_input_hash([row]))

    def test_profile_build_persists_crosscheck_without_replacing_image_facts(self):
        row = self.row()
        row["original_path"] = str(self.fixture.sources[0])
        before = (row["eye_color"], row["hair_description"], row["has_glasses_now"])
        with patch.object(self.backend, "call_subject_profile_normalizer",
                          side_effect=lambda rows, input_hash, total_count:
                          self.backend.fallback_subject_profile(rows, input_hash)):
            profile = self.backend.build_subject_profile([row])
        self.assertEqual(profile["initial_caption_evidence_summary"]["conflicts"],
                         {"eye_color": 1, "glasses": 1})
        self.assertEqual((row["eye_color"], row["hair_description"], row["has_glasses_now"]), before)
        self.assertTrue((self.fixture.output / "_subject_profile.json").is_file())


if __name__ == '__main__':
    unittest.main()
