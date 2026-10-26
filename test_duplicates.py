"""CLIP duplicate rules with synthetic embeddings: python -m unittest test_duplicates -v."""
import unittest
from unittest import mock

import numpy as np

import dataset_curator_v2 as curator


def pair_with_similarity(similarity):
    a = np.zeros(8, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(8, dtype=np.float32)
    b[0], b[1] = similarity, float(np.sqrt(max(0.0, 1.0 - similarity ** 2)))
    return a, b


def row(name, embedding, clothing, quality, background="bathroom mirror"):
    return {"original_filename": name, "base_status": "keep", "quality_total": quality, "shot_type": "full_body",
            "clip_embedding": embedding, "clothing_description": clothing, "background_description": background}


class ClipDuplicates(unittest.TestCase):
    def mark(self, similarity, clothing_b, background_b="bathroom mirror"):
        a, b = pair_with_similarity(similarity)
        rows = [row("a.jpg", a, "black hoodie and shorts", 90), row("b.jpg", b, clothing_b, 80, background_b)]
        with mock.patch.multiple(curator, USE_CLIP_DUPLICATE_SCORING=True, USE_PHASH_DUPLICATE_SCORING=False,
                                 CLIP_HARD_DUPLICATE_THRESHOLD=0.985, CLIP_COSINE_THRESHOLD=0.96):
            curator.mark_duplicates(rows)
        return rows[1]

    def test_harte_schwelle_liegt_standardmaessig_ueber_der_weichen(self):
        a, b = pair_with_similarity(0.93)
        rows = [row("a.jpg", a, "black hoodie and shorts", 90), row("b.jpg", b, "white dress", 80, "living room")]
        with mock.patch.multiple(curator, USE_CLIP_DUPLICATE_SCORING=True, USE_PHASH_DUPLICATE_SCORING=False):
            curator.mark_duplicates(rows)
        self.assertGreater(curator.CLIP_HARD_DUPLICATE_THRESHOLD, curator.CLIP_COSINE_THRESHOLD)
        self.assertEqual(rows[1]["base_status"], "keep")

    def test_spiegelselfie_mit_anderem_outfit_bleibt_erhalten(self):
        # Frueher fiel jedes Paar ab 0.92 ohne Abgleich heraus.
        second = self.mark(0.93, "white dress", "living room")
        self.assertEqual(second["base_status"], "keep")

    def test_gleiche_szene_ab_weicher_schwelle_ist_duplikat(self):
        second = self.mark(0.97, "black hoodie and shorts")
        self.assertEqual(second["base_status"], "reject")
        self.assertEqual(second["duplicate_method"], "clip")

    def test_ab_regler_wert_immer_duplikat(self):
        second = self.mark(0.99, "white dress", "beach")
        self.assertEqual(second["duplicate_method"], "clip_hard")

    def test_grund_nennt_das_duplikat(self):
        second = self.mark(0.99, "white dress", "beach")
        self.assertIn("duplicate of a.jpg (CLIP 0.990)", curator.post_audit_reject_reason(second))


class FinalSetGuard(unittest.TestCase):
    def test_anderes_outfit_im_selben_raum_ist_kein_endset_duplikat(self):
        a, b = pair_with_similarity(0.92)
        first = row("a.jpg", a, "black hoodie and shorts", 90)
        second = row("b.jpg", b, "white summer dress", 85)
        with mock.patch.object(curator, "USE_CLIP_DUPLICATE_SCORING", True):
            duplicate, _reason, _score = curator.is_final_scene_duplicate(first, second)
        self.assertFalse(duplicate)

    def test_gleiches_outfit_und_sehr_aehnlich_ist_endset_duplikat(self):
        a, b = pair_with_similarity(0.92)
        first = row("a.jpg", a, "black hoodie and shorts", 90)
        second = row("b.jpg", b, "black hoodie and shorts", 85)
        with mock.patch.object(curator, "USE_CLIP_DUPLICATE_SCORING", True):
            duplicate, reason, _score = curator.is_final_scene_duplicate(first, second)
        self.assertTrue(duplicate)
        self.assertEqual(reason, "final_clip_scene")


if __name__ == "__main__":
    unittest.main()
