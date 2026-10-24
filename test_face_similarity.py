"""Face similarity before selection, with synthetic embeddings only:
python -m unittest test_face_similarity -v."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import dataset_curator_v2 as curator


def vector(seed, base=None, noise=0.3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    v /= np.linalg.norm(v)
    if base is not None:
        v = base + noise * v
    return v / np.linalg.norm(v)


PERSON = vector(1)
OTHER = vector(2)


class FaceSimilarity(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.embeddings = {}

    def row(self, name, embedding, **extra):
        path = self.root / f"{name}.jpg"
        path.write_bytes(name.encode())
        self.embeddings[name] = embedding
        return {"original_filename": f"{name}.jpg", "original_path": str(path), "effective_image_path": str(path),
                "file_hash": name, "base_status": "keep", "shot_type": "headshot", "quality_total": 80, **extra}

    def check(self, rows, reference=None):
        with mock.patch.object(curator, "HAVE_INSIGHTFACE", True), \
                mock.patch.object(curator, "USE_ARCFACE_IDENTITY_CHECK", True), \
                mock.patch.object(curator, "compute_arcface_embedding", side_effect=lambda _p, h: self.embeddings[h]):
            return curator.run_identity_consistency_check(rows, reference_rows=reference)

    def test_andere_person_wird_vor_der_auswahl_erkannt(self):
        rows = [self.row(f"a{i}", vector(10 + i, PERSON)) for i in range(8)] + [self.row("b", OTHER)]
        summary = self.check(rows)
        self.assertEqual(summary["reference"], "all_candidates")
        self.assertEqual(rows[-1]["arcface_flag"], "hard")
        self.assertTrue(all(r["arcface_flag"] == "ok" for r in rows[:-1]))
        self.assertTrue(curator.face_outlier_blocked(rows[-1]))

    def test_favoriten_und_core_sind_die_referenz(self):
        # Most images show another person; the user's favorites decide who is right.
        rows = [self.row(f"o{i}", vector(30 + i, OTHER)) for i in range(6)]
        favorites = [self.row(f"p{i}", vector(50 + i, PERSON), priority_image=True) for i in range(3)]
        summary = self.check(rows + favorites, reference=favorites)
        self.assertEqual(summary["reference"], "priority_core")
        self.assertTrue(all(r["arcface_flag"] == "hard" for r in rows))
        self.assertTrue(all(r["arcface_flag"] == "ok" for r in favorites))

    def test_aehnlichkeit_wirkt_als_bonus_und_abzug(self):
        close = {"arcface_flag": "ok", "arcface_distance_to_centroid": 0.9}
        borderline = {"arcface_flag": "soft", "arcface_distance_to_centroid": 0.55}
        self.assertEqual(curator.identity_similarity_bonus(close), curator.IDENTITY_SIMILARITY_BONUS_MAX)
        self.assertLess(curator.identity_similarity_bonus(borderline), 0)
        self.assertEqual(curator.identity_similarity_bonus({"arcface_flag": "skipped"}), 0.0)

    def test_freigabe_durch_rolle_oder_favorit(self):
        outlier = {"arcface_flag": "hard", "identity_cluster_role": "review"}
        self.assertTrue(curator.face_outlier_blocked(outlier))
        self.assertFalse(curator.face_outlier_blocked(dict(outlier, identity_cluster_role="variation")))
        self.assertFalse(curator.face_outlier_blocked(dict(outlier, priority_image=True)))


class FaceOutlierBucket(unittest.TestCase):
    def profile(self):
        return {"identity_clusters": [
            {"cluster_id": "look_a", "role": "core", "members": ["1", "2", "3"], "asset_ids": [1, 2, 3],
             "filenames": ["a.jpg", "b.jpg", "c.jpg"], "image_paths": ["/a", "/b", "/c"], "n": 3}]}

    def rows(self, flagged):
        return [{"asset_id": i, "original_filename": f"{n}.jpg", "original_path": f"/{n}", "base_status": "keep",
                 "arcface_flag": "hard" if n in flagged else "ok", "arcface_distance_to_centroid": 0.4 if n in flagged else 0.8}
                for i, n in ((1, "a"), (2, "b"), (3, "c"))]

    def test_ausreisser_kommt_als_review_in_eigenen_bucket(self):
        profile = curator.attach_face_outlier_cluster_to_profile(self.profile(), self.rows({"b"}))
        clusters = {c["cluster_id"]: c for c in profile["identity_clusters"]}
        self.assertEqual(clusters["look_a"]["members"], ["1", "3"])
        self.assertEqual(clusters[curator.FACE_OUTLIER_CLUSTER_ID]["role"], "review")
        self.assertEqual(clusters[curator.FACE_OUTLIER_CLUSTER_ID]["members"], ["2"])
        self.assertEqual(profile["identity_cluster_member_roles"]["2"], "review")

    def test_vom_nutzer_verschobenes_bild_bleibt_wo_es_ist(self):
        profile = curator.attach_face_outlier_cluster_to_profile(self.profile(), self.rows({"b"}))
        # The user moves the image back into the core bucket.
        clusters = {c["cluster_id"]: c for c in profile["identity_clusters"]}
        clusters["look_a"]["members"].append("2")
        profile["identity_clusters"] = [clusters["look_a"]]
        again = curator.attach_face_outlier_cluster_to_profile(profile, self.rows({"b"}))
        self.assertEqual(again["identity_cluster_member_roles"]["2"], "core")
        self.assertNotIn(curator.FACE_OUTLIER_CLUSTER_ID, [c["cluster_id"] for c in again["identity_clusters"]])

    def test_kein_ausreisser_mehr_verlaesst_den_bucket(self):
        profile = curator.attach_face_outlier_cluster_to_profile(self.profile(), self.rows({"b"}))
        again = curator.attach_face_outlier_cluster_to_profile(profile, self.rows(set()))
        self.assertNotIn(curator.FACE_OUTLIER_CLUSTER_ID, [c["cluster_id"] for c in again["identity_clusters"]])
        self.assertEqual(again["face_outlier_assigned"], [])


if __name__ == "__main__":
    unittest.main()
