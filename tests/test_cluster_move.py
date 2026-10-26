"""Offline regressions for moving several images between identity buckets."""
import unittest

import dataset_curator_ui as ui


def profile():
    return {
        "identity_clusters": [
            {"cluster_id": "core_a", "cluster_kind": "appearance", "role": "core", "n": 3,
             "members": [1, 2, 3], "asset_ids": [1, 2, 3],
             "filenames": ["a.jpg", "b.jpg", "c.jpg"], "image_paths": ["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"]},
            {"cluster_id": "var_b", "cluster_kind": "appearance", "role": "variation", "n": 1,
             "members": [4], "asset_ids": [4], "filenames": ["d.jpg"], "image_paths": ["/x/d.jpg"]},
            {"cluster_id": "audited_rejects", "cluster_kind": "audited_rejects", "role": "exclude", "n": 1,
             "members": [5], "asset_ids": [5], "filenames": ["e.jpg"], "image_paths": ["/x/e.jpg"],
             "reject_reasons": ["blur"]},
        ]
    }


def refs(prof, cluster_id, positions):
    cluster = ui._identity_cluster_by_id(prof, cluster_id)
    records = ui._cluster_member_records(cluster)
    return [ui._selection_ref_for_record(records[i], i) for i in positions]


class MoveClusterMembers(unittest.TestCase):
    def test_several_images_move_into_one_new_bucket_with_chosen_role(self):
        prof = profile()
        prof, moved, dest, error = ui.move_cluster_members(prof, "core_a", refs(prof, "core_a", [0, 2]), ui.CLUSTER_MOVE_NEW_BUCKET, "exclude")
        self.assertEqual((moved, error), (2, ""))
        new = ui._identity_cluster_by_id(prof, dest)
        self.assertEqual(new["role"], "exclude")
        self.assertEqual(new["filenames"], ["a.jpg", "c.jpg"])
        self.assertEqual(ui._identity_cluster_by_id(prof, "core_a")["filenames"], ["b.jpg"])
        self.assertEqual(prof["identity_cluster_member_clusters"]["1"], dest)

    def test_images_move_into_an_existing_bucket(self):
        prof = profile()
        prof, moved, dest, error = ui.move_cluster_members(prof, "core_a", refs(prof, "core_a", [1]), "var_b", "")
        self.assertEqual((moved, dest, error), (1, "var_b", ""))
        self.assertEqual(ui._identity_cluster_by_id(prof, "var_b")["asset_ids"], [4, 2])

    def test_emptied_source_bucket_disappears(self):
        prof = profile()
        prof, moved, _dest, _error = ui.move_cluster_members(prof, "var_b", refs(prof, "var_b", [0]), "core_a", "")
        self.assertEqual(moved, 1)
        self.assertIsNone(next((c for c in prof["identity_clusters"] if c["cluster_id"] == "var_b"), None))

    def test_moving_into_reject_bucket_keeps_reason_list_aligned(self):
        prof = profile()
        prof, _moved, _dest, _error = ui.move_cluster_members(prof, "core_a", refs(prof, "core_a", [0]), "audited_rejects", "")
        rejects = ui._identity_cluster_by_id(prof, "audited_rejects")
        self.assertEqual(len(rejects["reject_reasons"]), len(rejects["members"]))

    def test_nothing_checked_changes_nothing(self):
        prof = profile()
        prof, moved, _dest, error = ui.move_cluster_members(prof, "core_a", [], "var_b", "")
        self.assertEqual((moved, error), (0, "nothing_selected"))
        self.assertEqual(ui._identity_cluster_by_id(prof, "core_a")["n"], 3)


if __name__ == "__main__":
    unittest.main()
