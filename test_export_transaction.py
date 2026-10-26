"""Offline regressions for the export swap on locked folders: python -m unittest test_export_transaction -v."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import curator_core


def _locked_staging(failures):
    """os.replace that fails ``failures`` times while the staged folder is locked."""
    real_replace = os.replace
    state = {"left": failures}

    def fake(source, target):
        if "._export_staging_" in str(source) and state["left"] > 0:
            state["left"] -= 1
            raise PermissionError(13, "Zugriff verweigert", str(source))
        return real_replace(source, target)
    return fake


class ExportTransaction(unittest.TestCase):
    def _transaction(self, folder):
        final = Path(folder) / "01_train_ready"
        final.mkdir()
        (final / "old.jpg").write_bytes(b"old")
        (final / "old.txt").write_text("old", encoding="utf-8")
        tx = curator_core.OutputTransaction(str(folder), {"TRAIN_READY_DIR": str(final)})
        namespace = {"TRAIN_READY_DIR": str(final)}
        tx.begin(namespace)
        staged = Path(namespace["TRAIN_READY_DIR"])
        (staged / "new.jpg").write_bytes(b"new")
        (staged / "new.txt").write_text("new", encoding="utf-8")
        return tx, namespace, final

    def test_kurz_gesperrter_ordner_wird_nach_wartezeit_uebernommen(self):
        with tempfile.TemporaryDirectory() as folder:
            tx, namespace, final = self._transaction(folder)
            with mock.patch.object(curator_core.os, "replace", _locked_staging(2)), \
                    mock.patch.object(curator_core.time, "sleep") as sleep:
                tx.commit(namespace)
            self.assertEqual(sorted(os.listdir(final)), ["new.jpg", "new.txt"])
            self.assertEqual(sleep.call_count, 2)

    def test_dauerhaft_gesperrter_ordner_stellt_den_alten_stand_wieder_her(self):
        with tempfile.TemporaryDirectory() as folder:
            tx, namespace, final = self._transaction(folder)
            with mock.patch.object(curator_core.os, "replace", _locked_staging(99)), \
                    mock.patch.object(curator_core.time, "sleep"):
                with self.assertRaises(PermissionError) as raised:
                    tx.commit(namespace)
            self.assertIn("Dropbox", str(raised.exception))
            self.assertEqual(sorted(os.listdir(final)), ["old.jpg", "old.txt"])

    def test_andere_fehler_werden_nicht_wiederholt(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(curator_core.time, "sleep") as sleep:
                with self.assertRaises(FileNotFoundError):
                    curator_core.replace_dir_with_retry(str(Path(folder) / "missing"), str(Path(folder) / "x"))
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
