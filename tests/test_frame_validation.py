"""Validation sampling and proposal-versus-automation regressions, offline."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from frame_cleanup import file_sha1
from frame_benchmark_sample import sample
from frame_benchmark_score import score
from frame_benchmark import build_review


class FrameValidation(unittest.TestCase):
    def test_sample_excludes_development_content_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, controls = root/'frames', root/'controls'
            frames.mkdir(); controls.mkdir()
            for folder, name, color in ((frames, 'confirmed.png', 'red'),
                    (frames, 'open.png', 'blue'), (controls, 'same_content.png', 'red'),
                    (controls, 'control.png', 'green')):
                Image.new('RGB', (100, 100), color).save(folder/name)
            exported = controls/'curated_test'
            exported.mkdir()
            Image.new('RGB', (100, 100), 'yellow').save(exported/'output.png')
            confirmed = root/'confirmed.json'
            confirmed.write_text(json.dumps({'records': [{'id': file_sha1(str(frames/'confirmed.png')),
                                                           'reference_confirmed': True}]}))
            selection = root/'selection.json'
            self.assertEqual(sample(frames, confirmed, controls, selection, 20),
                             {'unconfirmed_frames': 1, 'control_candidates': 1})
            images = json.loads(selection.read_text())['images']
            self.assertEqual({Path(r['source_path']).name for r in images}, {'open.png', 'control.png'})
            first = selection.read_bytes()
            sample(frames, confirmed, controls, selection, 20)
            self.assertEqual(first, selection.read_bytes())

    def test_modified_source_is_rejected_from_frozen_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'sources'; source.mkdir()
            image = source/'sample.png'
            Image.new('RGB', (100, 100), 'blue').save(image)
            selection = root/'selection.json'
            selection.write_text(json.dumps({'images': [{'source_path': str(image), 'id': file_sha1(str(image))}]}))
            Image.new('RGB', (100, 100), 'red').save(image)
            result = build_review(source, root/'output', selection=selection, blind=True)
            self.assertEqual(result['images'], 0)
            self.assertEqual(result['failures'], 1)

    def test_unnecessary_proposal_is_separate_from_automatic_crop(self):
        manifest = {'schema': 'frame-reference-v1', 'records': [
            {'id': 'suggested', 'filename': 'a.jpg', 'size': [100, 100], 'layout': 'uniform_canvas',
             'confidence_level': 'medium', 'local_bbox': [10, 10, 90, 90],
             'automatic_bbox': None, 'cohort': 'control_candidates'},
            {'id': 'automatic', 'filename': 'b.jpg', 'size': [100, 100], 'layout': 'uniform_canvas',
             'confidence_level': 'high', 'local_bbox': [10, 10, 90, 90],
             'automatic_bbox': [10, 10, 90, 90], 'cohort': 'control_candidates'}]}
        confirmed = {'schema': 'frame-reference-v1', 'records': [
            {'id': row['id'], 'size': [100, 100], 'reference_confirmed': True,
             'reference_decision': 'keep_original'} for row in manifest['records']]}
        result = score(manifest, confirmed)
        self.assertEqual(result['outcomes'], {'unnecessary_proposal': 2})
        self.assertEqual(result['automatic_outcomes'], {'no_automatic_crop': 1, 'unnecessary_automatic_crop': 1})
        self.assertEqual(result['by_cohort']['control_candidates'], {'unnecessary_proposal': 2})


if __name__ == '__main__':
    unittest.main()
