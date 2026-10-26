"""Offline integration: persisted Frames decisions through real caption exports.

Detector proposals/audit rows are controlled fixtures; crop files, captions,
reports, profile guards and transactional export replacement are real.
"""
import builtins
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import frame_cleanup as frames


class FramesExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import-time config must never touch the user's saved dataset or key.
        cls.import_root = tempfile.TemporaryDirectory()
        project = Path(__file__).resolve().parent.parent
        source = project/'app'/'dataset_curator_v2.py'
        config_path = str(project/'_ui_config.json')
        config = json.dumps({'INPUT_FOLDER': cls.import_root.name,
                             'TRIGGER_WORD': 'offline', 'API_KEY': '',
                             'ENABLE_CACHE': False, 'USE_AI_TRIGGERWORD_CHECK': False})
        real_open, real_exists = builtins.open, os.path.exists

        def isolated_open(path, *args, **kwargs):
            return io.StringIO(config) if str(path) == config_path else real_open(path, *args, **kwargs)

        spec = importlib.util.spec_from_file_location('frames_export_backend', source)
        cls.backend = importlib.util.module_from_spec(spec)
        with patch('builtins.open', isolated_open), patch('os.path.exists',
                lambda path: True if str(path) == config_path else real_exists(path)):
            spec.loader.exec_module(cls.backend)

    @classmethod
    def tearDownClass(cls):
        cls.import_root.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root/'curated_offline'
        self.output.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        values = {'INPUT_FOLDER': str(self.root), 'OUTPUT_ROOT': str(self.output),
                  'CACHE_DIR': str(self.output/'_cache'), 'SAFE_TRIGGER': 'offline',
                  'SUBJECT_PROFILE_CACHE_DIR': str(self.output/'_cache'/'subject_profiles'),
                  'ENABLE_CACHE': False, 'USE_CONTROLLED_BUCKETS': False,
                  'TRAINING_TARGET': 'ernie', 'USE_KREA_AI_CAPTIONING': False,
                  'ENABLE_CANON_REPRESENTATION_BONUS': False, 'TARGET_DATASET_SIZE': 3,
                  'EXPORT_REJECT_IMAGES': False, 'CONTINUE_FROM_PROFILE': False,
                  'USE_AI_TRIGGERWORD_CHECK': False, '_ACTIVE_OUTPUT_TRANSACTION': None}
        for key, value in values.items():
            self.stack.enter_context(patch.object(self.backend, key, value))
        self.stack.enter_context(patch('requests.sessions.Session.request',
                                       side_effect=AssertionError('Network forbidden in offline export test')))
        self.stack.enter_context(patch.object(self.backend, 'safe_print'))
        self.addCleanup(self.backend.cancel_output_transaction)
        self.sources = []
        for index, color in enumerate(('red', 'green', 'blue')):
            path = self.root/f'image{index}.png'
            image = Image.new('RGB', (200, 300), 'white')
            image.paste(color, (20, 30, 180, 270))
            image.save(path)
            self.sources.append(path)
        self.originals = {path: path.read_bytes() for path in self.sources}

    def profile(self):
        return {'profile_schema_version': self.backend.PROFILE_CACHE_SCHEMA_VERSION,
                'canonical_features': {}, 'profile_variability_stats': {
                    'glasses': {}, 'beard_pattern': {}},
                'identity_markers': {'piercing_inventory': []}}

    def stage(self, mode='suggest_only'):
        survivors, rejects = frames.filter_excluded_sources(map(str, self.sources), str(self.output))
        rows = []
        proposal = {'candidate_bbox': [20, 30, 180, 270], 'confidence_level': 'high',
                    'crop_type': 'uniform_canvas', 'layout_flags': [], 'recommendation': 'auto_accept'}
        for path in survivors:
            with patch.object(frames, 'analyze_frame_cleanup', return_value=dict(proposal)):
                resolved = frames.resolve_frame_cleanup(path, frames.file_sha1(path),
                    str(self.output/'_cache'), str(self.output), mode)
            rows.append({'original_filename': Path(path).name, 'original_path': path,
                         'base_status': 'keep', 'final_status': 'keep', 'quality_total': 90,
                         'shot_type': 'medium', 'face_visible': True, 'status_notes': [],
                         'selected': True, **resolved})
        profile = self.profile()
        payload = {'all_rows': rows+rejects, 'selected_sorted': rows, 'reject_items': rejects,
                   'review_items': [], 'unselected_keep': [], 'subject_profile': profile,
                   'valid_candidate_count': len(rows)}
        (self.output/'_caption_stage.json').write_text(json.dumps(payload), encoding='utf-8')
        (self.output/'_subject_profile.json').write_text(json.dumps(profile), encoding='utf-8')
        return payload

    def export(self):
        self.backend.continue_caption_from_profile()
        result = {}
        for path in (self.output/'01_train_ready').glob('*.jpg'):
            with Image.open(path) as image:
                result[path.name] = image.size
            self.assertTrue(path.with_suffix('.txt').read_text(encoding='utf-8').strip())
        return result

    def test_export_replaces_old_files_and_restoration_uses_current_crop(self):
        self.stage()
        self.assertEqual(list(self.export().values()), [(200, 300)]*3)
        frames.save_user_decision(str(self.output), frames.file_sha1(str(self.sources[0])),
                                  str(self.sources[0]), 'exclude')
        frames.save_user_decision(str(self.output), frames.file_sha1(str(self.sources[1])),
                                  str(self.sources[1]), 'manual', [25, 35, 175, 265])
        self.stage('auto_high_review_medium')
        exported = self.export()
        self.assertEqual(sorted(exported.values()), [(150, 230), (160, 240)])
        self.assertNotIn('offline_003.jpg', exported)
        self.assertFalse((self.output/'01_train_ready'/'offline_003.txt').exists())
        frames.save_user_decision(str(self.output), frames.file_sha1(str(self.sources[0])),
                                  str(self.sources[0]), 'keep_original')
        frames.save_user_decision(str(self.output), frames.file_sha1(str(self.sources[1])),
                                  str(self.sources[1]), 'manual', [40, 50, 160, 250])
        self.stage('auto_high_review_medium')
        self.assertEqual(sorted(self.export().values()), [(120, 200), (160, 240), (200, 300)])
        for path, before in self.originals.items():
            self.assertEqual(path.read_bytes(), before)

    def test_changed_decision_blocks_stale_profile_and_preserves_previous_export(self):
        import dataset_curator_ui as ui
        self.stage()
        self.export()
        previous = {p.name: p.read_bytes() for p in (self.output/'01_train_ready').iterdir()}
        path = self.sources[0]
        record = {'source_hash': frames.file_sha1(str(path)), 'source_path': str(path),
                  'filename': path.name, 'original_size': [200, 300], 'user_decision': 'auto'}
        state = {'records': [record], 'output_root': str(self.output),
                 'cache_dir': str(self.output/'_cache'), 'frame_mode': 'suggest_only'}
        ui.set_frame_review_decision_ui(state, 'all', '1', record['source_hash'],
                                       'manual', None, 25, 35, 175, 265)
        with self.assertRaisesRegex(RuntimeError, 'Frame decision changed'):
            self.backend.continue_caption_from_profile()
        self.assertEqual(previous, {p.name: p.read_bytes() for p in (self.output/'01_train_ready').iterdir()})
        self.stage()
        self.assertIn((150, 230), self.export().values())

    def test_excluded_priority_and_training_role_cannot_restore_image(self):
        path = self.sources[0]
        frames.save_user_decision(str(self.output), frames.file_sha1(str(path)), str(path), 'exclude')
        _, rejects = frames.filter_excluded_sources([str(path)], str(self.output))
        row = rejects[0]
        row.update(priority_image=True, identity_cluster_role='core', identity_cluster_id='old-manual')
        profile = self.profile()
        profile['priority_image_filenames'] = [path.name]
        self.assertFalse(self.backend.is_priority_image(row, profile))
        self.assertEqual(self.backend.merge_priority_images_into_selection([], [row], profile), [])
        with patch.object(self.backend, 'apply_identity_cluster_roles_to_rows'):
            self.assertEqual(self.backend.rebuild_selection_from_identity_roles([row], profile), ([], [], []))

    def test_failed_export_preserves_last_complete_dataset(self):
        self.stage()
        self.export()
        previous = {p.name: p.read_bytes() for p in (self.output/'01_train_ready').iterdir()}
        with patch.object(self.backend, 'build_caption', side_effect=RuntimeError('Interrupted caption export')):
            with self.assertRaisesRegex(RuntimeError, 'Interrupted caption export'):
                self.backend.continue_caption_from_profile()
        self.backend.cancel_output_transaction('offline failure check')
        self.assertEqual(previous, {p.name: p.read_bytes() for p in (self.output/'01_train_ready').iterdir()})

    def test_exclusion_precedes_duplicate_selection_and_changes_cache_fingerprint(self):
        class ReachedDuplicates(Exception):
            pass
        path = self.sources[0]
        original_fp = self.backend.dataset_fingerprint(list(map(str, self.sources)))
        frames.save_user_decision(str(self.output), frames.file_sha1(str(path)), str(path), 'exclude')
        observed = []
        def dedup(paths):
            observed.extend(paths)
            raise ReachedDuplicates()
        # The previous run chose the now-excluded representative and discarded
        # another image. Reusing that cached selection would lose both.
        with patch.object(self.backend, 'ENABLE_CACHE', True), \
             patch.object(self.backend, 'apply_early_static_rejects', side_effect=lambda paths: (paths, [])), \
             patch.object(self.backend, 'USE_EARLY_PHASH_DEDUP', True), \
             patch.object(self.backend, 'USE_PHASH_DUPLICATE_SCORING', True), \
             patch.object(self.backend, 'early_phash_dedup', side_effect=dedup), \
             patch.object(self.backend, 'update_run_manifest'):
            settings_fp = self.backend.early_result_settings_fingerprint()
            self.backend.save_cached_early_results({
                'schema_version': self.backend.EARLY_RESULT_CACHE_SCHEMA_VERSION,
                'dataset_fingerprint': original_fp, 'settings_fingerprint': settings_fp,
                'survivor_paths': [str(path), str(self.sources[2])],
                'early_duplicate_paths': [str(self.sources[1])], 'early_reject_rows': []})
            with self.assertRaises(ReachedDuplicates):
                self.backend.main()
        self.assertEqual(set(observed), set(map(str, self.sources[1:])))
        self.assertNotEqual(original_fp, self.backend.dataset_fingerprint(observed))


if __name__ == '__main__':
    unittest.main()
