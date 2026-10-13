"""Image target migration, shared audits and real offline export adapters."""
import json
import unittest
from unittest.mock import patch
from training_targets import IMAGE_TARGETS, normalize_training_target, caption_profile_for_training_target
from curator_core import normalize_run_config_payload, build_caption_fingerprint
import test_frames_export as export_fixtures


class ImageTargets(unittest.TestCase):
    def test_legacy_values_and_unknown_targets(self):
        for value, expected in [('shared_compact','ernie'),('Z-Image Base','z_image_base'),
                ('krea2_character','krea2'),('Krea 2 Raw','krea2'),('qwen_image_2','qwen_image_2_1')]:
            self.assertEqual(normalize_training_target(value),expected)
        for value in ['invented','ltx2.5','minimax_h3']:
            with self.assertRaises(ValueError): normalize_run_config_payload({'TRAINING_TARGET':value})
        self.assertEqual(normalize_training_target(''),'ernie')
        for key in IMAGE_TARGETS:
            config=normalize_run_config_payload({'TRAINING_TARGET':key})
            self.assertEqual(config['TRAINING_TARGET'],key)
            self.assertEqual(config['CAPTION_PROFILE'],caption_profile_for_training_target(key))


class SharedAuditsAndExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export_fixtures.FramesExport.setUpClass()

    @classmethod
    def tearDownClass(cls):
        export_fixtures.FramesExport.tearDownClass()

    def setUp(self):
        self.fixture=export_fixtures.FramesExport(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend

    def test_audit_and_profile_cache_keys_are_shared_but_export_config_is_not(self):
        audit_keys=[]; profile_keys=[]; config_keys=[]
        for key in IMAGE_TARGETS:
            with patch.object(self.backend,'TRAINING_TARGET',key):
                audit_keys.append(self.backend.audit_cache_key('same-source','same-model'))
                profile_keys.append(self.backend.subject_profile_settings_fingerprint())
            config_keys.append(normalize_run_config_payload({'TRAINING_TARGET':key})['RUN_CONFIG_FINGERPRINT'])
        self.assertEqual(len(set(audit_keys)),1)
        self.assertEqual(len(set(profile_keys)),1)
        self.assertEqual(len(set(config_keys)),len(IMAGE_TARGETS))
        self.assertNotEqual(self.backend.audit_cache_key('changed-source','same-model'),audit_keys[0])
        self.assertNotEqual(self.backend.audit_cache_key('same-source','changed-model'),audit_keys[0])
        self.assertNotEqual(self.backend.audit_cache_key('same-source','same-model',variant='crop'),audit_keys[0])
        with patch.object(self.backend,'TRIGGER_WORD','another-subject'):
            self.assertNotEqual(self.backend.audit_cache_key('same-source','same-model'),audit_keys[0])

    def test_ui_and_backend_agree_and_preserve_user_caption_switches(self):
        import dataset_curator_ui as ui
        for key in IMAGE_TARGETS:
            self.assertEqual(ui.caption_profile_for_training_target(key), self.backend.caption_profile_for_training_target(key))
            self.assertTrue(ui.get_caption_preset_values(key))
            policy=self.backend.enforce_caption_policy_profile(key,{'include_background':False})
            self.assertFalse(policy['include_background'])
        self.assertIn(('Krea 2 Raw','krea2'),ui.training_target_choices())
        self.assertIn(('Krea 2 Turbo','krea2_turbo'),ui.training_target_choices())

    def test_target_switch_changes_caption_preset_without_resetting_run_choices(self):
        import dataset_curator_ui as ui
        previous = (37, 0.2, 0.5, 0.3, 'chat-latest', 'medium', 'low',
                    'high', 'gpt-6-astra', 'medium', 'gpt-6-luna', 'none',
                    'single_pass')
        for target in IMAGE_TARGETS:
            with self.subTest(target=target):
                changed = ui.apply_training_target_defaults(target, [], *previous)
                self.assertEqual(changed[0], ui.get_caption_preset_values(target))
                self.assertEqual(changed[1:], previous)

    def test_all_image_targets_export_same_saved_stage_without_network_or_source_changes(self):
        self.fixture.stage()
        stage_before=(self.fixture.output/'_caption_stage.json').read_bytes()
        for key in IMAGE_TARGETS:
            with self.subTest(target=key), patch.object(self.backend,'TRAINING_TARGET',key), \
                 patch.object(self.backend,'CAPTION_PROFILE',caption_profile_for_training_target(key)), \
                 patch.object(self.backend,'USE_KREA_AI_CAPTIONING',False):
                exported=self.fixture.export()
                self.assertEqual(list(exported.values()),[(200,300)]*3)
                caption=(self.fixture.output/'01_train_ready'/'offline_001.txt').read_text(encoding='utf-8')
                if key=='ideogram4':
                    data=json.loads(caption)
                    self.assertIn('offline',data['high_level_description'])
                    self.assertEqual(data['compositional_deconstruction']['elements'][0]['type'],'obj')
                    self.assertNotIn('bbox',data['compositional_deconstruction']['elements'][0])
                else:
                    self.assertFalse(caption.startswith('{'))
                    self.assertIn('offline',caption)
                report=json.loads((self.fixture.output/'01_train_ready'/'image_target.json').read_text(encoding='utf-8'))
                self.assertEqual(report['key'],key)
                self.assertFalse(report['local_training_verified'])
        self.assertEqual((self.fixture.output/'_caption_stage.json').read_bytes(),stage_before)
        for path,before in self.fixture.originals.items(): self.assertEqual(path.read_bytes(),before)


if __name__=='__main__': unittest.main()
