"""Offline checks of image evidence -> profile -> actual caption policy."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import test_frames_export as export_fixtures


class ProfileCaptionPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export_fixtures.FramesExport.setUpClass()
        cls.backend = export_fixtures.FramesExport.backend

    @classmethod
    def tearDownClass(cls):
        export_fixtures.FramesExport.tearDownClass()

    def setUp(self):
        self.fixture = export_fixtures.FramesExport(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.b = self.backend

    def row(self, **changes):
        row = {'original_filename': 'example.jpg', 'base_status': 'keep',
               'face_visible': True, 'main_face_ratio': 0.1,
               'occlusion_type': 'none', 'eye_color': 'blue',
               'eye_appearance': 'natural_eyes', 'has_glasses_now': False,
               'shot_type': 'headshot', 'hair_description': 'blonde straight hair'}
        row.update(changes)
        return row

    def profile(self, rows, **changes):
        profile = {'stable_identity': {'eye_color': 'green'},
                   'identity_markers': {}, 'canonical_features': {}}
        profile.update(changes)
        profile['per_image_traits'] = {
            str(i): self.b.per_image_profile_traits(row, profile)
            for i, row in enumerate(rows)
        }
        return self.b.attach_profile_variability_policies(profile, rows)

    def test_blond_family_wins_by_full_frequency_and_raw_evidence_survives(self):
        rows = [self.row(hair_description=color + ' hair')
                for color in ['blonde'] * 20 + ['platinum'] * 20 + ['dark blonde'] * 20 + ['brown'] * 30]
        profile = self.profile(rows)
        self.assertEqual(profile['canonical_features']['hair_color'], 'blonde')
        self.assertEqual(profile['profile_variability_stats']['hair_color']['counts'], {'blonde': 60, 'brown': 30})
        self.assertEqual(profile['per_image_traits']['20']['hair_color_base'], 'platinum')
        state = self.b.get_hair_feature_state(rows[20], profile, profile['per_image_traits']['20'], {}, {'include_hair_when_variable': True}, 'zimage')
        self.assertFalse(state['must_caption'])

    def test_eye_outliers_normalize_in_real_caption_without_rewriting_observation(self):
        rows = [self.row(eye_color=color) for color in ['blue'] * 30 + ['green'] * 2]
        before = copy.deepcopy(rows)
        profile = self.profile(rows)
        self.assertEqual(profile['stable_identity']['eye_color'], 'blue')
        self.assertEqual(profile['per_image_traits']['30']['eye_color_observed'], 'green')
        self.assertFalse(profile['profile_variability_stats']['eye_color']['variation_detected'])
        item = dict(rows[30], profile_image_id='30')
        for target in ('ernie', 'zimage', 'krea2_character'):
            with self.subTest(target=target), patch.object(self.b, 'CAPTION_PROFILE', target), patch.object(self.b, 'CAPTION_POLICY', {'include_eye_color': True}):
                caption = self.b.build_local_caption(item, {}, profile)
                self.assertIn('blue eyes', caption)
                self.assertNotIn('green eyes', caption)
        self.assertEqual(rows, before)

    def test_lenses_remain_real_deviation_but_hidden_eyes_are_not_captioned(self):
        rows = [self.row()] * 10 + [self.row(eye_color='green', eye_appearance='colored_contact_lenses')]
        profile = self.profile(rows)
        state = self.b.get_eye_feature_state(rows[-1], profile, profile['per_image_traits']['10'], {'include_eye_color_when_variable': True})
        self.assertEqual(state['phrase'], 'green eyes')
        ambiguous = self.row(eye_color='vivid green', eye_appearance='unnatural_eye_color')
        state = self.b.get_eye_feature_state(ambiguous, profile, self.b.per_image_profile_traits(ambiguous, profile), {'include_eye_color_when_variable': True})
        self.assertFalse(state['must_caption'])
        for changes in ({'face_visible': False}, {'occlusion_type': 'sunglasses_occluding_eyes'}, {'gaze_category': 'eyes_closed'}):
            row = self.row(**changes)
            profile['per_image_traits']['hidden'] = self.b.per_image_profile_traits(row, profile)
            with patch.object(self.b, 'CAPTION_PROFILE', 'ernie'), patch.object(self.b, 'CAPTION_POLICY', {'include_eye_color': True}):
                caption = self.b.build_local_caption(dict(row, profile_image_id='hidden'), {}, profile)
                self.assertNotIn('blue eyes', caption)

    def test_confirmed_profile_values_survive_consolidation(self):
        confirmed = {'canonical_features': {'hair_color': 'brown', 'eye_color': 'green'}, 'stable_identity': {'eye_color': 'green'}}
        profile = self.profile([self.row()] * 10, user_confirmed_identity_overrides=confirmed)
        self.assertEqual(profile['canonical_features']['hair_color'], 'brown')
        self.assertEqual(profile['canonical_features']['eye_color'], 'green')
        confirmed['canonical_features']['eye_color'] = ''
        state = self.b.get_eye_feature_state(self.row(), profile, profile['per_image_traits']['0'], {'include_eye_color': True})
        self.assertFalse(state['must_caption'])

    def test_saving_json_preserves_explicit_decisions_for_rebuild(self):
        import dataset_curator_ui as ui
        path = self.fixture.root / 'saved_profile.json'
        profile = {'canonical_features': {'hair_color': 'brown'}, 'stable_identity': {'eye_color': 'green'}, 'identity_markers': {'piercing_inventory': [{'location': 'nose', 'role': 'canonical'}]}}
        with patch.object(ui, 'subject_profile_path_for', return_value=str(path)):
            ui.save_subject_profile_ui('offline', str(self.fixture.root), json.dumps(profile))
        saved = json.loads(Path(path).read_text(encoding='utf-8'))
        self.assertEqual(saved['user_confirmed_identity_overrides'], profile)

    def test_glasses_visibility_and_existing_switches(self):
        profile = {'identity_markers': {'glasses': {'wears_regularly': True, 'canonical_description': 'round eyeglasses'}}, 'canonical_features': {}, 'profile_variability_stats': {}}
        normal = self.row(has_glasses_now=True, glasses_description='round eyeglasses', glasses_position='on_face')
        traits = self.b.per_image_profile_traits(normal, profile)
        deviations = {'include_glasses': False, 'include_glasses_when_variable': True}
        self.assertFalse(self.b.get_glasses_feature_state(normal, profile, traits, deviations)['must_caption'])
        self.assertTrue(self.b.get_glasses_feature_state(normal, profile, traits, {'include_glasses': True})['must_caption'])
        sunglasses = self.row(has_glasses_now=True, glasses_description='sunglasses', glasses_position='on_face')
        self.assertTrue(self.b.get_glasses_feature_state(sunglasses, profile, self.b.per_image_profile_traits(sunglasses, profile), deviations)['must_caption'])
        for row in (self.row(face_visible=False, occlusion_type='object_covering_face'), self.row(), self.row(has_glasses_now=True, glasses_description='round eyeglasses', glasses_position='not_visible')):
            state = self.b.get_glasses_feature_state(row, profile, self.b.per_image_profile_traits(row, profile), deviations)
            self.assertFalse(state['must_caption'])
            self.assertEqual(state['phrase'], '')

    def test_hidden_glasses_do_not_count_as_absence(self):
        rows = [self.row(has_glasses_now=True, glasses_description='eyeglasses')] * 3 + [self.row(face_visible=False)] * 10 + [self.row()]
        profile = self.profile(rows)
        self.assertEqual(profile['profile_variability_stats']['glasses']['counts'], {'regular_glasses': 3, 'no_glasses': 1})
        self.assertTrue(profile['identity_markers']['glasses']['wears_regularly'])
        self.assertEqual(profile['identity_markers']['glasses']['frequency'], '3/4 observable images')

    def test_frequent_piercing_stays_controllable_and_switch_can_disable_it(self):
        rows = [self.row(piercing_inventory_now=[{'location': 'nose', 'description': 'nose stud'}])] * 6 + [self.row()] * 4
        profile = {'identity_markers': {'piercing_inventory': [{'location': 'nose', 'role': 'canonical'}]}}
        profile = self.b.attach_piercing_inventory(profile, rows)
        self.assertEqual(profile['identity_markers']['piercing_inventory'][0]['role'], 'variable')
        traits = self.b.per_image_profile_traits(rows[0], profile)
        self.assertTrue(self.b.get_visible_piercing_state(rows[0], profile, traits, {'include_piercings': True}, 'zimage')['must_caption'])
        self.assertFalse(self.b.get_visible_piercing_state(rows[0], profile, traits, {'include_piercings': False}, 'zimage')['must_caption'])
        profile['user_confirmed_identity_overrides'] = {'identity_markers': {'piercing_inventory': [{'location': 'nose', 'role': 'canonical'}]}}
        profile = self.b.attach_piercing_inventory(profile, rows)
        self.assertEqual(profile['identity_markers']['piercing_inventory'][0]['role'], 'canonical')


if __name__ == '__main__':
    unittest.main()
