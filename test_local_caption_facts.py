import copy
import json
import unittest

from local_caption_facts import TEXT_FIELDS, facts_to_profile_row, parse_person_facts


class LocalFacts(unittest.TestCase):
    def facts(self):
        return dict({k: '' for k in TEXT_FIELDS}, eye_color='', eye_visibility='uncertain',
                    eye_appearance='unclear', glasses_visibility='uncertain',
                    glasses_position='not_visible', shot_type='medium', face_visible=None,
                    mirror_selfie=None, main_subject_confident=True, person_count=1,
                    piercings=[], tattoos=[])

    def test_valid_facts_preserve_unknown_and_do_not_invent_audit_scores(self):
        facts = self.facts()
        parsed = parse_person_facts(json.dumps(facts))
        row = facts_to_profile_row(parsed, case_id='one', source_path='one.jpg', source_sha256='abc', engine='local')
        self.assertNotIn('has_glasses_now', row)
        self.assertNotIn('base_status', row)
        self.assertNotIn('quality_total', row)
        self.assertNotIn('main_face_ratio', row)
        self.assertEqual(row['glasses_observation'], 'uncertain')

    def test_hidden_features_and_false_absence_are_rejected(self):
        for changed in ({'eye_color': 'blue'}, {'glasses_visibility': 'absent'},
                        {'glasses_description': 'eyeglasses'}, {'glasses_position': 'on_face'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                parse_person_facts(json.dumps(dict(self.facts(), **changed)))

    def test_incomplete_malformed_and_extra_output_is_rejected(self):
        for raw in ('{}', 'Here is your JSON: ' + json.dumps(self.facts()),
                    json.dumps(dict(self.facts(), invented_quality=99))):
            with self.subTest(raw=raw[:20]), self.assertRaises(ValueError):
                parse_person_facts(raw)
        with self.assertRaises(ValueError):
            parse_person_facts(json.dumps(self.facts()), hit_token_limit=True)

    def test_only_explicit_json_literals_can_be_unquoted_with_provenance(self):
        facts = dict(self.facts(), face_visible='true', mirror_selfie='null', person_count='1', glasses_position='not visible')
        notes = []
        parsed = parse_person_facts(json.dumps(facts), transport_normalizations=notes)
        self.assertIs(parsed['face_visible'], True)
        self.assertIsNone(parsed['mirror_selfie'])
        self.assertEqual(parsed['person_count'], 1)
        self.assertEqual(parsed['glasses_position'], 'not_visible')
        self.assertEqual(len(notes), 4)
        with self.assertRaises(ValueError):
            parse_person_facts(json.dumps(dict(facts, face_visible='probably')))

    def test_caption_bridge_cannot_add_description_or_inventory(self):
        facts = self.facts()
        facts['clothing_description'] = 'a red shirt'
        parse_person_facts(json.dumps(facts), source_caption='She is wearing a red shirt.')
        with self.assertRaises(ValueError):
            parse_person_facts(json.dumps(facts), source_caption='She is wearing a blue shirt.')
        facts['clothing_description'] = 'a red shirt, blue jeans'
        parse_person_facts(json.dumps(facts), source_caption='She is wearing a red shirt. Her blue jeans have pockets.')
        with self.assertRaises(ValueError):
            parse_person_facts(json.dumps(facts), source_caption='She is wearing a red shirt. Her black jeans have pockets.')
        facts['clothing_description'] = 'a red shirt'
        facts['piercings'] = [{'location': 'nose_left', 'description': 'nose stud'}]
        with self.assertRaises(ValueError):
            parse_person_facts(json.dumps(facts), source_caption='She is wearing a red shirt.')
        facts = dict(self.facts(), face_visible=True, eye_visibility='visible', eye_color='blue')
        with self.assertRaises(ValueError):
            parse_person_facts(json.dumps(facts), source_caption='A woman wearing a blue shirt.')
        parse_person_facts(json.dumps(facts), source_caption='A woman with blue eyes.')

    def test_terse_pose_and_gaze_labels_format_without_changing_raw_facts(self):
        facts = dict(self.facts(), pose_description='hand near neck', gaze_direction='camera')
        row = facts_to_profile_row(facts, case_id='one', source_path='one.jpg', source_sha256='abc', engine='local')
        self.assertEqual(row['pose_description'], 'posing with hand near neck')
        self.assertEqual(row['gaze_direction'], 'looking at the camera')
        self.assertEqual(facts['pose_description'], 'hand near neck')
        for pose, expected in [('close-up portrait', ''), ('stands', 'standing')]:
            row = facts_to_profile_row(dict(facts, pose_description=pose, clothing_description='not wearing any visible clothing'), case_id='one', source_path='one.jpg', source_sha256='abc', engine='local')
            self.assertEqual(row['pose_description'], expected)
            self.assertEqual(row['clothing_description'], '')

    def test_unverified_fields_can_only_be_removed_with_recorded_reason(self):
        facts = dict(self.facts(), clothing_description='blue shirt', makeup_description='green eyes with eyeliner')
        rejected = []
        parsed = parse_person_facts(json.dumps(facts), source_caption='She wears a blue shirt and has green eyes with eyeliner.', rejected_fields=rejected)
        self.assertEqual(parsed['clothing_description'], 'blue shirt')
        self.assertEqual(parsed['makeup_description'], '')
        self.assertEqual(rejected[0]['field'], 'makeup_description')
        self.assertEqual(facts['makeup_description'], 'green eyes with eyeliner')

    def test_multiple_people_ambiguity_and_observations_survive_adapter(self):
        facts = self.facts()
        facts.update(main_subject_confident=False, person_count=2)
        before = copy.deepcopy(facts)
        row = facts_to_profile_row(facts, case_id='two', source_path='two.jpg', source_sha256='abc', engine='local')
        self.assertFalse(row['local_caption_facts']['main_subject_confident'])
        self.assertEqual(row['local_caption_facts']['person_count'], 2)
        self.assertEqual(facts, before)
        parsed = parse_person_facts(json.dumps(dict(facts, main_subject_confident=None)))
        self.assertIsNone(parsed['main_subject_confident'])

    def test_unknown_nose_and_ear_locations_stay_distinct_without_guessing_side(self):
        facts = self.facts()
        facts['piercings'] = [{'location': 'other', 'description': 'Small hoop on the visible nostril'}, {'location': 'other', 'description': 'Small earrings on the visible ear'}]
        before = copy.deepcopy(facts)
        row = facts_to_profile_row(facts, case_id='one', source_path='one.jpg', source_sha256='abc', engine='local')
        self.assertEqual([x['location'] for x in row['piercing_inventory_now']], ['nose_other', 'ear_other'])
        self.assertEqual(facts, before)
        self.assertEqual(len(row['local_caption_facts']['location_normalizations']), 2)


if __name__ == '__main__':
    unittest.main()
