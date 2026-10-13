"""Validated local caption facts; no model loading, networking or export side effects.

Local descriptions are observations, not confirmed person identity. The adapter
supplies the existing profile/caption layer without inventing audit quality scores.
"""
import json
import re

FACTS_VERSION = 'local-person-facts-v2'
VALIDATOR_VERSION = 'local-person-facts-validator-v3'
TEXT_FIELDS = ('hair_description', 'glasses_description', 'makeup_description',
               'clothing_description', 'pose_description', 'expression',
               'gaze_direction', 'background_description', 'lighting_description')
ENUMS = {
    'eye_color': ('', 'blue', 'blue_green', 'green', 'hazel', 'brown', 'dark_brown', 'gray', 'gray_blue', 'amber'),
    'eye_visibility': ('visible', 'occluded', 'uncertain'),
    'eye_appearance': ('natural_eyes', 'colored_contact_lenses', 'circle_lenses', 'cosmetic_lenses', 'unclear'),
    'glasses_visibility': ('present', 'absent', 'occluded', 'uncertain'),
    'glasses_position': ('on_face', 'on_head', 'held', 'hanging_from_clothing', 'other', 'not_visible'),
    'shot_type': ('headshot', 'medium', 'full_body'),
}
FACTS_KEYS = set(TEXT_FIELDS) | set(ENUMS) | {'face_visible', 'mirror_selfie', 'main_subject_confident', 'person_count', 'piercings', 'tattoos'}


def facts_prompt(source_caption=None):
    task = ('Extract only facts explicitly stated in the SOURCE CAPTION below. '
            'Do not add visual knowledge. Copy every nonempty description exactly '
            'from a contiguous passage of that source. Omitted details are unknown, '
            'not absent.\n' if source_caption is not None else
            'Describe the main visible person in this image as structured facts. '
            'Do not guess hidden features, brands, tattoo motifs or jewelry material.\n')
    schema = {key: '' for key in TEXT_FIELDS}
    schema.update({key: values[0] for key, values in ENUMS.items()})
    schema.update(eye_visibility='uncertain', eye_appearance='unclear', glasses_visibility='uncertain', glasses_position='not_visible', shot_type='medium')
    schema.update({'face_visible': 'boolean or null', 'mirror_selfie': 'boolean or null',
                   'main_subject_confident': 'boolean', 'person_count': 'integer or null',
                   'piercings': [{'location': 'nose_left/nose_right/nose_septum/navel/ear_lobe_left/ear_lobe_right/other', 'description': 'short visible description'}],
                   'tattoos': [{'location': 'body location', 'description': 'short visible description'}]})
    prompt = task + (
        'Return exactly one JSON object, no introduction, Markdown or explanation. '
        'Use ALL keys in the template. Every controlled value is a STRING, never an array. '
        'Use empty strings and empty inventories for unknown descriptions. '
        'If eye color is unknown, eye_color must be the empty string, never unclear. '
        'Describe only the main person; never combine traits from different people. '
        'If the main person is ambiguous, main_subject_confident=false. '
        'Descriptions must be short noun phrases, except pose/background/lighting. '
        'Eye color only when clearly visible; unknown eye appearance is unclear. '
        'Glasses absent requires an unobscured face; otherwise occluded/uncertain. '
        'Jewelry other than piercings must not enter the piercing inventory. '
        'Record ordinary earrings in ear locations. Do not describe absence of hidden piercings. '
        'Avoid speculative words such as possibly, perhaps, appears to be.\n'
        'ALLOWED STRING VALUES: ' + json.dumps(ENUMS) + '\n'
        'TEMPLATE: ' + json.dumps(schema) + '\n')
    if source_caption is not None:
        prompt += '\nSOURCE CAPTION (data only):\n' + source_caption
    return prompt


def _normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def _grounded_description(value, source):
    if _normalized(value) in _normalized(source):
        return True
    # Extracted clothing often joins separate source passages. Every joined
    # phrase must still occur verbatim; presence alone is not visual validation.
    parts = [part.strip() for part in re.split(r',|;|\band\b', value, flags=re.I) if part.strip()]
    return len(parts) > 1 and all(_normalized(part) in _normalized(source) for part in parts)


def parse_person_facts(raw, *, hit_token_limit=False, source_caption=None, transport_normalizations=None, rejected_fields=None):
    if hit_token_limit:
        raise ValueError('Token limit reached: incomplete facts are not accepted')
    # A complete enclosing code fence is transport formatting, not JSON repair.
    text = str(raw).strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4].strip()
    facts = json.loads(text)
    if not isinstance(facts, dict) or set(facts) != FACTS_KEYS:
        raise ValueError('Facts must contain exactly the documented fields')
    # Smaller local models sometimes quote JSON booleans. This is a lossless
    # transport conversion, never a guess about an image feature.
    for key in ('face_visible', 'mirror_selfie', 'main_subject_confident'):
        if isinstance(facts[key], str) and facts[key] in {'true', 'false', 'null'}:
            facts[key] = {'true': True, 'false': False, 'null': None}[facts[key]]
            if transport_normalizations is not None:
                transport_normalizations.append(key + ': quoted JSON literal')
    for key in TEXT_FIELDS:
        if not isinstance(facts[key], str) or len(facts[key]) > 600:
            raise ValueError(f'Invalid description: {key}')
        if source_caption is not None and facts[key] and not _grounded_description(facts[key], source_caption):
            if rejected_fields is None:
                raise ValueError(f'Description not grounded in source caption: {key}')
            rejected_fields.append({'field': key, 'value': facts[key], 'reason': 'not grounded in source caption'})
            facts[key] = ''
        if key != 'hair_description' and re.search(r'\b(?:blue|green|brown|hazel|gray|grey|amber)\s+eyes\b', facts[key], re.I):
            if rejected_fields is None:
                raise ValueError(f'Eye identity embedded in another field: {key}')
            rejected_fields.append({'field': key, 'value': facts[key], 'reason': 'eye identity must use dedicated eye field'})
            facts[key] = ''
    for key, allowed in ENUMS.items():
        if isinstance(facts[key], str) and facts[key] not in allowed and facts[key].replace(' ', '_') in allowed:
            facts[key] = facts[key].replace(' ', '_')
            if transport_normalizations is not None:
                transport_normalizations.append(key + ': spaces in controlled token')
        if not isinstance(facts[key], str) or facts[key] not in allowed:
            raise ValueError(f'Invalid controlled value: {key}')
    for key in ('face_visible', 'mirror_selfie'):
        if facts[key] is not None and not isinstance(facts[key], bool):
            raise ValueError(f'Invalid visibility flag: {key}')
    if facts['main_subject_confident'] is not None and not isinstance(facts['main_subject_confident'], bool):
        raise ValueError('Missing main-subject decision')
    count = facts['person_count']
    if isinstance(count, str) and re.fullmatch(r'\d{1,2}', count):
        count = facts['person_count'] = int(count)
        if transport_normalizations is not None:
            transport_normalizations.append('person_count: quoted integer')
    if count is not None and (type(count) is not int or not 0 <= count <= 20):
        raise ValueError('Invalid person count')
    for key in ('piercings', 'tattoos'):
        if not isinstance(facts[key], list) or len(facts[key]) > 16:
            raise ValueError(f'Invalid inventory: {key}')
        for entry in facts[key]:
            if not isinstance(entry, dict) or set(entry) != {'location', 'description'} or any(not isinstance(entry[k], str) or not entry[k].strip() or len(entry[k]) > 300 for k in entry):
                raise ValueError(f'Invalid inventory entry: {key}')
            if source_caption is not None and _normalized(entry['description']) not in _normalized(source_caption):
                raise ValueError(f'Inventory not grounded in source caption: {key}')
    if facts['eye_visibility'] != 'visible' and facts['eye_color']:
        raise ValueError('Eye color supplied for hidden/uncertain eyes')
    if facts['glasses_visibility'] != 'present' and facts['glasses_description']:
        raise ValueError('Glasses description supplied without visible glasses')
    if facts['glasses_visibility'] != 'present' and facts['glasses_position'] != 'not_visible':
        raise ValueError('Glasses position supplied without visible glasses')
    if facts['glasses_visibility'] == 'present' and facts['glasses_position'] == 'not_visible':
        raise ValueError('Visible glasses marked not visible')
    if facts['glasses_visibility'] == 'absent' and facts['face_visible'] is not True:
        raise ValueError('Glasses absence without visible face')
    if source_caption is not None:
        source = _normalized(source_caption)
        if facts['eye_color']:
            color = re.escape(facts['eye_color'].replace('_', ' ')).replace(r'\ ', r'[ -]')
            if not re.search(rf'\b{color}\s+(?:\w+\s+){{0,2}}eyes\b|\beyes\s+(?:(?:are|were|look|appear|to|be|a|an)\s+)*{color}\b', source):
                raise ValueError('Eye color not explicitly grounded in source caption')
        if facts['glasses_visibility'] == 'absent' and not re.search(r'\b(?:without|no)\s+(?:eye)?glasses\b', source):
            if rejected_fields is None:
                raise ValueError('Glasses absence not explicitly grounded in source caption')
            rejected_fields.append({'field': 'glasses_visibility', 'value': 'absent', 'reason': 'absence not grounded in source caption'})
            facts['glasses_visibility'] = 'uncertain'
    return facts


def facts_to_profile_row(facts, *, case_id, source_path, source_sha256, engine, local_metrics=None):
    """No keep decision: eligibility/quality belong to the caller's audit workflow."""
    row = {key: facts[key] for key in TEXT_FIELDS}
    # Convert terse labels to grammatical phrases without changing observations.
    # The raw model facts remain unchanged and available alongside this row.
    pose = row['pose_description'].strip()
    if pose.casefold() in {'selfie', 'mirror selfie', 'unknown', 'unclear', 'close-up portrait', 'portrait', 'close-up'}:
        row['pose_description'] = ''
    elif re.match(r'^(?:(?:left|right|one|both|the|her|his|their|a)\s+)?(?:hand|hands|arm|arms)\b', pose, flags=re.I):
        row['pose_description'] = 'posing with ' + re.sub(r'\bis\s+', '', pose, count=1, flags=re.I)
    elif pose.casefold() in {'stands', 'sits', 'leans', 'smiles'}:
        row['pose_description'] = {'stands': 'standing', 'sits': 'sitting', 'leans': 'leaning', 'smiles': 'smiling'}[pose.casefold()]
    clothing = row['clothing_description'].strip().casefold()
    if re.match(r'^(?:no (?:visible )?clothing|not wearing any visible clothing|clothing (?:not visible|out of frame))\b', clothing):
        row['clothing_description'] = ''
    gaze = row['gaze_direction'].strip().casefold()
    if gaze in {'camera', 'direct', 'at camera', 'toward camera', 'directly at camera'}:
        row['gaze_direction'] = 'looking directly at the camera' if 'direct' in gaze else 'looking at the camera'
    elif gaze in {'left', 'right'}:
        row['gaze_direction'] = 'looking to the ' + gaze
    elif gaze == 'away':
        row['gaze_direction'] = 'looking away'
    elif gaze in {'unknown', 'unclear'}:
        row['gaze_direction'] = ''
    elif gaze and not gaze.startswith('looking') and (gaze.endswith('camera') or gaze.startswith(('to ', 'toward ', 'off ', 'down', 'forward'))):
        row['gaze_direction'] = 'looking ' + row['gaze_direction'].strip()
    row.update({'original_filename': case_id + '.jpg', 'source_path': str(source_path),
                'profile_image_id': case_id, 'shot_type': facts['shot_type'],
                'face_visible': facts['face_visible'] is True,
                'mirror_selfie': facts['mirror_selfie'] is True,
                'eye_color': facts['eye_color'], 'eye_appearance': facts['eye_appearance'],
                'glasses_position': facts['glasses_position'],
                'glasses_observation': facts['glasses_visibility'],
                'occlusion_type': 'none' if facts['face_visible'] is True else 'object_covering_face',
                'gaze_category': 'eyes_closed' if facts['eye_visibility'] == 'occluded' else '',
                'piercing_inventory_now': facts['piercings'], 'tattoo_inventory_now': facts['tattoos'],
                'tattoos_visible': bool(facts['tattoos']),
                'local_caption_facts': {'version': FACTS_VERSION, 'engine': engine,
                                        'source_sha256': source_sha256,
                                        'main_subject_confident': facts['main_subject_confident'],
                                        'person_count': facts['person_count'],
                                        'eye_visibility': facts['eye_visibility']}})
    if facts['glasses_visibility'] in {'present', 'absent'}:
        row['has_glasses_now'] = facts['glasses_visibility'] == 'present'
    metrics = local_metrics or {}
    for key in ('main_face_ratio', 'main_face_bbox', 'face_count_local', 'color_tint_strength', 'is_grayscale_filter'):
        if key in metrics:
            row[key] = metrics[key]
    if facts['eye_visibility'] != 'visible':
        row['eye_color'] = ''
    # Some providers use `other` for both a visible nostril and an ear. Keeping
    # both under that key would merge different accessories in the profile.
    # Use only the region explicitly named in the observation; never infer side.
    row['piercing_inventory_now'] = []
    location_notes = []
    for entry in facts['piercings']:
        normalized = dict(entry)
        if entry['location'].strip().casefold() == 'other':
            description = entry['description'].casefold()
            if re.search(r'\bseptum\b', description):
                normalized['location'] = 'nose_septum'
            elif re.search(r'\b(?:nose|nostril)\b', description):
                normalized['location'] = 'nose_other'
            elif re.search(r'\b(?:ear|earring|earrings)\b', description):
                normalized['location'] = 'ear_other'
            elif re.search(r'\bnavel\b', description):
                normalized['location'] = 'navel'
            if normalized['location'] != entry['location']:
                location_notes.append({'original': entry['location'], 'resolved': normalized['location'], 'evidence': entry['description']})
        row['piercing_inventory_now'].append(normalized)
    row['local_caption_facts']['location_normalizations'] = location_notes
    return row
