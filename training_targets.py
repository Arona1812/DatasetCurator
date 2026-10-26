"""Image-only export targets, independent of visible-fact extraction.

Toolkit support checked 2026-10-11 against Ostris model docs and training UI.
New profiles are export integrations, not claims of measured LoRA quality.
"""
from dataclasses import dataclass
import json

TARGET_REGISTRY_VERSION = 'image-targets-v1'
AUDIT_FACTS_VERSION = 'visible-image-facts-v1'


@dataclass(frozen=True)
class ImageTarget:
    key: str
    label: str
    toolkit_arch: str
    checkpoint: str
    caption_format: str = 'natural'
    experimental: bool = False
    local_training_verified: bool = False


IMAGE_TARGETS = {
    t.key: t for t in (
        ImageTarget('ernie', 'ERNIE Image', 'ernie_image', 'baidu/ERNIE-Image'),
        ImageTarget('z_image_base', 'Z-Image Base', 'zimage', 'Tongyi-MAI/Z-Image'),
        ImageTarget('krea2', 'Krea 2 Raw', 'krea2', 'krea/Krea-2-Raw'),
        ImageTarget('krea2_turbo', 'Krea 2 Turbo', 'krea2:turbo', 'krea/Krea-2-Turbo'),
        ImageTarget('qwen_image_2_1', 'Qwen-Image 2.1', 'qwen_image_2', 'Comfy-Org/Qwen-Image-2.1'),
        ImageTarget('ideogram4', 'Ideogram 4 · experimental', 'ideogram4', 'ideogram-ai/ideogram-4-fp8', 'ideogram_json', True),
        ImageTarget('flux2', 'FLUX.2 dev', 'flux2', 'black-forest-labs/FLUX.2-dev'),
        ImageTarget('flux2_klein_4b', 'FLUX.2 Klein 4B Base', 'flux2_klein_4b', 'black-forest-labs/FLUX.2-klein-base-4B'),
        ImageTarget('flux2_klein_9b', 'FLUX.2 Klein 9B Base', 'flux2_klein_9b', 'black-forest-labs/FLUX.2-klein-base-9B'),
    )
}
ALIASES = {'shared_compact':'ernie', 'ernie_image':'ernie', 'zimage':'z_image_base',
    'z_image':'z_image_base', 'krea_2':'krea2', 'krea2_character':'krea2',
    'krea_2_character':'krea2', 'krea2_raw':'krea2', 'krea_2_raw':'krea2',
    'qwen_image_2.1':'qwen_image_2_1', 'qwen_image_2':'qwen_image_2_1',
    'ideogram_4':'ideogram4', 'flux_2':'flux2'}


def normalize_training_target(value):
    key = str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    if not key:
        return 'ernie'
    key = ALIASES.get(key, key)
    if key not in IMAGE_TARGETS:
        raise ValueError(f'Unknown image training target: {value!r}. Select a supported image target.')
    return key


def image_target(value):
    return IMAGE_TARGETS[normalize_training_target(value)]


def is_krea_target(value):
    return normalize_training_target(value) in {'krea2', 'krea2_turbo'}


def caption_profile_for_training_target(value):
    key = normalize_training_target(value)
    return 'krea2_character' if is_krea_target(key) else key


def adapt_caption_for_target(value, natural_caption):
    """Minimal official Ideogram shape; never invent object boxes or palettes."""
    target = image_target(value)
    text = str(natural_caption or '').strip()
    if not text:
        raise ValueError('Cannot export an empty caption')
    if target.caption_format != 'ideogram_json':
        return text
    return json.dumps({'high_level_description': text,
        'compositional_deconstruction': {'elements': [{'type':'obj', 'desc':text}]}},
        ensure_ascii=False, separators=(',', ':'))


def target_export_metadata(value):
    from dataclasses import asdict
    return {'registry_version':TARGET_REGISTRY_VERSION, **asdict(image_target(value)),
            'scope':'single_image_rgb', 'reference_pairs_supported':False,
            'rgba_export_supported':False, 'video_supported':False}


NEUTRAL_AUDIT_GUIDANCE = (
    'Record visible facts for a reusable person-image dataset, independently of the '
    'downstream generator. Preserve scene facts, full-person usefulness, orientation, '
    'framing, temporary appearance and visible accessories. Separate observed facts '
    'from uncertainty; never invent hidden identity features or discard useful views '
    'because a particular training model prefers different captions.'
)
NEUTRAL_PROFILE_GUIDANCE = (
    'Build a generator-independent identity and terminology layer. Separate stable '
    'identity from variable scene traits. Record recurring accessories in inventories '
    'without forcing them into images where they are not visible. Caption policy is '
    'applied later by the export target.'
)
