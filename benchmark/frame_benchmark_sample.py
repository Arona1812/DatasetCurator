"""Freeze a validation selection without using detector scores or proposals."""
import os as _os
import sys as _sys

# The detector modules live in ../app.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "app"))
import argparse
import hashlib
import json
from pathlib import Path

from frame_cleanup import IMAGE_EXTENSIONS, file_sha1


def sample(frames: Path, confirmed: Path, controls: Path, output: Path, count=20):
    if count < 1:
        raise ValueError('Control count must be positive')
    references = json.loads(confirmed.read_text(encoding='utf-8'))
    confirmed_ids = {r['id'] for r in references['records'] if r.get('reference_confirmed') is True}
    entries, frame_ids = [], set()
    for path in sorted(frames.iterdir(), key=lambda p: p.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in (*IMAGE_EXTENSIONS, '.heic', '.heif'):
            continue
        digest = file_sha1(str(path))
        frame_ids.add(digest)
        if digest not in confirmed_ids:
            entries.append({'source_path': str(path.resolve()), 'id': digest, 'cohort': 'unconfirmed_frames'})
    # Root images only: never include curated exports, caches or derived crops.
    candidates = [p for p in controls.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    candidates.sort(key=lambda p: hashlib.sha256(('frame-validation-v1:'+p.name).encode()).hexdigest())
    seen = frame_ids | confirmed_ids
    selected = 0
    for path in candidates:
        digest = file_sha1(str(path))
        if digest in seen:
            continue
        seen.add(digest)
        entries.append({'source_path': str(path.resolve()), 'id': digest, 'cohort': 'control_candidates'})
        selected += 1
        if selected >= count:
            break
    result = {'schema': 'frame-validation-selection-v1', 'images': entries,
              'confirmed_reference_ids_excluded': sorted(confirmed_ids),
              'control_sampling': 'SHA256(frame-validation-v1:filename), unique content, root originals only',
              'notes': 'Frame references have no confirmed ground truth; some were seen in earlier development. '
                       'Controls are sampled before detection and are not assumed to be borderless. '
                       'All references require human confirmation; this is not an unseen-corpus acceptance claim.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'unconfirmed_frames': len(entries)-selected, 'control_candidates': selected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frames', 'confirmed', 'controls', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(sample(args.frames, args.confirmed, args.controls, args.output, args.count)))
