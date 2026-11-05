"""Explains, image by image, why the primary audit cache hits or misses.

Uses the settings of the last run (_ui_config.json in the program folder) and
makes no API call. For every input image it rebuilds the audit cache key the
way the pipeline does and, on a miss, looks for the closest existing entry:
the uncropped original, another frame crop or another reasoning effort.

    python benchmark/audit_cache_report.py
"""
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

import dataset_curator_v2 as v2  # noqa: E402
import frame_cleanup as fc  # noqa: E402

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def cached(file_hash: str, effort: str = None) -> bool:
    key = v2.audit_cache_key(file_hash, v2.AI_MODEL, "primary_audit", reasoning_effort=effort)
    return os.path.exists(v2.cache_path_for_file(key))


def other_crops(source_hash: str, current_path: str) -> list:
    folder = os.path.join(v2.CACHE_DIR, fc.FRAME_CACHE_SUBDIR, fc.FRAME_CROP_SUBDIR)
    paths = glob.glob(os.path.join(folder, f"{source_hash}_*.jpg"))
    return [p for p in paths if os.path.normcase(p) != os.path.normcase(current_path or "")]


def explain(path: str, asset_id: int) -> tuple:
    source_hash = v2.file_sha1(path)
    frame = v2.frame_cleanup_for_pipeline(path, source_hash=source_hash, asset_id=asset_id)
    cropped = bool(frame.get("frame_cleanup_applied"))
    effective = str(frame.get("effective_image_path") or path) if cropped else path
    effective_hash = v2.file_sha1(effective)
    label = "crop" if cropped else "original"
    if cached(effective_hash):
        return f"hit ({label})", ""
    if cropped and cached(source_hash):
        return "miss: cached for the uncropped original", frame.get("frame_cleanup_applied_by", "")
    if not cropped and any(cached(v2.file_sha1(p)) for p in other_crops(source_hash, "")):
        return "miss: cached for a frame crop, now uncropped", frame.get("frame_cleanup_applied_by", "")
    if cropped and any(cached(v2.file_sha1(p)) for p in other_crops(source_hash, effective)):
        return "miss: cached for a different frame crop", frame.get("frame_cleanup_applied_by", "")
    for effort in EFFORTS:
        if cached(effective_hash, effort):
            return f"miss: cached with reasoning effort '{effort}'", ""
    return f"miss: never audited ({label})", frame.get("frame_cleanup_applied_by", "")


def main() -> None:
    print(f"Input:  {v2.INPUT_FOLDER}")
    print(f"Cache:  {v2.CACHE_DIR}")
    print(f"Model:  {v2.AI_MODEL} | effort {v2.AUDIT_REASONING_EFFORT} | trigger {v2.TRIGGER_WORD}")
    print(f"Frames: {v2.FRAME_CLEANUP_MODE}" + ("" if v2.ENABLE_IG_FRAME_CROP else " | skipped"))
    paths = v2.iter_input_images(v2.INPUT_FOLDER)
    _, asset_ids = v2.ensure_project_asset_registry(v2.INPUT_FOLDER, v2.TRIGGER_WORD, image_paths=paths)
    paths, _ = v2.filter_excluded_sources(paths, v2.OUTPUT_ROOT)
    results = Counter()
    examples = []
    for path in paths:
        asset_id = int(asset_ids.get(os.path.normcase(os.path.abspath(path)), 0) or 0)
        outcome, detail = explain(path, asset_id)
        results[outcome] += 1
        if outcome.startswith("miss") and len(examples) < 12:
            examples.append(f"  {os.path.basename(path)}: {outcome} {detail}".rstrip())
    print(f"\nImages: {len(paths)}")
    for outcome, count in results.most_common():
        print(f"  {count:>5}  {outcome}")
    if examples:
        print("\nExamples:")
        print("\n".join(examples))
    v2.flush_file_hash_index()


if __name__ == "__main__":
    main()
