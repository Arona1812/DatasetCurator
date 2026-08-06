#!/usr/bin/env python3
"""Local project preflight helpers for Dataset Curator.

The preflight layer deliberately avoids OpenAI calls. It owns the project
workspace metadata and the first pHash duplicate pass that runs before frame
analysis. Results are stored inside curated_<Trigger> so the normal pipeline can
reuse them without moving caches into the user profile.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from curator_core import atomic_write_json as core_atomic_write_json, natural_sort_key

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PIL_HARD_IMAGE_PIXEL_LIMIT = 350_000_000
PROJECT_STATE_FILENAME = "_project_workspace.json"
PREFLIGHT_STATE_FILENAME = "_preflight_state.json"
EARLY_RESULT_FILENAME = "early_results.json"
EARLY_RESULT_SCHEMA = "v1"
ASSET_REGISTRY_SCHEMA = "asset-registry-v1"


@dataclass(frozen=True)
class PHashSettings:
    enabled: bool = True
    loop1_enabled: bool = True
    loop1_threshold: int = 1
    loop1_keep: int = 1
    loop2_enabled: bool = True
    loop2_threshold: int = 4
    loop2_keep: int = 2
    min_side_px: int = 768
    use_min_filesize: bool = True
    min_filesize_kb: float = 80.0

    def as_cache_settings(self) -> Dict[str, Any]:
        return {
            "HARD_MIN_SIDE_PX": int(self.min_side_px),
            "USE_MIN_FILESIZE_FILTER": bool(self.use_min_filesize),
            "HARD_MIN_FILESIZE_KB": float(self.min_filesize_kb),
            "USE_EARLY_PHASH_DEDUP": bool(self.enabled),
            "USE_PHASH_DUPLICATE_SCORING": True,
            "USE_EARLY_PHASH_LOOP1": bool(self.loop1_enabled),
            "EARLY_PHASH_HAMMING_THRESHOLD_1": int(self.loop1_threshold),
            "EARLY_PHASH_KEEP_PER_GROUP_1": int(self.loop1_keep),
            "EARLY_PHASH_LOOP1_PREFER_RESOLUTION": True,
            "USE_EARLY_PHASH_LOOP2": bool(self.loop2_enabled),
            "EARLY_PHASH_HAMMING_THRESHOLD_2": int(self.loop2_threshold),
            "EARLY_PHASH_KEEP_PER_GROUP_2": int(self.loop2_keep),
            "EARLY_PHASH_HAMMING_THRESHOLD": int(self.loop2_threshold),
            "EARLY_PHASH_KEEP_PER_GROUP": int(self.loop2_keep),
        }


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    core_atomic_write_json(path, payload)


def output_root_for(input_folder: str, trigger_word: str) -> str:
    safe = "".join(ch for ch in str(trigger_word).strip() if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")
    return os.path.join(os.path.abspath(input_folder), f"curated_{safe or 'subject'}")


def cache_dir_for(input_folder: str, trigger_word: str) -> str:
    return os.path.join(output_root_for(input_folder, trigger_word), "_cache")


def scan_images(input_folder: str, output_root: Optional[str] = None) -> List[str]:
    output_root_abs = os.path.normcase(os.path.abspath(output_root)) if output_root else ""
    result: List[str] = []
    for name in sorted(os.listdir(input_folder), key=natural_sort_key):
        path = os.path.abspath(os.path.join(input_folder, name))
        if not os.path.isfile(path) or not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        if output_root_abs and os.path.normcase(path).startswith(output_root_abs + os.sep):
            continue
        result.append(path)
    return result


def _asset_relative_key(input_folder: str, path: str) -> str:
    """Stable project-local key used to preserve integer asset IDs."""
    try:
        relative = os.path.relpath(os.path.abspath(path), os.path.abspath(input_folder))
    except Exception:
        relative = os.path.basename(str(path or ""))
    return os.path.normcase(relative.replace("\\", "/"))


def _load_json_dict(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def build_asset_registry(
    input_folder: str,
    image_paths: Iterable[str],
    existing_workspace: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Assign persistent, never-reused integer IDs to project images.

    IDs are preserved by project-relative path across repeated preflight runs.
    Removed files stay in the registry as inactive entries so their IDs are not
    accidentally reused. Paths and hashes remain metadata/cache keys; UI and
    workflow selection use ``asset_id`` as the authoritative identity.
    """
    workspace = dict(existing_workspace or {})
    previous = workspace.get("asset_registry") or {}
    previous_items = previous.get("items") if isinstance(previous, dict) else []
    if not isinstance(previous_items, list):
        previous_items = []

    by_key: Dict[str, Dict[str, Any]] = {}
    highest = 0
    for raw in previous_items:
        if not isinstance(raw, dict):
            continue
        try:
            asset_id = int(raw.get("asset_id"))
        except Exception:
            continue
        if asset_id <= 0:
            continue
        key = str(raw.get("relative_key") or "")
        if not key:
            source_path = str(raw.get("source_path") or "")
            if source_path:
                key = _asset_relative_key(input_folder, source_path)
        if not key:
            continue
        item = dict(raw)
        item["asset_id"] = asset_id
        item["relative_key"] = key
        by_key[key] = item
        highest = max(highest, asset_id)

    try:
        next_asset_id = max(highest + 1, int(previous.get("next_asset_id", 1) or 1))
    except Exception:
        next_asset_id = highest + 1

    active_keys = set()
    active_map: Dict[str, int] = {}
    for path in image_paths:
        absolute = os.path.abspath(str(path))
        key = _asset_relative_key(input_folder, absolute)
        active_keys.add(key)
        item = by_key.get(key)
        if item is None:
            item = {
                "asset_id": next_asset_id,
                "relative_key": key,
            }
            by_key[key] = item
            next_asset_id += 1
        item.update({
            "source_path": absolute,
            "filename": os.path.basename(absolute),
            "active": True,
        })
        active_map[os.path.normcase(absolute)] = int(item["asset_id"])

    for key, item in by_key.items():
        if key not in active_keys:
            item["active"] = False

    items = sorted(by_key.values(), key=lambda item: int(item.get("asset_id", 0) or 0))
    registry = {
        "schema_version": ASSET_REGISTRY_SCHEMA,
        "next_asset_id": int(next_asset_id),
        "items": items,
    }
    return registry, active_map


def ensure_asset_registry(
    input_folder: str,
    trigger_word: str,
    image_paths: Optional[Iterable[str]] = None,
    workspace: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Load/migrate and persist the integer asset registry for a project."""
    output_root = output_root_for(input_folder, trigger_word)
    workspace_path = os.path.join(output_root, PROJECT_STATE_FILENAME)
    current = dict(workspace or _load_json_dict(workspace_path))
    paths = list(image_paths) if image_paths is not None else scan_images(input_folder, output_root)
    registry, active_map = build_asset_registry(input_folder, paths, current)
    if current.get("asset_registry") != registry:
        current["asset_registry"] = registry
        current["schema_version"] = "workspace-v2-asset-ids"
        current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        current.setdefault("input_folder", os.path.abspath(input_folder))
        current.setdefault("trigger_word", str(trigger_word).strip())
        current.setdefault("output_root", output_root)
        atomic_write_json(workspace_path, current)
    return current, active_map


def image_dimensions(path: str) -> Tuple[int, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            width, height = image.size
            if int(width) * int(height) > PIL_HARD_IMAGE_PIXEL_LIMIT:
                raise ValueError(f"image_too_large_{width}x{height}")
            try:
                orientation = int(image.getexif().get(274, 1) or 1)
            except Exception:
                orientation = 1
            return (int(height), int(width)) if orientation in (5, 6, 7, 8) else (int(width), int(height))


def _load_small_gray(path: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            try:
                source.draft("L", (64, 64))
            except Exception:
                pass
            image = ImageOps.exif_transpose(source).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
            return np.asarray(image, dtype=np.float32)


def compute_phash(path: str) -> int:
    arr = _load_small_gray(path)
    if cv2 is not None:
        dct = cv2.dct(arr)
        low = dct[:8, :8]
        med = np.median(low[1:, 1:])
        bits = low > med
    else:
        med = np.median(arr)
        bits = arr[:8, :8] > med
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def dataset_fingerprint(image_paths: Iterable[str], input_folder: str) -> str:
    rows: List[List[Any]] = []
    for path in sorted(image_paths, key=str.lower):
        try:
            st = os.stat(path)
            rel = os.path.relpath(path, input_folder).replace("\\", "/")
            rows.append([rel, int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))])
        except OSError:
            rows.append([path, -1, -1])
    return hashlib.sha1(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def settings_fingerprint(settings: PHashSettings) -> str:
    raw = json.dumps(settings.as_cache_settings(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _quality_tuple(path: str) -> Tuple[int, int, float, str]:
    """Cheap deterministic winner score: pixels, bytes, small-image sharpness."""
    try:
        width, height = image_dimensions(path)
        pixels = int(width) * int(height)
    except Exception:
        pixels = 0
    try:
        size = int(os.path.getsize(path))
    except OSError:
        size = 0
    sharpness = 0.0
    try:
        arr = _load_small_gray(path)
        if cv2 is not None:
            sharpness = float(cv2.Laplacian(arr, cv2.CV_32F).var())
        else:
            sharpness = float(np.var(np.diff(arr, axis=0)) + np.var(np.diff(arr, axis=1)))
    except Exception:
        pass
    return pixels, size, sharpness, os.path.basename(path).lower()


def _group_pass(paths: List[str], hashes: Dict[str, int], threshold: int, keep_n: int) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    groups: List[Dict[str, Any]] = []
    for path in sorted(paths, key=str.lower):
        value = hashes.get(path)
        if value is None:
            groups.append({"anchor": None, "members": [path]})
            continue
        for group in groups:
            anchor = group.get("anchor")
            if anchor is not None and hamming_distance(value, anchor) <= int(threshold):
                group["members"].append(path)
                break
        else:
            groups.append({"anchor": value, "members": [path]})
    survivors: List[str] = []
    duplicates: List[str] = []
    details: List[Dict[str, Any]] = []
    keep_n = max(1, int(keep_n))
    for group_index, group in enumerate(groups, start=1):
        members = list(group["members"])
        ranked = sorted(members, key=_quality_tuple, reverse=True)
        kept = ranked[:keep_n]
        removed = ranked[keep_n:]
        survivors.extend(kept)
        duplicates.extend(removed)
        if len(members) > 1:
            details.append({
                "group_id": group_index,
                "threshold": int(threshold),
                "kept": kept,
                "removed": removed,
                "members": ranked,
            })
    survivor_set = set(survivors)
    return [p for p in paths if p in survivor_set], duplicates, details


def run_preflight(
    input_folder: str,
    trigger_word: str,
    settings: PHashSettings,
    *,
    frame_enabled: bool,
    frame_mode: str,
    frame_auto_accept_types: List[str],
    frame_pause_on_medium: bool,
    post_frame_duplicate_refresh: bool,
) -> Dict[str, Any]:
    output_root = output_root_for(input_folder, trigger_word)
    cache_dir = os.path.join(output_root, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    images = scan_images(input_folder, output_root)
    existing_workspace = _load_json_dict(os.path.join(output_root, PROJECT_STATE_FILENAME))
    asset_registry, asset_id_by_path = build_asset_registry(input_folder, images, existing_workspace)
    dataset_fp = dataset_fingerprint(images, input_folder)

    def asset_id_for(path: str) -> int:
        return int(asset_id_by_path.get(os.path.normcase(os.path.abspath(path)), 0) or 0)

    early_reject_rows: List[Dict[str, Any]] = []
    survivors: List[str] = []
    for path in images:
        name = os.path.basename(path)
        try:
            width, height = image_dimensions(path)
            size_kb = os.path.getsize(path) / 1024.0
            reason = ""
            if min(width, height) < int(settings.min_side_px):
                reason = f"hard_pass_too_small_{width}x{height}"
            elif settings.use_min_filesize and size_kb < float(settings.min_filesize_kb):
                reason = f"filesize_too_small_{size_kb:.0f}kb"
            if reason:
                early_reject_rows.append({
                    "asset_id": asset_id_for(path),
                    "asset_id": asset_id_for(path),
                "source_asset_id": asset_id_for(path),
                    "original_filename": name,
                    "original_path": path,
                    "width": width,
                    "height": height,
                    "quality_total": 0,
                    "base_status": "reject",
                    "final_status": "reject",
                    "short_reason": reason,
                    "local_override_reasons": [reason],
                    "status_notes": ["early_static_reject"],
                    "selected": False,
                    "output_bucket": "",
                    "new_basename": "",
                })
            else:
                survivors.append(path)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            early_reject_rows.append({
                "asset_id": asset_id_for(path),
                "source_asset_id": asset_id_for(path),
                "original_filename": name,
                "original_path": path,
                "width": 0,
                "height": 0,
                "quality_total": 0,
                "base_status": "reject",
                "final_status": "reject",
                "short_reason": "unreadable_or_corrupt_image",
                "local_override_reasons": [str(exc)],
                "status_notes": ["early_static_reject"],
                "selected": False,
                "output_bucket": "",
                "new_basename": "",
            })

    hashes: Dict[str, int] = {}
    if settings.enabled:
        for path in survivors:
            try:
                hashes[path] = compute_phash(path)
            except Exception:
                pass

    duplicate_paths: List[str] = []
    groups: List[Dict[str, Any]] = []
    if settings.enabled and settings.loop1_enabled:
        survivors, removed, details = _group_pass(survivors, hashes, settings.loop1_threshold, settings.loop1_keep)
        duplicate_paths.extend(removed)
        for item in details:
            item["pass"] = "loop1"
        groups.extend(details)
    if settings.enabled and settings.loop2_enabled:
        survivors, removed, details = _group_pass(survivors, hashes, settings.loop2_threshold, settings.loop2_keep)
        duplicate_paths.extend(removed)
        for item in details:
            item["pass"] = "loop2"
        groups.extend(details)

    duplicate_paths = list(dict.fromkeys(duplicate_paths))
    early_cache = {
        "schema_version": EARLY_RESULT_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_fingerprint": dataset_fp,
        "settings_fingerprint": settings_fingerprint(settings),
        "settings": settings.as_cache_settings(),
        "survivor_paths": survivors,
        "survivor_asset_ids": [asset_id_for(path) for path in survivors],
        "early_duplicate_paths": duplicate_paths,
        "early_duplicate_asset_ids": [asset_id_for(path) for path in duplicate_paths],
        "asset_ids_by_path": {
            os.path.abspath(path): asset_id_for(path)
            for path in images
        },
        "asset_registry": asset_registry,
        "phash_cache": {p: hashes[p] for p in hashes if os.path.exists(p)},
        "early_reject_rows": early_reject_rows,
        "created_by": "workspace_preflight",
    }
    atomic_write_json(os.path.join(cache_dir, EARLY_RESULT_FILENAME), early_cache)

    workspace = {
        "schema_version": "workspace-v2-asset-ids",
        "asset_registry": asset_registry,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_folder": os.path.abspath(input_folder),
        "trigger_word": str(trigger_word).strip(),
        "output_root": output_root,
        "preflight": {
            "completed": True,
            "input_count": len(images),
            "survivor_count": len(survivors),
            "early_reject_count": len(early_reject_rows),
            "duplicate_count": len(duplicate_paths),
            "duplicate_groups": groups,
            "dataset_fingerprint": dataset_fp,
            "settings_fingerprint": early_cache["settings_fingerprint"],
        },
        "frame": {
            "enabled": bool(frame_enabled),
            "mode": str(frame_mode or "suggest_only"),
            "auto_accept_types": list(frame_auto_accept_types or []),
            "pause_on_medium": bool(frame_pause_on_medium),
            "post_frame_duplicate_refresh": bool(post_frame_duplicate_refresh),
        },
    }
    atomic_write_json(os.path.join(output_root, PROJECT_STATE_FILENAME), workspace)
    atomic_write_json(os.path.join(output_root, PREFLIGHT_STATE_FILENAME), workspace["preflight"])
    return workspace


def load_workspace(input_folder: str, trigger_word: str) -> Dict[str, Any]:
    path = os.path.join(output_root_for(input_folder, trigger_word), PROJECT_STATE_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
