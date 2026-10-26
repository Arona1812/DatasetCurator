"""Profile-independent first descriptions written by GPT-6 Luna and cached per image."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from curator_core import atomic_write_json, stable_hash

VERSION = "initial-caption-v1"
ENGINE = "gpt-6-luna"
ENGINES = (ENGINE,)
# Local Florence 2 and JoyCaption runs were removed; saved settings move to Luna.
LEGACY_ENGINES = {"florence2": ENGINE, "joycaption": ENGINE}
PROMPT = (
    "Write a detailed, objective description of this image in English. Describe visible "
    "people, their poses, expressions, clothing, accessories, piercings, tattoos, framing, "
    "background and lighting. Assign attributes to the correct person when there is more "
    "than one. Do not invent details, identity, camera settings, or absence of features hidden from view."
)


def image_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_engine(value):
    """Return the supported provider for a saved value; empty stays empty (no first descriptions)."""
    engine = str(value or "").strip()
    if not engine:
        return ""
    engine = LEGACY_ENGINES.get(engine, engine)
    if engine not in ENGINES:
        raise ValueError(f"Unknown initial caption provider: {value}")
    return engine


def settings(engine=ENGINE):
    """Cache settings of a first description. The values match earlier releases, so stored Luna texts stay valid."""
    if engine not in ENGINES:
        raise ValueError(f"Unknown initial caption provider: {engine}")
    return {"version": VERSION, "engine": engine, "revision": "",
            "prompt": PROMPT, "max_new_tokens": 4096,
            "do_sample": False, "seed": 20261012,
            "quantization": "api",
            "image_preparation": "exif-rgb-full-resolution-v1"}


def caption_key(image_hash, config):
    # No profile, trigger, target, selection or caption switches belong here.
    return stable_hash({"image_sha256": image_hash, "settings": config})


def cached_caption(cache_dir, image_hash, config):
    path = Path(cache_dir) / (caption_key(image_hash, config) + ".json")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_record(record, image_hash, config)
        return record
    except (OSError, ValueError, TypeError, KeyError):
        return None


def validate_record(record, image_hash, config):
    if (not isinstance(record, dict) or record.get("image_sha256") != image_hash
            or record.get("settings") != config or record.get("status") != "ok"
            or record.get("hit_token_limit") or not str(record.get("caption") or "").strip()):
        raise ValueError("Incomplete, empty or mismatched initial caption")


def store_caption(cache_dir, image_hash, config, result):
    record = dict(result, image_sha256=image_hash, settings=config,
                  fingerprint=caption_key(image_hash, config))
    validate_record(record, image_hash, config)
    atomic_write_json(str(Path(cache_dir) / (record["fingerprint"] + ".json")), record)
    return record
