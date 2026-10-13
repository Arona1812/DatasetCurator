"""Profile-independent first captions. GPU dependencies live in a worker process."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from curator_core import atomic_write_json, stable_hash

VERSION = "initial-caption-v1"
ENGINES = ("florence2", "joycaption", "gpt-6-luna")
FLORENCE_TASKS = ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")
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


def settings(engine, task="<MORE_DETAILED_CAPTION>", revision=""):
    if engine not in ENGINES:
        raise ValueError(f"Unknown initial caption provider: {engine}")
    if engine == "florence2" and task not in FLORENCE_TASKS:
        raise ValueError(f"Unknown Florence task: {task}")
    return {"version": VERSION, "engine": engine, "revision": revision,
            "prompt": task if engine == "florence2" else PROMPT,
            "max_new_tokens": 512 if engine != "gpt-6-luna" else 4096,
            "do_sample": False, "seed": 20261012,
            "quantization": "fp16" if engine == "florence2" else "language-only-nf4-bf16" if engine == "joycaption" else "api",
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


def local_resources(root, engine, python_override="", weights_override=""):
    root = Path(root)
    python = Path(python_override) if python_override else root / (
        ".venv_florence" if engine == "florence2" else ".venv_caption") / "Scripts" / "python.exe"
    candidates = [root / "_cache" / "caption_models" / f"{engine}_weights.json",
                  root / "Datasets" / "caption_benchmark" / f"{engine}_weights.json"]
    weights_path = Path(weights_override) if weights_override else next((p for p in candidates if p.is_file()), candidates[0])
    if not python.is_file() or not weights_path.is_file():
        raise RuntimeError(f"{engine}: local runtime/model missing. Configure INITIAL_CAPTION_PYTHON and INITIAL_CAPTION_WEIGHTS; see CAPTION_PIPELINE.md.")
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    if not weights.get("revision") or not Path(weights.get("snapshot", "")).is_dir():
        raise RuntimeError(f"{engine}: weights must reference an existing pinned local snapshot.")
    return str(python), weights
