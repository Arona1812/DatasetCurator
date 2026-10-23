"""Read-only composition using the production backend in an isolated process."""
import argparse
import contextlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

from curator_core import atomic_write_json


def preview(request):
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="caption_preview_") as scratch:
        config_path = Path(scratch) / "config.json"
        config = dict(request["config"], INPUT_FOLDER=scratch, API_KEY="", ENABLE_CACHE=False,
                      USE_AI_TRIGGERWORD_CHECK=False, INITIAL_CAPTION_ENGINE="")
        atomic_write_json(str(config_path), config)
        old = os.environ.get("DATASET_CURATOR_CONFIG_PATH")
        os.environ["DATASET_CURATOR_CONFIG_PATH"] = str(config_path)
        try:
            spec = importlib.util.spec_from_file_location("caption_preview_backend", root / "dataset_curator_v2.py")
            backend = importlib.util.module_from_spec(spec)
            with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
                spec.loader.exec_module(backend)
        finally:
            if old is None:
                os.environ.pop("DATASET_CURATOR_CONFIG_PATH", None)
            else:
                os.environ["DATASET_CURATOR_CONFIG_PATH"] = old
        profile = request["profile"]
        if profile.get("profile_rebuild_required"):
            raise ValueError("Identity roles changed: rebuild the profile before previewing captions.")
        row = request["row"]
        if request["config"].get("CAPTION_COMPOSITION_MODE") == "luna_text" and int(row.get("face_count_local") or 0) <= 1:
            from caption_details import cached_details, enriched_row
            raw = str((row.get("initial_caption") or {}).get("caption") or "")
            details = cached_details(request["cache_dir"], raw) if raw else None
            if not details:
                raise ValueError("Luna text details are not prepared yet. Use the explicit API preparation button, or select local composition.")
            row = enriched_row(row, details)
        backend.USE_KREA_AI_CAPTIONING = False
        text = backend.adapt_caption_for_target(backend.TRAINING_TARGET,
            backend.build_local_caption(row, request.get("global_rules", {}), profile))
        if request.get("preview_image_path"):
            view = backend.body_aware_crop(backend.row_effective_image_path(row), row)
            view.thumbnail((1200, 1200))
            view.save(request["preview_image_path"], "JPEG", quality=92)
        return {"caption": text, "composition_source": "audited_observations_and_subject_profile"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("result")
    args = parser.parse_args()
    atomic_write_json(args.result, preview(json.loads(Path(args.request).read_text(encoding="utf-8"))))
