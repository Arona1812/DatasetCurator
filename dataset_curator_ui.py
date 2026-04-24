#!/usr/bin/env python3
"""
LoRA Dataset Curator – Gradio UI
================================
Steuert dataset_curator_v2.py und video_Processor.py ueber eine Weboberflaeche.
Schreibt _ui_config.json / _ui_video_config.json, startet die Skripte als
Subprocess und streamt Log + Bildvorschau live zurueck.

Einstellungen werden in _ui_settings.json gespeichert und beim naechsten Start
automatisch wiederhergestellt.
"""

import os
import re
import json
import subprocess
import sys
import threading
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
from PIL import Image

# ============================================================
# UI LANGUAGE / I18N
# ============================================================

# Default UI language. Is overwritten during build_ui() from settings.
UI_LANG = "en"  # "en" | "de"


def tr(de: str, en: str) -> str:
    """Very small translation helper (German/English)."""
    return en if UI_LANG == "en" else de


_LANG_LABEL_TO_CODE = {"English": "en", "Deutsch": "de"}
_LANG_CODE_TO_LABEL = {"en": "English", "de": "Deutsch"}


def _normalize_lang(value: Optional[str]) -> str:
    if not value:
        return "en"
    v = str(value).strip().lower()
    if v in ("en", "english"):
        return "en"
    if v in ("de", "deutsch", "german"):
        return "de"
    return "en"

# ============================================================
# PFADE
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURATOR_SCRIPT = os.path.join(SCRIPT_DIR, "dataset_curator_v2.py")
VIDEO_SCRIPT = os.path.join(SCRIPT_DIR, "video_Processor.py")
CURATOR_CONFIG = os.path.join(SCRIPT_DIR, "_ui_config.json")
VIDEO_CONFIG = os.path.join(SCRIPT_DIR, "_ui_video_config.json")
SETTINGS_PATH = os.path.join(SCRIPT_DIR, "_ui_settings.json")

# Used to temporarily apply language for the next UI start only.
# This keeps English as the default on a fresh start, while still allowing
# switching the UI language for the current session (by restarting the UI).
LANG_OVERRIDE_PATH = os.path.join(SCRIPT_DIR, "_ui_language_override.json")

VENV_PYTHON = os.path.join(SCRIPT_DIR, "curator_env", "Scripts", "python.exe")
if not os.path.isfile(VENV_PYTHON):
    VENV_PYTHON = sys.executable

_active_process: Optional[subprocess.Popen] = None


# ============================================================
# PERSISTENT SETTINGS
# ============================================================

# Alle Standardwerte an einer Stelle – wird sowohl fuer Defaults als
# auch fuer Save/Load verwendet.
SHARED_COMPACT_CAPTION_FIELDS: List[str] = [
    "include_glasses",
    "include_piercings",
    "include_makeup",
    "include_background",
    "include_lighting",
    "include_gaze",
    "include_expression",
    "include_hair_always",
    "include_beard_when_variable",
    "include_mirror_selfie_marker",
]

DEFAULTS: Dict[str, Any] = {
    # UI
    "ui_language": "en",
    # Curator Basis
    "c_trigger": "",
    "c_input": r"",
    "c_target": 30,
    "c_api_key": "",
    "c_model": "gpt-5.4-nano",
    "c_use_trigger_check": False,
    "c_trigger_model": "gpt-5.4-nano",
    "c_use_review_escalation": True,
    "c_review_escalation_model": "",
    "c_review_escalation_score_min": 50,
    "c_review_escalation_score_max": 65,
    "c_escalate_on_review": True,
    "c_escalate_on_conflict": True,
    "c_escalate_smart_crop": True,
    "c_smart_crop_escalation_delta": 10,
    # Shot-Verteilung
    "c_ratio_h": 0.50,
    "c_ratio_m": 0.35,
    "c_ratio_f": 0.15,
    # Qualitaet
    "c_keep_min": 55,
    "c_reject": 30,
    "c_min_side": 768,
    # Vorfilter
    "c_use_filesize": True,
    "c_min_filesize": 80,
    "c_use_blur": True,
    "c_min_blur": 25,
    "c_face_min_blur": 45,
    "c_blur_norm_edge": 512,
    "c_use_early_phash": True,
    # Subject-Sanity-Check (Gliedmassen-Filter)
    "c_subject_sanity": True,
    "c_subject_min_torso": 2,
    # IG-Frame-Detection
    "c_ig_frame_crop": True,
    "c_ig_two_stage_bar": True,
    # Duplikate
    "c_use_clip": True,
    "c_use_phash": True,
    "c_phash_thresh": 8,
    "c_clip_thresh": 0.985,
    # Smart Crop
    "c_smart_crop": True,
    "c_crop_gain": 8,
    "c_crop_pad": 1.5,
    # Clustering
    "c_use_cluster": True,
    "c_max_outfit": 4,
    "c_max_session": 5,
    "c_use_diversity": True,
    # Captions
    "c_caption_profile": "shared_compact",
    "c_captions": list(SHARED_COMPACT_CAPTION_FIELDS),
    # Export
    "c_exp_review": True,
    "c_exp_reject": True,
    "c_exp_compare": True,
    # Video Processor
    "v_source": r".\00_videos",
    "v_target": r".\00_input",
    "v_ref": r".\referenz.jpg",
    "v_fpm": 5,
    "v_fps": 2,
    "v_sim": 0.45,
    "v_sharp": 50,
}

CAPTION_FIELD_CHOICES: List[str] = [
    "include_gender_class",
    "include_skin_tone",
    "include_body_build",
    "include_tattoos",
    "include_glasses",
    "include_piercings",
    "include_makeup",
    "include_background",
    "include_lighting",
    "include_gaze",
    "include_expression",
    "include_hair_always",
    "include_hair_when_variable",
    "include_beard_always",
    "include_beard_when_variable",
    "include_mirror_selfie_marker",
    "include_eye_color",
]

CAPTION_PROFILE_PRESETS: Dict[str, List[str]] = {
    "shared_compact": list(SHARED_COMPACT_CAPTION_FIELDS),
}


def normalize_caption_profile(value: Optional[str]) -> str:
    v = (value or "").strip().lower()
    if v in {"shared_compact", "z_image_base", "ernie"}:
        return "shared_compact"
    if v in CAPTION_PROFILE_PRESETS:
        return v
    return "custom"


def caption_profile_choices() -> List[Tuple[str, str]]:
    return [
        (tr("Shared Compact (Z-Image + ERNIE)", "Shared Compact (Z-Image + ERNIE)"), "shared_compact"),
        (tr("Custom", "Custom"), "custom"),
    ]


def get_caption_preset_values(profile: Optional[str]) -> List[str]:
    normalized = normalize_caption_profile(profile)
    if normalized in CAPTION_PROFILE_PRESETS:
        return list(CAPTION_PROFILE_PRESETS[normalized])
    return list(DEFAULTS["c_captions"])


def detect_caption_profile(selected_fields: Optional[List[str]]) -> str:
    selected = set(selected_fields or [])
    for profile, preset_fields in CAPTION_PROFILE_PRESETS.items():
        if selected == set(preset_fields):
            return profile
    return "custom"


def load_settings() -> Dict[str, Any]:
    """Laedt gespeicherte Einstellungen, ergaenzt fehlende mit Defaults."""
    settings = dict(DEFAULTS)
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception:
            pass

    # If settings file exists but is missing keys (e.g. only language was saved),
    # ensure missing keys are filled from DEFAULTS.
    for k, v in DEFAULTS.items():
        settings.setdefault(k, v)

    # Language behavior:
    # - English should be the default on a fresh start.
    # - Switching to German/English should still be possible.
    # We implement this by using a one-shot override file that is written when
    # the user changes language; the next UI start consumes & deletes it.
    base_lang = "en"
    override_lang: Optional[str] = None
    if os.path.isfile(LANG_OVERRIDE_PATH):
        try:
            with open(LANG_OVERRIDE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "ui_language" in data:
                override_lang = _normalize_lang(data.get("ui_language"))
        except Exception:
            override_lang = None
        try:
            os.remove(LANG_OVERRIDE_PATH)
        except Exception:
            pass

    settings["ui_language"] = override_lang or base_lang

    # API Key aus Umgebungsvariable wenn nicht gespeichert
    if not settings.get("c_api_key"):
        settings["c_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    return settings


def save_ui_language(lang_code: str) -> str:
    """Persist only the UI language selection in _ui_settings.json."""
    global UI_LANG
    lang_code = _normalize_lang(lang_code)
    UI_LANG = lang_code
    try:
        # One-shot language override for the next UI start.
        with open(LANG_OVERRIDE_PATH, "w", encoding="utf-8") as f:
            json.dump({"ui_language": lang_code}, f, ensure_ascii=False, indent=2)
        return tr(
            "💾 Sprache gespeichert.",
            "💾 Language saved.",
        )
    except Exception as e:
        return tr(
            f"⚠️ Speichern fehlgeschlagen: {e}",
            f"⚠️ Save failed: {e}",
        )


def request_ui_restart() -> str:
    """Exit the process so the launcher can restart the UI."""

    def _exit_soon():
        time.sleep(0.5)
        # Exit code 5 is used by start_curator.bat to auto-restart.
        os._exit(5)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return tr(
        "🔄 UI startet neu...",
        "🔄 Restarting UI...",
    )


def save_language_and_restart(lang_code: str) -> str:
    """Save language selection and restart the UI so all labels update."""
    _ = save_ui_language(lang_code)
    return request_ui_restart()


def save_settings_fn(
    ui_language,
    # Curator
    c_trigger, c_input, c_target, c_api_key, c_model, c_use_trigger_check, c_trigger_model,
    c_use_review_escalation, c_review_escalation_model,
    c_review_escalation_score_min, c_review_escalation_score_max,
    c_escalate_on_review, c_escalate_on_conflict, c_escalate_smart_crop, c_smart_crop_escalation_delta,
    c_ratio_h, c_ratio_m, c_ratio_f,
    c_keep_min, c_reject, c_min_side,
    c_use_filesize, c_min_filesize,
    c_use_blur, c_min_blur, c_face_min_blur, c_blur_norm_edge,
    c_use_early_phash,
    c_subject_sanity, c_subject_min_torso,
    c_ig_frame_crop, c_ig_two_stage_bar,
    c_use_clip, c_use_phash, c_phash_thresh, c_clip_thresh,
    c_smart_crop, c_crop_gain, c_crop_pad,
    c_use_cluster, c_max_outfit, c_max_session, c_use_diversity,
    c_caption_profile,
    c_captions,
    c_exp_review, c_exp_reject, c_exp_compare,
    # Video
    v_source, v_target, v_ref, v_fpm, v_fps, v_sim, v_sharp,
):
    """Speichert alle aktuellen UI-Werte in _ui_settings.json."""
    data = {
        # Keep English as default in persisted settings.
        "ui_language": "en",
        "c_trigger": c_trigger, "c_input": c_input, "c_target": c_target,
        "c_api_key": c_api_key, "c_model": c_model,
        "c_use_trigger_check": c_use_trigger_check,
        "c_trigger_model": c_trigger_model,
        "c_use_review_escalation": c_use_review_escalation,
        "c_review_escalation_model": c_review_escalation_model,
        "c_review_escalation_score_min": c_review_escalation_score_min,
        "c_review_escalation_score_max": c_review_escalation_score_max,
        "c_escalate_on_review": c_escalate_on_review,
        "c_escalate_on_conflict": c_escalate_on_conflict,
        "c_escalate_smart_crop": c_escalate_smart_crop,
        "c_smart_crop_escalation_delta": c_smart_crop_escalation_delta,
        "c_ratio_h": c_ratio_h, "c_ratio_m": c_ratio_m, "c_ratio_f": c_ratio_f,
        "c_keep_min": c_keep_min, "c_reject": c_reject, "c_min_side": c_min_side,
        "c_use_filesize": c_use_filesize, "c_min_filesize": c_min_filesize,
        "c_use_blur": c_use_blur, "c_min_blur": c_min_blur,
        "c_face_min_blur": c_face_min_blur, "c_blur_norm_edge": c_blur_norm_edge,
        "c_use_early_phash": c_use_early_phash,
        "c_subject_sanity": c_subject_sanity,
        "c_subject_min_torso": c_subject_min_torso,
        "c_ig_frame_crop": c_ig_frame_crop,
        "c_ig_two_stage_bar": c_ig_two_stage_bar,
        "c_use_clip": c_use_clip, "c_use_phash": c_use_phash,
        "c_phash_thresh": c_phash_thresh, "c_clip_thresh": c_clip_thresh,
        "c_smart_crop": c_smart_crop, "c_crop_gain": c_crop_gain, "c_crop_pad": c_crop_pad,
        "c_use_cluster": c_use_cluster, "c_max_outfit": c_max_outfit,
        "c_max_session": c_max_session, "c_use_diversity": c_use_diversity,
        "c_caption_profile": normalize_caption_profile(c_caption_profile),
        "c_captions": c_captions,
        "c_exp_review": c_exp_review, "c_exp_reject": c_exp_reject,
        "c_exp_compare": c_exp_compare,
        "v_source": v_source, "v_target": v_target, "v_ref": v_ref,
        "v_fpm": v_fpm, "v_fps": v_fps, "v_sim": v_sim, "v_sharp": v_sharp,
    }
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return tr("💾 Einstellungen gespeichert.", "💾 Settings saved.")
    except Exception as e:
        return tr(f"⚠️ Speichern fehlgeschlagen: {e}", f"⚠️ Save failed: {e}")


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def _cleanup_stale_configs():
    for cfg in (CURATOR_CONFIG, VIDEO_CONFIG):
        try:
            if os.path.exists(cfg):
                os.remove(cfg)
        except Exception:
            pass


_cleanup_stale_configs()


def scan_images(folder: str, limit: int = 60) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(glob(os.path.join(folder, ext)))
    imgs.sort(key=os.path.getmtime, reverse=True)
    return imgs[:limit]


def load_gallery_image(path: str, max_size: Tuple[int, int] = (1600, 1600)) -> Optional[Image.Image]:
    """Load an image into memory so Gradio does not need direct filesystem access."""
    try:
        with Image.open(path) as img:
            preview = img.convert("RGB")
            resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            preview.thumbnail(max_size, resampling)
            return preview
    except Exception:
        return None


def load_gallery_images(paths: List[str], max_size: Tuple[int, int] = (1600, 1600)) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        preview = load_gallery_image(path, max_size=max_size)
        if preview is not None:
            images.append(preview)
    return images


def build_gallery_with_captions(
    paths: List[str],
    captions: List[str],
    max_size: Tuple[int, int] = (1600, 1600),
) -> List[Tuple[Image.Image, str]]:
    gallery_data: List[Tuple[Image.Image, str]] = []
    for path, caption in zip(paths, captions):
        preview = load_gallery_image(path, max_size=max_size)
        if preview is not None:
            gallery_data.append((preview, caption))
    return gallery_data


def output_root_for(input_folder: str, trigger_word: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", trigger_word.strip(), flags=re.UNICODE).strip("_") or "subject"
    return os.path.join(input_folder, f"curated_{safe}")


def parse_progress(line: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"\[(\d+)/(\d+)\]", line)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def kill_process():
    global _active_process
    if _active_process and _active_process.poll() is None:
        try:
            _active_process.terminate()
            _active_process.wait(timeout=5)
        except Exception:
            try:
                _active_process.kill()
            except Exception:
                pass
    _active_process = None
    _cleanup_stale_configs()
    return tr("⏹ Prozess abgebrochen.", "⏹ Process cancelled.")


# ============================================================
# PROZESS-RUNNER
# ============================================================

def run_script(
    script_path: str,
    config_path: str,
    config_data: dict,
    image_scan_folder: Optional[str] = None,
) -> Generator:
    global _active_process

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Ensure the curator script can always access the key via environment as well.
    # (It still reads _ui_config.json, but env provides a robust fallback.)
    api_key = (config_data.get("API_KEY") or "").strip()
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    popen_kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=SCRIPT_DIR,
        env=env,
    )
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    _active_process = subprocess.Popen(
        [VENV_PYTHON, script_path],
        **popen_kwargs,
    )

    log_lines: List[str] = []
    progress = 0.0
    last_gallery_update = 0
    images: List[Any] = []

    try:
        for line in _active_process.stdout:
            line = line.rstrip("\n\r")
            log_lines.append(line)

            p = parse_progress(line)
            if p:
                idx, total = p
                progress = idx / max(1, total)

            processed_count = sum(1 for l in log_lines if re.match(r"\s*\[\d+/\d+\]", l))
            if image_scan_folder and processed_count - last_gallery_update >= 5:
                images = load_gallery_images(scan_images(image_scan_folder))
                last_gallery_update = processed_count

            log_text = "\n".join(log_lines[-500:])
            yield log_text, images, progress, tr(
                f"⏳ Läuft... ({int(progress*100)}%)",
                f"⏳ Running... ({int(progress*100)}%)",
            )

    except Exception as e:
        log_lines.append(tr(f"\n⚠️ Fehler: {e}", f"\n⚠️ Error: {e}"))

    try:
        rc = _active_process.wait(timeout=30)
    except Exception:
        _active_process.kill()
        rc = -1

    _active_process = None

    try:
        if os.path.exists(config_path):
            os.remove(config_path)
    except Exception:
        pass

    if image_scan_folder:
        images = load_gallery_images(scan_images(image_scan_folder))

    log_text = "\n".join(log_lines[-500:])
    status = (
        tr(
            f"✅ Fertig! ({len(log_lines)} Zeilen)",
            f"✅ Done! ({len(log_lines)} lines)",
        )
        if rc == 0
        else tr(f"❌ Fehlercode {rc}", f"❌ Exit code {rc}")
    )
    yield log_text, images, 1.0, status


# ============================================================
# CURATOR LAUNCHER
# ============================================================

def start_curator(
    trigger_word, input_folder, target_size, api_key, ai_model, use_trigger_check, trigger_check_model,
    use_review_escalation, review_escalation_model,
    review_escalation_score_min, review_escalation_score_max,
    escalate_on_review, escalate_on_conflict, escalate_smart_crop, smart_crop_escalation_delta,
    ratio_h, ratio_m, ratio_f,
    keep_score_min, hard_reject_score, hard_min_side,
    use_filesize_filter, min_filesize_kb,
    use_blur_filter, min_blur_variance, face_min_blur_variance, blur_norm_edge,
    use_early_phash,
    subject_sanity, subject_min_torso,
    ig_frame_crop, ig_two_stage_bar,
    use_clip, use_phash, phash_threshold, clip_threshold,
    enable_smart_crop, crop_min_gain, crop_padding,
    use_clustering, max_outfit, max_session, use_diversity,
    caption_profile,
    caption_options,
    export_review, export_reject, export_crop_compare,
):
    if not trigger_word.strip():
        yield tr("Bitte ein Triggerwort eingeben.", "Please enter a trigger word."), [], 0, tr("❌ Fehler", "❌ Error")
        return
    if not os.path.isdir(input_folder):
        yield tr(
            f"Input-Ordner existiert nicht: {input_folder}",
            f"Input folder does not exist: {input_folder}",
        ), [], 0, tr("❌ Fehler", "❌ Error")
        return
    if not api_key.strip():
        yield tr("Bitte einen OpenAI API Key eingeben.", "Please enter an OpenAI API key."), [], 0, tr("❌ Fehler", "❌ Error")
        return

    all_caption_keys = list(CAPTION_FIELD_CHOICES)
    caption_policy = {k: (k in caption_options) for k in all_caption_keys}

    config = {
        "TRIGGER_WORD": trigger_word.strip(),
        "INPUT_FOLDER": input_folder.strip(),
        "TARGET_DATASET_SIZE": int(target_size),
        "API_KEY": api_key.strip(),
        "AI_MODEL": ai_model.strip(),
        "USE_AI_TRIGGERWORD_CHECK": use_trigger_check,
        "TRIGGER_CHECK_MODEL": trigger_check_model.strip() or ai_model.strip(),
        "USE_REVIEW_ESCALATION": use_review_escalation,
        "REVIEW_ESCALATION_MODEL": review_escalation_model.strip(),
        "REVIEW_ESCALATION_SCORE_MIN": int(review_escalation_score_min),
        "REVIEW_ESCALATION_SCORE_MAX": int(review_escalation_score_max),
        "ESCALATE_ON_REVIEW_STATUS": escalate_on_review,
        "ESCALATE_ON_STATUS_CONFLICT": escalate_on_conflict,
        "ESCALATE_SMART_CROP_CLOSE_CALLS": escalate_smart_crop,
        "SMART_CROP_ESCALATION_MAX_DELTA": float(smart_crop_escalation_delta),
        "RATIO_HEADSHOT": round(ratio_h, 2),
        "RATIO_MEDIUM": round(ratio_m, 2),
        "RATIO_FULL_BODY": round(ratio_f, 2),
        "KEEP_SCORE_MIN": int(keep_score_min),
        "HARD_REJECT_SCORE": int(hard_reject_score),
        "HARD_MIN_SIDE_PX": int(hard_min_side),
        "USE_MIN_FILESIZE_FILTER": use_filesize_filter,
        "HARD_MIN_FILESIZE_KB": int(min_filesize_kb),
        "USE_BLUR_FILTER": use_blur_filter,
        "HARD_MIN_BLUR_VARIANCE": float(min_blur_variance),
        "FACE_MIN_BLUR_VARIANCE": float(face_min_blur_variance),
        "BLUR_NORMALIZE_LONG_EDGE": int(blur_norm_edge),
        "ENABLE_SUBJECT_SANITY_CHECK": subject_sanity,
        "SUBJECT_MIN_TORSO_LANDMARKS": int(subject_min_torso),
        "ENABLE_IG_FRAME_CROP": ig_frame_crop,
        "IG_FRAME_TWO_STAGE_BAR_DETECT": ig_two_stage_bar,
        "USE_EARLY_PHASH_DEDUP": use_early_phash,
        "USE_CLIP_DUPLICATE_SCORING": use_clip,
        "USE_PHASH_DUPLICATE_SCORING": use_phash,
        "PHASH_HAMMING_THRESHOLD": int(phash_threshold),
        "CLIP_COSINE_THRESHOLD": float(clip_threshold),
        "ENABLE_SMART_PRECROP": enable_smart_crop,
        "SMART_PRECROP_MIN_GAIN": float(crop_min_gain),
        "SMART_PRECROP_PADDING_FACTOR": float(crop_padding),
        "USE_SESSION_OUTFIT_CLUSTERING": use_clustering,
        "MAX_PER_OUTFIT_CLUSTER": int(max_outfit),
        "MAX_PER_SESSION_CLUSTER": int(max_session),
        "ENABLE_DIVERSITY_PENALTIES": use_diversity,
        "CAPTION_PROFILE": normalize_caption_profile(caption_profile),
        "CAPTION_POLICY": caption_policy,
        "EXPORT_REVIEW_IMAGES": export_review,
        "EXPORT_REJECT_IMAGES": export_reject,
        "EXPORT_SMART_CROP_COMPARISON": export_crop_compare,
        "SEND_TEXT_IMAGES_TO_CAPTION_REMOVE": True,
        "INTERACTIVE_CAPTION_OVERRIDE": False,
    }

    train_dir = os.path.join(output_root_for(input_folder, trigger_word), "01_train_ready")
    yield from run_script(CURATOR_SCRIPT, CURATOR_CONFIG, config, train_dir)


# ============================================================
# VIDEO PROCESSOR LAUNCHER
# ============================================================

def start_video(
    source_folder, target_folder, reference_image,
    frames_per_min, sample_fps, similarity, min_sharpness,
):
    if not os.path.isdir(source_folder):
        yield tr(
            f"Video-Ordner existiert nicht: {source_folder}",
            f"Video folder does not exist: {source_folder}",
        ), [], 0, tr("❌ Fehler", "❌ Error")
        return
    if not os.path.isfile(reference_image):
        yield tr(
            f"Referenzbild nicht gefunden: {reference_image}",
            f"Reference image not found: {reference_image}",
        ), [], 0, tr("❌ Fehler", "❌ Error")
        return

    config = {
        "SOURCE_FOLDER": source_folder.strip(),
        "TARGET_FOLDER": target_folder.strip(),
        "REFERENCE_IMAGE": reference_image.strip(),
        "FRAMES_PER_MINUTE": int(frames_per_min),
        "SAMPLE_FPS": int(sample_fps),
        "SIMILARITY_THRESHOLD": float(similarity),
        "MIN_SHARPNESS": float(min_sharpness),
    }
    os.makedirs(target_folder, exist_ok=True)
    yield from run_script(VIDEO_SCRIPT, VIDEO_CONFIG, config, target_folder)


# ============================================================
# ERGEBNIS-BROWSER
# ============================================================

def load_results(input_folder, trigger_word, subfolder):
    root = output_root_for(input_folder, trigger_word)
    # Use stable internal values so UI labels can be translated.
    folder_map = {
        "train_ready": "01_train_ready",
        "caption_remove": "02_caption_remove",
        "review": "03_review",
        "reject": "04_reject",
        "smart_crop_pairs": "07_smart_crop_pairs",

        # Backward compatibility (older UI values)
        "Train Ready": "01_train_ready",
        "Caption Remove": "02_caption_remove",
        "Review": "03_review",
        "Reject": "04_reject",
        "Smart Crop Paare": "07_smart_crop_pairs",
    }
    target = os.path.join(root, folder_map.get(subfolder, "01_train_ready"))
    image_paths = scan_images(target, limit=100)

    captions = []
    for img_path in image_paths:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                captions.append(f.read().strip()[:200])
        else:
            captions.append("")

    gallery_data = build_gallery_with_captions(image_paths, captions)

    safe = re.sub(r"[^\w\-]+", "_", trigger_word.strip()).strip("_") or "subject"
    report_path = os.path.join(root, f"dataset_report_{safe}.md")
    report = ""
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

    info = tr(
        f"📁 {target}\n📷 {len(image_paths)} Bilder gefunden",
        f"📁 {target}\n📷 {len(image_paths)} images found",
    )
    return gallery_data, report, info


# ============================================================
# GRADIO LAYOUT
# ============================================================

def build_ui() -> gr.Blocks:

    S = load_settings()

    # Make translations in this UI build consistent.
    global UI_LANG
    UI_LANG = _normalize_lang(S.get("ui_language"))

    with gr.Blocks(
        title=tr("LoRA Dataset Curator", "LoRA Dataset Curator"),
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=".log-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 12px !important; }",
    ) as app:

        with gr.Row():
            ui_language = gr.Dropdown(
                label=tr("Sprache", "Language"),
                choices=[("English", "en"), ("Deutsch", "de")],
                value=UI_LANG,
                info=tr(
                    "UI-Sprache wählen (wird gespeichert) – die UI startet danach automatisch neu.",
                    "Select UI language (will be saved) – UI will auto-restart afterwards.",
                ),
                scale=1,
            )
            ui_lang_status = gr.Textbox(
                label=tr("Hinweis", "Notice"),
                interactive=False,
                max_lines=2,
                scale=3,
            )

            ui_restart_btn = gr.Button(
                tr("🔄 UI neu starten", "🔄 Restart UI"),
                variant="secondary",
                scale=1,
            )

        ui_language.change(fn=save_language_and_restart, inputs=[ui_language], outputs=[ui_lang_status])
        ui_restart_btn.click(fn=request_ui_restart, outputs=[ui_lang_status])

        gr.Markdown(tr("# 🖼️ LoRA Dataset Curator", "# 🖼️ LoRA Dataset Curator"))
        gr.Markdown(
            tr(
                "Dataset-Aufbereitung und Video-Extraktion für LoRA-Training with [AI Toolkit](https://github.com/ostris/ai-toolkit)",
                "Dataset curation and video extraction for LoRA training with [AI Toolkit](https://github.com/ostris/ai-toolkit)",
            )
        )

        with gr.Tabs():

            # ==============================================================
            # TAB 1: DATASET CURATOR
            # ==============================================================
            with gr.TabItem(tr("📸 Dataset Curator", "📸 Dataset Curator")):

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(tr("### Basis-Einstellungen", "### Basic Settings"))
                        c_trigger = gr.Textbox(
                            label=tr("Trigger Word", "Trigger Word"),
                            value=S["c_trigger"],
                            info=tr(
                                "Eindeutiges Wort, das im Training die Person identifiziert. Sollte bei bestimmten Modellen kein realer Name oder Alltagsbegriff sein.",
                                "Unique word that identifies the subject during training. For certain models this should not be a real name or common word.",
                            ),
                            max_lines=1,
                        )
                        c_input = gr.Textbox(
                            label=tr("Input-Ordner (Bilder)", "Input folder (images)"),
                            value=S["c_input"],
                            info=tr(
                                "Ordner mit den Quellbildern. Unterordner werden nicht durchsucht.",
                                "Folder containing the source images. Subfolders are not scanned.",
                            ),
                            max_lines=1,
                        )
                        c_target = gr.Slider(
                            label=tr("Ziel-Datensatzgröße", "Target dataset size"),
                            minimum=5,
                            maximum=200,
                            step=1,
                            value=S["c_target"],
                            info=tr(
                                "Wie viele Bilder das finale Training-Set haben soll. Qualität geht vor Füllmaterial.",
                                "How many images the final training set should contain. Quality over filler images.",
                            ),
                        )
                        c_api_key = gr.Textbox(
                            label=tr("OpenAI API Key", "OpenAI API Key"),
                            value=S["c_api_key"],
                            type="password",
                            info=tr(
                                "Wird für die Bildanalyse benötigt. Kann auch als Umgebungsvariable OPENAI_API_KEY gesetzt werden.",
                                "Required for image analysis. Can also be set via environment variable OPENAI_API_KEY.",
                            ),
                            max_lines=1,
                        )
                        c_model = gr.Textbox(
                            label=tr("Primäres AI-Modell", "Primary AI model"),
                            value=S["c_model"],
                            info=tr(
                                "Hauptmodell für den ersten Audit-Durchlauf. Günstigere Modelle sind schneller, bewerten aber oft ungenauer.",
                                "Main model for the first audit pass. Cheaper models are faster, but often less accurate.",
                            ),
                            max_lines=1,
                        )
                        c_use_trigger_check = gr.Checkbox(
                            label=tr("Trigger-Check aktivieren", "Enable trigger check"),
                            value=S["c_use_trigger_check"],
                            info=tr(
                                "Prüft das Triggerwort per KI auf Kollisionen oder problematische Ähnlichkeiten. Wenn deaktiviert, wird die Prüfung komplett übersprungen.",
                                "Checks the trigger word via AI for collisions or problematic similarities. If disabled, the check is skipped entirely.",
                            ),
                        )
                        c_trigger_model = gr.Textbox(
                            label=tr("Trigger-Check-Modell", "Trigger-check model"),
                            value=S["c_trigger_model"],
                            info=tr(
                                "Optional separates Modell für die Triggerwort-Prüfung. Leer = primäres Modell verwenden. Wird nur genutzt, wenn der Trigger-Check aktiviert ist.",
                                "Optional separate model for trigger-word checks. Empty = use primary model. Only used when trigger check is enabled.",
                            ),
                            max_lines=1,
                        )

                with gr.Accordion(tr("🧠 Modellstrategie & Eskalation", "🧠 Model strategy & escalation"), open=False):
                    c_use_review_escalation = gr.Checkbox(
                        label=tr("Eskalation für schwierige Fälle aktivieren", "Enable escalation for difficult cases"),
                        value=S["c_use_review_escalation"],
                        info=tr(
                            "Schickt Grenzfälle optional an ein stärkeres zweites Modell.",
                            "Optionally sends borderline cases to a stronger second model.",
                        ),
                    )
                    with gr.Row():
                        c_review_escalation_model = gr.Textbox(
                            label=tr("Eskalationsmodell", "Escalation model"),
                            value=S["c_review_escalation_model"],
                            info=tr(
                                "Stärkeres Modell für Review-/Grenzfälle. Leer = Eskalation effektiv aus.",
                                "Stronger model for review/borderline cases. Empty = escalation effectively off.",
                            ),
                            max_lines=1,
                        )
                    with gr.Row():
                        c_review_escalation_score_min = gr.Slider(
                            label=tr("Eskalation Score-Min", "Escalation score min"),
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=S["c_review_escalation_score_min"],
                        )
                        c_review_escalation_score_max = gr.Slider(
                            label=tr("Eskalation Score-Max", "Escalation score max"),
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=S["c_review_escalation_score_max"],
                        )
                    with gr.Row():
                        c_escalate_on_review = gr.Checkbox(
                            label=tr("Bei Review eskalieren", "Escalate on review"),
                            value=S["c_escalate_on_review"],
                        )
                        c_escalate_on_conflict = gr.Checkbox(
                            label=tr("Bei Konflikt eskalieren", "Escalate on conflict"),
                            value=S["c_escalate_on_conflict"],
                        )
                        c_escalate_smart_crop = gr.Checkbox(
                            label=tr("Knappes Smart-Crop-Duell eskalieren", "Escalate close smart-crop duel"),
                            value=S["c_escalate_smart_crop"],
                        )
                    c_smart_crop_escalation_delta = gr.Slider(
                        label=tr("Max. Crop-Differenz für Eskalation", "Max crop delta for escalation"),
                        minimum=0,
                        maximum=30,
                        step=1,
                        value=S["c_smart_crop_escalation_delta"],
                        info=tr(
                            "Wenn Original und Crop nur so viele Punkte auseinanderliegen, wird optional das stärkere Modell gefragt.",
                            "If original and crop are only this many points apart, the stronger model can optionally decide.",
                        ),
                    )

                    with gr.Column(scale=1):
                        gr.Markdown(tr("### Shot-Verteilung", "### Shot distribution"))
                        c_ratio_h = gr.Slider(
                            label=tr("Headshot", "Headshot"),
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=S["c_ratio_h"],
                            info=tr(
                                "Anteil Nahaufnahmen. Wichtigste Kategorie für Identitätslernen.",
                                "Share of close-ups. Most important category for identity learning.",
                            ),
                        )
                        c_ratio_m = gr.Slider(
                            label=tr("Medium", "Medium"),
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=S["c_ratio_m"],
                            info=tr(
                                "Anteil Oberkörper-Aufnahmen. Hilft dem Modell Stil, Körperbau und Haltung zu lernen.",
                                "Share of upper-body shots. Helps the model learn style, body and posture.",
                            ),
                        )
                        c_ratio_f = gr.Slider(
                            label=tr("Full Body", "Full Body"),
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=S["c_ratio_f"],
                            info=tr(
                                "Anteil Ganzkörper-Aufnahmen. Weniger nötig, aber wichtig für vollständige Darstellung.",
                                "Share of full-body shots. Less needed, but important for full representation.",
                            ),
                        )
                        gr.Markdown(tr("*⚠️ Summe sollte 1.0 ergeben*", "*⚠️ Sum should be 1.0*"))

                with gr.Accordion(tr("⚙️ Qualität & Schwellwerte", "⚙️ Quality & thresholds"), open=False):
                    with gr.Row():
                        c_keep_min = gr.Slider(
                            label=tr("Keep Score Min", "Keep score min"),
                            minimum=0,
                            maximum=100,
                            step=5,
                            value=S["c_keep_min"],
                            info=tr(
                                "Mindest-Score (0–100) damit ein Bild als 'keep' gilt. Darunter wird es 'review'.",
                                "Minimum score (0–100) for an image to be considered 'keep'. Below that it becomes 'review'.",
                            ),
                        )
                        c_reject = gr.Slider(
                            label=tr("Hard Reject Score", "Hard reject score"),
                            minimum=0,
                            maximum=100,
                            step=5,
                            value=S["c_reject"],
                            info=tr(
                                "Unter diesem Score wird ein Bild sofort verworfen, ohne Review-Chance.",
                                "Below this score an image is rejected immediately (no review chance).",
                            ),
                        )
                        c_min_side = gr.Slider(
                            label=tr("Min. Seitenlänge (px)", "Min side length (px)"),
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=S["c_min_side"],
                            info=tr(
                                "Bilder mit kürzerer Seite werden sofort verworfen.",
                                "Images with a shorter side are rejected immediately.",
                            ),
                        )

                with gr.Accordion(tr("🔍 Vorfilter (lokal, ohne API-Kosten)", "🔍 Pre-filters (local, no API cost)"), open=False):
                    gr.Markdown(tr(
                        "*Diese Filter laufen vor dem API-Call und sparen Kosten bei großen Datensätzen.*",
                        "*These filters run before the API call and save cost for large datasets.*",
                    ))
                    with gr.Row():
                        c_use_filesize = gr.Checkbox(
                            label=tr("Dateigröße-Filter", "File size filter"),
                            value=S["c_use_filesize"],
                            info=tr(
                                "Verwirft sehr kleine Dateien, die meist stark komprimiert und für Training unbrauchbar sind.",
                                "Rejects very small files that are usually heavily compressed and not useful for training.",
                            ),
                        )
                        c_min_filesize = gr.Slider(
                            label=tr("Min. KB", "Min KB"),
                            minimum=10,
                            maximum=500,
                            step=10,
                            value=S["c_min_filesize"],
                            info=tr(
                                "Dateien unter diesem Wert (in Kilobyte) werden verworfen.",
                                "Files below this value (in kilobytes) are rejected.",
                            ),
                        )
                    with gr.Row():
                        c_use_blur = gr.Checkbox(
                            label=tr("Unschärfe-Filter (zweistufig)", "Blur filter (two-stage)"),
                            value=S["c_use_blur"],
                            info=tr(
                                "Stufe 1 prüft das ganze Bild vor der API (laxer Totalausfall-Filter). Stufe 2 prüft gezielt die Gesichtsregion nach der Gesichtserkennung. Beide Messungen laufen auf der normierten Bildgröße.",
                                "Stage 1 checks the whole image before the API (lax total-failure filter). Stage 2 targets the face region after face detection. Both measurements run on the normalized size.",
                            ),
                        )
                        c_min_blur = gr.Slider(
                            label=tr("Stufe 1: Min. Varianz (Gesamtbild)", "Stage 1: min variance (full image)"),
                            minimum=5,
                            maximum=200,
                            step=5,
                            value=S["c_min_blur"],
                            info=tr(
                                "Totalausfall-Schwelle auf dem normierten Gesamtbild. Niedriger = milder. 25 = nur komplett verwackelte Bilder werden gefiltert.",
                                "Total-failure threshold on the normalized full image. Lower = milder. 25 = only completely blurry images are rejected.",
                            ),
                        )
                    with gr.Row():
                        c_face_min_blur = gr.Slider(
                            label=tr("Stufe 2: Min. Varianz (Gesichts-Bbox)", "Stage 2: min variance (face bbox)"),
                            minimum=10,
                            maximum=200,
                            step=5,
                            value=S["c_face_min_blur"],
                            info=tr(
                                "Schärfe-Schwelle in der Face-Bbox nach API+Gesichtserkennung. 45 = konservativ, lässt Beauty-Filter-Selfies durch. 70+ = streng.",
                                "Sharpness threshold inside the face bbox after API+face detection. 45 = conservative, keeps beauty-filter selfies. 70+ = strict.",
                            ),
                        )
                        c_blur_norm_edge = gr.Slider(
                            label=tr("Normierungs-Kantenlänge (px)", "Normalization edge size (px)"),
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=S["c_blur_norm_edge"],
                            info=tr(
                                "Vor der Blur-Messung werden alle Bilder auf diese längste Kante skaliert. Macht Schwellenwerte auflösungsunabhängig. 512 ist Standard.",
                                "Before blur measurement, all images are resized to this long edge. Makes thresholds resolution-independent. 512 is the default.",
                            ),
                        )
                    c_use_early_phash = gr.Checkbox(
                        label=tr("Early pHash-Dedup (vor API)", "Early pHash dedup (pre-API)"),
                        value=S["c_use_early_phash"],
                        info=tr(
                            "Findet pixelnahe Duplikate lokal VOR der API. Spart API-Kosten.",
                            "Finds near pixel-duplicates locally before the API. Saves API cost.",
                        ),
                    )

                with gr.Accordion(tr("🖼️ Instagram-Frame / UI-Rand-Entfernung", "🖼️ Instagram frame / UI border removal"), open=False):
                    gr.Markdown(tr(
                        "*Erkennt und entfernt IG-Story-Rahmen, Schatten-Gradienten und Android-Nav-Bars VOR der API, damit die Analyse auf dem bereinigten Bild läuft.*",
                        "*Detects and removes IG-story frames, drop-shadow gradients and Android nav bars BEFORE the API so the analysis runs on the cleaned image.*",
                    ))
                    with gr.Row():
                        c_ig_frame_crop = gr.Checkbox(
                            label=tr("IG-Frame-Crop aktivieren", "Enable IG frame crop"),
                            value=S["c_ig_frame_crop"],
                            info=tr(
                                "Hauptschalter. Bei 'aus' werden Bilder unverändert weitergegeben.",
                                "Main switch. When off, images are passed through unchanged.",
                            ),
                        )
                        c_ig_two_stage_bar = gr.Checkbox(
                            label=tr("Zweistufige Bar-Detection (oben/unten)", "Two-stage bar detection (top/bottom)"),
                            value=S["c_ig_two_stage_bar"],
                            info=tr(
                                "Zusätzliche Erkennung für Android-Nav-Bars (schwarz + UI-Icons) und Drop-Shadow-Gradienten. Triggert nur, wenn schon ein Seitenrand gefunden wurde – verhindert False-Positives bei dunklen Kissen/Haaren.",
                                "Additional detection for Android nav bars (black + UI icons) and drop-shadow gradients. Only triggers when a side border was already detected – prevents false positives on dark pillows/hair.",
                            ),
                        )

                with gr.Accordion(tr("🧍 Subject-Sanity-Check (Gliedmaßen-Filter)", "🧍 Subject sanity check (limb filter)"), open=False):
                    gr.Markdown(tr(
                        "*Verwirft Bilder ohne sichtbares Gesicht UND ohne erkennbaren Torso (nur Füße, Hände o.ä.). Rückenansichten mit klarem Torso bleiben erhalten.*",
                        "*Rejects images with neither a visible face nor a recognizable torso (feet only, hands only, etc.). Back-view shots with a clear torso are kept.*",
                    ))
                    with gr.Row():
                        c_subject_sanity = gr.Checkbox(
                            label=tr("Sanity-Check aktivieren", "Enable sanity check"),
                            value=S["c_subject_sanity"],
                            info=tr(
                                "Nutzt MediaPipe-Pose und prüft Schulter/Hüft-Landmarks, wenn kein Gesicht sichtbar ist.",
                                "Uses MediaPipe pose and checks shoulder/hip landmarks when no face is visible.",
                            ),
                        )
                        c_subject_min_torso = gr.Slider(
                            label=tr("Min. Torso-Landmarks (von 4)", "Min torso landmarks (of 4)"),
                            minimum=1,
                            maximum=4,
                            step=1,
                            value=S["c_subject_min_torso"],
                            info=tr(
                                "Wie viele der 4 Landmarks (2 Schultern, 2 Hüften) sichtbar sein müssen. 2 = Standard: halber Torso reicht. 4 = sehr streng.",
                                "How many of the 4 landmarks (2 shoulders, 2 hips) must be visible. 2 = default: half torso is enough. 4 = very strict.",
                            ),
                        )

                with gr.Accordion(tr("🔗 Duplikaterkennung (nach API)", "🔗 Duplicate detection (post-API)"), open=False):
                    gr.Markdown(tr(
                        "*Erkennt Duplikate und zu ähnliche Bilder nach der Analyse.*",
                        "*Detects duplicates and overly similar images after analysis.*",
                    ))
                    with gr.Row():
                        c_use_clip = gr.Checkbox(
                            label=tr("CLIP Semantik-Dedup", "CLIP semantic dedup"),
                            value=S["c_use_clip"],
                            info=tr(
                                "Erkennt inhaltlich ähnliche Bilder (gleiches Outfit, ähnliche Pose) per CLIP ViT-L-14.",
                                "Detects semantically similar images (same outfit, similar pose) using CLIP ViT-L-14.",
                            ),
                        )
                        c_use_phash = gr.Checkbox(
                            label=tr("pHash Pixel-Dedup", "pHash pixel dedup"),
                            value=S["c_use_phash"],
                            info=tr(
                                "Erkennt visuell nahezu identische Bilder per perceptual Hash.",
                                "Detects visually near-identical images using perceptual hash (pHash).",
                            ),
                        )
                    with gr.Row():
                        c_phash_thresh = gr.Slider(
                            label=tr("pHash Hamming-Schwelle", "pHash hamming threshold"),
                            minimum=2,
                            maximum=20,
                            step=1,
                            value=S["c_phash_thresh"],
                            info=tr(
                                "Max. Hamming-Distanz. Niedriger = strenger. 8 = guter Kompromiss.",
                                "Max hamming distance. Lower = stricter. 8 = good compromise.",
                            ),
                        )
                        c_clip_thresh = gr.Slider(
                            label=tr("CLIP Cosine-Schwelle", "CLIP cosine threshold"),
                            minimum=0.90,
                            maximum=1.0,
                            step=0.005,
                            value=S["c_clip_thresh"],
                            info=tr(
                                "Ab dieser Similarity gelten zwei Bilder als identisch. 0.985 = konservativ.",
                                "At/above this similarity two images are considered identical. 0.985 = conservative.",
                            ),
                        )

                with gr.Accordion(tr("✂️ Smart Pre-Crop", "✂️ Smart pre-crop"), open=False):
                    gr.Markdown(tr(
                        "*Bei Full-Body/Medium-Bildern mit kleinem Gesicht wird automatisch ein Headshot-Crop erzeugt und gegen das Original bewertet.*",
                        "*For full-body/medium images with small face, an automatic headshot crop is generated and compared against the original.*",
                    ))
                    c_smart_crop = gr.Checkbox(
                        label=tr("Smart Pre-Crop aktivieren", "Enable smart pre-crop"),
                        value=S["c_smart_crop"],
                        info=tr(
                            "Erzeugt automatisch Headshot-Zuschnitte für Bilder mit kleinem Gesicht.",
                            "Automatically generates headshot crops for images with a small face.",
                        ),
                    )
                    with gr.Row():
                        c_crop_gain = gr.Slider(
                            label=tr("Min. Score-Gewinn", "Min score gain"),
                            minimum=0,
                            maximum=30,
                            step=1,
                            value=S["c_crop_gain"],
                            info=tr(
                                "Crop muss so viele Punkte besser sein als Original, um übernommen zu werden.",
                                "Crop must score this many points higher than the original to be accepted.",
                            ),
                        )
                        c_crop_pad = gr.Slider(
                            label=tr("Padding-Faktor", "Padding factor"),
                            minimum=0.5,
                            maximum=3.0,
                            step=0.1,
                            value=S["c_crop_pad"],
                            info=tr(
                                "Rand um das Gesicht beim Crop. 1.5 = 1.5× Gesichtsgröße als Rahmen.",
                                "Margin around the face during cropping. 1.5 = 1.5× face size as margin.",
                            ),
                        )

                with gr.Accordion(tr("📊 Clustering & Diversität", "📊 Clustering & diversity"), open=False):
                    gr.Markdown(tr(
                        "*Begrenzt zu viele ähnliche Bilder derselben Session/desselben Outfits.*",
                        "*Limits too many similar images from the same session/outfit.*",
                    ))
                    c_use_cluster = gr.Checkbox(
                        label=tr("Session/Outfit-Clustering", "Session/outfit clustering"),
                        value=S["c_use_cluster"],
                        info=tr(
                            "Gruppiert nach Kleidung, Hintergrund und Aufnahmezeit.",
                            "Groups by clothing, background, and capture time.",
                        ),
                    )
                    with gr.Row():
                        c_max_outfit = gr.Slider(
                            label=tr("Max pro Outfit", "Max per outfit"),
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=S["c_max_outfit"],
                            info=tr(
                                "Maximal so viele Bilder mit demselben Outfit.",
                                "Maximum number of images with the same outfit.",
                            ),
                        )
                        c_max_session = gr.Slider(
                            label=tr("Max pro Session", "Max per session"),
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=S["c_max_session"],
                            info=tr(
                                "Maximal so viele Bilder aus derselben Foto-Session.",
                                "Maximum number of images from the same photo session.",
                            ),
                        )
                    c_use_diversity = gr.Checkbox(
                        label=tr("Diversity-Penalties", "Diversity penalties"),
                        value=S["c_use_diversity"],
                        info=tr(
                            "Bestraft zu ähnliche Kandidaten bei der Endauswahl. Fördert Vielfalt.",
                            "Penalizes overly similar candidates during final selection. Encourages variety.",
                        ),
                    )

                with gr.Accordion(tr("📝 Caption-Regeln", "📝 Caption rules"), open=False):
                    gr.Markdown(tr(
                        "*Welche Merkmale in die automatische Bildbeschreibung kommen. 'immer' = in jeder Caption. 'bei Abweichung' = nur wenn es vom Durchschnitt abweicht.*",
                        "*Which attributes should be included in the automatic caption. 'always' = in every caption. 'when variable' = only when it differs from the average.*",
                    ))
                    c_caption_profile = gr.Dropdown(
                        label=tr("Caption-Preset", "Caption preset"),
                        choices=caption_profile_choices(),
                        value=normalize_caption_profile(S.get("c_caption_profile")) or "shared_compact",
                        info=tr(
                            "Gemeinsames kompaktes Character-LoRA-Schema für Z-Image und ERNIE. Danach kannst du die Felder unten weiterhin individuell anpassen.",
                            "Shared compact Character-LoRA schema for Z-Image and ERNIE. You can still fine-tune the fields below afterwards.",
                        ),
                    )
                    c_captions = gr.CheckboxGroup(
                        label=tr("Aktive Caption-Felder", "Active caption fields"),
                        choices=CAPTION_FIELD_CHOICES,
                        value=S["c_captions"],
                        info=tr(
                            "Aktivierte Felder werden in die Trainingsbeschreibung aufgenommen.",
                            "Enabled fields will be included in the training captions.",
                        ),
                    )
                    c_caption_profile.change(
                        fn=lambda profile: get_caption_preset_values(profile),
                        inputs=[c_caption_profile],
                        outputs=[c_captions],
                    )
                    c_captions.change(
                        fn=detect_caption_profile,
                        inputs=[c_captions],
                        outputs=[c_caption_profile],
                    )

                with gr.Accordion(tr("💾 Export-Optionen", "💾 Export options"), open=False):
                    with gr.Row():
                        c_exp_review = gr.Checkbox(
                            label=tr("Review exportieren", "Export review"),
                            value=S["c_exp_review"],
                            info=tr(
                                "Review-Bilder in separaten Ordner zur manuellen Prüfung.",
                                "Save review images to a separate folder for manual inspection.",
                            ),
                        )
                        c_exp_reject = gr.Checkbox(
                            label=tr("Reject exportieren", "Export reject"),
                            value=S["c_exp_reject"],
                            info=tr(
                                "Verworfene Bilder mit Begründung speichern.",
                                "Save rejected images with reasons.",
                            ),
                        )
                        c_exp_compare = gr.Checkbox(
                            label=tr("Crop-Vergleiche", "Crop comparisons"),
                            value=S["c_exp_compare"],
                            info=tr(
                                "Original vs. Headshot-Crop Paare mit Scores.",
                                "Original vs headshot crop pairs with scores.",
                            ),
                        )

                # ── Aktionen ──
                gr.Markdown("---")
                with gr.Row():
                    c_start_btn = gr.Button(tr("▶ Curator starten", "▶ Start curator"), variant="primary", scale=3)
                    c_stop_btn = gr.Button(tr("⏹ Abbrechen", "⏹ Cancel"), variant="stop", scale=1)
                    c_save_btn = gr.Button(tr("💾 Einstellungen speichern", "💾 Save settings"), variant="secondary", scale=2)

                c_status = gr.Textbox(label=tr("Status", "Status"), interactive=False, max_lines=1)
                c_progress = gr.Slider(label=tr("Fortschritt", "Progress"), minimum=0, maximum=1, step=0.01, value=0, interactive=False)

                with gr.Row():
                    with gr.Column(scale=3):
                        c_log = gr.Textbox(label=tr("Live-Log", "Live log"), lines=18, max_lines=18, interactive=False, elem_classes=["log-box"])
                    with gr.Column(scale=2):
                        c_gallery = gr.Gallery(label=tr("Train-Ready Vorschau", "Train-ready preview"), columns=3, rows=3, height=380, object_fit="cover")

                # Alle Curator-Inputs als Liste (fuer Save und Start)
                curator_inputs = [
                    c_trigger, c_input, c_target, c_api_key, c_model, c_use_trigger_check, c_trigger_model,
                    c_use_review_escalation, c_review_escalation_model,
                    c_review_escalation_score_min, c_review_escalation_score_max,
                    c_escalate_on_review, c_escalate_on_conflict, c_escalate_smart_crop, c_smart_crop_escalation_delta,
                    c_ratio_h, c_ratio_m, c_ratio_f,
                    c_keep_min, c_reject, c_min_side,
                    c_use_filesize, c_min_filesize,
                    c_use_blur, c_min_blur, c_face_min_blur, c_blur_norm_edge,
                    c_use_early_phash,
                    c_subject_sanity, c_subject_min_torso,
                    c_ig_frame_crop, c_ig_two_stage_bar,
                    c_use_clip, c_use_phash, c_phash_thresh, c_clip_thresh,
                    c_smart_crop, c_crop_gain, c_crop_pad,
                    c_use_cluster, c_max_outfit, c_max_session, c_use_diversity,
                    c_caption_profile,
                    c_captions,
                    c_exp_review, c_exp_reject, c_exp_compare,
                ]

                c_start_btn.click(fn=start_curator, inputs=curator_inputs, outputs=[c_log, c_gallery, c_progress, c_status])
                c_stop_btn.click(fn=kill_process, outputs=[c_status])

            # ==============================================================
            # TAB 2: VIDEO PROCESSOR
            # ==============================================================
            with gr.TabItem(tr("🎬 Video Processor", "🎬 Video Processor")):

                gr.Markdown(tr("### Video-Frames extrahieren", "### Extract video frames"))
                gr.Markdown(tr(
                    "Erkennt die Zielperson per InsightFace-Referenzbild und extrahiert die schärfsten, vielfältigsten Frames pro Video-Minute.",
                    "Detects the target person via an InsightFace reference image and extracts the sharpest, most diverse frames per minute.",
                ))

                with gr.Row():
                    with gr.Column():
                        v_source = gr.Textbox(
                            label=tr("Video-Ordner", "Video folder"),
                            value=S["v_source"],
                            info=tr(
                                "Ordner mit Video-Dateien (mp4, mov, mkv, avi).",
                                "Folder containing video files (mp4, mov, mkv, avi).",
                            ),
                            max_lines=1,
                        )
                        v_target = gr.Textbox(
                            label=tr("Ausgabe-Ordner", "Output folder"),
                            value=S["v_target"],
                            info=tr(
                                "Hierhin werden die Frames gespeichert. Kann direkt als Curator-Input dienen.",
                                "Extracted frames are saved here. Can be used directly as curator input.",
                            ),
                            max_lines=1,
                        )
                        v_ref = gr.Textbox(
                            label=tr("Referenzbild (Zielperson)", "Reference image (target person)"),
                            value=S["v_ref"],
                            info=tr(
                                "Klares Foto der Person. Gutes Licht, Gesicht frontal, keine Brille ideal.",
                                "A clear photo of the person. Good lighting, frontal face, ideally no glasses.",
                            ),
                            max_lines=1,
                        )
                    with gr.Column():
                        v_fpm = gr.Slider(
                            label=tr("Frames pro Minute", "Frames per minute"),
                            minimum=1,
                            maximum=30,
                            step=1,
                            value=S["v_fpm"],
                            info=tr(
                                "Maximal extrahierte Bilder pro Video-Minute. 5 ist ein guter Startwert.",
                                "Maximum extracted images per video minute. 5 is a good starting point.",
                            ),
                        )
                        v_fps = gr.Slider(
                            label=tr("Sample-FPS", "Sample FPS"),
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=S["v_fps"],
                            info=tr(
                                "Analysierte Frames pro Sekunde. Höher = genauer aber langsamer.",
                                "Frames analyzed per second. Higher = more accurate but slower.",
                            ),
                        )
                        v_sim = gr.Slider(
                            label=tr("Similarity-Schwelle", "Similarity threshold"),
                            minimum=0.2,
                            maximum=0.8,
                            step=0.05,
                            value=S["v_sim"],
                            info=tr(
                                "Ab welcher Cosine-Similarity ein Gesicht als Zielperson gilt.",
                                "Cosine similarity at/above which a face is considered the target person.",
                            ),
                        )
                        v_sharp = gr.Slider(
                            label=tr("Min. Schärfe", "Min sharpness"),
                            minimum=10,
                            maximum=200,
                            step=10,
                            value=S["v_sharp"],
                            info=tr(
                                "Mindest-Laplacian-Varianz. 50 = mild, 100+ = streng.",
                                "Minimum Laplacian variance. 50 = mild, 100+ = strict.",
                            ),
                        )

                with gr.Row():
                    v_start_btn = gr.Button(tr("▶ Video-Extraktion starten", "▶ Start video extraction"), variant="primary", scale=3)
                    v_stop_btn = gr.Button(tr("⏹ Abbrechen", "⏹ Cancel"), variant="stop", scale=1)

                v_status = gr.Textbox(label=tr("Status", "Status"), interactive=False, max_lines=1)
                v_progress = gr.Slider(label=tr("Fortschritt", "Progress"), minimum=0, maximum=1, step=0.01, value=0, interactive=False)

                with gr.Row():
                    with gr.Column(scale=3):
                        v_log = gr.Textbox(label=tr("Live-Log", "Live log"), lines=15, max_lines=15, interactive=False, elem_classes=["log-box"])
                    with gr.Column(scale=2):
                        v_gallery = gr.Gallery(label=tr("Extrahierte Frames", "Extracted frames"), columns=3, rows=3, height=340, object_fit="cover")

                video_inputs = [v_source, v_target, v_ref, v_fpm, v_fps, v_sim, v_sharp]
                v_start_btn.click(fn=start_video, inputs=video_inputs, outputs=[v_log, v_gallery, v_progress, v_status])
                v_stop_btn.click(fn=kill_process, outputs=[v_status])

            # ==============================================================
            # TAB 3: ERGEBNISSE
            # ==============================================================
            with gr.TabItem(tr("📊 Ergebnisse", "📊 Results")):
                gr.Markdown(tr("### Datensatz durchsuchen", "### Browse dataset"))
                gr.Markdown(tr(
                    "Lade Ergebnisse eines früheren Curator-Laufs. Bilder werden mit Captions angezeigt.",
                    "Load results from a previous curator run. Images are shown with captions.",
                ))

                with gr.Row():
                    r_trigger = gr.Textbox(
                        label=tr("Trigger Word", "Trigger Word"),
                        value=S["c_trigger"],
                        info=tr("Triggerwort des Laufs.", "Trigger word of the run."),
                        max_lines=1,
                        scale=2,
                    )
                    r_input = gr.Textbox(
                        label=tr("Input-Ordner", "Input folder"),
                        value=S["c_input"],
                        info=tr("Der Original-Input-Ordner.", "Original input folder."),
                        max_lines=1,
                        scale=3,
                    )
                    r_subfolder = gr.Dropdown(
                        label=tr("Kategorie", "Category"),
                        choices=[
                            (tr("Train Ready", "Train Ready"), "train_ready"),
                            (tr("Caption Remove", "Caption Remove"), "caption_remove"),
                            (tr("Review", "Review"), "review"),
                            (tr("Reject", "Reject"), "reject"),
                            (tr("Smart Crop Paare", "Smart crop pairs"), "smart_crop_pairs"),
                        ],
                        value="train_ready",
                        info=tr("Welche Ergebnis-Kategorie.", "Which result category to load."),
                        scale=2,
                    )
                    r_load_btn = gr.Button(tr("🔄 Laden", "🔄 Load"), variant="primary", scale=1)

                r_info = gr.Textbox(label=tr("Info", "Info"), interactive=False, max_lines=2)
                r_gallery = gr.Gallery(label=tr("Bilder (mit Captions)", "Images (with captions)"), columns=4, rows=3, height=420, object_fit="cover")
                r_report = gr.Markdown(label=tr("Report", "Report"))

                r_load_btn.click(fn=load_results, inputs=[r_input, r_trigger, r_subfolder], outputs=[r_gallery, r_report, r_info])

            # ── Save-Button Event (braucht Zugriff auf ALLE Inputs) ──
            all_save_inputs = [ui_language] + curator_inputs + video_inputs
            c_save_btn.click(fn=save_settings_fn, inputs=all_save_inputs, outputs=[c_status])

    return app


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    missing = []
    if not os.path.isfile(CURATOR_SCRIPT):
        missing.append(f"  - {CURATOR_SCRIPT}")
    if not os.path.isfile(VIDEO_SCRIPT):
        missing.append(f"  - {VIDEO_SCRIPT}")
    if missing:
        print("⚠️ Fehlende Skripte:")
        for m in missing:
            print(m)
        print("Bitte alle Dateien in denselben Ordner legen.\n")

    venv_ok = os.path.isfile(os.path.join(SCRIPT_DIR, "curator_env", "Scripts", "python.exe"))

    # Avoid UnicodeEncodeError on Windows consoles (cp1252) by not printing emojis.
    print(f"Python:        {VENV_PYTHON}")
    print(f"Venv gefunden: {'Ja' if venv_ok else 'Nein'}")
    print(f"Settings:      {SETTINGS_PATH} ({'vorhanden' if os.path.isfile(SETTINGS_PATH) else 'neu'})")
    print(f"Skript-Ordner: {SCRIPT_DIR}\n")

    app = build_ui()
    app.queue()

    # Port fallback: if 7860 is occupied, try a few next ports.
    base_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launched = False
    last_err: Optional[Exception] = None
    for port in range(base_port, base_port + 20):
        try:
            app.launch(
                server_name="127.0.0.1",
                server_port=port,
                inbrowser=True,
                share=False,
            )
            launched = True
            break
        except OSError as e:
            last_err = e
            continue

    if not launched:
        raise last_err or OSError("Could not find a free port to launch Gradio UI")
