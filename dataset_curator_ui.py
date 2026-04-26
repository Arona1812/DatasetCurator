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
import inspect
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

UI_THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
UI_CSS = ".log-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 12px !important; }"


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
    "include_beard_when_variable",
    "include_mirror_selfie_marker",
    "include_eye_color",
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
    "c_review_escalation_score_max": 58,
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
    # Frühe pHash-Vorfilterung (zwei Schleifen)
    "c_use_early_phash": True,
    # Loop 1: exakte Duplikate
    "c_use_early_phash_loop1": True,
    "c_early_phash_thresh_1": 1,
    "c_early_phash_keep_1": 1,
    # Loop 2: aggressiver Bulk-Filter
    "c_use_early_phash_loop2": True,
    "c_early_phash_thresh_2": 4,
    "c_early_phash_keep_2": 2,
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
    # Pose-Diversity
    "c_use_pose_diversity": True,
    "c_pose_soft_limit": 2,
    "c_pose_penalty_weight": 4.0,
    # Identity-Konsistenz-Check (ArcFace)
    "c_use_arcface": True,
    "c_arcface_hard": 0.50,
    "c_arcface_soft": 0.65,
    "c_arcface_trim": 0.10,
    "c_arcface_min_faces": 5,
    "c_arcface_model": "buffalo_l",
    "c_arcface_det_size": 640,
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


def resolve_caption_fields_for_profile(
    profile: Optional[str],
    current_fields: Optional[List[str]] = None,
) -> List[str]:
    normalized = normalize_caption_profile(profile)
    if normalized == "custom":
        return list(current_fields) if current_fields is not None else list(DEFAULTS["c_captions"])
    return get_caption_preset_values(normalized)


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
    c_use_early_phash_loop1, c_early_phash_thresh_1, c_early_phash_keep_1,
    c_use_early_phash_loop2, c_early_phash_thresh_2, c_early_phash_keep_2,
    c_subject_sanity, c_subject_min_torso,
    c_ig_frame_crop, c_ig_two_stage_bar,
    c_use_clip, c_use_phash, c_phash_thresh, c_clip_thresh,
    c_smart_crop, c_crop_gain, c_crop_pad,
    c_use_cluster, c_max_outfit, c_max_session, c_use_diversity,
    c_use_pose_diversity, c_pose_soft_limit, c_pose_penalty_weight,
    c_use_arcface, c_arcface_hard, c_arcface_soft, c_arcface_trim,
    c_arcface_min_faces, c_arcface_model, c_arcface_det_size,
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
        "c_use_early_phash_loop1": c_use_early_phash_loop1,
        "c_early_phash_thresh_1": c_early_phash_thresh_1,
        "c_early_phash_keep_1": c_early_phash_keep_1,
        "c_use_early_phash_loop2": c_use_early_phash_loop2,
        "c_early_phash_thresh_2": c_early_phash_thresh_2,
        "c_early_phash_keep_2": c_early_phash_keep_2,
        "c_subject_sanity": c_subject_sanity,
        "c_subject_min_torso": c_subject_min_torso,
        "c_ig_frame_crop": c_ig_frame_crop,
        "c_ig_two_stage_bar": c_ig_two_stage_bar,
        "c_use_clip": c_use_clip, "c_use_phash": c_use_phash,
        "c_phash_thresh": c_phash_thresh, "c_clip_thresh": c_clip_thresh,
        "c_smart_crop": c_smart_crop, "c_crop_gain": c_crop_gain, "c_crop_pad": c_crop_pad,
        "c_use_cluster": c_use_cluster, "c_max_outfit": c_max_outfit,
        "c_max_session": c_max_session, "c_use_diversity": c_use_diversity,
        "c_use_pose_diversity": c_use_pose_diversity,
        "c_pose_soft_limit": c_pose_soft_limit,
        "c_pose_penalty_weight": c_pose_penalty_weight,
        "c_use_arcface": c_use_arcface,
        "c_arcface_hard": c_arcface_hard,
        "c_arcface_soft": c_arcface_soft,
        "c_arcface_trim": c_arcface_trim,
        "c_arcface_min_faces": c_arcface_min_faces,
        "c_arcface_model": c_arcface_model,
        "c_arcface_det_size": c_arcface_det_size,
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
    use_early_phash_loop1, early_phash_threshold_1, early_phash_keep_per_group_1,
    use_early_phash_loop2, early_phash_threshold_2, early_phash_keep_per_group_2,
    subject_sanity, subject_min_torso,
    ig_frame_crop, ig_two_stage_bar,
    use_clip, use_phash, phash_threshold, clip_threshold,
    enable_smart_crop, crop_min_gain, crop_padding,
    use_clustering, max_outfit, max_session, use_diversity,
    use_pose_diversity, pose_soft_limit, pose_penalty_weight,
    use_arcface, arcface_hard, arcface_soft, arcface_trim,
    arcface_min_faces, arcface_model, arcface_det_size,
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
        "USE_EARLY_PHASH_LOOP1": bool(use_early_phash_loop1),
        "EARLY_PHASH_HAMMING_THRESHOLD_1": int(early_phash_threshold_1),
        "EARLY_PHASH_KEEP_PER_GROUP_1": int(early_phash_keep_per_group_1),
        "USE_EARLY_PHASH_LOOP2": bool(use_early_phash_loop2),
        "EARLY_PHASH_HAMMING_THRESHOLD_2": int(early_phash_threshold_2),
        "EARLY_PHASH_KEEP_PER_GROUP_2": int(early_phash_keep_per_group_2),
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
        "ENABLE_POSE_DIVERSITY": bool(use_pose_diversity),
        "POSE_DIVERSITY_SOFT_LIMIT": int(pose_soft_limit),
        "POSE_DIVERSITY_PENALTY_WEIGHT": float(pose_penalty_weight),
        "USE_ARCFACE_IDENTITY_CHECK": bool(use_arcface),
        "ARCFACE_HARD_THRESHOLD": float(arcface_hard),
        "ARCFACE_SOFT_THRESHOLD": float(arcface_soft),
        "ARCFACE_TRIM_FRACTION": float(arcface_trim),
        "ARCFACE_MIN_FACES_FOR_CENTROID": int(arcface_min_faces),
        "ARCFACE_MODEL_PACK": str(arcface_model).strip() or "buffalo_l",
        "ARCFACE_DET_SIZE": int(arcface_det_size),
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

    blocks_kwargs = {
        "title": tr("LoRA Dataset Curator", "LoRA Dataset Curator"),
    }

    # Gradio <=5 expects theme/css on Blocks(), while Gradio >=6 moves them to
    # launch(). Detect support dynamically so the UI works on both old and new
    # versions without warnings.
    blocks_signature = inspect.signature(gr.Blocks.__init__)
    if "theme" in blocks_signature.parameters:
        blocks_kwargs["theme"] = UI_THEME
    if "css" in blocks_signature.parameters:
        blocks_kwargs["css"] = UI_CSS

    with gr.Blocks(**blocks_kwargs) as app:

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
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Was bedeutet Eskalation und wann brauche ich das?</b></summary>"
                        "\n\n"
                        "Standardmäßig bewertet ein einziges, günstiges Modell (z. B. `gpt-5.4-nano`) "
                        "alle Bilder. Das ist für 80–90 % der Fälle völlig ausreichend. Bei "
                        "**Grenzfällen** – also Bildern, die das Hauptmodell nicht klar als "
                        "'gut genug' oder 'rauswerfen' einordnen kann – kann der Curator diese "
                        "Bilder optional an ein **stärkeres zweites Modell** weiterleiten "
                        "(z. B. `gpt-5.4` oder `claude-opus-4.7`).\n\n"
                        "**Drei Auslöser für Eskalation:**\n\n"
                        "**1. Bei Review:** Das Hauptmodell hat das Bild auf 'review' gesetzt "
                        "(also: 'ich kann mich nicht entscheiden') oder die Bewertung liegt im "
                        "konfigurierten Score-Fenster. Das stärkere Modell entscheidet dann.\n\n"
                        "**2. Bei Konflikt:** Wenn lokale Filter (z. B. Unschärfe-Erkennung) und "
                        "Hauptmodell unterschiedlicher Meinung sind. Vermeidet, dass ein "
                        "technisch unscharfes aber inhaltlich gutes Bild verloren geht.\n\n"
                        "**3. Bei knappem Smart-Crop-Duell:** Wenn Original und Crop weniger "
                        "Punkte auseinander liegen als die Eskalations-Differenz, entscheidet "
                        "das stärkere Modell, welcher Schnitt besser passt.\n\n"
                        "**Kosten:** Pro eskaliertem Bild ein zusätzlicher API-Call zum teureren "
                        "Modell. Bei normalem Setup landen 5–15 % der Bilder in der Eskalation."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ What is escalation and when do I need it?</b></summary>"
                        "\n\n"
                        "By default, a single cheap model (e.g. `gpt-5.4-nano`) scores all "
                        "images. That's enough for 80–90 % of cases. For **borderline cases** – "
                        "images the main model can't clearly classify as 'keep' or 'reject' – "
                        "the curator can optionally forward these images to a **stronger second "
                        "model** (e.g. `gpt-5.4` or `claude-opus-4.7`).\n\n"
                        "**Three escalation triggers:**\n\n"
                        "**1. On review:** Main model marked the image as 'review' (i.e. "
                        "'undecided') or the score falls inside the configured window. The "
                        "stronger model then decides.\n\n"
                        "**2. On conflict:** When local filters (e.g. blur detection) disagree "
                        "with the main model. Prevents losing a technically blurry but "
                        "content-wise good image.\n\n"
                        "**3. On close smart-crop duel:** If original and crop are within the "
                        "escalation delta, the stronger model decides which framing is better.\n\n"
                        "**Cost:** One extra API call to the more expensive model per escalated "
                        "image. With a normal setup, 5–15 % of images end up escalated."
                        "</details>",
                    ))
                    c_use_review_escalation = gr.Checkbox(
                        label=tr("Eskalation für schwierige Fälle aktivieren", "Enable escalation for difficult cases"),
                        value=S["c_use_review_escalation"],
                        info=tr(
                            "Empfohlen: an. Erhöht die Qualität bei Grenzfällen merklich, kostet aber pro eskaliertem Bild einen zusätzlichen API-Call zum stärkeren Modell.",
                            "Recommended: on. Noticeably improves quality on borderline cases, but each escalated image costs an extra API call to the stronger model.",
                        ),
                    )
                    with gr.Row():
                        c_review_escalation_model = gr.Textbox(
                            label=tr("Eskalationsmodell", "Escalation model"),
                            value=S["c_review_escalation_model"],
                            info=tr(
                                "Stärkeres Modell für die Eskalation. Leer = Eskalation effektiv aus, auch wenn der Schalter oben an ist. Empfohlen: ein Modell der nächsthöheren Klasse (z. B. `gpt-5.4` wenn das Hauptmodell `gpt-5.4-nano` ist).",
                                "Stronger model for escalation. Empty = escalation effectively off, even if the switch above is on. Recommended: a model from the next-higher tier (e.g. `gpt-5.4` if the main model is `gpt-5.4-nano`).",
                            ),
                            max_lines=1,
                        )
                    with gr.Row():
                        c_review_escalation_score_min = gr.Slider(
                            label=tr("Score-Fenster: Minimum", "Score window: minimum"),
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=S["c_review_escalation_score_min"],
                            info=tr(
                                "Untere Grenze des Score-Fensters für Eskalation. Bilder mit Score zwischen Minimum und Maximum werden eskaliert. Empfohlen: 35.",
                                "Lower bound of the escalation score window. Images scoring between min and max are escalated. Recommended: 35.",
                            ),
                        )
                        c_review_escalation_score_max = gr.Slider(
                            label=tr("Score-Fenster: Maximum", "Score window: maximum"),
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=S["c_review_escalation_score_max"],
                            info=tr(
                                "Obere Grenze des Score-Fensters. Bilder über diesem Wert sind 'eindeutig gut' und werden nicht eskaliert. Empfohlen: 58.",
                                "Upper bound of the score window. Images above this are 'clearly good' and won't be escalated. Recommended: 58.",
                            ),
                        )
                    with gr.Row():
                        c_escalate_on_review = gr.Checkbox(
                            label=tr("Bei Review eskalieren", "Escalate on review"),
                            value=S["c_escalate_on_review"],
                            info=tr(
                                "Eskaliert Bilder, die das Hauptmodell als 'unentschieden' markiert hat. Empfohlen: an.",
                                "Escalates images flagged as 'undecided' by the main model. Recommended: on.",
                            ),
                        )
                        c_escalate_on_conflict = gr.Checkbox(
                            label=tr("Bei Konflikt eskalieren", "Escalate on conflict"),
                            value=S["c_escalate_on_conflict"],
                            info=tr(
                                "Eskaliert, wenn lokale Filter und Hauptmodell unterschiedlicher Meinung sind (z. B. lokal als unscharf erkannt, KI sagt: scharf). Empfohlen: an.",
                                "Escalates when local filters and main model disagree (e.g. flagged as blurry locally, AI says sharp). Recommended: on.",
                            ),
                        )
                        c_escalate_smart_crop = gr.Checkbox(
                            label=tr("Knappes Smart-Crop-Duell eskalieren", "Escalate close smart-crop duel"),
                            value=S["c_escalate_smart_crop"],
                            info=tr(
                                "Eskaliert, wenn Original und Smart-Crop fast gleich gut bewertet werden. Empfohlen: an, wenn dir die Smart-Crop-Auswahl wichtig ist.",
                                "Escalates when original and smart crop score almost equally. Recommended: on if smart crop selection matters to you.",
                            ),
                        )
                    c_smart_crop_escalation_delta = gr.Slider(
                        label=tr("Max. Punktdifferenz für Crop-Eskalation", "Max point delta for crop escalation"),
                        minimum=0,
                        maximum=30,
                        step=1,
                        value=S["c_smart_crop_escalation_delta"],
                        info=tr(
                            "Wenn Original und Crop weniger als so viele Punkte auseinanderliegen, entscheidet das stärkere Modell. Empfohlen: 8. Höher (15+) eskaliert mehr Crop-Duelle, niedriger (4) nur die wirklich knappen.",
                            "If original and crop are within this many points of each other, the stronger model decides. Recommended: 8. Higher (15+) escalates more duels, lower (4) only the very close ones.",
                        ),
                    )

                    with gr.Column(scale=1):
                        gr.Markdown(tr(
                            "<details>"
                            "<summary><b>ℹ️ Wie funktioniert die Shot-Verteilung?</b></summary>"
                            "\n\n"
                            "Bestimmt, mit welchem Verhältnis die drei Aufnahmetypen (Nahaufnahme, "
                            "Oberkörper, Ganzkörper) im finalen Trainings-Set landen sollen.\n\n"
                            "**Headshot (Nahaufnahme):** Gesicht füllt einen Großteil des Bildes. "
                            "Wichtigste Kategorie für Identitätslernen – das Modell lernt hier, "
                            "wie das Gesicht aussieht.\n\n"
                            "**Medium (Oberkörper):** Gesicht plus Schultern und Oberkörper. "
                            "Hilft dem Modell, Körperbau, Haltung und typische Outfit-Schnitte "
                            "zu lernen.\n\n"
                            "**Full Body (Ganzkörper):** Komplette Person inklusive Beine. Damit "
                            "das LoRA später nicht nur Brustportraits, sondern auch Ganzkörper-"
                            "Generierungen sauber hinbekommt.\n\n"
                            "**Empfohlen für Person-LoRAs:** 0.45 / 0.30 / 0.25 (Headshot-lastig). "
                            "Wenn dein Material überwiegend Selfies enthält, kannst du Full Body "
                            "auf 0.10 senken; wenn du Mode/Outfit lernen willst, eher 0.30 / 0.40 / 0.30.\n\n"
                            "**Wichtig:** Die drei Werte sollten zusammen 1.0 ergeben. Wenn nicht, "
                            "normalisiert der Curator sie automatisch."
                            "</details>",
                            "<details>"
                            "<summary><b>ℹ️ How does shot distribution work?</b></summary>"
                            "\n\n"
                            "Defines what ratio of the three shot types (close-up, upper body, "
                            "full body) the final training set should aim for.\n\n"
                            "**Headshot (close-up):** Face fills most of the frame. Most "
                            "important category for identity learning – this is where the model "
                            "learns what the face looks like.\n\n"
                            "**Medium (upper body):** Face plus shoulders and torso. Helps the "
                            "model learn body type, posture and typical outfit cuts.\n\n"
                            "**Full Body:** Entire person including legs. So the LoRA can later "
                            "generate full-body shots cleanly, not just bust portraits.\n\n"
                            "**Recommended for person LoRAs:** 0.45 / 0.30 / 0.25 (headshot-"
                            "heavy). If your material is mostly selfies, you can drop full body "
                            "to 0.10; if you want to learn fashion/outfits, try 0.30 / 0.40 / 0.30.\n\n"
                            "**Important:** The three values should add up to 1.0. If they "
                            "don't, the curator normalizes them automatically."
                            "</details>",
                        ))
                        gr.Markdown(tr("### Shot-Verteilung", "### Shot distribution"))
                        c_ratio_h = gr.Slider(
                            label=tr("Headshot (Nahaufnahme)", "Headshot (close-up)"),
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=S["c_ratio_h"],
                            info=tr(
                                "Anteil enger Gesichts-Aufnahmen. Wichtigste Kategorie für Identitätslernen. Empfohlen: 0.45.",
                                "Share of tight face shots. Most important category for identity learning. Recommended: 0.45.",
                            ),
                        )
                        c_ratio_m = gr.Slider(
                            label=tr("Medium (Oberkörper)", "Medium (upper body)"),
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=S["c_ratio_m"],
                            info=tr(
                                "Anteil Oberkörper-Aufnahmen. Hilft beim Lernen von Körperbau, Haltung und Outfit-Schnitten. Empfohlen: 0.30.",
                                "Share of upper-body shots. Helps learn body type, posture and outfit cuts. Recommended: 0.30.",
                            ),
                        )
                        c_ratio_f = gr.Slider(
                            label=tr("Full Body (Ganzkörper)", "Full Body"),
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=S["c_ratio_f"],
                            info=tr(
                                "Anteil Ganzkörper-Aufnahmen. Weniger nötig, aber wichtig damit das LoRA später vollständige Personen generieren kann. Empfohlen: 0.25.",
                                "Share of full-body shots. Less needed, but important so the LoRA can generate complete persons. Recommended: 0.25.",
                            ),
                        )
                        gr.Markdown(tr("*⚠️ Summe sollte 1.0 ergeben*", "*⚠️ Sum should be 1.0*"))

                with gr.Accordion(tr("⚙️ Qualität & Schwellwerte", "⚙️ Quality & thresholds"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Wie funktionieren die Score-Schwellen?</b></summary>"
                        "\n\n"
                        "Jedes Bild bekommt nach der KI-Analyse einen Gesamtscore von 0 bis 100, "
                        "der sich aus Schärfe, Beleuchtung, Komposition und Identitäts-"
                        "Nützlichkeit zusammensetzt. Anhand zweier Schwellen wird das Bild dann "
                        "klassifiziert:\n\n"
                        "**Keep-Schwelle (oben):** Ab diesem Score zählt ein Bild als 'gut "
                        "genug' und kommt in den Pool für die Endauswahl. Bilder darunter (aber "
                        "über der Reject-Schwelle) landen als 'review' im Ordner "
                        "`02_keep_unused` – sie sind nicht aussortiert, gehen aber nur ins "
                        "Trainings-Set, wenn sonst nicht genug Material da ist.\n\n"
                        "**Reject-Schwelle (unten):** Unter diesem Score wird das Bild "
                        "**direkt verworfen** und landet in `03_rejected`. Keine zweite Chance.\n\n"
                        "**Faustregel:** Lass zwischen Reject und Keep einen Abstand von "
                        "mindestens 20 Punkten. Sonst gibt es kaum Bilder im Review-Bereich, "
                        "und wenn dein Material knapp wird, hast du keinen Puffer.\n\n"
                        "**Mindestseitenlänge:** Wirkt vor allem als Schutz gegen Thumbnails "
                        "und versehentlich kleine Bilder. Für Training auf Auflösung 1024 "
                        "reicht 768 als Mindestmaß; für 512 entsprechend 512."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ How do score thresholds work?</b></summary>"
                        "\n\n"
                        "Each image gets an overall score from 0 to 100 after AI analysis, "
                        "composed of sharpness, lighting, composition and identity usefulness. "
                        "Two thresholds then classify the image:\n\n"
                        "**Keep threshold (upper):** Above this score, an image counts as "
                        "'good enough' and goes into the final-selection pool. Images below "
                        "this (but above reject) land as 'review' in `02_keep_unused` – not "
                        "rejected, but only used in the training set if there's a shortage of "
                        "better material.\n\n"
                        "**Reject threshold (lower):** Below this score, the image is "
                        "**immediately rejected** and lands in `03_rejected`. No second chance.\n\n"
                        "**Rule of thumb:** Leave at least 20 points between reject and keep. "
                        "Otherwise you'll have almost no images in the review range, and if "
                        "you run short on material later, you'll have no buffer.\n\n"
                        "**Minimum side length:** Mainly a safety net against thumbnails and "
                        "accidentally tiny images. For training at resolution 1024, 768 is a "
                        "good minimum; for 512, use 512 accordingly."
                        "</details>",
                    ))
                    with gr.Row():
                        c_keep_min = gr.Slider(
                            label=tr("Keep-Schwelle", "Keep threshold"),
                            minimum=0,
                            maximum=100,
                            step=5,
                            value=S["c_keep_min"],
                            info=tr(
                                "Mindest-Score (0–100) damit ein Bild direkt als 'keep' gilt. Empfohlen: 55. Darunter (aber über Reject) landet das Bild im Review-Bereich.",
                                "Minimum score (0–100) for an image to count as 'keep' directly. Recommended: 55. Below that (but above reject) the image goes into the review area.",
                            ),
                        )
                        c_reject = gr.Slider(
                            label=tr("Reject-Schwelle", "Reject threshold"),
                            minimum=0,
                            maximum=100,
                            step=5,
                            value=S["c_reject"],
                            info=tr(
                                "Unter diesem Score wird ein Bild sofort verworfen, ohne Review-Chance. Empfohlen: 30. Höher = strenger (bei guter Materiallage). Niedriger = mehr Bilder durchlassen (bei knappem Material).",
                                "Below this score, an image is rejected immediately, no review chance. Recommended: 30. Higher = stricter (when material is plentiful). Lower = let more through (when short).",
                            ),
                        )
                        c_min_side = gr.Slider(
                            label=tr("Min. Seitenlänge (px)", "Min side length (px)"),
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=S["c_min_side"],
                            info=tr(
                                "Kürzeste Bildseite in Pixeln. Bilder darunter werden sofort verworfen. Empfohlen: 768 für Training auf 1024, 512 für Training auf 512.",
                                "Shortest image side in pixels. Images below are rejected immediately. Recommended: 768 for training at 1024, 512 for training at 512.",
                            ),
                        )

                with gr.Accordion(tr("🔍 Vorfilter (lokal, ohne API-Kosten)", "🔍 Pre-filters (local, no API cost)"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Wozu Vorfilter?</b></summary>"
                        "\n\n"
                        "Bevor irgendein Bild zur kostenpflichtigen KI-Bewertung geschickt wird, "
                        "laufen mehrere lokale Filter rein auf deinem Rechner – ohne API-Kosten. "
                        "Sie killen offensichtliche Ausschuss-Bilder (verwackelt, zu klein, "
                        "Duplikate), bevor du dafür bezahlst.\n\n"
                        "**Bei einem Dataset von 100 Bildern** sind die Ersparnisse marginal. "
                        "**Bei 3000 Bildern** macht das den Unterschied zwischen 5 € und 50 € "
                        "API-Kosten. Lass die Filter im Zweifel an – sie sind so konfiguriert, "
                        "dass sie nur klare Fälle aussortieren und im Grenzfall lieber das "
                        "Bild durchlassen als zu strikt sein.\n\n"
                        "**Unschärfe-Filter (zweistufig):**\n\n"
                        "**Stufe 1** prüft das ganze Bild *vor* der API – nur eine "
                        "Totalausfall-Erkennung für komplett verwackelte oder verwaschene "
                        "Bilder. Bewusst lax, weil bei Stufe 1 das Gesicht noch nicht lokalisiert "
                        "ist und ein scharfer Hintergrund mit unscharfem Gesicht durchrutschen "
                        "könnte.\n\n"
                        "**Stufe 2** prüft *gezielt die Gesichtsregion* nach der "
                        "Gesichtserkennung. Hier kann die Schwelle deutlich strenger sein, "
                        "weil wir wissen wo wir hinschauen. **Achtung:** Beauty-Filter-Selfies "
                        "(starkes Hautglätten) können hier fälschlich als 'unscharf' gelten – "
                        "siehe Erklärung im Hauptmodell-Eskalations-Bereich.\n\n"
                        "**Frühe pHash-Vorfilterung:** Erkennt offensichtliche Duplikate "
                        "(gleiches Foto mehrfach hochgeladen, identische Re-Uploads) schon "
                        "*vor* der API-Bewertung und behält pro Duplikatsgruppe nur ein paar "
                        "Bilder zur Bewertung. Sehr effektiv bei Bulk-Datasets aus Social Media."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ Why pre-filters?</b></summary>"
                        "\n\n"
                        "Before any image is sent to the paid AI scoring, several local "
                        "filters run on your machine – at no API cost. They kill obvious junk "
                        "images (blurry, too small, duplicates) before you pay for them.\n\n"
                        "**For a 100-image dataset** the savings are marginal. **For a 3000-"
                        "image dataset** this is the difference between $5 and $50 in API "
                        "costs. When in doubt, leave the filters on – they're tuned to only "
                        "reject clear cases and to err toward letting an image through rather "
                        "than being too strict.\n\n"
                        "**Blur filter (two-stage):**\n\n"
                        "**Stage 1** checks the whole image *before* the API – pure total-"
                        "failure detection for completely blurry or smeared images. "
                        "Intentionally lax, because at stage 1 the face hasn't been located "
                        "yet, and a sharp background with a blurry face could slip through.\n\n"
                        "**Stage 2** targets the *face region* after face detection. Here the "
                        "threshold can be much stricter because we know where to look. "
                        "**Caveat:** Beauty-filter selfies (heavy skin smoothing) can falsely "
                        "register as 'blurry' here.\n\n"
                        "**Early pHash pre-filter:** Detects obvious duplicates (same photo "
                        "uploaded multiple times, identical re-uploads) *before* AI scoring "
                        "and keeps only a few images per duplicate group for scoring. Very "
                        "effective on bulk social-media datasets."
                        "</details>",
                    ))
                    with gr.Row():
                        c_use_filesize = gr.Checkbox(
                            label=tr("Dateigröße-Filter", "File size filter"),
                            value=S["c_use_filesize"],
                            info=tr(
                                "Verwirft sehr kleine Dateien, die meist stark komprimiert und für Training unbrauchbar sind. Empfohlen: an.",
                                "Rejects very small files that are usually heavily compressed and unusable for training. Recommended: on.",
                            ),
                        )
                        c_min_filesize = gr.Slider(
                            label=tr("Min. Dateigröße (KB)", "Min file size (KB)"),
                            minimum=10,
                            maximum=500,
                            step=10,
                            value=S["c_min_filesize"],
                            info=tr(
                                "Dateien unter dieser Größe werden direkt verworfen. Empfohlen: 50 KB. Höher (100+) bei reinen Foto-Datasets, niedriger (20) wenn auch kleinere Bilder erlaubt sein sollen.",
                                "Files below this size are rejected immediately. Recommended: 50 KB. Higher (100+) for pure photo datasets, lower (20) if smaller images should pass.",
                            ),
                        )
                    with gr.Row():
                        c_use_blur = gr.Checkbox(
                            label=tr("Unschärfe-Filter (zweistufig)", "Blur filter (two-stage)"),
                            value=S["c_use_blur"],
                            info=tr(
                                "Erkennt unscharfe Bilder in zwei Stufen. Empfohlen: an. Details siehe Erklärung oben.",
                                "Detects blurry images in two stages. Recommended: on. See explanation above for details.",
                            ),
                        )
                        c_min_blur = gr.Slider(
                            label=tr("Stufe 1: Min. Varianz (Gesamtbild)", "Stage 1: min variance (full image)"),
                            minimum=5,
                            maximum=200,
                            step=5,
                            value=S["c_min_blur"],
                            info=tr(
                                "Totalausfall-Schwelle auf dem ganzen Bild. Empfohlen: 25 (nur komplett verwackelte Bilder fliegen raus). Höher = strenger, kann aber Bilder mit unscharfem Hintergrund fälschlich treffen.",
                                "Total-failure threshold on the full image. Recommended: 25 (only completely blurry images get rejected). Higher = stricter, but may falsely reject images with intentionally blurred backgrounds.",
                            ),
                        )
                    with gr.Row():
                        c_face_min_blur = gr.Slider(
                            label=tr("Stufe 2: Min. Schärfe (Gesicht)", "Stage 2: min sharpness (face)"),
                            minimum=10,
                            maximum=200,
                            step=5,
                            value=S["c_face_min_blur"],
                            info=tr(
                                "Schärfe-Schwelle in der Gesichts-Region. Empfohlen: 45 (konservativ, lässt Beauty-Filter-Selfies durch). 70+ = streng (filtert auch leicht weichgezeichnete Gesichter raus).",
                                "Sharpness threshold inside the face region. Recommended: 45 (conservative, keeps beauty-filter selfies). 70+ = strict (also filters slightly soft-focused faces).",
                            ),
                        )
                        c_blur_norm_edge = gr.Slider(
                            label=tr("Normierungs-Kantenlänge (px)", "Normalization edge size (px)"),
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=S["c_blur_norm_edge"],
                            info=tr(
                                "Vor der Schärfe-Messung werden alle Bilder auf diese längste Kante skaliert. Macht die Schwellen unabhängig von der Original-Auflösung. Empfohlen: 512 (Standard, in Bezug zu den Schwellen oben kalibriert).",
                                "Before sharpness measurement, all images get resized to this longest edge. Makes thresholds independent of the original resolution. Recommended: 512 (default, calibrated against the thresholds above).",
                            ),
                        )
                    c_use_early_phash = gr.Checkbox(
                        label=tr("Frühe Duplikat-Vorfilterung (vor API)", "Early duplicate pre-filter (pre-API)"),
                        value=S["c_use_early_phash"],
                        info=tr(
                            "Master-Schalter für die zwei pHash-Schleifen unten. Wenn aus, werden beide Schleifen übersprungen. Empfohlen: an, besonders bei großen Datasets aus Social Media oder Video-Frame-Extraktionen.",
                            "Master switch for the two pHash loops below. When off, both loops are skipped. Recommended: on, especially for large social-media datasets or video frame extractions.",
                        ),
                    )
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Wie funktionieren die zwei Schleifen?</b></summary>"
                        "\n\n"
                        "Zwei aufeinanderfolgende pHash-Vorfilter-Durchgänge mit "
                        "unterschiedlichen Schwellen, weil ein einzelner Durchgang nie "
                        "beide Anwendungsfälle gleichzeitig gut bedienen kann.\n\n"
                        "**Schleife 1 (exakte Duplikate):**\n\n"
                        "Sehr strenge Schwelle (Hamming 1, 1 pro Gruppe behalten). "
                        "Findet praktisch nur bit-identische Re-Uploads, Screenshots "
                        "vom selben Foto und identische Kompressions-Varianten. Bei "
                        "Datasets aus Story-Highlights, wo dieselben Bilder mehrmals "
                        "in unterschiedlichen Reposts auftauchen, räumt das massiv auf.\n\n"
                        "**Schleife 2 (Bulk-Filter):**\n\n"
                        "Lockerere Schwelle (Hamming 4, 2 pro Gruppe behalten). "
                        "Greift Bulk-Aufnahmen ab, bei denen aus einem Video oder "
                        "einer Burst-Aufnahme viele fast-identische Frames extrahiert "
                        "wurden. Pro Bulk-Gruppe bleiben zwei Bilder durch, damit du "
                        "noch leichte Varianz behältst.\n\n"
                        "**Reihenfolge:** Schleife 1 läuft zuerst auf allen Bildern, "
                        "Schleife 2 dann auf den Überlebenden. Die pHashes werden "
                        "zwischen den Schleifen wiederverwendet, kein doppeltes Hashen.\n\n"
                        "**Empfohlen:** Beide an. Wer kein Video-Material und keine "
                        "Bulk-Aufnahmen hat, kann Schleife 2 ausschalten – Schleife 1 "
                        "alleine bringt schon einen großen Teil der Ersparnis."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ How do the two loops work?</b></summary>"
                        "\n\n"
                        "Two sequential pHash pre-filter passes with different "
                        "thresholds, because no single pass can serve both use cases "
                        "well at the same time.\n\n"
                        "**Loop 1 (exact duplicates):**\n\n"
                        "Very strict threshold (hamming 1, keep 1 per group). Finds "
                        "essentially only bit-identical re-uploads, screenshots of "
                        "the same photo and identical compression variants. For "
                        "datasets from story highlights, where the same images appear "
                        "in multiple reposts, this clears massive amounts.\n\n"
                        "**Loop 2 (bulk filter):**\n\n"
                        "Looser threshold (hamming 4, keep 2 per group). Catches "
                        "bulk shots where many near-identical frames were extracted "
                        "from a video or burst capture. Two images per bulk group "
                        "survive, so you keep some slight variation.\n\n"
                        "**Order:** Loop 1 runs first on all images, loop 2 then on "
                        "the survivors. pHashes are reused between loops – no double "
                        "hashing.\n\n"
                        "**Recommended:** Both on. Without video material or burst "
                        "shots you can turn loop 2 off – loop 1 alone already brings "
                        "a big part of the savings."
                        "</details>",
                    ))
                    with gr.Row():
                        c_use_early_phash_loop1 = gr.Checkbox(
                            label=tr("Schleife 1 aktivieren (exakte Duplikate)", "Enable loop 1 (exact duplicates)"),
                            value=S["c_use_early_phash_loop1"],
                            info=tr(
                                "Empfohlen: an. Sehr günstig (kostet praktisch nichts) und filtert garantierte Duplikate.",
                                "Recommended: on. Very cheap (practically no cost) and filters guaranteed duplicates.",
                            ),
                        )
                        c_early_phash_thresh_1 = gr.Slider(
                            label=tr("Schleife 1: Hamming-Schwelle", "Loop 1: hamming threshold"),
                            minimum=0,
                            maximum=4,
                            step=1,
                            value=S["c_early_phash_thresh_1"],
                            info=tr(
                                "Wie ähnlich Bilder sein müssen, um als exaktes Duplikat zu gelten. Empfohlen: 1 (nur bit-identische plus minimale Kompressions-Unterschiede). 0 = wirklich nur exakt identisch. 2–3 = lockerer.",
                                "How similar images must be to count as exact duplicates. Recommended: 1 (only bit-identical plus minor compression differences). 0 = truly only identical. 2–3 = looser.",
                            ),
                        )
                        c_early_phash_keep_1 = gr.Slider(
                            label=tr("Schleife 1: pro Gruppe behalten", "Loop 1: keep per group"),
                            minimum=1,
                            maximum=3,
                            step=1,
                            value=S["c_early_phash_keep_1"],
                            info=tr(
                                "Wie viele Bilder pro Duplikat-Gruppe überleben. Empfohlen: 1 (bei exakten Duplikaten reicht eins). Höhere Werte sind hier eher sinnlos, weil die Bilder ohnehin praktisch identisch sind.",
                                "How many images per duplicate group survive. Recommended: 1 (with exact duplicates one is enough). Higher values are pointless here since the images are practically identical anyway.",
                            ),
                        )
                    with gr.Row():
                        c_use_early_phash_loop2 = gr.Checkbox(
                            label=tr("Schleife 2 aktivieren (Bulk-Filter)", "Enable loop 2 (bulk filter)"),
                            value=S["c_use_early_phash_loop2"],
                            info=tr(
                                "Empfohlen: an bei Datasets mit Video-Frames oder Burst-Aufnahmen. Bei reinen Foto-Datasets aus Studio/DSLR kannst du es ausschalten.",
                                "Recommended: on for datasets with video frames or burst shots. For pure photo datasets from studio/DSLR you can turn it off.",
                            ),
                        )
                        c_early_phash_thresh_2 = gr.Slider(
                            label=tr("Schleife 2: Hamming-Schwelle", "Loop 2: hamming threshold"),
                            minimum=2,
                            maximum=12,
                            step=1,
                            value=S["c_early_phash_thresh_2"],
                            info=tr(
                                "Wie ähnlich Bilder sein müssen, um als Bulk-Duplikat zu gelten. Empfohlen: 4 (typische Video-Frame-Ähnlichkeit). 6–8 für aggressiveres Aufräumen, kann aber leichte Pose-Varianten verlieren.",
                                "How similar images must be to count as bulk duplicates. Recommended: 4 (typical video-frame similarity). 6–8 for more aggressive cleanup, but may lose minor pose variants.",
                            ),
                        )
                        c_early_phash_keep_2 = gr.Slider(
                            label=tr("Schleife 2: pro Gruppe behalten", "Loop 2: keep per group"),
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=S["c_early_phash_keep_2"],
                            info=tr(
                                "Wie viele Bilder pro Bulk-Gruppe überleben und zur API kommen. Empfohlen: 2 (lässt zwei Varianten als Sicherheit durch). 1 = streng (spart maximal Kosten), 3+ = locker (mehr API-Kosten, mehr Vielfalt).",
                                "How many images per bulk group survive and get sent to the API. Recommended: 2 (lets two variants through for safety). 1 = strict (max cost savings), 3+ = loose (more API cost, more variety).",
                            ),
                        )

                with gr.Accordion(tr("🖼️ Instagram-Frame / UI-Rand-Entfernung", "🖼️ Instagram frame / UI border removal"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Was wird hier entfernt?</b></summary>"
                        "\n\n"
                        "Viele Bilder aus Social Media – besonders Instagram-Stories und "
                        "Screenshots vom Handy – haben **künstliche Ränder, die nicht zum "
                        "eigentlichen Bild gehören**:\n\n"
                        "**Instagram-Story-Frames:** Weiße oder farbige Balken oben und unten, "
                        "die das eigentliche Foto zentrieren. Beim LoRA-Training würde das "
                        "Modell lernen, dass solche Ränder zur Person gehören.\n\n"
                        "**Drop-Shadow-Gradienten:** Halb-transparente dunkle Verläufe am "
                        "oberen oder unteren Rand für lesbaren UI-Text.\n\n"
                        "**Android-Nav-Bars:** Schwarze Leisten mit Statusbar-Icons (Akku, "
                        "Uhrzeit) oder System-Buttons.\n\n"
                        "Der Curator erkennt diese Ränder und schneidet sie *vor* der KI-"
                        "Bewertung weg. Damit wird das Bild auf dem bereinigten Inhalt "
                        "bewertet, und das spätere LoRA lernt nicht versehentlich Telefon-"
                        "Interfaces als Teil der Person.\n\n"
                        "**Konservativ kalibriert:** Die Erkennung ist so eingestellt, dass "
                        "sie bei Unsicherheit lieber nichts wegschneidet, statt einen echten "
                        "dunklen Hintergrund (schwarze Wand, Haare) als Nav-Bar misszuver-"
                        "stehen."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ What gets removed here?</b></summary>"
                        "\n\n"
                        "Many social-media images – especially Instagram stories and phone "
                        "screenshots – have **artificial borders that don't belong to the "
                        "actual photo**:\n\n"
                        "**Instagram story frames:** White or colored bars top and bottom that "
                        "center the actual photo. During LoRA training, the model would learn "
                        "that such borders are part of the person.\n\n"
                        "**Drop-shadow gradients:** Semi-transparent dark gradients at the top "
                        "or bottom for readable UI text.\n\n"
                        "**Android nav bars:** Black bars with status icons (battery, clock) "
                        "or system buttons.\n\n"
                        "The curator detects these borders and crops them out *before* AI "
                        "scoring. The image is then evaluated on cleaned content, and the "
                        "later LoRA doesn't accidentally learn phone UI as part of the "
                        "person.\n\n"
                        "**Conservatively calibrated:** Detection is set up to err on not "
                        "cutting anything if uncertain, rather than mistaking a real dark "
                        "background (black wall, hair) for a nav bar."
                        "</details>",
                    ))
                    with gr.Row():
                        c_ig_frame_crop = gr.Checkbox(
                            label=tr("IG-Frame-/UI-Rand-Entfernung aktivieren", "Enable IG frame / UI border removal"),
                            value=S["c_ig_frame_crop"],
                            info=tr(
                                "Hauptschalter für die Rand-Erkennung und das Wegschneiden. Empfohlen: an, sobald deine Bilder aus Social Media stammen. Bei reinen DSLR-/Studio-Fotos kannst du es ausschalten.",
                                "Main switch for border detection and cropping. Recommended: on as soon as your images come from social media. For pure DSLR/studio photos you can turn it off.",
                            ),
                        )
                        c_ig_two_stage_bar = gr.Checkbox(
                            label=tr("Zusätzlich Nav-Bars und Schatten erkennen", "Also detect nav bars and shadows"),
                            value=S["c_ig_two_stage_bar"],
                            info=tr(
                                "Zusätzliche Erkennung für Android-Nav-Bars (schwarz + UI-Icons) und Drop-Shadow-Gradienten. Triggert nur, wenn bereits ein Seitenrand gefunden wurde – das verhindert, dass dunkle Kissen oder Haare als Nav-Bar erkannt werden. Empfohlen: an.",
                                "Additional detection for Android nav bars (black + UI icons) and drop-shadow gradients. Only triggers when a side border was already detected – prevents dark pillows or hair from being misidentified as nav bars. Recommended: on.",
                            ),
                        )

                with gr.Accordion(tr("🧍 Subject-Sanity-Check (Gliedmaßen-Filter)", "🧍 Subject sanity check (limb filter)"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Wann fliegen Bilder hier raus?</b></summary>"
                        "\n\n"
                        "Manche Bilder zeigen zwar *etwas Menschliches*, aber nicht die "
                        "Person, um die es geht: nur Füße auf einem Strand-Foto, nur Hände "
                        "beim Kochen, ein Detail-Shot vom Schmuck. Die Gesichtserkennung "
                        "findet kein Gesicht, also würde das Bild ohne diesen Filter trotzdem "
                        "weiterlaufen und am Ende ggf. die KI fragen.\n\n"
                        "Der Sanity-Check verwirft solche Bilder direkt: Wenn weder ein "
                        "Gesicht erkannt wird, noch ein Torso (Schultern + Hüften), ist das "
                        "Bild für ein Person-LoRA wertlos.\n\n"
                        "**Wichtig:** Rückenansichten mit klar erkennbarem Torso (Schultern "
                        "sichtbar) bleiben erhalten – die sind für Pose-Diversität wertvoll. "
                        "Es geht wirklich nur um Bilder *ohne erkennbare Person als Ganzes*."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ When do images get rejected here?</b></summary>"
                        "\n\n"
                        "Some images show *something human* but not the actual person: just "
                        "feet on a beach photo, just hands while cooking, a jewelry detail "
                        "shot. Face detection finds no face, so without this filter the "
                        "image would still proceed and possibly be sent to the AI.\n\n"
                        "The sanity check rejects such images directly: if neither a face "
                        "nor a torso (shoulders + hips) is detected, the image is worthless "
                        "for a person LoRA.\n\n"
                        "**Important:** Back views with a clearly visible torso (shoulders "
                        "visible) are kept – they're valuable for pose diversity. This is "
                        "really only about images *without a recognizable person as a whole*."
                        "</details>",
                    ))
                    with gr.Row():
                        c_subject_sanity = gr.Checkbox(
                            label=tr("Sanity-Check aktivieren", "Enable sanity check"),
                            value=S["c_subject_sanity"],
                            info=tr(
                                "Empfohlen: an. Fängt 'fehlgeschlagene' Bilder (nur Hände/Füße) ab, bevor sie API-Kosten verursachen.",
                                "Recommended: on. Catches 'failed' images (hands/feet only) before they cause API costs.",
                            ),
                        )
                        c_subject_min_torso = gr.Slider(
                            label=tr("Min. Torso-Punkte (von 4)", "Min torso landmarks (of 4)"),
                            minimum=1,
                            maximum=4,
                            step=1,
                            value=S["c_subject_min_torso"],
                            info=tr(
                                "Wie viele der 4 Körperpunkte (2 Schultern, 2 Hüften) sichtbar sein müssen, damit ein Bild ohne Gesicht trotzdem bleibt. Empfohlen: 2 (halber Torso reicht). 4 = sehr streng, nur frontale Standards. 1 = sehr locker.",
                                "How many of the 4 body landmarks (2 shoulders, 2 hips) must be visible for an image without a face to be kept. Recommended: 2 (half torso is enough). 4 = very strict, only frontal standards. 1 = very loose.",
                            ),
                        )

                with gr.Accordion(tr("🔗 Duplikaterkennung (nach Bewertung)", "🔗 Duplicate detection (post-API)"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Wozu zwei verschiedene Methoden zur Duplikat-Erkennung?</b></summary>"
                        "\n\n"
                        "Die zwei Verfahren erkennen unterschiedliche Arten von Duplikaten – "
                        "und kein einzelnes Verfahren fängt beide ab. Deshalb arbeiten sie "
                        "parallel.\n\n"
                        "**pHash (Pixel-Vergleich):**\n\n"
                        "Vergleicht Bilder als Pixel-Strukturen. Berechnet einen kompakten "
                        "'Fingerabdruck' und vergleicht, wie viele Bits sich zwischen zwei "
                        "Fingerabdrücken unterscheiden (Hamming-Distanz). Erkennt Bilder, die "
                        "sich nur durch Kompression, leichte Crops oder Helligkeit unter-"
                        "scheiden – also klassische Duplikate, Re-Uploads, Screenshots vom "
                        "selben Foto.\n\n"
                        "**CLIP (Bedeutungs-Vergleich):**\n\n"
                        "Vergleicht Bilder semantisch. CLIP ist ein KI-Modell, das Bilder als "
                        "Vektor-Bedeutung darstellt. Erkennt Bilder mit gleichem Motiv, "
                        "Outfit, Setting – auch wenn sie aus unterschiedlichen Winkeln "
                        "stammen oder leichte Pose-Variationen haben. Für pHash sind solche "
                        "Bilder schon zu unterschiedlich, semantisch sind sie aber redundant.\n\n"
                        "**Beide zusammen:** Bei nur pHash bleiben semantische Duplikate "
                        "übrig (z. B. drei Selfies vom selben Outfit, leicht andere Köpfe). "
                        "Bei nur CLIP rutschen Pixel-Duplikate durch (z. B. dasselbe Foto "
                        "in verschiedenen Auflösungen). Empfohlen ist immer beide laufen "
                        "zu lassen.\n\n"
                        "**Hinweis zur frühen pHash-Vorfilterung:** Im Vorfilter-Bereich "
                        "läuft schon vor der API-Bewertung eine erste pHash-Runde, die nur "
                        "*offensichtliche* Duplikate aussortiert. Diese hier ist die "
                        "*finale* Runde nach der Bewertung, die mit Score-Information "
                        "entscheiden kann, welches der ähnlichen Bilder behalten wird."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ Why two different duplicate-detection methods?</b></summary>"
                        "\n\n"
                        "The two methods detect different kinds of duplicates – and no single "
                        "method catches both. That's why they run in parallel.\n\n"
                        "**pHash (pixel comparison):**\n\n"
                        "Compares images as pixel structures. Computes a compact "
                        "'fingerprint' and counts how many bits differ between two "
                        "fingerprints (hamming distance). Detects images that only differ in "
                        "compression, slight crops or brightness – classic duplicates, "
                        "re-uploads, screenshots of the same photo.\n\n"
                        "**CLIP (semantic comparison):**\n\n"
                        "Compares images by meaning. CLIP is an AI model that represents "
                        "images as a meaning vector. Detects images with the same subject, "
                        "outfit, setting – even from different angles or with slight pose "
                        "variations. To pHash these images are already too different, "
                        "semantically they're redundant.\n\n"
                        "**Both together:** With pHash only, semantic duplicates remain "
                        "(e.g. three selfies of the same outfit with slight head tilts). "
                        "With CLIP only, pixel duplicates slip through (e.g. the same photo "
                        "at different resolutions). Recommended is to run both.\n\n"
                        "**Note on early pHash:** In the pre-filter section there's an "
                        "earlier pHash round before AI scoring that only removes *obvious* "
                        "duplicates. This here is the *final* round after scoring, which "
                        "can use score information to decide which of the similar images "
                        "to keep."
                        "</details>",
                    ))
                    with gr.Row():
                        c_use_clip = gr.Checkbox(
                            label=tr("Bedeutungs-Vergleich (CLIP)", "Meaning-based (CLIP)"),
                            value=S["c_use_clip"],
                            info=tr(
                                "Erkennt Bilder mit gleichem Inhalt (Outfit, Pose, Setting), auch wenn sie aus leicht anderen Winkeln aufgenommen wurden. Empfohlen: an. Braucht CLIP ViT-L-14 (wird beim ersten Lauf heruntergeladen).",
                                "Detects images with the same content (outfit, pose, setting), even from slightly different angles. Recommended: on. Requires CLIP ViT-L-14 (auto-downloaded on first run).",
                            ),
                        )
                        c_use_phash = gr.Checkbox(
                            label=tr("Pixel-Vergleich (pHash)", "Pixel-based (pHash)"),
                            value=S["c_use_phash"],
                            info=tr(
                                "Erkennt visuell fast identische Bilder (Re-Uploads, gleiches Foto in verschiedenen Auflösungen). Empfohlen: an. Kostet praktisch keine Rechenzeit.",
                                "Detects visually near-identical images (re-uploads, same photo in different resolutions). Recommended: on. Costs practically no compute.",
                            ),
                        )
                    with gr.Row():
                        c_phash_thresh = gr.Slider(
                            label=tr("pHash-Schwelle (Hamming)", "pHash threshold (hamming)"),
                            minimum=2,
                            maximum=20,
                            step=1,
                            value=S["c_phash_thresh"],
                            info=tr(
                                "Wie ähnlich Bilder sein müssen, um als Duplikat zu gelten. Niedriger = strenger. Empfohlen: 8 (guter Kompromiss). Bei vielen Re-Uploads/Screenshots: 12. Bei knappem Material wo nichts verloren gehen darf: 4–6.",
                                "How similar images must be to count as duplicates. Lower = stricter. Recommended: 8 (good compromise). With many re-uploads/screenshots: 12. With scarce material where nothing should be lost: 4–6.",
                            ),
                        )
                        c_clip_thresh = gr.Slider(
                            label=tr("CLIP-Ähnlichkeits-Schwelle", "CLIP similarity threshold"),
                            minimum=0.90,
                            maximum=1.0,
                            step=0.005,
                            value=S["c_clip_thresh"],
                            info=tr(
                                "Ab welcher Ähnlichkeit (0–1) zwei Bilder als inhaltlich gleich gelten. Empfohlen: 0.985 (konservativ, nur wirklich ähnliche werden gefiltert). Niedriger als 0.97 wird aggressiv und kann unterschiedliche Bilder derselben Person zusammenwerfen.",
                                "At what similarity (0–1) two images count as the same content. Recommended: 0.985 (conservative, only truly similar images get filtered). Below 0.97 becomes aggressive and may merge different images of the same person.",
                            ),
                        )

                with gr.Accordion(tr("✂️ Smart Pre-Crop", "✂️ Smart pre-crop"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Was macht Smart Pre-Crop und wann brauche ich das?</b></summary>"
                        "\n\n"
                        "Bei großen Bildern (4K-Fotos, DSLR-Aufnahmen), auf denen die Person "
                        "klein im Bild ist – z. B. Ganzkörper-Aufnahmen mit viel Hintergrund – "
                        "erzeugt der Curator automatisch einen engen Headshot-Ausschnitt rund um "
                        "das Gesicht und schickt diesen *zusätzlich* zur KI-Bewertung. Beide "
                        "Versionen (Original und Crop) konkurrieren dann um den Platz im "
                        "Trainings-Set; die besser bewertete gewinnt.\n\n"
                        "**Wozu das gut ist:** Aus einem Foto, das als Ganzkörper-Aufnahme nur "
                        "mittelmäßig ist (unruhiger Hintergrund, ungünstige Pose), kannst du so "
                        "trotzdem ein gutes Identitäts-Bild fürs Training herausziehen – ohne "
                        "ein zweites Foto-Set zu brauchen.\n\n"
                        "**Wann es ausgelöst wird:** Nur bei Bildern, die groß genug sind "
                        "(mindestens 2 Megapixel) und bei denen das Gesicht weniger als 7 % "
                        "der Bildfläche einnimmt. Bei reinen Headshots oder Selfies passiert "
                        "nichts – da gibt es nichts zu zoomen.\n\n"
                        "**Kosten:** Pro ausgelöstem Pre-Crop ein zusätzlicher API-Call. Bei "
                        "den meisten Datasets sind das nur 10–30 % der Bilder."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ What does Smart Pre-Crop do and when do I need it?</b></summary>"
                        "\n\n"
                        "For large images (4K photos, DSLR shots) where the person is small in "
                        "the frame – e.g. full-body shots with a lot of background – the curator "
                        "automatically generates a tight headshot crop around the face and sends "
                        "it *additionally* to the AI for scoring. Both versions (original and "
                        "crop) then compete for a spot in the training set; the better-scored "
                        "one wins.\n\n"
                        "**Why it matters:** From a photo that's only mediocre as a full-body "
                        "shot (busy background, awkward pose), you can still extract a good "
                        "identity image for training – without needing a second photo set.\n\n"
                        "**When it triggers:** Only on images that are large enough (at least "
                        "2 megapixels) and where the face takes up less than 7 % of the image "
                        "area. Pure headshots or selfies are skipped – nothing to zoom into.\n\n"
                        "**Cost:** One additional API call per triggered pre-crop. For most "
                        "datasets that's only 10–30 % of the images."
                        "</details>",
                    ))
                    c_smart_crop = gr.Checkbox(
                        label=tr("Smart Pre-Crop aktivieren", "Enable smart pre-crop"),
                        value=S["c_smart_crop"],
                        info=tr(
                            "Empfohlen: an. Wird ohnehin nur bei großen Bildern mit kleinem Gesicht ausgelöst – kostet bei kleinen Datasets fast nichts extra.",
                            "Recommended: on. Only triggers for large images with a small face – costs almost nothing extra on small datasets.",
                        ),
                    )
                    with gr.Row():
                        c_crop_gain = gr.Slider(
                            label=tr("Mindestvorsprung des Crops", "Min crop score gain"),
                            minimum=0,
                            maximum=30,
                            step=1,
                            value=S["c_crop_gain"],
                            info=tr(
                                "Wie viele Punkte besser der Crop bewertet sein muss als das Original, damit er übernommen wird. Empfohlen: 8. Niedriger (4–6) lässt mehr Crops gewinnen, gut wenn du Headshots brauchst und deine Originale meist Full-Body sind. Höher (12+) ist konservativ.",
                                "How many points the crop must score above the original to be accepted. Recommended: 8. Lower (4–6) lets more crops win — good if you need headshots and your originals are mostly full-body. Higher (12+) is conservative.",
                            ),
                        )
                        c_crop_pad = gr.Slider(
                            label=tr("Padding um das Gesicht", "Padding around the face"),
                            minimum=0.3,
                            maximum=1.5,
                            step=0.05,
                            value=S["c_crop_pad"],
                            info=tr(
                                "Rand pro Seite um das Gesicht, gemessen in Vielfachen der Gesichtsgröße. Empfohlen: 0.6 (ergibt Gesicht + Haare + obere Schultern, klassischer Headshot). 0.4 = enger Gesichts-Crop. 0.8+ = lockerer mit Schulter-Anteil.",
                                "Padding per side around the face, measured in multiples of face size. Recommended: 0.6 (face + hair + upper shoulders, classic headshot). 0.4 = tight face crop. 0.8+ = looser with shoulder area.",
                            ),
                        )

                with gr.Accordion(tr("📊 Clustering & Diversität", "📊 Clustering & diversity"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Was machen Clustering und Diversitäts-Penalty?</b></summary>"
                        "\n\n"
                        "Wenn deine Quellbilder typisch aus Social Media kommen, hast du oft "
                        "10–30 Fotos aus *derselben Foto-Session* (gleiche Kleidung, gleicher "
                        "Ort, alle in 2 Minuten geschossen). Ohne Begrenzung würde dein "
                        "Trainings-Set hauptsächlich aus dieser einen Session bestehen, und "
                        "das LoRA lernt dann sehr engmaschig genau dieses eine Outfit – "
                        "schlecht für die spätere Generierungs-Vielfalt.\n\n"
                        "**Clustering:**\n\n"
                        "Der Curator gruppiert Bilder anhand von Kleidung, Hintergrund und "
                        "Aufnahmezeitpunkt zu **Outfit-Clustern** und **Session-Clustern**. "
                        "Du kannst pro Cluster ein Maximum festlegen – z. B. 'höchstens 4 "
                        "Bilder mit demselben Outfit'.\n\n"
                        "**Diversitäts-Penalty:**\n\n"
                        "Zusätzlich gibt es bei der Endauswahl Punktabzug, wenn ähnliche "
                        "Bilder bereits ausgewählt wurden – betrifft Outfit, Hintergrund, "
                        "Beleuchtung, Gesichtsausdruck und (wenn die Kopfpose-Diversität "
                        "aktiv ist) auch Kopfposen. Bei zwei ähnlich guten Bildern gewinnt "
                        "dann das mit den selteneren Eigenschaften.\n\n"
                        "**Empfohlen:** Beide an. Das Clustering verhindert harte "
                        "Überrepräsentationen, die Penalty sorgt für feine Vielfalt zwischen "
                        "den Bildern, die durch das Clustering schon gefiltert sind."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ What do clustering and diversity penalty do?</b></summary>"
                        "\n\n"
                        "Source images from social media often include 10–30 photos from the "
                        "*same photo session* (same outfit, same location, all shot within 2 "
                        "minutes). Without limits, your training set would mostly come from "
                        "this one session, and the LoRA would learn this one outfit very "
                        "tightly – bad for later generation variety.\n\n"
                        "**Clustering:**\n\n"
                        "The curator groups images by clothing, background and capture time "
                        "into **outfit clusters** and **session clusters**. You can set a "
                        "max per cluster – e.g. 'at most 4 images with the same outfit'.\n\n"
                        "**Diversity penalty:**\n\n"
                        "Additionally, during final selection, similar images get a score "
                        "deduction if similar ones are already picked – covers outfit, "
                        "background, lighting, facial expression and (when head pose "
                        "diversity is on) also head pose. Between two similarly-good images, "
                        "the one with rarer attributes wins.\n\n"
                        "**Recommended:** Both on. Clustering prevents hard over-"
                        "representation, the penalty creates fine variety between the "
                        "images already filtered through clustering."
                        "</details>",
                    ))
                    c_use_cluster = gr.Checkbox(
                        label=tr("Outfit-/Session-Clustering aktivieren", "Enable outfit/session clustering"),
                        value=S["c_use_cluster"],
                        info=tr(
                            "Gruppiert ähnliche Bilder nach Kleidung, Hintergrund und Aufnahmezeit. Empfohlen: an. Verhindert dass eine einzelne Foto-Session den Trainings-Set dominiert.",
                            "Groups similar images by clothing, background and capture time. Recommended: on. Prevents a single photo session from dominating the training set.",
                        ),
                    )
                    with gr.Row():
                        c_max_outfit = gr.Slider(
                            label=tr("Max. Bilder pro Outfit", "Max images per outfit"),
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=S["c_max_outfit"],
                            info=tr(
                                "Höchstens so viele Bilder mit demselben Outfit landen im Final-Set. Empfohlen: 4. Niedriger (2–3) zwingt mehr Outfit-Vielfalt, höher (6+) ist locker (für Datasets mit ohnehin viel Outfit-Wechsel).",
                                "At most this many images with the same outfit end up in the final set. Recommended: 4. Lower (2–3) forces more outfit variety, higher (6+) is loose (for datasets with lots of outfit changes anyway).",
                            ),
                        )
                        c_max_session = gr.Slider(
                            label=tr("Max. Bilder pro Foto-Session", "Max images per photo session"),
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=S["c_max_session"],
                            info=tr(
                                "Höchstens so viele Bilder aus derselben Aufnahme-Session (gleicher Ort, gleicher Tag). Empfohlen: 5. Etwas höher als das Outfit-Limit, weil Sessions oft mehrere Outfits enthalten.",
                                "At most this many images from the same shoot session (same location, same day). Recommended: 5. Slightly higher than the outfit limit because sessions often contain multiple outfits.",
                            ),
                        )
                    c_use_diversity = gr.Checkbox(
                        label=tr("Diversitäts-Punktabzug bei Endauswahl", "Diversity penalty during final selection"),
                        value=S["c_use_diversity"],
                        info=tr(
                            "Bei der finalen Endauswahl bekommen Bilder mit häufiger Kombination aus Outfit/Hintergrund/Licht/Gesichtsausdruck Punktabzug. Empfohlen: an.",
                            "During final selection, images with frequent combinations of outfit/background/lighting/expression get a score deduction. Recommended: on.",
                        ),
                    )

                with gr.Accordion(tr("🧭 Kopfpose-Diversität", "🧭 Head pose diversity"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Wozu Kopfpose-Diversität?</b></summary>"
                        "\n\n"
                        "Wenn dein Trainings-Set aus 25 Frontal-Aufnahmen und nur 5 anderen "
                        "Posen besteht, lernt das LoRA Frontalansichten hervorragend, aber "
                        "3/4-Profile schlecht. Bei späteren Generierungen siehst du das oft "
                        "als 'Identitätsbruch' – sobald die Pose vom Frontalen abweicht, "
                        "passt das Gesicht nicht mehr ganz.\n\n"
                        "Die Kopfpose-Diversität sorgt dafür, dass bei der Endauswahl Bilder "
                        "mit gleicher Kopfpose (frontal, 3/4-Profil-links, 3/4-Profil-rechts, "
                        "Profil, von oben, von unten, von hinten) Punktabzug bekommen, sobald "
                        "schon genug von dieser Pose im Set sind. Bei zwei ähnlich guten "
                        "Bildern gewinnt dann das mit der unterrepräsentierten Pose.\n\n"
                        "**Wichtig:** Kein Hard-Reject – qualitativ deutlich schlechtere "
                        "Bilder mit seltener Pose werden nicht stur bevorzugt. Der Abzug "
                        "wirkt nur, wenn die Bilder ähnlich gut sind.\n\n"
                        "**Empfohlen:** an. Die KI bewertet die Kopfpose ohnehin im normalen "
                        "Bewertungs-Schritt mit – also keine zusätzlichen API-Kosten."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ Why head pose diversity?</b></summary>"
                        "\n\n"
                        "If your training set is 25 frontal shots and only 5 other poses, "
                        "the LoRA learns frontal views excellently but three-quarter views "
                        "poorly. In later generation you see this as 'identity break' – as "
                        "soon as the pose deviates from frontal, the face doesn't quite "
                        "match.\n\n"
                        "Head pose diversity ensures that during final selection, images "
                        "with the same head pose (frontal, three-quarter-left, three-"
                        "quarter-right, profile, looking up, looking down, back) get a "
                        "score deduction once enough of that pose is already in the set. "
                        "Between two similarly-good images, the one with the under-"
                        "represented pose wins.\n\n"
                        "**Important:** Not a hard reject – clearly worse images with rare "
                        "poses aren't blindly preferred. The penalty only matters when "
                        "images are similarly good.\n\n"
                        "**Recommended:** on. The AI scores head pose anyway during normal "
                        "scoring – so no additional API costs."
                        "</details>",
                    ))
                    c_use_pose_diversity = gr.Checkbox(
                        label=tr("Pose-Diversität aktivieren", "Enable pose diversity"),
                        value=S["c_use_pose_diversity"],
                        info=tr(
                            "Empfohlen: an. Nutzt die KI-Klassifikation der Kopfpose – kostet keine zusätzlichen API-Calls.",
                            "Recommended: on. Uses the AI's head pose classification – no extra API calls.",
                        ),
                    )
                    with gr.Row():
                        c_pose_soft_limit = gr.Slider(
                            label=tr("Erlaubte Anzahl pro Pose (ohne Abzug)", "Allowed per pose (without penalty)"),
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=S["c_pose_soft_limit"],
                            info=tr(
                                "Bis zu wie vielen Bildern pro Pose es noch keinen Punktabzug gibt. Empfohlen: 2. Bei kleinen Datasets (<20 Bilder) macht 1 mehr Druck Richtung Vielfalt; bei großen (>50) eher 3.",
                                "Up to how many images per pose receive no penalty. Recommended: 2. For small datasets (<20 images), 1 pushes harder toward variety; for large ones (>50), 3.",
                            ),
                        )
                        c_pose_penalty_weight = gr.Slider(
                            label=tr("Stärke des Punktabzugs", "Penalty strength"),
                            minimum=0.0,
                            maximum=10.0,
                            step=0.5,
                            value=S["c_pose_penalty_weight"],
                            info=tr(
                                "Wie deutlich überzählige Posen abgewertet werden. Empfohlen: 4.0. Höher (6.0+) bevorzugt Pose-Vielfalt fast um jeden Preis – gut bei sehr selfie-lastigem Material. Niedriger (2.0) macht den Effekt subtil.",
                                "How strongly excess poses get penalized. Recommended: 4.0. Higher (6.0+) prefers pose variety almost at any cost – good for very selfie-heavy material. Lower (2.0) makes the effect subtle.",
                            ),
                        )

                with gr.Accordion(tr("🪪 Identitäts-Konsistenz-Check (ArcFace)", "🪪 Identity consistency check (ArcFace)"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Was macht der Identitäts-Konsistenz-Check?</b></summary>"
                        "\n\n"
                        "Bei Person-LoRAs gibt es eine Klasse von Bug, die unsichtbar bleibt "
                        "und das Training trotzdem ruiniert: **Ein einzelnes Bild der falschen "
                        "Person mischt sich in dein Dataset.** Das passiert leichter als man "
                        "denkt – die Schwester sieht ähnlich aus, ein altes Foto stammt noch "
                        "aus der Pubertät, oder Smart Pre-Crop hat versehentlich die Bekannte "
                        "im Hintergrund gezoomt.\n\n"
                        "29 von 30 Bildern korrekt, 1 falsche Person dabei → das LoRA "
                        "produziert visuell oft 'fast richtige' Gesichter, die irgendwie nicht "
                        "ganz die Person treffen. Du siehst auf den ersten Blick nicht, woran "
                        "es liegt.\n\n"
                        "**Wie der Check funktioniert:**\n\n"
                        "ArcFace ist ein KI-Modell, das speziell auf Gesichts-Identität "
                        "trainiert ist. Es berechnet pro Bild einen Identitäts-Vektor, der "
                        "bei Bildern *derselben Person* sehr ähnlich ist – auch bei anderen "
                        "Posen, Beleuchtung oder Alter. Der Curator berechnet diese Vektoren "
                        "für alle Bilder im Final-Set, mittelt sie zu einer 'Set-Identität' "
                        "und vergleicht jedes einzelne Bild damit.\n\n"
                        "**Outlier-Trimming:** Vor der Mittelung werden die schlechtesten "
                        "Vektoren verworfen. Das verhindert, dass 2–3 falsche Bilder den "
                        "Mittelwert in ihre Richtung ziehen und der Check dann unzuverlässig "
                        "wird.\n\n"
                        "**Drei Klassifikationen pro Bild:**\n\n"
                        "**Hard-Flag** (Ähnlichkeit unter Hard-Schwelle): Wahrscheinlich eine "
                        "andere Person. Das Bild wird **aus 01_train_ready entfernt** und mit "
                        "Präfix `IDCHECK_` nach 06_needs_manual_review kopiert. Captions "
                        "bleiben unangetastet.\n\n"
                        "**Soft-Flag** (Ähnlichkeit zwischen Hard und Soft): Grenzfall, "
                        "könnte dieselbe Person mit Beauty-Filter / extremem Makeup / Brille "
                        "sein. Das Bild **bleibt** im Train-Set, wird aber im Markdown-Report "
                        "markiert für deinen visuellen Check.\n\n"
                        "**OK** (Ähnlichkeit über Soft-Schwelle): Klar dieselbe Person.\n\n"
                        "**Voraussetzung:** Mindestens 5 Gesichter im Set müssen erkennbar "
                        "sein, sonst ist der Mittelwert nicht aussagekräftig und der Check "
                        "wird übersprungen.\n\n"
                        "**Installation:** `pip install insightface onnxruntime-gpu` (oder "
                        "`onnxruntime` ohne GPU). Beim ersten Lauf werden ~250 MB Modell-"
                        "Daten nach `~/.insightface/models/` heruntergeladen. Wenn du "
                        "ComfyUI mit FaceID/ReActor benutzt, sind die Modelle wahrscheinlich "
                        "schon da.\n\n"
                        "**Lizenz-Hinweis:** Die ArcFace-Modelle (`buffalo_l` etc.) sind nur "
                        "für nicht-kommerzielle Forschung freigegeben. Für private LoRA-"
                        "Erstellung ist das unproblematisch."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ What does the identity consistency check do?</b></summary>"
                        "\n\n"
                        "Person LoRAs have a class of bug that stays invisible but ruins "
                        "training anyway: **a single image of the wrong person sneaking into "
                        "your dataset.** It happens more easily than you'd think – a sister "
                        "looks similar, an old photo is from puberty, or smart pre-crop "
                        "accidentally zoomed in on the friend in the background.\n\n"
                        "29 out of 30 images correct, 1 wrong person → the LoRA often "
                        "produces 'almost right' faces that somehow don't quite hit the "
                        "person. At first glance, you can't tell why.\n\n"
                        "**How the check works:**\n\n"
                        "ArcFace is an AI model trained specifically on face identity. It "
                        "computes one identity vector per image that's very similar for "
                        "*the same person* across different poses, lighting or ages. The "
                        "curator computes these vectors for all final-set images, averages "
                        "them into a 'set identity' and compares each image against it.\n\n"
                        "**Outlier trimming:** Before averaging, the worst vectors are "
                        "dropped. This prevents 2–3 wrong images from pulling the average "
                        "toward their identity and making the check unreliable.\n\n"
                        "**Three classifications per image:**\n\n"
                        "**Hard flag** (similarity below hard threshold): Likely a different "
                        "person. The image is **removed from 01_train_ready** and copied to "
                        "06_needs_manual_review with prefix `IDCHECK_`. Captions stay "
                        "untouched.\n\n"
                        "**Soft flag** (similarity between hard and soft): Borderline, could "
                        "be the same person with a beauty filter / heavy makeup / glasses. "
                        "The image **stays** in the train set but gets marked in the "
                        "markdown report for your visual check.\n\n"
                        "**OK** (similarity above soft threshold): Clearly the same person.\n\n"
                        "**Requirement:** At least 5 detected faces in the set, otherwise "
                        "the average is not meaningful and the check gets skipped.\n\n"
                        "**Installation:** `pip install insightface onnxruntime-gpu` (or "
                        "`onnxruntime` without GPU). On first run, ~250 MB of model data "
                        "auto-download to `~/.insightface/models/`. If you use ComfyUI with "
                        "FaceID/ReActor, the models are probably already there.\n\n"
                        "**License note:** The ArcFace models (`buffalo_l` etc.) are "
                        "licensed for non-commercial research only. For private LoRA "
                        "creation that's not an issue."
                        "</details>",
                    ))
                    c_use_arcface = gr.Checkbox(
                        label=tr("Identitäts-Check aktivieren", "Enable identity check"),
                        value=S["c_use_arcface"],
                        info=tr(
                            "Empfohlen: an. Wird automatisch übersprungen, wenn insightface nicht installiert ist – also kein Problem, wenn du das erst später nachinstallierst.",
                            "Recommended: on. Automatically skipped if insightface isn't installed – no problem if you install it later.",
                        ),
                    )
                    with gr.Row():
                        c_arcface_hard = gr.Slider(
                            label=tr("Hard-Schwelle (Bild fliegt raus)", "Hard threshold (image gets removed)"),
                            minimum=0.20,
                            maximum=0.80,
                            step=0.01,
                            value=S["c_arcface_hard"],
                            info=tr(
                                "Unter diesem Ähnlichkeits-Wert wird das Bild aus dem Train-Set entfernt. Empfohlen: 0.50 (konservativ – nur klare Mismatches). 0.40 = sehr tolerant. 0.60 = streng (kann auch echte Treffer mit starkem Beauty-Filter rauswerfen).",
                                "Below this similarity, the image is removed from the train set. Recommended: 0.50 (conservative – only clear mismatches). 0.40 = very tolerant. 0.60 = strict (may also reject real matches with heavy beauty filters).",
                            ),
                        )
                        c_arcface_soft = gr.Slider(
                            label=tr("Soft-Schwelle (Bild bleibt, wird markiert)", "Soft threshold (image stays, gets marked)"),
                            minimum=0.30,
                            maximum=0.90,
                            step=0.01,
                            value=S["c_arcface_soft"],
                            info=tr(
                                "Zwischen Hard und Soft gilt ein Bild als Grenzfall. Empfohlen: 0.65. Niedriger = nur deutliche Grenzfälle werden markiert. Höher = mehr Bilder bekommen den Soft-Flag (gut wenn du genauer hinschauen willst).",
                                "Between hard and soft, an image counts as borderline. Recommended: 0.65. Lower = only clear borderline cases get marked. Higher = more images get a soft flag (good if you want to inspect more carefully).",
                            ),
                        )
                    with gr.Row():
                        c_arcface_trim = gr.Slider(
                            label=tr("Anteil verworfener Ausreißer", "Outlier-trim fraction"),
                            minimum=0.0,
                            maximum=0.30,
                            step=0.01,
                            value=S["c_arcface_trim"],
                            info=tr(
                                "Welcher Anteil der schlechtesten Identitäts-Vektoren vor der Mittelwert-Berechnung verworfen wird. Empfohlen: 0.10 (10 %, fängt 1–2 falsche Bilder im 30er-Set ab). 0 = kein Trimming. 0.20 = bei Datasets, in denen du mehrere falsche Bilder vermutest.",
                                "Fraction of worst identity vectors dropped before centroid calculation. Recommended: 0.10 (10 %, catches 1–2 wrong images in a 30-image set). 0 = no trimming. 0.20 = for datasets where you suspect several wrong images.",
                            ),
                        )
                        c_arcface_min_faces = gr.Slider(
                            label=tr("Min. erkannte Gesichter im Set", "Min detected faces in set"),
                            minimum=3,
                            maximum=15,
                            step=1,
                            value=S["c_arcface_min_faces"],
                            info=tr(
                                "Wenn weniger Gesichter im Final-Set erkannt werden, wird der Check übersprungen (zu unzuverlässig). Empfohlen: 5. Bei sehr kleinen Datasets (Ziel <15 Bilder) kannst du auf 3 senken.",
                                "If fewer faces are detected in the final set, the check is skipped (not reliable enough). Recommended: 5. For very small datasets (target <15 images) you can lower to 3.",
                            ),
                        )
                    with gr.Row():
                        c_arcface_model = gr.Dropdown(
                            label=tr("ArcFace-Modell", "ArcFace model"),
                            choices=["buffalo_l", "buffalo_s", "buffalo_m", "antelopev2"],
                            value=S["c_arcface_model"],
                            info=tr(
                                "Empfohlen: buffalo_l (höchste Genauigkeit). buffalo_s ist schneller und kleiner, aber weniger genau – nur wählen wenn du auf CPU läufst und Geschwindigkeit zählt.",
                                "Recommended: buffalo_l (highest accuracy). buffalo_s is faster and smaller but less accurate – only pick if you run on CPU and speed matters.",
                            ),
                        )
                        c_arcface_det_size = gr.Slider(
                            label=tr("Auflösung der Gesichtserkennung (px)", "Face detection resolution (px)"),
                            minimum=320,
                            maximum=1024,
                            step=32,
                            value=S["c_arcface_det_size"],
                            info=tr(
                                "Auf welche Größe Bilder vor der Gesichtserkennung skaliert werden. Empfohlen: 640 (balanciert). 320 = schneller, kann aber kleine Gesichter verpassen. 1024 = genauer, aber deutlich langsamer.",
                                "Resolution images get scaled to before face detection. Recommended: 640 (balanced). 320 = faster but may miss small faces. 1024 = more accurate but noticeably slower.",
                            ),
                        )

                with gr.Accordion(tr("📝 Caption-Regeln", "📝 Caption rules"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Wie funktionieren die Captions?</b></summary>"
                        "\n\n"
                        "Beim LoRA-Training bekommt das Modell pro Bild eine Text-"
                        "Beschreibung (Caption) mitgeliefert. Daran lernt es, was zur Person "
                        "gehört (intrinsisch, immer gleich) und was austauschbar ist "
                        "(situativ, ändert sich).\n\n"
                        "Der Curator generiert die Captions automatisch aus der KI-Analyse "
                        "der Bilder. Du kannst auswählen, welche Merkmale in die Captions "
                        "aufgenommen werden – das beeinflusst, wie das LoRA später auf "
                        "Prompts reagiert.\n\n"
                        "**Caption-Preset:**\n\n"
                        "Ein Voreinstellungs-Bündel, das passende Felder für ein bestimmtes "
                        "Basis-Modell vorbelegt. Du kannst danach die einzelnen Felder "
                        "trotzdem nachjustieren.\n\n"
                        "**Aktive Caption-Felder:**\n\n"
                        "Welche Eigenschaften sollen in die Beschreibung? Hier gibt es zwei "
                        "Denkschulen, je nach Basismodell:\n\n"
                        "**Alles inkludieren** (empfohlen für ERNIE und allgemein gut für "
                        "Anfänger): Auch permanente Eigenschaften wie Hautfarbe, konstante "
                        "Frisur, Augenfarbe werden in jede Caption geschrieben. Vorteil: "
                        "Das LoRA hat redundante Anker und ist robust. Nachteil: Wenn du "
                        "im späteren Prompt nicht alle Anker erwähnst, kann das Modell "
                        "verwirrt sein.\n\n"
                        "**Nur Veränderliches** (für Z-Image fortgeschritten): Permanente "
                        "Eigenschaften werden weggelassen, weil das LoRA von selbst lernen "
                        "soll, dass sie zur Person gehören. Nur situative Dinge wie "
                        "Kleidung, Brille, Hintergrund kommen rein. Vorteil: Sauberere "
                        "Trennung von 'Person' und 'Situation' in den späteren "
                        "Generierungen. Nachteil: Sensibler gegen Caption-Fehler.\n\n"
                        "**Wenn du unsicher bist:** Lass das Preset entscheiden – es ist "
                        "auf typische Anwendungsfälle voreingestellt."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ How do captions work?</b></summary>"
                        "\n\n"
                        "During LoRA training, each image gets a text description (caption) "
                        "alongside it. From this the model learns what's intrinsic to the "
                        "person (always the same) and what's swappable (situational, "
                        "changes).\n\n"
                        "The curator generates captions automatically from the AI image "
                        "analysis. You pick which attributes go into the captions – this "
                        "affects how the LoRA later reacts to prompts.\n\n"
                        "**Caption preset:**\n\n"
                        "A bundle preselecting fields suitable for a particular base model. "
                        "You can fine-tune individual fields afterwards.\n\n"
                        "**Active caption fields:**\n\n"
                        "Which attributes go into the description? Two schools of thought, "
                        "depending on the base model:\n\n"
                        "**Include everything** (recommended for ERNIE and generally safer "
                        "for beginners): Persistent attributes like skin tone, consistent "
                        "hair, eye color are written into every caption. Pro: The LoRA has "
                        "redundant anchors and is robust. Con: If your later prompts don't "
                        "mention all anchors, the model may get confused.\n\n"
                        "**Only changeable** (advanced, for Z-Image): Persistent attributes "
                        "are omitted so the LoRA learns by itself that they belong to the "
                        "person. Only situational things like clothing, glasses, background "
                        "go in. Pro: Cleaner separation of 'person' vs 'situation' in "
                        "later generations. Con: More sensitive to caption errors.\n\n"
                        "**If unsure:** Let the preset decide – it's preset for typical "
                        "use cases."
                        "</details>",
                    ))
                    c_caption_profile = gr.Dropdown(
                        label=tr("Caption-Voreinstellung", "Caption preset"),
                        choices=caption_profile_choices(),
                        value=normalize_caption_profile(S.get("c_caption_profile")) or "shared_compact",
                        info=tr(
                            "Voreingestelltes Schema für ein Basis-Modell. Empfohlen: 'Shared Compact' für Z-Image und ERNIE. Die einzelnen Felder unten kannst du nach Auswahl trotzdem manuell anpassen.",
                            "Preset schema for a base model. Recommended: 'Shared Compact' for Z-Image and ERNIE. You can still manually tweak the individual fields below after selection.",
                        ),
                    )
                    c_captions = gr.CheckboxGroup(
                        label=tr("Aktive Caption-Felder", "Active caption fields"),
                        choices=CAPTION_FIELD_CHOICES,
                        value=S["c_captions"],
                        info=tr(
                            "Welche Merkmale in die Trainings-Captions aufgenommen werden. Empfehlung hängt vom Basis-Modell ab: Bei ERNIE meistens alle Felder einschließen (das Basis-Dataset ist asiatisch geprägt, redundante Anker helfen). Bei Z-Image Base sind die Meinungen geteilt – manche empfehlen ebenfalls alles, andere nur veränderliche Merkmale (Kleidung, Brille) und permanente Eigenschaften (Hautfarbe, konstante Frisur, Tattoos) weglassen, damit das LoRA selbst lernt was zur Person gehört. Im Zweifel auf das Preset oben verlassen.",
                            "Which attributes go into the training captions. The recommendation depends on the base model: for ERNIE, usually include all fields (its base dataset is Asia-leaning, redundant anchors help). For Z-Image Base, opinions vary – some recommend including everything too, others only changeable attributes (clothing, glasses) and dropping persistent ones (skin tone, consistent hairstyle, tattoos) so the LoRA learns by itself what belongs to the person. When in doubt, trust the preset above.",
                        ),
                    )
                    c_caption_profile.change(
                        fn=resolve_caption_fields_for_profile,
                        inputs=[c_caption_profile, c_captions],
                        outputs=[c_captions],
                    )
                    c_captions.change(
                        fn=detect_caption_profile,
                        inputs=[c_captions],
                        outputs=[c_caption_profile],
                    )

                with gr.Accordion(tr("💾 Export-Optionen", "💾 Export options"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Welche Ordner und Exporte gibt es?</b></summary>"
                        "\n\n"
                        "Der Curator legt im Ausgabeverzeichnis mehrere Unterordner an, "
                        "abhängig von den hier ausgewählten Optionen:\n\n"
                        "**01_train_ready** (immer): Die finalen Bilder mit Captions, "
                        "fertig fürs LoRA-Training. Das ist das eigentliche Ergebnis.\n\n"
                        "**02_keep_unused** (immer): Bilder, die qualitativ gut genug "
                        "wären, aber wegen der Cluster-Limits oder Diversitäts-Regeln nicht "
                        "ins finale Set kamen. Falls dir später Material fehlt, kannst du "
                        "von hier nachschöpfen.\n\n"
                        "**03_rejected** (optional): Verworfene Bilder mit Begründung in "
                        "der Caption-Datei (z. B. 'rejected: face_blur_too_low'). Nützlich "
                        "zum Debuggen, ob deine Schwellen zu streng sind. Bei großen "
                        "Datasets kann das viele MB werden.\n\n"
                        "**05_review_candidates** (optional): Bilder die das Hauptmodell "
                        "als 'review' markiert hat – Grenzfälle, die du visuell prüfen "
                        "kannst. Wenn die Eskalation an ist, sind das die Bilder, die "
                        "zusätzlich vom stärkeren Modell entschieden wurden.\n\n"
                        "**06_needs_manual_review** (immer, wenn nötig): Sammelt alles, "
                        "was menschliches Auge braucht – z. B. NSFW-Verdachtsfälle (Präfix "
                        "`NSFW_`) und Hard-Flag-Identitätsmismatches (Präfix `IDCHECK_`).\n\n"
                        "**08_smart_crop_pairs** (optional): Pärchen aus Original und "
                        "Smart-Pre-Crop-Variante mit beiden Scores im Dateinamen. Wertvoll "
                        "zum Debuggen, ob deine Smart-Crop-Einstellungen sinnvoll sind. "
                        "Kostet Speicherplatz, aber bei Bedarf sehr aufschlussreich."
                        "</details>",
                        "<details>"
                        "<summary><b>ℹ️ What folders and exports exist?</b></summary>"
                        "\n\n"
                        "The curator creates several subfolders in the output directory, "
                        "depending on the options selected here:\n\n"
                        "**01_train_ready** (always): The final images with captions, "
                        "ready for LoRA training. This is the actual deliverable.\n\n"
                        "**02_keep_unused** (always): Images that would be quality-wise "
                        "good enough but didn't make the final set due to cluster limits "
                        "or diversity rules. If you later need more material, you can pull "
                        "from here.\n\n"
                        "**03_rejected** (optional): Rejected images with reason in the "
                        "caption file (e.g. 'rejected: face_blur_too_low'). Useful for "
                        "debugging whether your thresholds are too strict. For large "
                        "datasets this can grow to many MB.\n\n"
                        "**05_review_candidates** (optional): Images the main model "
                        "flagged as 'review' – borderline cases for visual inspection. "
                        "When escalation is on, these are the images additionally "
                        "decided by the stronger model.\n\n"
                        "**06_needs_manual_review** (always, when needed): Collects "
                        "everything that needs human eyes – e.g. NSFW suspects (prefix "
                        "`NSFW_`) and hard-flagged identity mismatches (prefix "
                        "`IDCHECK_`).\n\n"
                        "**08_smart_crop_pairs** (optional): Pairs of original and "
                        "smart-pre-crop variant with both scores in the filename. "
                        "Valuable for debugging whether your smart-crop settings make "
                        "sense. Costs disk space, but very informative when needed."
                        "</details>",
                    ))
                    with gr.Row():
                        c_exp_review = gr.Checkbox(
                            label=tr("Review-Kandidaten exportieren", "Export review candidates"),
                            value=S["c_exp_review"],
                            info=tr(
                                "Speichert Bilder im Grenzbereich in `05_review_candidates` zur manuellen Sichtung. Empfohlen: an, besonders wenn dir der Curator gerade neue Schwellen lernt.",
                                "Saves borderline images to `05_review_candidates` for manual review. Recommended: on, especially while you're tuning the curator's thresholds.",
                            ),
                        )
                        c_exp_reject = gr.Checkbox(
                            label=tr("Verworfene Bilder exportieren", "Export rejected images"),
                            value=S["c_exp_reject"],
                            info=tr(
                                "Speichert verworfene Bilder mit Reject-Grund in `03_rejected`. Empfohlen: an für die ersten Läufe (zum Debuggen). Bei großen Produktions-Datasets kannst du es ausschalten, um Platz zu sparen.",
                                "Saves rejected images with reject reason to `03_rejected`. Recommended: on for initial runs (for debugging). For large production datasets you can turn it off to save space.",
                            ),
                        )
                        c_exp_compare = gr.Checkbox(
                            label=tr("Smart-Crop-Vergleichspaare exportieren", "Export smart-crop comparison pairs"),
                            value=S["c_exp_compare"],
                            info=tr(
                                "Speichert Original und Smart-Crop nebeneinander in `08_smart_crop_pairs`, mit beiden Bewertungen im Dateinamen. Empfohlen: an, wenn du Smart Pre-Crop nutzt – essentiell zum Debuggen, falls die Crops nicht so aussehen wie erwartet.",
                                "Saves original and smart crop side-by-side in `08_smart_crop_pairs`, with both scores in the filename. Recommended: on if you use smart pre-crop – essential for debugging when crops don't look as expected.",
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
                    c_use_early_phash_loop1, c_early_phash_thresh_1, c_early_phash_keep_1,
                    c_use_early_phash_loop2, c_early_phash_thresh_2, c_early_phash_keep_2,
                    c_subject_sanity, c_subject_min_torso,
                    c_ig_frame_crop, c_ig_two_stage_bar,
                    c_use_clip, c_use_phash, c_phash_thresh, c_clip_thresh,
                    c_smart_crop, c_crop_gain, c_crop_pad,
                    c_use_cluster, c_max_outfit, c_max_session, c_use_diversity,
                    c_use_pose_diversity, c_pose_soft_limit, c_pose_penalty_weight,
                    c_use_arcface, c_arcface_hard, c_arcface_soft, c_arcface_trim,
                    c_arcface_min_faces, c_arcface_model, c_arcface_det_size,
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
        print("WARNING: Missing scripts:")
        for m in missing:
            print(m)
        print("Please place all files in the same folder.\n")

    venv_ok = os.path.isfile(os.path.join(SCRIPT_DIR, "curator_env", "Scripts", "python.exe"))

    # Avoid UnicodeEncodeError on Windows consoles (cp1252) by not printing emojis.
    print(f"Python:        {VENV_PYTHON}")
    print(f"Venv found:    {'Yes' if venv_ok else 'No'}")
    print(f"Settings:      {SETTINGS_PATH} ({'present' if os.path.isfile(SETTINGS_PATH) else 'new'})")
    print(f"Script folder: {SCRIPT_DIR}\n")

    app = build_ui()
    app.queue()

    # Port fallback: if 7860 is occupied, try a few next ports.
    base_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launched = False
    last_err: Optional[Exception] = None
    launch_signature = inspect.signature(app.launch)
    for port in range(base_port, base_port + 20):
        try:
            launch_kwargs = {
                "server_name": "127.0.0.1",
                "server_port": port,
                "inbrowser": True,
                "share": False,
            }

            if "theme" in launch_signature.parameters:
                launch_kwargs["theme"] = UI_THEME
            if "css" in launch_signature.parameters:
                launch_kwargs["css"] = UI_CSS

            app.launch(**launch_kwargs)
            launched = True
            break
        except OSError as e:
            last_err = e
            continue

    if not launched:
        raise last_err or OSError("Could not find a free port to launch Gradio UI")
