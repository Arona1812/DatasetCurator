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
import hashlib
import subprocess
import sys
import threading
import signal
import time
import uuid
import warnings
from glob import glob
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

from curator_core import (
    APP_VERSION,
    atomic_write_json as core_atomic_write_json,
    normalize_run_config_payload,
)

from preflight import (
    PHashSettings as WorkspacePHashSettings,
    dataset_fingerprint as preflight_dataset_fingerprint,
    load_workspace as load_project_workspace,
    output_root_for as preflight_output_root_for,
    run_preflight as run_project_preflight,
    scan_images as scan_preflight_images,
)

from frame_cleanup import (
    DetectorSettings as SmartFrameDetectorSettings,
    analyze_frame_cleanup,
    build_review_preview,
    decision_summary as frame_decision_summary,
    file_sha1 as frame_file_sha1,
    image_dimensions as frame_image_dimensions,
    load_user_decisions as load_frame_user_decisions,
    reset_detector_cache as reset_frame_detector_cache,
    reset_user_decisions as reset_frame_user_decisions,
    save_user_decision as save_frame_user_decision,
    scan_source_images as scan_frame_source_images,
)

# Pillow warns above ~89 MP by default. The curator works with trusted local
# source files and can legitimately encounter stitched panoramas or very
# high-resolution photographs. Keep a finite safety ceiling, but avoid noisy
# warnings for images that are still reasonable for local curation.
PIL_HARD_IMAGE_PIXEL_LIMIT = 350_000_000
Image.MAX_IMAGE_PIXELS = PIL_HARD_IMAGE_PIXEL_LIMIT // 2

# Make the UI process itself tolerant of characters that the active Windows
# console code page cannot represent. Child-process log streaming has its own
# explicit UTF-8 decoder below.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")
    except Exception:
        pass

# ============================================================
# UI LANGUAGE / I18N
# ============================================================

# Default UI language. Is overwritten during build_ui() from settings.
UI_LANG = "en"  # "en" | "de"

UI_THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
UI_CSS = """
.log-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 12px !important; }
.frame-comparison-gallery .grid-wrap {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-gutter: stable;
}
.frame-comparison-gallery .grid-container {
    display: flex !important;
    flex-wrap: nowrap !important;
    width: max-content !important;
    min-width: 100% !important;
    align-items: stretch !important;
}
.frame-comparison-gallery .gallery-item {
    flex: 0 0 clamp(260px, 31vw, 430px) !important;
    min-width: 260px !important;
    height: 510px !important;
    border-radius: 10px;
}
.frame-comparison-gallery .thumbnail-item {
    pointer-events: none !important;
    cursor: default !important;
}
.frame-comparison-gallery .thumbnail-item:hover {
    --ring-color: transparent !important;
    border-color: var(--border-color-primary) !important;
    filter: none !important;
}
.frame-comparison-gallery .caption-label {
    white-space: normal !important;
    max-width: 95% !important;
}
.frame-option-selector [role="radiogroup"],
.frame-option-selector .wrap {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 0.65rem !important;
}
.frame-option-selector label {
    flex: 1 1 220px !important;
    min-width: 180px !important;
}
.frame-viewer-position {
    text-align: center;
    align-self: center;
    min-width: 160px;
}
"""


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
_active_process_lock = threading.RLock()
_cancel_requested = threading.Event()

# Cross-thread / cross-worker run registry.  Gradio callbacks normally share the
# same process, but relying only on a Python global made cancellation fragile.
# The PID and cooperative cancel marker are therefore persisted in one tiny
# state file beside the application.
ACTIVE_RUN_STATE_PATH = os.path.join(SCRIPT_DIR, "_active_run_state.json")


# ============================================================
# PERSISTENT SETTINGS
# ============================================================

# Alle Standardwerte an einer Stelle – wird sowohl fuer Defaults als
# auch fuer Save/Load verwendet.
SHARED_COMPACT_CAPTION_FIELDS: List[str] = [
    "include_gender_class",
    "include_skin_tone",
    "include_body_build",
    "include_freckles",
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
    "include_eye_color_when_variable",
    "include_costume_accessories",
    "include_beard_always",
    "include_beard_when_variable",
    "include_mirror_selfie_marker",
    "include_eye_color",
    "include_visual_style",
]

# Z-Image_Base-Profil: bewusst reduziert. Stabile Identitaetsmerkmale
# (Hautfarbe, Augenfarbe, Koerperbau, konstante Frisur, Geschlechts-Klasse)
# werden NICHT captioniert, damit der LoRA-Trigger sie als Person-Identitaet
# absorbiert statt sie als austauschbare Caption-Tokens zu lernen.
# Variable Attribute (Hair-when-variable, abweichende Brillen, variable
# Piercings/Ohrschmuck, Make-up, Kleidung, Pose, Gaze, Ausdruck, Hintergrund,
# Beleuchtung, Bildstil) bleiben drin, weil sie zwischen Bildern wechseln
# und das LoRA sie als situative Marker lernen soll.
# Begruendung: Z-Image_Base ist ein 6B-Parameter Single-Stream DiT, dessen
# T5-basierter Text-Encoder Standard-Konzepte wie 'blonde hair', 'blue eyes',
# 'fair skin' bereits sehr gut kennt. Diese in jeder Caption zu wiederholen
# konkurriert mit der Trigger-Identitaet bei Inferenz und reduziert die
# Steuerbarkeit (z.B. 'Kathi mit roten Haaren' wird unsauber, weil der
# Trigger 'blonde' mitschleppt).
Z_IMAGE_BASE_CAPTION_FIELDS: List[str] = [
    "include_glasses_when_variable",
    "include_piercings",
    "include_makeup",
    "include_background",
    "include_lighting",
    "include_gaze",
    "include_expression",
    "include_hair_when_variable",
    "include_eye_color_when_variable",
    "include_costume_accessories",
    "include_beard_when_variable",
    "include_mirror_selfie_marker",
    "include_visual_style",
]

# Krea 2 character profile: stable physical identity is recorded in the
# subject profile for QC but deliberately omitted from captions. Captions focus
# on visible scene-specific attributes and are generated as natural language.
KREA2_CHARACTER_CAPTION_FIELDS: List[str] = [
    "include_glasses_when_variable",
    "include_piercings",
    "include_makeup",
    "include_background",
    "include_lighting",
    "include_gaze",
    "include_expression",
    "include_hair_when_variable",
    "include_eye_color_when_variable",
    "include_costume_accessories",
    "include_beard_when_variable",
    "include_mirror_selfie_marker",
    "include_visual_style",
]


DEFAULTS: Dict[str, Any] = {
    # UI
    "ui_language": "en",
    # Curator Basis
    "c_trigger": "",
    "c_input": r"",
    "c_target": 30,
    "c_api_key": "",
    "c_model": "gpt-5.6-luna",
    "c_audit_reasoning_effort": "none",
    "c_openai_token_limit": 0,
    "c_use_trigger_check": False,
    "c_trigger_model": "gpt-5.6-luna",
    "c_trigger_reasoning_effort": "none",
    "c_use_review_escalation": False,
    "c_review_escalation_model": "",
    "c_review_escalation_reasoning_effort": "low",
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
    "c_face_min_blur_headshot": 25,
    "c_face_min_blur_medium": 35,
    "c_face_min_blur_full_body": 45,
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
    "c_frame_cleanup_mode": "suggest_only",
    "c_frame_pause_on_medium": False,
    "c_frame_auto_accept_types": ["uniform_canvas", "story_bars"],
    "c_post_frame_phash_refresh": True,
    # Duplikate
    "c_use_clip": True,
    "c_use_phash": True,
    "c_phash_thresh": 8,
    "c_clip_thresh": 0.985,
    # Smart Crop
    "c_smart_crop": True,
    "c_crop_gain": 8,
    "c_crop_pad": 1.5,
    "c_medium_rescue_crop": True,
    "c_medium_rescue_gain": 4,
    # Clustering
    "c_use_cluster": True,
    "c_max_outfit": 4,
    "c_max_session": 5,
    "c_use_diversity": True,
    # Weiche Repräsentation der im Subject Profile bestätigten Canon-Haarfarbe
    "c_use_canon_representation": True,
    "c_canon_representation_target": 3,
    "c_canon_max_quality_gap": 5.0,
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
    "c_training_target": "ernie",
    "c_caption_profile": "ernie",  # legacy settings key
    "c_captions": list(SHARED_COMPACT_CAPTION_FIELDS),
    "c_variable_feature_mode": "canonical_deviations",
    "c_krea_caption_model": "gpt-5.6-luna",
    "c_krea_caption_reasoning_effort": "none",
    "c_use_krea_caption_repair": True,
    "c_krea_caption_repair_model": "gpt-5.6-terra",
    "c_krea_caption_repair_reasoning_effort": "low",
    # Subject Profile / Phase 2
    "c_pipeline_mode": "single_pass",
    "c_profile_normalizer_model": "gpt-5.6-terra",
    "c_profile_reasoning_effort": "low",
    "c_profile_sample_threshold": 100,
    "c_profile_sample_size": 80,
    # Export
    "c_exp_review": True,
    "c_exp_reject": True,
    "c_exp_compare": True,
    "c_controlled_buckets": False,
    # Video Processor
    "v_source": r".\00_videos",
    "v_target": r".\00_input",
    "v_ref": r".\referenz.jpg",
    "v_fpm": 5,
    "v_fps": 2,
    "v_sim": 0.45,
    "v_sharp": 50,
}

REASONING_EFFORT_CHOICES: List[str] = [
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

OPENAI_MODEL_PRESET_CHOICES: List[str] = [
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5.6",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4",
    "gpt-5.5",
]

CAPTION_FIELD_CHOICES: List[str] = [
    "include_gender_class",
    "include_skin_tone",
    "include_body_build",
    "include_freckles",
    "include_tattoos",
    "include_glasses",
    "include_glasses_when_variable",
    "include_piercings",
    "include_makeup",
    "include_background",
    "include_lighting",
    "include_gaze",
    "include_expression",
    "include_hair_always",
    "include_hair_when_variable",
    "include_eye_color_when_variable",
    "include_costume_accessories",
    "include_beard_always",
    "include_beard_when_variable",
    "include_mirror_selfie_marker",
    "include_eye_color",
    "include_visual_style",
]

CAPTION_PROFILE_PRESETS: Dict[str, List[str]] = {
    # ERNIE: volle Beschreibung inkl. stabiler Identitaetsmerkmale.
    # Begruendung: ERNIE-Image hat im Default-Output einen asiatischen Bias,
    # explizite Anker (blonde hair, blue eyes, fair skin) sind noetig damit
    # nicht-asiatische Personen sauber generiert werden.
    "ernie": list(SHARED_COMPACT_CAPTION_FIELDS),
    # Z-Image_Base: nur variable Attribute. Trigger-Wort uebernimmt die
    # stabile Person-Identitaet, Captions beschreiben nur was zwischen
    # Bildern wechselt.
    "z_image_base": list(Z_IMAGE_BASE_CAPTION_FIELDS),
    "krea2": list(KREA2_CHARACTER_CAPTION_FIELDS),
    # Legacy-Alias: bestehende Configs mit "shared_compact" werden
    # automatisch wie ERNIE behandelt (das war urspruenglich das einzige
    # Profil, faktisch ERNIE-Style).
    "shared_compact": list(SHARED_COMPACT_CAPTION_FIELDS),
}


def normalize_training_target(value: Optional[str]) -> str:
    v = (value or "").strip().lower()
    if v in {"ernie", "shared_compact"}:
        return "ernie"
    if v in {"z_image_base", "z-image_base", "zimage", "z_image"}:
        return "z_image_base"
    if v in {"krea2", "krea_2", "krea2_character", "krea_2_character"}:
        return "krea2"
    return "ernie"


def caption_profile_for_training_target(value: Optional[str]) -> str:
    target = normalize_training_target(value)
    return "krea2_character" if target == "krea2" else target


def training_target_choices() -> List[Tuple[str, str]]:
    return [
        (tr("ERNIE Image", "ERNIE Image"), "ernie"),
        (tr("Z-Image Base", "Z-Image Base"), "z_image_base"),
        (tr("Krea 2", "Krea 2"), "krea2"),
    ]


def get_caption_preset_values(target: Optional[str]) -> List[str]:
    normalized = normalize_training_target(target)
    if normalized in CAPTION_PROFILE_PRESETS:
        return list(CAPTION_PROFILE_PRESETS[normalized])
    return list(DEFAULTS["c_captions"])


def resolve_caption_fields_for_target(
    target: Optional[str],
    current_fields: Optional[List[str]] = None,
) -> List[str]:
    normalized = normalize_training_target(target)
    return get_caption_preset_values(normalized)


def apply_training_target_defaults(
    target: Optional[str],
    current_fields: Optional[List[str]],
    target_size: int,
    ratio_h: float,
    ratio_m: float,
    ratio_f: float,
    audit_model: str,
    audit_reasoning_effort: str,
    trigger_reasoning_effort: str,
    escalation_reasoning_effort: str,
    profile_model: str,
    profile_reasoning_effort: str,
    krea_caption_model: str,
    krea_caption_reasoning_effort: str,
    pipeline_mode: str,
):
    """Load target defaults. Later checkbox edits never change the target."""
    normalized = normalize_training_target(target)
    fields = resolve_caption_fields_for_target(normalized, current_fields)
    if normalized == "krea2":
        return (
            fields, 20, 0.40, 0.35, 0.25,
            "gpt-5.6-luna", "none", "none", "low",
            "gpt-5.6-terra", "low", "gpt-5.6-luna", "none",
            "profile_then_caption",
        )
    return (
        fields, target_size, ratio_h, ratio_m, ratio_f,
        audit_model, audit_reasoning_effort, trigger_reasoning_effort,
        escalation_reasoning_effort, profile_model, profile_reasoning_effort,
        krea_caption_model, krea_caption_reasoning_effort,
        pipeline_mode,
    )


def caption_policy_adjustment_note(target: Optional[str], selected_fields: Optional[List[str]]) -> str:
    normalized = normalize_training_target(target)
    preset = set(CAPTION_PROFILE_PRESETS.get(normalized, []))
    selected = set(selected_fields or [])
    if selected == preset:
        return tr("✅ Standardregeln des Trainingsziels aktiv.", "✅ Training-target default rules active.")
    return tr(
        "🛠️ Caption-Regeln individuell angepasst. Trainingsziel und Caption-Engine bleiben unverändert.",
        "🛠️ Caption rules customized. Training target and caption engine remain unchanged.",
    )


def load_settings() -> Dict[str, Any]:
    """Laedt gespeicherte Einstellungen, ergaenzt fehlende mit Defaults."""
    settings = dict(DEFAULTS)
    saved: Dict[str, Any] = {}
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                saved = loaded
                settings.update(saved)
        except Exception:
            saved = {}

    # If settings file exists but is missing keys (e.g. only language was saved),
    # ensure missing keys are filled from DEFAULTS.
    for k, v in DEFAULTS.items():
        settings.setdefault(k, v)

    # Language is persistent. A temporary override file is still understood
    # for compatibility with older releases, but the saved language remains
    # authoritative on all later starts.
    base_lang = _normalize_lang(saved.get("ui_language") or settings.get("ui_language") or "en")
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
    legacy_target = saved.get("c_training_target") or saved.get("c_caption_profile") or settings.get("c_training_target")
    settings["c_training_target"] = normalize_training_target(legacy_target)
    settings["c_caption_profile"] = settings["c_training_target"]
    return settings


def save_ui_language(lang_code: str) -> str:
    """Persist only the UI language selection in _ui_settings.json."""
    global UI_LANG
    lang_code = _normalize_lang(lang_code)
    UI_LANG = lang_code
    try:
        settings = load_settings()
        settings["ui_language"] = lang_code
        core_atomic_write_json(SETTINGS_PATH, settings)
        # Keep the compatibility override so an immediate restart uses the new
        # language even when an old launcher still expects this marker.
        core_atomic_write_json(LANG_OVERRIDE_PATH, {"ui_language": lang_code})
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
    c_trigger, c_input, c_target, c_api_key, c_model, c_audit_reasoning_effort, c_openai_token_limit, c_use_trigger_check, c_trigger_model, c_trigger_reasoning_effort,
    c_use_review_escalation, c_review_escalation_model, c_review_escalation_reasoning_effort,
    c_review_escalation_score_min, c_review_escalation_score_max,
    c_escalate_on_review, c_escalate_on_conflict, c_escalate_smart_crop, c_smart_crop_escalation_delta,
    c_ratio_h, c_ratio_m, c_ratio_f,
    c_keep_min, c_reject, c_min_side,
    c_use_filesize, c_min_filesize,
    c_use_blur, c_min_blur, c_face_min_blur, c_blur_norm_edge,
    c_face_min_blur_headshot, c_face_min_blur_medium, c_face_min_blur_full_body,
    c_use_early_phash,
    c_use_early_phash_loop1, c_early_phash_thresh_1, c_early_phash_keep_1,
    c_use_early_phash_loop2, c_early_phash_thresh_2, c_early_phash_keep_2,
    c_subject_sanity, c_subject_min_torso,
    c_ig_frame_crop, c_ig_two_stage_bar, c_frame_cleanup_mode, c_frame_pause_on_medium,
    c_use_clip, c_use_phash, c_phash_thresh, c_clip_thresh,
    c_smart_crop, c_crop_gain, c_crop_pad,
    c_medium_rescue_crop, c_medium_rescue_gain,
    c_use_cluster, c_max_outfit, c_max_session, c_use_diversity,
    c_use_canon_representation, c_canon_representation_target, c_canon_max_quality_gap,
    c_use_pose_diversity, c_pose_soft_limit, c_pose_penalty_weight,
    c_use_arcface, c_arcface_hard, c_arcface_soft, c_arcface_trim,
    c_arcface_min_faces, c_arcface_model, c_arcface_det_size,
    c_training_target,
    c_captions, c_variable_feature_mode, c_krea_caption_model, c_krea_caption_reasoning_effort,
    c_use_krea_caption_repair, c_krea_caption_repair_model, c_krea_caption_repair_reasoning_effort,
    c_pipeline_mode, c_profile_normalizer_model, c_profile_reasoning_effort,
    c_profile_sample_threshold, c_profile_sample_size,
    c_exp_review, c_exp_reject, c_exp_compare, c_controlled_buckets,
    # Video
    v_source, v_target, v_ref, v_fpm, v_fps, v_sim, v_sharp,
):
    """Speichert alle aktuellen UI-Werte in _ui_settings.json."""
    data = {
        "ui_language": _normalize_lang(ui_language),
        "c_trigger": c_trigger, "c_input": c_input, "c_target": c_target,
        "c_api_key": c_api_key, "c_model": c_model,
        "c_audit_reasoning_effort": c_audit_reasoning_effort,
        "c_openai_token_limit": int(c_openai_token_limit or 0),
        "c_use_trigger_check": c_use_trigger_check,
        "c_trigger_model": c_trigger_model,
        "c_trigger_reasoning_effort": c_trigger_reasoning_effort,
        "c_use_review_escalation": c_use_review_escalation,
        "c_review_escalation_model": c_review_escalation_model,
        "c_review_escalation_reasoning_effort": c_review_escalation_reasoning_effort,
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
        "c_face_min_blur_headshot": c_face_min_blur_headshot,
        "c_face_min_blur_medium": c_face_min_blur_medium,
        "c_face_min_blur_full_body": c_face_min_blur_full_body,
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
        "c_frame_cleanup_mode": str(c_frame_cleanup_mode or "suggest_only"),
        "c_frame_pause_on_medium": bool(c_frame_pause_on_medium),
        "c_use_clip": c_use_clip, "c_use_phash": c_use_phash,
        "c_phash_thresh": c_phash_thresh, "c_clip_thresh": c_clip_thresh,
        "c_smart_crop": c_smart_crop, "c_crop_gain": c_crop_gain, "c_crop_pad": c_crop_pad,
        "c_medium_rescue_crop": c_medium_rescue_crop,
        "c_medium_rescue_gain": c_medium_rescue_gain,
        "c_use_cluster": c_use_cluster, "c_max_outfit": c_max_outfit,
        "c_max_session": c_max_session, "c_use_diversity": c_use_diversity,
        "c_use_canon_representation": bool(c_use_canon_representation),
        "c_canon_representation_target": int(c_canon_representation_target),
        "c_canon_max_quality_gap": float(c_canon_max_quality_gap),
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
        "c_training_target": normalize_training_target(c_training_target),
        "c_caption_profile": normalize_training_target(c_training_target),
        "c_captions": c_captions,
        "c_variable_feature_mode": c_variable_feature_mode,
        "c_krea_caption_model": c_krea_caption_model,
        "c_krea_caption_reasoning_effort": c_krea_caption_reasoning_effort,
        "c_use_krea_caption_repair": bool(c_use_krea_caption_repair),
        "c_krea_caption_repair_model": c_krea_caption_repair_model,
        "c_krea_caption_repair_reasoning_effort": c_krea_caption_repair_reasoning_effort,
        "c_pipeline_mode": c_pipeline_mode,
        "c_profile_normalizer_model": c_profile_normalizer_model,
        "c_profile_reasoning_effort": c_profile_reasoning_effort,
        "c_profile_sample_threshold": int(c_profile_sample_threshold),
        "c_profile_sample_size": int(c_profile_sample_size),
        "c_exp_review": c_exp_review, "c_exp_reject": c_exp_reject,
        "c_exp_compare": c_exp_compare,
        "c_controlled_buckets": c_controlled_buckets,
        "v_source": v_source, "v_target": v_target, "v_ref": v_ref,
        "v_fpm": v_fpm, "v_fps": v_fps, "v_sim": v_sim, "v_sharp": v_sharp,
    }
    try:
        core_atomic_write_json(SETTINGS_PATH, data)
        return tr("💾 Einstellungen gespeichert.", "💾 Settings saved.")
    except Exception as e:
        return tr(f"⚠️ Speichern fehlgeschlagen: {e}", f"⚠️ Save failed: {e}")


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    core_atomic_write_json(path, payload)


def _read_active_run_state() -> Dict[str, Any]:
    try:
        with open(ACTIVE_RUN_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pid_is_running(pid: int) -> bool:
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            # Keep tasklist output as bytes. Windows command-line tools may
            # emit text in an OEM or legacy code page that differs from
            # Python's current ANSI codec. Decoding it via text=True can make
            # subprocess._readerthread crash with UnicodeDecodeError.  We only
            # need the ASCII PID, so no text decoding is necessary here.
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            return str(pid).encode("ascii") in (result.stdout or b"")
        except Exception:
            return True  # do not discard a potentially valid registry entry
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _write_active_run_state(proc: subprocess.Popen, run_id: str, cancel_file: str, config_path: str) -> None:
    _atomic_write_json(ACTIVE_RUN_STATE_PATH, {
        "schema_version": "v2",
        "pid": int(proc.pid),
        "run_id": run_id,
        "cancel_file": cancel_file,
        "config_path": config_path,
        "started_at": time.time(),
    })


def _clear_active_run_state(expected_pid: Optional[int] = None) -> None:
    state = _read_active_run_state()
    if expected_pid is not None and state:
        try:
            if int(state.get("pid", -1)) != int(expected_pid):
                return
        except Exception:
            return
    try:
        if os.path.exists(ACTIVE_RUN_STATE_PATH):
            os.remove(ACTIVE_RUN_STATE_PATH)
    except Exception:
        pass


def _touch_cancel_marker(path: str, reason: str = "ui_cancel") -> bool:
    if not path:
        return False
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"cancelled_at": time.time(), "reason": reason}, f)
        return True
    except Exception:
        return False


def _cleanup_stale_configs():
    for cfg in (CURATOR_CONFIG, VIDEO_CONFIG):
        try:
            if os.path.exists(cfg):
                os.remove(cfg)
        except Exception:
            pass

    # A state file from a previous crashed UI must not survive a new UI start.
    state = _read_active_run_state()
    pid = state.get("pid")
    if not pid or not _pid_is_running(pid):
        cancel_file = str(state.get("cancel_file") or "")
        try:
            if cancel_file and os.path.exists(cancel_file):
                os.remove(cancel_file)
        except Exception:
            pass
        _clear_active_run_state()


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
    """Load a bounded preview while preserving the original source file.

    ``Image.draft`` lets Pillow decode large JPEGs close to preview resolution
    instead of expanding (for example) a 108 MP source completely in memory.
    PNG and other formats still use Pillow's normal decoder, but remain guarded
    by a finite manual pixel ceiling.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                width, height = img.size
                if int(width) * int(height) > PIL_HARD_IMAGE_PIXEL_LIMIT:
                    return None

                # JPEG draft decoding is substantially cheaper for large local
                # originals. A 2x margin keeps enough detail for LANCZOS downscale.
                draft_size = (max(64, int(max_size[0]) * 2), max(64, int(max_size[1]) * 2))
                try:
                    img.draft("RGB", draft_size)
                except Exception:
                    pass

                preview = ImageOps.exif_transpose(img)
                resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                preview.thumbnail(max_size, resampling)
                return preview.convert("RGB").copy()
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




def caption_stage_path_for(input_folder: str, trigger_word: str) -> str:
    return os.path.join(output_root_for(input_folder, trigger_word), "_caption_stage.json")


def subject_profile_path_for(input_folder: str, trigger_word: str) -> str:
    return os.path.join(output_root_for(input_folder, trigger_word), "_subject_profile.json")


def _load_caption_stage_for_profile_ui(trigger_word: str, input_folder: str) -> Dict[str, Any]:
    path = caption_stage_path_for(input_folder, trigger_word)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _profile_summary_markdown(profile: Dict[str, Any]) -> str:
    if not profile:
        return tr("Kein Profil geladen.", "No profile loaded.")
    stable = profile.get("stable_identity", {}) or {}
    canonical = profile.get("canonical_features", {}) or {}
    markers = profile.get("identity_markers", {}) or {}
    glasses = markers.get("glasses", {}) or {}
    freckles = markers.get("freckles", {}) or {}
    tattoos = markers.get("tattoo_inventory", []) or []
    piercings = markers.get("piercing_inventory", []) or markers.get("piercing_baseline", []) or []
    per_image = profile.get("per_image_traits", {}) or {}
    lines = [
        f"### {profile.get('subject_id', '') or 'Subject'}",
        "",
        f"**Gender:** {stable.get('gender', '-')}",
        f"**Skin tone:** {stable.get('skin_tone', '-')}",
        f"**Eye color:** {stable.get('eye_color', '-')}",
        f"**Hair texture:** {stable.get('hair_texture', '-')}",
        f"**Body build:** {stable.get('body_build', '-')}",
        f"**Body height impression:** {stable.get('body_height_impression', '-')}",
        f"**Appearance mode:** {profile.get('profile_appearance_mode', '-')}",
        f"**Canonical hair color:** {canonical.get('hair_color', '-')}",
        f"**Canonical beard:** {canonical.get('beard_pattern', '-') or '-'} / {canonical.get('beard_color', '-') or '-'}",
        "",
        f"**Glasses:** {glasses.get('canonical_description', '-') if glasses.get('wears_regularly') else 'not regular'}",
        f"**Freckles:** {freckles.get('canonical_description', '-') if freckles.get('has_freckles') else 'not regular'}",
        f"**Tattoos in inventory:** {len(tattoos)}",
        f"**Piercings/accessories in inventory:** {len(piercings)}",
        f"**Per-image trait rows:** {len(per_image)}",
        f"**Force only when visible:** {profile.get('force_only_when_visible', True)}",
    ]
    notes = profile.get("normalizer_notes", []) or []
    if notes:
        lines.append("")
        lines.append("**Notes:**")
        for note in notes[:8]:
            lines.append(f"- {note}")
    return "\n".join(lines)


def load_subject_profile_ui(trigger_word: str, input_folder: str) -> Tuple[str, str, str]:
    path = subject_profile_path_for(input_folder, trigger_word)
    stage_path = caption_stage_path_for(input_folder, trigger_word)
    if not os.path.exists(path):
        return "", tr(
            f"Kein _subject_profile.json gefunden. Erwarteter Pfad: `{path}`",
            f"No _subject_profile.json found. Expected path: `{path}`",
        ), tr("⚠️ Kein Profil geladen.", "⚠️ No profile loaded.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        text = json.dumps(profile, ensure_ascii=False, indent=2)
        summary = _profile_summary_markdown(profile)
        if os.path.exists(stage_path):
            status = tr("✅ Profil und Caption-Stage geladen.", "✅ Profile and caption stage loaded.")
        else:
            status = tr("⚠️ Profil geladen, aber _caption_stage.json fehlt.", "⚠️ Profile loaded, but _caption_stage.json is missing.")
        return text, summary, status
    except Exception as e:
        return "", tr(f"Profil konnte nicht gelesen werden: {e}", f"Could not read profile: {e}"), tr("❌ Fehler", "❌ Error")


def save_subject_profile_ui(trigger_word: str, input_folder: str, profile_json_text: str) -> str:
    path = subject_profile_path_for(input_folder, trigger_word)
    if not profile_json_text.strip():
        return tr("⚠️ Kein JSON zum Speichern vorhanden.", "⚠️ No JSON to save.")
    try:
        profile = json.loads(profile_json_text)
        if not isinstance(profile, dict):
            return tr("❌ Profil muss ein JSON-Objekt sein.", "❌ Profile must be a JSON object.")
        profile["force_only_when_visible"] = True
        os.makedirs(os.path.dirname(path), exist_ok=True)
        core_atomic_write_json(path, profile)
        return tr(f"✅ Profil gespeichert: {path}", f"✅ Profile saved: {path}")
    except Exception as e:
        return tr(f"❌ Speichern fehlgeschlagen: {e}", f"❌ Save failed: {e}")


# ============================================================
# SUBJECT PROFILE TAB - VOCAB & HELPERS
# ============================================================
# Diese Listen spiegeln die kanonischen Vocabs aus dataset_curator_v2.py.
# Bei Aenderungen dort auch hier anpassen.

PROFILE_VOCAB_GENDER: List[str] = ["woman", "man", "girl", "boy", "person"]
PROFILE_VOCAB_SKIN: List[str] = ["very_fair", "fair", "light", "medium", "tan", "olive", "brown", "dark", "deep", "unclear"]
PROFILE_VOCAB_EYES: List[str] = ["blue", "blue_green", "green", "hazel", "brown", "dark_brown", "gray", "gray_blue", "amber", "not_visible", "unclear"]
PROFILE_VOCAB_HAIR_TEXTURE: List[str] = ["straight", "wavy", "curly", "coily", "afro_textured"]
PROFILE_VOCAB_BODY: List[str] = ["petite", "slim", "average", "athletic", "curvy", "plus_size", "muscular", "broad_build", "unclear"]
PROFILE_VOCAB_BODY_HEIGHT: List[str] = ["short", "average_height", "tall", "unclear"]
PROFILE_VOCAB_HAIR_COLOR: List[str] = [
    "black", "dark_brown", "brown", "light_brown", "dark_blonde", "blonde",
    "platinum", "strawberry_blonde", "red", "copper", "auburn", "burgundy",
    "gray", "silver", "white", "blue", "pink", "purple", "green",
    "dyed_other", "multicolor", "ombre", "highlights", "not_visible", "unclear",
]
PROFILE_VOCAB_BEARD_PATTERN: List[str] = [
    "clean_shaven", "stubble", "designer_stubble", "short_beard", "full_beard",
    "long_beard", "goatee", "mustache_only", "mustache_goatee", "chin_strap",
    "mutton_chops", "soul_patch", "circle_beard", "handlebar_mustache", "neckbeard", "other",
]
PROFILE_VOCAB_BEARD_COLOR: List[str] = ["", "dark", "brown", "blonde", "red", "gray", "white", "salt_pepper", "other"]

PROFILE_VOCAB_HAIR_FORM: List[str] = [
    "loose_straight", "loose_wavy", "loose_curly", "loose_coily",
    "afro_natural", "ponytail", "low_ponytail", "high_ponytail",
    "pigtails", "two_braids", "single_braid", "box_braids",
    "knotless_braids", "cornrows", "dreadlocks", "bun", "low_bun",
    "high_bun", "messy_bun", "updo", "half_up", "pulled_back",
    "pixie_cut", "bob_cut", "lob_cut", "short_cut", "buzz_cut",
    "shaved_head", "undercut", "side_shaved", "bangs", "curtain_bangs",
    "covered_hair", "other",
]
PROFILE_VOCAB_MAKEUP: List[str] = ["none", "minimal", "natural", "defined", "full", "dramatic", "stage_makeup", "costume_makeup", "face_paint", "unclear"]
PROFILE_VOCAB_MAKEUP_STYLE: List[str] = [
    "natural_makeup", "gyaru_makeup", "cosplay_makeup",
    "anime_inspired_makeup", "dramatic_eyeliner", "smoky_eye_makeup",
    "false_eyelashes", "glossy_lips", "face_paint", "fantasy_makeup", "unclear",
]
PROFILE_VOCAB_LOOK_CONTEXT: List[str] = [
    "regular_photo", "fashion", "glamour", "gyaru_style", "cosplay",
    "character_costume", "fantasy_costume", "stage_costume",
    "swimwear_costume", "lingerie_costume", "unclear",
]
PROFILE_VOCAB_APPEARANCE_MODE: List[str] = [
    "natural_identity", "fashion_identity", "cosplay_identity", "high_variation_model_identity",
]


def _conf_level(profile: Dict[str, Any], field: str) -> str:
    """Liest den Confidence-Level robust aus.
    Akzeptiert sowohl das neue Object-Format ({level, reasoning, outliers})
    als auch das alte String-only-Format.
    """
    conf = (profile or {}).get("confidence", {})
    val = conf.get(field, "")
    if isinstance(val, dict):
        return str(val.get("level", "") or "")
    return str(val or "")


def _conf_reasoning(profile: Dict[str, Any], field: str) -> str:
    conf = (profile or {}).get("confidence", {})
    val = conf.get(field, "")
    if isinstance(val, dict):
        return str(val.get("reasoning", "") or "")
    return ""


def _conf_emoji(level: str) -> str:
    l = (level or "").strip().lower()
    if l == "high":
        return "✅"
    if l == "medium":
        return "🟡"
    if l == "low":
        return "⚠️"
    if l == "fallback":
        return "🔁"
    return "⚪"


def _normalize_dropdown_choices(vocab: List[str], current: str) -> List[str]:
    """Sortiert das Vocab so, dass der aktuelle Wert ganz oben steht,
    gefolgt von den uebrigen Tokens in stabiler Reihenfolge.
    """
    cur = (current or "").strip()
    if cur and cur in vocab:
        return [cur] + [v for v in vocab if v != cur]
    if cur and cur not in vocab:
        return [cur] + list(vocab)
    return list(vocab)


def aggregate_per_image_traits(profile: Dict[str, Any]) -> Dict[str, List[Tuple[str, int]]]:
    """Bucket-Counts ueber per_image_traits, sortiert nach Haeufigkeit absteigend."""
    per_image = (profile or {}).get("per_image_traits", {}) or {}
    fields = ["hair_color_base", "hair_color_modifier", "hair_form", "eye_color_base", "eye_appearance", "makeup_intensity", "makeup_style", "look_context", "glasses_position"]
    result: Dict[str, List[Tuple[str, int]]] = {}
    for field in fields:
        counts: Dict[str, int] = {}
        for traits in per_image.values():
            v = (traits or {}).get(field, "") or ""
            v = str(v).strip()
            if not v:
                continue
            counts[v] = counts.get(v, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        result[field] = sorted_counts
    return result


def _bucket_summary_markdown(label: str, counts: List[Tuple[str, int]], vocab: List[str]) -> str:
    if not counts:
        return f"**{label}:** _keine Daten_"
    total = sum(c for _, c in counts)
    lines = [f"**{label}** (N={total})"]
    for value, n in counts:
        marker = "" if value in vocab else " 🟠"
        lines.append(f"  • `{value}`{marker} — **{n}**")
    return "\n".join(lines)


def _empty_editor_payload(status_msg: str) -> Tuple:
    """Liefert eine leere Editor-Payload mit dem gegebenen Status."""
    empty_dropdown = gr.update(choices=[], value="")
    empty_info = "—"
    return (
        {}, "",  # state, raw_json
        empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown,
        empty_info, empty_info, empty_info, empty_info, empty_info, empty_info,
        empty_dropdown, empty_dropdown, empty_dropdown,  # canonical hair / beard
        False, "",  # glasses
        False, "",  # freckles
        "_kein Profil_", "_kein Profil_", "_kein Profil_",
        "_kein Profil_", [], "_kein Profil_",
        # Bucket-edit dropdowns (3x from + 3x to)
        empty_dropdown, empty_dropdown, empty_dropdown,
        empty_dropdown, empty_dropdown, empty_dropdown,
        tr("_Keine Identity-Cluster geladen._", "_No identity clusters loaded._"), [],
        tr("_Kein Cluster ausgewählt._", "_No cluster selected._"), [],
        "", gr.update(choices=IDENTITY_CLUSTER_ROLE_CHOICES, value="variation"),
        status_msg,
    )


def _remove_early_duplicates_from_profile_reject_bucket_ui(
    profile: Dict[str, Any], trigger_word: str, input_folder: str
) -> Dict[str, Any]:
    """Filter stale early-pHash duplicates from legacy saved reject buckets."""
    if not isinstance(profile, dict):
        return profile
    stage_path = caption_stage_path_for(input_folder, trigger_word)
    try:
        with open(stage_path, "r", encoding="utf-8") as handle:
            stage = json.load(handle)
    except Exception:
        return profile
    excluded_filenames = set()
    excluded_paths = set()
    for row in (stage.get("all_rows", []) or []) if isinstance(stage, dict) else []:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("short_reason", "") or "").strip().lower()
        notes = row.get("status_notes") or []
        if isinstance(notes, str):
            notes = [notes]
        notes_lower = {str(v or "").strip().lower() for v in notes}
        duplicate_method = str(row.get("duplicate_method", "") or "").strip().lower()
        if not (
            reason == "early_phash_duplicate"
            or "early_phash_dedup" in notes_lower
            or duplicate_method.startswith("early_phash")
        ):
            continue
        filename = str(row.get("original_filename", "") or "")
        path = str(row.get("original_path", "") or row.get("source_original_path", "") or "")
        if filename:
            excluded_filenames.add(filename)
        if path:
            excluded_paths.add(os.path.normcase(os.path.abspath(path)))
    if not excluded_filenames and not excluded_paths:
        return profile

    new_clusters = []
    removed_member_ids = set()
    for cluster in profile.get("identity_clusters", []) or []:
        if not isinstance(cluster, dict):
            continue
        if str(cluster.get("cluster_id", "") or "") != AUDITED_REJECT_CLUSTER_ID and str(cluster.get("cluster_kind", "") or "") != "audited_rejects":
            new_clusters.append(cluster)
            continue
        records = _cluster_member_records(cluster)
        kept = []
        for record in records:
            filename = str(record.get("filename", "") or "")
            path = str(record.get("image_path", "") or "")
            normalized_path = os.path.normcase(os.path.abspath(path)) if path else ""
            if filename in excluded_filenames or (normalized_path and normalized_path in excluded_paths):
                removed_member_ids.add(str(record.get("image_id", "") or ""))
                continue
            kept.append(record)
        if kept:
            updated = dict(cluster)
            updated["members"] = [r.get("image_id", "") for r in kept]
            updated["filenames"] = [r.get("filename", "") for r in kept]
            updated["image_paths"] = [r.get("image_path", "") for r in kept]
            updated["reject_reasons"] = [r.get("reject_reason", "") for r in kept]
            updated["n"] = len(kept)
            new_clusters.append(updated)
    profile["identity_clusters"] = new_clusters
    for field in ("identity_cluster_member_roles", "identity_cluster_member_clusters"):
        mapping = profile.get(field)
        if isinstance(mapping, dict):
            for member_id in removed_member_ids:
                mapping.pop(member_id, None)
    return profile


def load_profile_for_editor(trigger_word: str, input_folder: str):
    """Laedt ein Profil und fuellt alle UI-Komponenten."""
    path = subject_profile_path_for(input_folder, trigger_word)
    if not os.path.exists(path):
        return _empty_editor_payload(
            tr(f"⚠️ Kein _subject_profile.json gefunden: {path}",
               f"⚠️ No _subject_profile.json found: {path}"),
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if not isinstance(profile, dict):
            raise ValueError("profile is not a JSON object")
        profile = _remove_early_duplicates_from_profile_reject_bucket_ui(profile, trigger_word, input_folder)
        profile = _normalize_audited_reject_role_ui(profile)
    except Exception as e:
        return _empty_editor_payload(
            tr(f"❌ Profil-Lesefehler: {e}", f"❌ Profile read error: {e}"),
        )

    stable = profile.get("stable_identity", {}) or {}
    canonical = profile.get("canonical_features", {}) or {}
    markers = profile.get("identity_markers", {}) or {}
    glasses = markers.get("glasses", {}) or {}
    freckles = markers.get("freckles", {}) or {}

    def info(field: str) -> str:
        level = _conf_level(profile, field)
        reasoning = _conf_reasoning(profile, field)
        emoji = _conf_emoji(level)
        if reasoning:
            return f"{emoji} {level} — {reasoning}"
        return f"{emoji} {level}" if level else "—"

    gender_val = (stable.get("gender") or "").strip()
    skin_val = (stable.get("skin_tone") or "").strip()
    eyes_val = (stable.get("eye_color") or "").strip()
    hair_tex_val = (stable.get("hair_texture") or "").strip()
    body_val = (stable.get("body_build") or "").strip()
    body_height_val = (stable.get("body_height_impression") or "").strip()
    variability = profile.get("profile_variability_stats", {}) or {}
    hair_baseline_val = (canonical.get("hair_color") or (variability.get("hair_color", {}) or {}).get("mode") or "").strip()
    beard_pattern_val = (canonical.get("beard_pattern") or (variability.get("beard_pattern", {}) or {}).get("mode") or "").strip()
    beard_color_val = (canonical.get("beard_color") or (variability.get("beard_color", {}) or {}).get("mode") or "").strip()

    gender_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_GENDER, gender_val), value=gender_val)
    skin_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_SKIN, skin_val), value=skin_val)
    eyes_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_EYES, eyes_val), value=eyes_val)
    hair_tex_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_HAIR_TEXTURE, hair_tex_val), value=hair_tex_val)
    body_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_BODY, body_val), value=body_val)
    body_height_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_BODY_HEIGHT, body_height_val), value=body_height_val)
    hair_baseline_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_HAIR_COLOR, hair_baseline_val), value=hair_baseline_val)
    beard_pattern_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_BEARD_PATTERN, beard_pattern_val), value=beard_pattern_val)
    beard_color_dd = gr.update(choices=_normalize_dropdown_choices(PROFILE_VOCAB_BEARD_COLOR, beard_color_val), value=beard_color_val)

    counts = aggregate_per_image_traits(profile)
    hair_color_md = _bucket_summary_markdown(
        tr("Haarfarbe (per Bild)", "Hair color (per image)"),
        counts.get("hair_color_base", []),
        PROFILE_VOCAB_HAIR_COLOR,
    )
    hair_modifier_counts = counts.get("hair_color_modifier", [])
    if hair_modifier_counts:
        hair_color_md += "\n\n" + _bucket_summary_markdown(
            tr("Haarfarb-Modifier", "Hair-color modifiers"),
            hair_modifier_counts,
            ["highlights", "blonde_highlights", "red_highlights", "ombre", "balayage"],
        )
    hair_form_md = _bucket_summary_markdown(
        tr("Frisur / Form (per Bild)", "Hair form (per image)"),
        counts.get("hair_form", []),
        PROFILE_VOCAB_HAIR_FORM,
    )
    makeup_md = _bucket_summary_markdown(
        tr("Makeup-Intensität (per Bild)", "Makeup intensity (per image)"),
        counts.get("makeup_intensity", []),
        PROFILE_VOCAB_MAKEUP,
    )
    eye_color_md = _bucket_summary_markdown(
        tr("Augenfarbe (per Bild)", "Eye color (per image)"),
        counts.get("eye_color_base", []),
        PROFILE_VOCAB_EYES,
    )
    eye_appearance_md = _bucket_summary_markdown(
        tr("Augen-/Linsenlook (per Bild)", "Eye/lens appearance (per image)"),
        counts.get("eye_appearance", []),
        ["natural_eyes", "colored_contact_lenses", "circle_lenses", "cosmetic_lenses", "unnatural_eye_color", "unclear"],
    )
    makeup_style_md = _bucket_summary_markdown(
        tr("Makeup-Stil (per Bild)", "Makeup style (per image)"),
        counts.get("makeup_style", []),
        PROFILE_VOCAB_MAKEUP_STYLE,
    )
    look_context_md = _bucket_summary_markdown(
        tr("Look-Kontext (per Bild)", "Look context (per image)"),
        counts.get("look_context", []),
        PROFILE_VOCAB_LOOK_CONTEXT,
    )

    # Re-bucket dropdowns: from-Werte sind die im Profil vorkommenden Tokens
    color_present = [v for v, _ in counts.get("hair_color_base", [])]
    form_present = [v for v, _ in counts.get("hair_form", [])]
    makeup_present = [v for v, _ in counts.get("makeup_intensity", [])]

    color_from_dd = gr.update(choices=color_present, value=(color_present[-1] if color_present else ""))
    form_from_dd = gr.update(choices=form_present, value=(form_present[-1] if form_present else ""))
    makeup_from_dd = gr.update(choices=makeup_present, value=(makeup_present[-1] if makeup_present else ""))

    color_to_dd = gr.update(choices=PROFILE_VOCAB_HAIR_COLOR, value=(color_present[0] if color_present else PROFILE_VOCAB_HAIR_COLOR[0]))
    form_to_dd = gr.update(choices=PROFILE_VOCAB_HAIR_FORM, value=(form_present[0] if form_present else PROFILE_VOCAB_HAIR_FORM[0]))
    makeup_to_dd = gr.update(choices=PROFILE_VOCAB_MAKEUP, value=(makeup_present[0] if makeup_present else PROFILE_VOCAB_MAKEUP[0]))

    tattoo_inv = markers.get("tattoo_inventory", []) or []
    piercing_inv = markers.get("piercing_inventory", []) or markers.get("piercing_baseline", []) or []
    if tattoo_inv:
        tattoo_md = "**" + tr("Tattoo-Inventar", "Tattoo inventory") + f"** ({len(tattoo_inv)})\n\n"
        for t in tattoo_inv:
            tattoo_md += f"- `{t.get('location','')}` — {t.get('canonical_description','')} _({t.get('frequency','')})_\n"
    else:
        tattoo_md = tr("**Tattoo-Inventar:** _keine erfasst_", "**Tattoo inventory:** _none recorded_")

    piercing_rows = []
    for p_item in piercing_inv:
        if not isinstance(p_item, dict):
            continue
        piercing_rows.append([
            str(p_item.get("location", "") or ""),
            str(p_item.get("canonical_description", "") or ""),
            str(p_item.get("frequency", "") or ""),
            str(p_item.get("category", "body_piercing") or "body_piercing"),
            str(p_item.get("role", "variable") or "variable"),
        ])

    notes = profile.get("normalizer_notes", []) or []
    if notes:
        notes_md = "**" + tr("Normalizer-Notizen", "Normalizer notes") + ":**\n\n" + "\n".join(f"- {n}" for n in notes[:12])
    else:
        notes_md = tr("_Keine Normalizer-Notizen._", "_No normalizer notes._")

    raw_json = json.dumps(profile, ensure_ascii=False, indent=2)

    sample = profile.get("sample_size", "?")
    total = profile.get("total_usable_images", "?")
    model = profile.get("normalizer_model", "?")
    schema = profile.get("profile_schema_version", "?")
    retry_count = int(profile.get("normalizer_retry_count", 0) or 0)
    primary_error = str(profile.get("normalizer_primary_error", "") or "").strip()
    if not primary_error:
        for note in profile.get("normalizer_notes", []) or []:
            note_text = str(note or "").strip()
            if note_text.lower().startswith("fallback profile used"):
                primary_error = note_text
                break
    if str(model).strip().lower() == "fallback_local":
        short_error = primary_error[:240] + ("…" if len(primary_error) > 240 else "")
        status = tr(
            f"⚠️ Lokales Fallback-Profil geladen — Terra-Normalisierung fehlgeschlagen. "
            f"Sample {sample}/{total} | schema {schema}"
            + (f" | Ursache: {short_error}" if short_error else ""),
            f"⚠️ Local fallback profile loaded — Terra normalization failed. "
            f"Sample {sample}/{total} | schema {schema}"
            + (f" | Cause: {short_error}" if short_error else ""),
        )
    elif retry_count:
        status = tr(
            f"✅ Profil geladen — automatische Normalizer-Reparatur erfolgreich | "
            f"Sample {sample}/{total} | {model} | schema {schema}",
            f"✅ Profile loaded — automatic normalizer repair succeeded | "
            f"sample {sample}/{total} | {model} | schema {schema}",
        )
    else:
        status = tr(
            f"✅ Profil geladen — Sample {sample}/{total} | {model} | schema {schema}",
            f"✅ Profile loaded — sample {sample}/{total} | {model} | schema {schema}",
        )

    return (
        profile, raw_json,
        gender_dd, skin_dd, eyes_dd, hair_tex_dd, body_dd, body_height_dd,
        info("gender"), info("skin_tone"), info("eye_color"), info("hair_texture"), info("body_build"), info("body_height_impression"),
        hair_baseline_dd, beard_pattern_dd, beard_color_dd,
        bool(glasses.get("wears_regularly", False)),
        str(glasses.get("canonical_description", "") or ""),
        bool(freckles.get("has_freckles", False)),
        str(freckles.get("canonical_description", "") or ""),
        hair_color_md, hair_form_md, makeup_md,
        tattoo_md, piercing_rows, notes_md,
        color_from_dd, form_from_dd, makeup_from_dd,
        color_to_dd, form_to_dd, makeup_to_dd,
        _identity_clusters_markdown(profile), _identity_clusters_table(profile),
        _identity_cluster_preview_markdown(profile), _identity_cluster_preview_gallery(profile, trigger_word, input_folder),
        str((_identity_cluster_by_id(profile) or {}).get("cluster_id", "") or ""),
        gr.update(
            choices=IDENTITY_CLUSTER_ROLE_CHOICES,
            value=_identity_cluster_role_for_id(profile, str((_identity_cluster_by_id(profile) or {}).get("cluster_id", "") or "")),
        ),
        status,
    )


def save_profile_from_editor(
    trigger_word: str,
    input_folder: str,
    raw_profile_json: str,
    gender: str,
    skin_tone: str,
    eye_color: str,
    hair_texture: str,
    body_build: str,
    body_height_impression: str,
    hair_color_baseline: str,
    beard_pattern: str,
    beard_color: str,
    glasses_regular: bool,
    glasses_desc: str,
    freckles_present: bool,
    freckles_desc: str,
    piercing_table: Any,
) -> str:
    """Speichert die Editor-Werte zurueck ins _subject_profile.json.

    Wir behalten den Rest des Profils (per_image_traits, identity_markers,
    confidence, notes) und ueberschreiben gezielt die vom User bearbeiteten
    Felder. raw_profile_json ist der Backup-Zustand fuer den Fall, dass die
    Datei zwischenzeitlich geloescht wurde.
    """
    path = subject_profile_path_for(input_folder, trigger_word)
    profile: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f) or {}
        except Exception:
            profile = {}
    if not isinstance(profile, dict):
        profile = {}

    if not profile and raw_profile_json.strip():
        try:
            backup = json.loads(raw_profile_json)
            if isinstance(backup, dict):
                profile = backup
        except Exception:
            pass

    if not profile:
        return tr("❌ Kein Profil zum Speichern vorhanden.", "❌ No profile to save.")

    stable = profile.setdefault("stable_identity", {})
    stable["gender"] = (gender or "").strip()
    stable["skin_tone"] = (skin_tone or "").strip()
    stable["eye_color"] = (eye_color or "").strip()
    stable["hair_texture"] = (hair_texture or "").strip()
    stable["body_build"] = (body_build or "").strip()  # explicit "" erlaubt
    stable["body_height_impression"] = (body_height_impression or "").strip()

    canonical = profile.setdefault("canonical_features", {})
    canonical["hair_color"] = (hair_color_baseline or "").strip()
    canonical["eye_color"] = (eye_color or "").strip()
    canonical["beard_pattern"] = (beard_pattern or "").strip()
    canonical["beard_color"] = (beard_color or "").strip()

    markers = profile.setdefault("identity_markers", {})
    glasses = markers.setdefault("glasses", {"wears_regularly": False, "canonical_description": "", "frequency": ""})
    glasses["wears_regularly"] = bool(glasses_regular)
    glasses["canonical_description"] = (glasses_desc or "").strip()
    freckles = markers.setdefault("freckles", {"has_freckles": False, "canonical_description": "", "frequency": ""})
    freckles["has_freckles"] = bool(freckles_present)
    freckles["canonical_description"] = (freckles_desc or "").strip()

    piercing_records: List[Dict[str, Any]] = []
    if hasattr(piercing_table, "to_dict"):
        try:
            piercing_records = list(piercing_table.to_dict("records"))
        except Exception:
            piercing_records = []
    elif isinstance(piercing_table, dict) and "data" in piercing_table:
        headers = piercing_table.get("headers") or ["location", "canonical_description", "frequency", "category", "role"]
        for row in piercing_table.get("data") or []:
            piercing_records.append(row if isinstance(row, dict) else {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))})
    elif isinstance(piercing_table, list):
        headers = ["location", "canonical_description", "frequency", "category", "role"]
        for row in piercing_table:
            if isinstance(row, dict):
                piercing_records.append(row)
            elif isinstance(row, (list, tuple)):
                piercing_records.append({headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))})

    clean_inventory = []
    for rec in piercing_records:
        loc = str(rec.get("location", "") or "").strip()
        desc = str(rec.get("canonical_description", "") or "").strip()
        if not loc or not desc:
            continue
        category = str(rec.get("category", "body_piercing") or "body_piercing").strip()
        if category not in {"body_piercing", "ear_jewelry"}:
            category = "ear_jewelry" if loc.startswith("ear_") else "body_piercing"
        role = str(rec.get("role", "variable") or "variable").strip()
        if role not in {"canonical", "variable", "accessory", "ignore"}:
            role = "accessory" if category == "ear_jewelry" else "variable"
        clean_inventory.append({
            "location": loc,
            "canonical_description": desc,
            "frequency": str(rec.get("frequency", "") or ""),
            "category": category,
            "role": role,
        })
    markers["piercing_inventory"] = clean_inventory
    markers["piercing_baseline"] = [
        {"location": x["location"], "canonical_description": x["canonical_description"], "frequency": x["frequency"]}
        for x in clean_inventory if x.get("role") == "canonical"
    ]

    profile["force_only_when_visible"] = True
    # Preserve explicit user identity decisions across later Priority/Core
    # profile rebuilds. Automatic weighted consolidation is applied first;
    # these confirmed values remain authoritative afterwards.
    profile["user_confirmed_identity_overrides"] = {
        "stable_identity": dict(stable),
        "canonical_features": dict(canonical),
        "identity_markers": {
            "glasses": dict(glasses),
            "freckles": dict(freckles),
            "piercing_inventory": list(clean_inventory),
            "piercing_baseline": list(markers.get("piercing_baseline", []) or []),
        },
    }

    notes = profile.setdefault("normalizer_notes", [])
    if isinstance(notes, list):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        notes.append(f"User-edited via UI at {stamp}.")
        profile["normalizer_notes"] = notes[-20:]

    try:
        core_atomic_write_json(path, profile)
        return tr(f"✅ Profil gespeichert: {path}", f"✅ Profile saved: {path}")
    except Exception as e:
        return tr(f"❌ Speichern fehlgeschlagen: {e}", f"❌ Save failed: {e}")


def rebucket_per_image_field(
    trigger_word: str,
    input_folder: str,
    field_name: str,
    from_value: str,
    to_value: str,
) -> str:
    """Verschiebt alle Per-Image-Eintraege eines Buckets in einen anderen.
    Anwendungsfall: Normalizer hat 3 Bilder als 'red' klassifiziert, in
    Wahrheit ist es Lichtartefakt -> alle auf 'blonde' setzen.
    """
    if field_name not in {"hair_color_base", "hair_form", "makeup_intensity"}:
        return tr("❌ Unbekanntes Feld.", "❌ Unknown field.")

    from_value = (from_value or "").strip()
    to_value = (to_value or "").strip()
    if not from_value or not to_value:
        return tr("⚠️ Quell- und Zielwert müssen gesetzt sein.", "⚠️ From and to values required.")
    if from_value == to_value:
        return tr("ℹ️ Quelle und Ziel sind identisch.", "ℹ️ Source and target are identical.")

    path = subject_profile_path_for(input_folder, trigger_word)
    if not os.path.exists(path):
        return tr(f"❌ Kein Profil unter {path}", f"❌ No profile at {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception as e:
        return tr(f"❌ Lesefehler: {e}", f"❌ Read error: {e}")

    per_image = profile.get("per_image_traits", {}) or {}
    moved = 0
    for image_id, traits in per_image.items():
        if not isinstance(traits, dict):
            continue
        if str(traits.get(field_name, "")).strip() == from_value:
            traits[field_name] = to_value
            moved += 1

    if moved == 0:
        return tr(f"ℹ️ Keine Bilder mit {field_name}={from_value} gefunden.",
                  f"ℹ️ No images with {field_name}={from_value} found.")

    notes = profile.setdefault("normalizer_notes", [])
    if isinstance(notes, list):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        notes.append(f"User-rebucket via UI at {stamp}: {field_name} '{from_value}' -> '{to_value}' ({moved} images).")
        profile["normalizer_notes"] = notes[-20:]

    try:
        core_atomic_write_json(path, profile)
        return tr(f"✅ {moved} Bilder umgebuchtet ({field_name}: {from_value} → {to_value}).",
                  f"✅ {moved} images re-bucketed ({field_name}: {from_value} → {to_value}).")
    except Exception as e:
        return tr(f"❌ Schreibfehler: {e}", f"❌ Write error: {e}")


def reset_profile_from_backup(trigger_word: str, input_folder: str, raw_profile_json: str) -> str:
    """Stellt das Profile aus dem Editor-Textfeld wieder her (Reset-Button)."""
    if not raw_profile_json.strip():
        return tr("⚠️ Kein Backup verfügbar.", "⚠️ No backup available.")
    try:
        profile = json.loads(raw_profile_json)
        if not isinstance(profile, dict):
            return tr("❌ Backup ist kein JSON-Objekt.", "❌ Backup is not a JSON object.")
    except Exception as e:
        return tr(f"❌ Backup-JSON ungültig: {e}", f"❌ Backup JSON invalid: {e}")

    path = subject_profile_path_for(input_folder, trigger_word)
    try:
        core_atomic_write_json(path, profile)
        return tr(f"↩️ Reset abgeschlossen: {path}", f"↩️ Reset complete: {path}")
    except Exception as e:
        return tr(f"❌ Schreibfehler: {e}", f"❌ Write error: {e}")



def _candidate_image_paths_for_cluster_preview(trigger_word: str, input_folder: str, filename_or_path: str) -> List[str]:
    """Return likely filesystem locations for a cluster preview image.

    Profiles may contain either original filenames, absolute original paths, or
    cached/exported paths depending on which Curator version created them. The
    UI should be forgiving and show previews whenever the image still exists in
    the input folder or one of the curated output buckets.
    """
    value = str(filename_or_path or "").strip()
    if not value:
        return []

    candidates: List[str] = []
    seen: set = set()

    def add(path: str) -> None:
        if not path:
            return
        p = os.path.normpath(path)
        if p not in seen:
            seen.add(p)
            candidates.append(p)

    # Absolute / explicit path first.
    add(value)

    basename = os.path.basename(value)
    if input_folder:
        add(os.path.join(input_folder, value))
        add(os.path.join(input_folder, basename))

    root = output_root_for(input_folder, trigger_word)
    for sub in (
        "01_train_ready",
        "02_keep_unused",
        "03_caption_remove",
        "04_review",
        "05_reject",
        "06_needs_manual_review",
        os.path.join("_cache", "ig_frame_crops"),
    ):
        add(os.path.join(root, sub, value))
        add(os.path.join(root, sub, basename))

    return candidates


def _find_cluster_preview_image(trigger_word: str, input_folder: str, filename_or_path: str) -> Optional[str]:
    for path in _candidate_image_paths_for_cluster_preview(trigger_word, input_folder, filename_or_path):
        if os.path.isfile(path):
            return path

    # Last-resort bounded recursive search by basename in the input folder and
    # curated output root. This keeps older profiles useful without forcing a
    # full re-run. The search is intentionally filename-only and stops early.
    basename = os.path.basename(str(filename_or_path or "").strip())
    if not basename:
        return None
    roots = []
    if input_folder and os.path.isdir(input_folder):
        roots.append(input_folder)
    out_root = output_root_for(input_folder, trigger_word)
    if os.path.isdir(out_root) and out_root not in roots:
        roots.append(out_root)

    scanned = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            scanned += 1
            if scanned > 500:
                return None
            if basename in filenames:
                return os.path.join(dirpath, basename)
    return None


def _identity_clusters_gallery(
    profile: Dict[str, Any],
    trigger_word: str,
    input_folder: str,
    max_per_cluster: int = 4,
    max_total: int = 80,
) -> List[Tuple[Image.Image, str]]:
    """Build preview thumbnails for identity clusters.

    Shows a small representative sample per cluster. These are previews only;
    the editable table remains the source of truth for cluster roles.
    """
    gallery: List[Tuple[Image.Image, str]] = []
    clusters = profile.get("identity_clusters", []) or []
    if not isinstance(clusters, list):
        return gallery

    for c in clusters:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("cluster_id", "") or "")
        role = str(c.get("role", "variation") or "variation")
        summary = str(c.get("summary", "") or "")
        filenames = c.get("filenames", []) or []
        image_paths = c.get("image_paths", []) or []
        preview_sources: List[str] = []
        for src in list(image_paths) + list(filenames):
            s = str(src or "").strip()
            if s and s not in preview_sources:
                preview_sources.append(s)

        shown = 0
        for src in preview_sources:
            if shown >= max_per_cluster or len(gallery) >= max_total:
                break
            path = _find_cluster_preview_image(trigger_word, input_folder, src)
            if not path:
                continue
            img = load_gallery_image(path, max_size=(512, 512))
            if img is None:
                continue
            caption = f"{cid} | {role} | {summary}\n{os.path.basename(path)}"
            gallery.append((img, caption))
            shown += 1
        if len(gallery) >= max_total:
            break
    return gallery


AUDITED_REJECT_CLUSTER_ID = "audited_rejects"
CLUSTER_GALLERY_PAGE_SIZE = 36


def _parse_cluster_gallery_page(value: Any) -> int:
    try:
        return max(1, int(float(str(value or "1").strip())))
    except Exception:
        return 1


def _cluster_gallery_page_count(cluster: Optional[Dict[str, Any]], page_size: int = CLUSTER_GALLERY_PAGE_SIZE) -> int:
    total = len(_cluster_member_records(cluster or {}))
    return max(1, (total + max(1, int(page_size)) - 1) // max(1, int(page_size)))


def _cluster_gallery_page_update(profile: Dict[str, Any], cluster_id: str, page: Any = 1) -> Any:
    cluster = _identity_cluster_by_id(profile or {}, cluster_id)
    total_pages = _cluster_gallery_page_count(cluster)
    current = min(_parse_cluster_gallery_page(page), total_pages)
    choices = [(f"{i} / {total_pages}", str(i)) for i in range(1, total_pages + 1)]
    return gr.update(choices=choices, value=str(current), interactive=total_pages > 1)


def _profile_member_id_ui(row: Dict[str, Any]) -> str:
    existing = str((row or {}).get("profile_member_id") or "").strip()
    if existing:
        return existing
    content_id = str((row or {}).get("profile_image_id") or (row or {}).get("file_hash") or "").strip()
    src = str((row or {}).get("original_path") or (row or {}).get("original_filename") or "").strip()
    if not content_id:
        content_id = hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()
    normalized = os.path.normcase(os.path.abspath(src)) if src else str((row or {}).get("original_filename") or content_id)
    suffix = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{content_id}::{suffix}"


def _selection_ref_for_record(record: Dict[str, Any], record_index: int = -1) -> str:
    return json.dumps({
        "image_id": str(record.get("image_id", "") or ""),
        "filename": str(record.get("filename", "") or ""),
        "image_path": str(record.get("image_path", "") or ""),
        "record_index": int(record_index),
    }, ensure_ascii=False, separators=(",", ":"))


def _parse_selection_ref(value: Any) -> Dict[str, Any]:
    raw = str(value or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Backward compatibility with the earlier UI state, which stored only ID.
    return {"image_id": raw}


def _resolve_cluster_record(cluster: Dict[str, Any], selected_ref: Any) -> Tuple[int, Optional[Dict[str, str]]]:
    records = _cluster_member_records(cluster or {})
    ref = _parse_selection_ref(selected_ref)
    if not ref:
        return -1, None
    try:
        idx = int(ref.get("record_index", -1))
    except Exception:
        idx = -1
    if 0 <= idx < len(records):
        candidate = records[idx]
        checks = [
            ("image_path", str(ref.get("image_path", "") or "")),
            ("filename", str(ref.get("filename", "") or "")),
            ("image_id", str(ref.get("image_id", "") or "")),
        ]
        if all(not expected or str(candidate.get(field, "") or "") == expected for field, expected in checks):
            return idx, candidate

    image_path = str(ref.get("image_path", "") or "")
    filename = str(ref.get("filename", "") or "")
    image_id = str(ref.get("image_id", "") or "")
    for i, record in enumerate(records):
        if image_path and str(record.get("image_path", "") or "") == image_path:
            return i, record
    for i, record in enumerate(records):
        if filename and image_id and str(record.get("filename", "") or "") == filename and str(record.get("image_id", "") or "") == image_id:
            return i, record
    for i, record in enumerate(records):
        if filename and str(record.get("filename", "") or "") == filename:
            return i, record
    for i, record in enumerate(records):
        if image_id and str(record.get("image_id", "") or "") == image_id:
            return i, record
    return -1, None


def _normalize_audited_reject_role_ui(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return profile
    member_roles = profile.setdefault("identity_cluster_member_roles", {})
    if not isinstance(member_roles, dict):
        member_roles = {}
        profile["identity_cluster_member_roles"] = member_roles
    overrides = profile.setdefault("identity_cluster_role_overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}
        profile["identity_cluster_role_overrides"] = overrides
    for cluster in profile.get("identity_clusters", []) or []:
        if not isinstance(cluster, dict):
            continue
        cid = str(cluster.get("cluster_id", "") or "")
        kind = str(cluster.get("cluster_kind", "") or "")
        if cid == AUDITED_REJECT_CLUSTER_ID or kind == "audited_rejects":
            cluster["role"] = "exclude"
            overrides[cid or AUDITED_REJECT_CLUSTER_ID] = "exclude"
            for mid in cluster.get("members", []) or []:
                member_roles[str(mid)] = "exclude"
    return profile


def _sorted_identity_clusters(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    profile = _normalize_audited_reject_role_ui(profile or {})
    clusters = [c for c in ((profile or {}).get("identity_clusters", []) or []) if isinstance(c, dict)]
    return sorted(
        clusters,
        key=lambda c: (
            1 if (
                str(c.get("cluster_id", "") or "") == AUDITED_REJECT_CLUSTER_ID
                or str(c.get("cluster_kind", "") or "") == "audited_rejects"
                or bool(c.get("sort_last"))
            ) else 0,
        ),
    )


def _cluster_member_records(cluster: Dict[str, Any]) -> List[Dict[str, str]]:
    members = list(cluster.get("members", []) or [])
    filenames = list(cluster.get("filenames", []) or [])
    paths = list(cluster.get("image_paths", []) or [])
    reject_reasons = list(cluster.get("reject_reasons", []) or [])
    total = max(len(members), len(filenames), len(paths), len(reject_reasons))
    records: List[Dict[str, str]] = []
    for idx in range(total):
        records.append({
            "image_id": str(members[idx] if idx < len(members) else ""),
            "filename": str(filenames[idx] if idx < len(filenames) else ""),
            "image_path": str(paths[idx] if idx < len(paths) else ""),
            "reject_reason": str(reject_reasons[idx] if idx < len(reject_reasons) else ""),
            "record_index": str(idx),
        })
    return records




def _missing_cluster_preview_image() -> Image.Image:
    """Return a neutral tile so every bucket member stays selectable."""
    return Image.new("RGB", (384, 384), (72, 72, 72))

def _identity_cluster_preview_entries(
    profile: Dict[str, Any],
    trigger_word: str,
    input_folder: str,
    cluster_id: Optional[str] = None,
    page: Any = 1,
    page_size: int = CLUSTER_GALLERY_PAGE_SIZE,
) -> List[Dict[str, Any]]:
    """Build one deterministic page of preview records.

    Large reject buckets used to create hundreds of PIL thumbnails at once.
    Apart from being slow, a transient image decode failure could shift gallery
    indices between the select callback and the detach callback.  We now page
    the *record list first* and return file paths to Gradio, so the browser/UI
    loads only one small stable page and every selection keeps the original
    member record index.
    """
    entries: List[Dict[str, Any]] = []
    cluster = _identity_cluster_by_id(profile, cluster_id)
    if not cluster:
        return entries
    role = str(cluster.get("role", "variation") or "variation")
    summary = str(cluster.get("summary", "") or "")
    priority_ids = {str(x) for x in ((profile or {}).get("priority_image_ids", []) or [])}
    priority_names = {str(x) for x in ((profile or {}).get("priority_image_filenames", []) or [])}
    row_lookup = _stage_rows_by_profile_id(trigger_word, input_folder)
    is_reject_bucket = str(cluster.get("cluster_kind", "") or "") == "audited_rejects"

    records = _cluster_member_records(cluster)
    size = max(1, int(page_size))
    total_pages = max(1, (len(records) + size - 1) // size)
    current_page = min(_parse_cluster_gallery_page(page), total_pages)
    start_index = (current_page - 1) * size
    page_records = records[start_index:start_index + size]

    for offset, record in enumerate(page_records):
        record_index = start_index + offset
        source = record.get("image_path") or record.get("filename")
        path = _find_cluster_preview_image(trigger_word, input_folder, source)
        filename = record.get("filename") or (os.path.basename(path) if path else str(source or "unknown"))
        preview = load_gallery_image(path, max_size=(384, 384)) if path else None
        preview_available = preview is not None
        if preview is None:
            preview = _missing_cluster_preview_image()
        content_id = str(record.get("image_id", "") or "").split("::", 1)[0]
        is_priority = record.get("image_id") in priority_ids or content_id in priority_ids or filename in priority_names
        marker = "⭐ PRIORITY | " if is_priority else ""
        if is_reject_bucket:
            marker += "REJECTED/AUDITED | "
        row = _row_for_cluster_record(row_lookup, record)
        reject_reason = str(record.get("reject_reason", "") or "") or _reject_reason_from_row_ui(row)
        caption = f"{marker}{role} | {summary}\n{filename}"
        if not preview_available:
            caption += "\n⚠ Preview unavailable – item remains selectable"
        if reject_reason:
            caption += f"\nReject: {reject_reason}"
        entries.append({
            **record,
            "record_index": str(record_index),
            "filename": filename,
            "resolved_path": path,
            # Only one small page is decoded at a time. Missing/unreadable files
            # receive a placeholder so record positions never shift.
            "image": preview,
            "caption": caption,
            "priority": is_priority,
            "reject_reason": reject_reason,
            "selection_ref": _selection_ref_for_record(record, record_index),
            "page": current_page,
            "total_pages": total_pages,
            "preview_available": preview_available,
        })
    return entries


def _identity_cluster_selector_choices(profile: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Choices for selecting exactly one identity cluster in the UI preview."""
    choices: List[Tuple[str, str]] = []
    clusters = _sorted_identity_clusters(profile)
    for idx, c in enumerate(clusters, start=1):
        cid = str(c.get("cluster_id", "") or "").strip()
        if not cid:
            continue
        role = str(c.get("role", "variation") or "variation")
        n = int(c.get("n", 0) or 0)
        summary = str(c.get("summary", "") or cid)
        prefix = "⬇ REJECTED | " if str(c.get("cluster_kind", "") or "") == "audited_rejects" else ""
        label = f"{idx:02d}. {prefix}{role} | n={n} | {summary}"
        choices.append((label, cid))
    return choices

def _identity_cluster_by_id(profile: Dict[str, Any], cluster_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    clusters = _sorted_identity_clusters(profile)
    if not clusters:
        return None
    wanted = str(cluster_id or "").strip()
    if wanted:
        for c in clusters:
            if str(c.get("cluster_id", "") or "") == wanted:
                return c
    return clusters[0]

def _identity_cluster_preview_markdown(profile: Dict[str, Any], cluster_id: Optional[str] = None, page: Any = 1) -> str:
    c = _identity_cluster_by_id(profile, cluster_id)
    if not c:
        return tr("_Kein Cluster ausgewählt._", "_No cluster selected._")
    cid = str(c.get("cluster_id", "") or "")
    role = str(c.get("role", "variation") or "variation")
    summary = str(c.get("summary", "") or "")
    n = int(c.get("n", 0) or 0)
    quality = str(c.get("avg_quality_total", "") or "")
    identity = str(c.get("avg_identity_usefulness", "") or "")
    style_counts = c.get("style_counts", {}) or {}
    shot_counts = c.get("shot_counts", {}) or {}
    filenames = c.get("filenames", []) or []
    cluster_kind = str(c.get("cluster_kind", "") or "")
    page_count = _cluster_gallery_page_count(c)
    current_page = min(_parse_cluster_gallery_page(page), page_count)
    lines = [
        f"### {tr('Vorschau', 'Preview')}: `{cid}`",
        f"**Role:** `{role}`  ",
        f"**N:** {n}  ",
        f"**Summary:** {summary}  ",
        f"**" + tr("Galerieseite", "Gallery page") + f":** {current_page}/{page_count} · " + tr(f"bis zu {CLUSTER_GALLERY_PAGE_SIZE} Bilder je Seite", f"up to {CLUSTER_GALLERY_PAGE_SIZE} images per page") + "  ",
    ]
    if cluster_kind == "audited_rejects":
        lines.append("**" + tr("Sonder-Bucket", "Special bucket") + ":** " + tr(
            "Vollständig auditierte Rejects; immer am Ende und standardmäßig `exclude`. Einzelbilder können gelöst oder als Priority markiert werden.  ",
            "Fully audited rejects; always listed last and default to `exclude`. Individual images can be detached or marked Priority.  ",
        ))
    if quality or identity:
        lines.append(f"**Quality / Identity:** {quality or '-'} / {identity or '-'}  ")
    if shot_counts:
        lines.append("**Shots:** " + ", ".join(f"`{k}`={v}" for k, v in sorted(shot_counts.items())))
    if style_counts:
        lines.append("**Style:** " + ", ".join(f"`{k}`={v}" for k, v in sorted(style_counts.items())))
    reject_reasons = [str(x) for x in (c.get("reject_reasons", []) or []) if str(x)]
    if reject_reasons:
        reason_counts = Counter(reject_reasons)
        lines.append("**" + tr("Häufigste Reject-Gründe", "Most common reject reasons") + ":** " + ", ".join(
            f"`{reason}`={count}" for reason, count in reason_counts.most_common(8)
        ))
    if filenames:
        lines.append("")
        lines.append("**" + tr("Dateien", "Files") + ":** " + ", ".join(str(x) for x in filenames[:10]) + (" …" if len(filenames) > 10 else ""))
    return "\n".join(lines)


def _identity_cluster_preview_gallery(
    profile: Dict[str, Any],
    trigger_word: str,
    input_folder: str,
    cluster_id: Optional[str] = None,
    page: Any = 1,
) -> List[Tuple[Any, str]]:
    """Build one stable, lazily loaded page for the selected cluster."""
    return [
        (entry["image"], entry["caption"])
        for entry in _identity_cluster_preview_entries(
            profile, trigger_word, input_folder, cluster_id, page=page
        )
    ]


def select_identity_cluster_image_ui(
    profile: Dict[str, Any],
    trigger_word: str,
    input_folder: str,
    cluster_id: str,
    gallery_page: Any,
    evt: gr.SelectData,
) -> Tuple[str, str]:
    entries = _identity_cluster_preview_entries(profile or {}, trigger_word, input_folder, cluster_id, page=gallery_page)
    idx = getattr(evt, "index", None)
    try:
        idx = idx[0] if isinstance(idx, (tuple, list)) else idx
        selected_index = int(idx)
    except Exception:
        selected_index = -1
    if selected_index < 0 or selected_index >= len(entries):
        return "", tr("⚠️ Bildauswahl konnte nicht bestimmt werden.", "⚠️ Could not resolve selected image.")
    entry = entries[selected_index]
    image_id = str(entry.get("image_id", "") or "")
    filename = str(entry.get("filename", "") or "")
    priority = bool(entry.get("priority"))
    reject_reason = str(entry.get("reject_reason", "") or "")
    status = (
        f"**{tr('Ausgewähltes Bild', 'Selected image')}:** `{filename}`  \n"
        f"**Member ID:** `{image_id or '-'}`  \n"
        f"**" + tr("Galerieseite", "Gallery page") + f":** {_parse_cluster_gallery_page(gallery_page)}  \n"
        f"**Priority:** {'⭐ yes' if priority else 'no'}"
    )
    if reject_reason:
        status += f"  \n**{tr('Reject-Grund', 'Reject reason')}:** `{reject_reason}`"
    return str(entry.get("selection_ref", "") or ""), status


def _load_profile_for_bucket_edit(trigger_word: str, input_folder: str, profile_state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    path = subject_profile_path_for(input_folder, trigger_word)
    profile: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                profile = _normalize_audited_reject_role_ui(loaded)
        except Exception:
            profile = {}
    if not profile and isinstance(profile_state, dict):
        profile = json.loads(json.dumps(profile_state))
    return profile, path


def _save_profile_bucket_edit(profile: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    core_atomic_write_json(path, profile)


def _stage_rows_by_profile_id(trigger_word: str, input_folder: str) -> Dict[str, Dict[str, Any]]:
    path = caption_stage_path_for(input_folder, trigger_word)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            stage = json.load(f)
        rows = stage.get("all_rows", []) if isinstance(stage, dict) else []
    except Exception:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        image_id = str(row.get("profile_image_id", "") or row.get("file_hash", "") or "")
        member_id = _profile_member_id_ui(row)
        filename = str(row.get("original_filename", "") or "")
        image_path = str(row.get("original_path", "") or "")
        if member_id:
            result[member_id] = row
        if image_id and image_id not in result:
            result[image_id] = row
        if filename:
            result[f"filename::{filename}"] = row
        if image_path:
            result[f"path::{image_path}"] = row
    return result


def _row_for_cluster_record(row_lookup: Dict[str, Dict[str, Any]], record: Dict[str, Any]) -> Dict[str, Any]:
    for key in (
        str(record.get("image_id", "") or ""),
        f"path::{str(record.get('image_path', '') or '')}",
        f"filename::{str(record.get('filename', '') or '')}",
    ):
        if key and key in row_lookup:
            return row_lookup[key]
    return {}


def _reject_reason_from_row_ui(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict) or not row:
        return ""
    parts: List[str] = []
    short_reason = str(row.get("short_reason", "") or "").strip()
    if short_reason:
        parts.append(short_reason)
    if str(row.get("arcface_flag", "") or "").strip().lower() == "hard":
        parts.append("ArcFace hard identity flag")
    for field in ("local_override_reasons", "issues"):
        value = row.get(field)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = [x.strip() for x in re.split(r"[;,|]", value) if x.strip()]
        if isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item or "").strip()
                if text and text not in parts:
                    parts.append(text)
    return " | ".join(parts[:4])


def _refresh_cluster_stats_ui(cluster: Dict[str, Any], row_lookup: Dict[str, Dict[str, Any]]) -> None:
    members = [str(x) for x in (cluster.get("members", []) or [])]
    rows = [row_lookup[mid] for mid in members if mid in row_lookup]
    cluster["n"] = len(members)
    if not rows:
        return
    qualities = [float(r.get("quality_total", 0) or 0) for r in rows]
    identities = [float(r.get("quality_identity_usefulness", 0) or 0) for r in rows]
    cluster["avg_quality_total"] = round(sum(qualities) / max(1, len(qualities)), 1)
    cluster["avg_identity_usefulness"] = round(sum(identities) / max(1, len(identities)), 1)
    shot_counts: Dict[str, int] = {}
    style_counts: Dict[str, int] = {}
    for row in rows:
        shot = str(row.get("shot_type", "") or "unknown")
        shot_counts[shot] = shot_counts.get(shot, 0) + 1
        style = "bw" if bool(row.get("is_grayscale_filter")) else (str(row.get("color_tint_label", "") or "clean") or "clean")
        style_counts[style] = style_counts.get(style, 0) + 1
    cluster["shot_counts"] = shot_counts
    cluster["style_counts"] = style_counts


def _rebuild_cluster_member_maps_ui(profile: Dict[str, Any]) -> None:
    roles: Dict[str, str] = {}
    clusters_by_member: Dict[str, str] = {}
    for cluster in _sorted_identity_clusters(profile):
        cid = str(cluster.get("cluster_id", "") or "")
        role = str(cluster.get("role", "variation") or "variation")
        for mid in cluster.get("members", []) or []:
            roles[str(mid)] = role
            clusters_by_member[str(mid)] = cid
    profile["identity_cluster_member_roles"] = roles
    profile["identity_cluster_member_clusters"] = clusters_by_member


def detach_selected_image_from_cluster_ui(
    trigger_word: str,
    input_folder: str,
    profile_state: Dict[str, Any],
    selected_cluster_id: str,
    selected_image_id: str,
    gallery_page: Any,
) -> Tuple[Dict[str, Any], str, str, List[List[Any]], str, List[Tuple[Any, str]], str, Any, Any, str, str, str]:
    profile, path = _load_profile_for_bucket_edit(trigger_word, input_folder, profile_state)
    cid = str(selected_cluster_id or "").strip()
    selected_ref = str(selected_image_id or "").strip()
    if not profile or not cid or not selected_ref:
        return profile, json.dumps(profile, ensure_ascii=False, indent=2), _identity_clusters_markdown(profile), _identity_clusters_table(profile), _identity_cluster_preview_markdown(profile, cid, page=gallery_page), _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cid, page=gallery_page), cid, gr.update(choices=IDENTITY_CLUSTER_ROLE_CHOICES, value=_identity_cluster_role_for_id(profile, cid)), _cluster_gallery_page_update(profile, cid, gallery_page), selected_ref, tr("_Kein Bild ausgewählt._", "_No image selected._"), tr("⚠️ Erst ein Bild in der Vorschau anklicken.", "⚠️ Select an image in the preview first.")

    clusters = _sorted_identity_clusters(profile)
    source = next((c for c in clusters if str(c.get("cluster_id", "") or "") == cid), None)
    if not source:
        return profile, json.dumps(profile, ensure_ascii=False, indent=2), _identity_clusters_markdown(profile), _identity_clusters_table(profile), _identity_cluster_preview_markdown(profile, cid, page=gallery_page), _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cid, page=gallery_page), cid, gr.update(choices=IDENTITY_CLUSTER_ROLE_CHOICES, value="variation"), _cluster_gallery_page_update(profile, cid, gallery_page), "", tr("_Kein Bild ausgewählt._", "_No image selected._"), tr("❌ Bucket nicht gefunden.", "❌ Bucket not found.")

    selected_index, record = _resolve_cluster_record(source, selected_ref)
    if selected_index < 0 or not record:
        return profile, json.dumps(profile, ensure_ascii=False, indent=2), _identity_clusters_markdown(profile), _identity_clusters_table(profile), _identity_cluster_preview_markdown(profile, cid, page=gallery_page), _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cid, page=gallery_page), cid, gr.update(choices=IDENTITY_CLUSTER_ROLE_CHOICES, value=_identity_cluster_role_for_id(profile, cid)), _cluster_gallery_page_update(profile, cid, gallery_page), "", tr("_Kein Bild ausgewählt._", "_No image selected._"), tr("❌ Bild ist nicht mehr in diesem Bucket.", "❌ Image is no longer in this bucket.")

    image_id = str(record.get("image_id", "") or "")
    members = list(source.get("members", []) or [])
    filenames = list(source.get("filenames", []) or [])
    paths = list(source.get("image_paths", []) or [])
    reject_reasons = list(source.get("reject_reasons", []) or [])
    members.pop(selected_index)
    filename = filenames.pop(selected_index) if selected_index < len(filenames) else record.get("filename", "")
    image_path = paths.pop(selected_index) if selected_index < len(paths) else record.get("image_path", "")
    reject_reason = reject_reasons.pop(selected_index) if selected_index < len(reject_reasons) else str(record.get("reject_reason", "") or "")
    source["members"], source["filenames"], source["image_paths"] = members, filenames, paths
    if reject_reasons or "reject_reasons" in source:
        source["reject_reasons"] = reject_reasons

    row_lookup = _stage_rows_by_profile_id(trigger_word, input_folder)
    _refresh_cluster_stats_ui(source, row_lookup)
    if not members:
        clusters = [c for c in clusters if c is not source]

    stem = re.sub(r"[^a-zA-Z0-9_\-]+", "_", os.path.splitext(os.path.basename(filename or image_id))[0]).strip("_").lower() or "image"
    existing_ids = {str(c.get("cluster_id", "") or "") for c in clusters}
    new_cid = f"manual_{stem}"
    counter = 2
    while new_cid in existing_ids:
        new_cid = f"manual_{stem}_{counter}"
        counter += 1
    detached_from_reject = str(source.get("cluster_kind", "") or "") == "audited_rejects" or cid == AUDITED_REJECT_CLUSTER_ID
    inherited_role = "exclude" if detached_from_reject else str(source.get("role", "variation") or "variation")
    row = _row_for_cluster_record(row_lookup, record)
    reject_reason = reject_reason or _reject_reason_from_row_ui(row)
    new_cluster = {
        "cluster_id": new_cid,
        "cluster_kind": "manual_singleton",
        "role": inherited_role,
        "n": 1,
        "summary": f"manually detached | {filename or image_id}",
        "avg_quality_total": round(float(row.get("quality_total", 0) or 0), 1) if row else "",
        "avg_identity_usefulness": round(float(row.get("quality_identity_usefulness", 0) or 0), 1) if row else "",
        "shot_counts": {str(row.get("shot_type", "") or "unknown"): 1} if row else {},
        "style_counts": {},
        "members": [image_id],
        "filenames": [filename],
        "image_paths": [image_path],
        "reject_reasons": [reject_reason] if reject_reason else [],
        "origin_cluster_id": cid,
        "detached_from_reject": bool(detached_from_reject),
    }
    # Keep audited rejects at the absolute bottom.
    insert_at = next((i for i, c in enumerate(clusters) if str(c.get("cluster_kind", "") or "") == "audited_rejects"), len(clusters))
    clusters.insert(insert_at, new_cluster)
    profile["identity_clusters"] = clusters
    _rebuild_cluster_member_maps_ui(profile)
    notes = profile.setdefault("normalizer_notes", [])
    if isinstance(notes, list):
        notes.append(f"User detached image {filename or image_id} from {cid} into {new_cid} at {time.strftime('%Y-%m-%dT%H:%M:%S')}.")
        profile["normalizer_notes"] = notes[-40:]
    _save_profile_bucket_edit(profile, path)
    role_update = gr.update(choices=IDENTITY_CLUSTER_ROLE_CHOICES, value=inherited_role)
    return (
        profile,
        json.dumps(profile, ensure_ascii=False, indent=2),
        _identity_clusters_markdown(profile),
        gr.update(value=_identity_clusters_table(profile)),
        _identity_cluster_preview_markdown(profile, new_cid, page=1),
        gr.update(value=_identity_cluster_preview_gallery(profile, trigger_word, input_folder, new_cid, page=1)),
        new_cid,
        role_update,
        _cluster_gallery_page_update(profile, new_cid, 1),
        "",
        tr("_Kein Bild ausgewählt._", "_No image selected._"),
        tr(f"✅ Bild aus `{cid}` gelöst und in neuen Bucket `{new_cid}` verschoben.", f"✅ Image detached from `{cid}` and moved into new bucket `{new_cid}`."),
    )


def refresh_identity_cluster_panel_ui(
    trigger_word: str,
    input_folder: str,
    profile_state: Dict[str, Any],
    preferred_cluster_id: str,
) -> Tuple[Dict[str, Any], str, str, Any, str, Any, str, Any, Any, str, str]:
    """Force-refresh all cluster UI components from the persisted profile.

    Gradio 6 may keep an interactive Dataframe/Gallery visually unchanged when a
    callback returns only a raw replacement value.  This helper reloads the
    just-saved profile and returns explicit component updates.  It is chained
    after detach so the new singleton bucket is visible immediately.
    """
    profile, _path = _load_profile_for_bucket_edit(trigger_word, input_folder, profile_state)
    clusters = _sorted_identity_clusters(profile)
    wanted = str(preferred_cluster_id or "").strip()
    cluster_ids = {str(c.get("cluster_id", "") or "") for c in clusters}
    if wanted not in cluster_ids:
        wanted = str((clusters[0] if clusters else {}).get("cluster_id", "") or "")
    role = _identity_cluster_role_for_id(profile, wanted) if wanted else "variation"
    table_rows = _identity_clusters_table(profile)
    gallery_rows = _identity_cluster_preview_gallery(profile, trigger_word, input_folder, wanted, page=1)
    return (
        profile,
        json.dumps(profile, ensure_ascii=False, indent=2),
        _identity_clusters_markdown(profile),
        gr.update(value=table_rows),
        _identity_cluster_preview_markdown(profile, wanted, page=1),
        gr.update(value=gallery_rows),
        wanted,
        gr.update(choices=IDENTITY_CLUSTER_ROLE_CHOICES, value=role),
        _cluster_gallery_page_update(profile, wanted, 1),
        "",
        tr("_Kein Bild ausgewählt._", "_No image selected._"),
    )


def set_selected_image_priority_ui(
    trigger_word: str,
    input_folder: str,
    profile_state: Dict[str, Any],
    selected_cluster_id: str,
    selected_image_id: str,
    gallery_page: Any,
    make_priority: bool,
    confirm_hazards: bool = False,
) -> Tuple[Dict[str, Any], str, str, List[Tuple[Any, str]], str, str]:
    profile, path = _load_profile_for_bucket_edit(trigger_word, input_folder, profile_state)
    cid = str(selected_cluster_id or "").strip()
    selected_ref = str(selected_image_id or "").strip()
    cluster = _identity_cluster_by_id(profile, cid)
    _record_index, record = _resolve_cluster_record(cluster or {}, selected_ref)
    if not record:
        return profile, json.dumps(profile, ensure_ascii=False, indent=2), _identity_cluster_preview_markdown(profile, cid, page=gallery_page), _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cid, page=gallery_page), tr("_Kein Bild ausgewählt._", "_No image selected._"), tr("⚠️ Erst ein Bild in der Vorschau anklicken.", "⚠️ Select an image in the preview first.")
    image_id = str(record.get("image_id", "") or "")
    content_image_id = image_id.split("::", 1)[0]
    filename = str(record.get("filename", "") or "")
    reject_reason = str(record.get("reject_reason", "") or "")

    hazards: List[str] = []
    if make_priority:
        stage = _load_caption_stage_for_profile_ui(trigger_word, input_folder)
        matching = None
        for candidate in (stage.get("all_rows", []) or []):
            if not isinstance(candidate, dict):
                continue
            candidate_name = str(candidate.get("original_filename", "") or "")
            candidate_member = str(candidate.get("profile_member_id", "") or "")
            candidate_image = str(candidate.get("profile_image_id", "") or "")
            matches_filename = bool(filename and candidate_name == filename)
            matches_full_id = bool(image_id and image_id in {candidate_member, candidate_image})
            matches_content_id = bool(content_image_id and content_image_id in {candidate_member, candidate_image})
            if matches_filename or matches_full_id or matches_content_id:
                matching = candidate
                break
        if matching:
            source = str(matching.get("original_path", "") or "")
            if source and not os.path.isfile(source):
                hazards.append("Quelldatei fehlt")
            if bool(matching.get("multiple_people")):
                hazards.append("mehrere Personen erkannt")
            if str(matching.get("arcface_flag", "") or "").lower() == "hard":
                hazards.append("ArcFace Hard-Flag")
            reason = str(matching.get("short_reason", "") or "").lower()
            duplicate_method = str(matching.get("duplicate_method", "") or "").lower()
            if "duplicate" in reason or duplicate_method or matching.get("duplicate_of"):
                hazards.append("Duplikat-/Near-Duplicate-Hinweis")
            if reason.startswith(("hard_pass_too_small", "filesize_too_small", "script_error")):
                hazards.append(f"technischer Ausschluss: {reason}")
        if hazards and not bool(confirm_hazards):
            warning = tr(
                "⚠️ Priority-Warnung für `" + (filename or image_id) + "`: " + ", ".join(hazards) + ". Aktiviere die Bestätigung unter den Buttons und klicke erneut, wenn das Bild trotzdem zwingend ins Training soll.",
                "⚠️ Priority warning for `" + (filename or image_id) + "`: " + ", ".join(hazards) + ". Enable the confirmation below the buttons and click again to force it into training anyway.",
            )
            return profile, json.dumps(profile, ensure_ascii=False, indent=2), _identity_cluster_preview_markdown(profile, cid, page=gallery_page), _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cid, page=gallery_page), warning, warning

    ids = [str(x) for x in (profile.get("priority_image_ids", []) or []) if str(x)]
    names = [str(x) for x in (profile.get("priority_image_filenames", []) or []) if str(x)]
    if make_priority:
        if content_image_id and content_image_id not in ids:
            ids.append(content_image_id)
        if filename and filename not in names:
            names.append(filename)
    else:
        ids = [x for x in ids if x not in {image_id, content_image_id}]
        names = [x for x in names if x != filename]
    profile["priority_image_ids"] = ids
    profile["priority_image_filenames"] = names
    profile["profile_rebuild_required"] = True
    profile["profile_rebuild_reason"] = "Priority image selection changed"
    notes = profile.setdefault("normalizer_notes", [])
    if isinstance(notes, list):
        notes.append(f"User {'marked' if make_priority else 'unmarked'} Priority image {filename or image_id} at {time.strftime('%Y-%m-%dT%H:%M:%S')}.")
        profile["normalizer_notes"] = notes[-40:]
    _save_profile_bucket_edit(profile, path)
    selected_info = (
        f"**{tr('Ausgewähltes Bild', 'Selected image')}:** `{filename}`  \n"
        f"**Image ID:** `{image_id}`  \n"
        f"**Priority:** {'⭐ yes' if make_priority else 'no'}"
    )
    if reject_reason:
        selected_info += f"  \n**{tr('Reject-Grund', 'Reject reason')}:** `{reject_reason}`"
    status = tr(
        f"✅ `{filename}` ist jetzt {'Priority und wird zwingend in Train Ready übernommen' if make_priority else 'nicht mehr Priority'}.",
        f"✅ `{filename}` is now {'Priority and will be forced into Train Ready' if make_priority else 'no longer Priority'}.",
    )
    return profile, json.dumps(profile, ensure_ascii=False, indent=2), _identity_cluster_preview_markdown(profile, cid, page=gallery_page), _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cid, page=gallery_page), selected_info, status


def _table_records_from_gradio(table_data: Any) -> List[Dict[str, Any]]:
    """Normalize gr.Dataframe payloads across Gradio versions."""
    if table_data is None:
        return []
    try:
        if hasattr(table_data, "to_dict"):
            return list(table_data.to_dict("records"))
    except Exception:
        pass
    if isinstance(table_data, dict) and "data" in table_data:
        headers = table_data.get("headers") or ["cluster_id", "role", "n", "summary", "avg_quality_total", "avg_identity_usefulness"]
        records: List[Dict[str, Any]] = []
        for row in table_data.get("data") or []:
            if isinstance(row, dict):
                records.append(row)
            else:
                records.append({headers[i]: row[i] if i < len(row) else None for i in range(len(headers))})
        return records
    if isinstance(table_data, list):
        # Some Gradio versions return a raw list of rows.
        records = []
        headers = ["cluster_id", "role", "n", "summary", "avg_quality_total", "avg_identity_usefulness"]
        for row in table_data:
            if isinstance(row, dict):
                records.append(row)
            elif isinstance(row, (list, tuple)):
                records.append({headers[i]: row[i] if i < len(row) else None for i in range(len(headers))})
        return records
    return []


def preview_identity_cluster_from_dropdown_ui(
    profile: Dict[str, Any],
    trigger_word: str,
    input_folder: str,
    cluster_id: str,
) -> Tuple[str, List[Tuple[Any, str]], Any]:
    profile = profile or {}
    return (
        _identity_cluster_preview_markdown(profile, cluster_id, page=1),
        _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cluster_id, page=1),
        _cluster_gallery_page_update(profile, cluster_id, 1),
    )


def _identity_cluster_id_from_table_select(
    profile: Dict[str, Any],
    table_data: Any,
    evt: Optional[gr.SelectData],
) -> str:
    """Resolve the selected identity-cluster id from a Gradio Dataframe click.

    Different Gradio versions expose Dataframe selection data differently:
    - evt.index may be (row, col), row, or sometimes a string-like row key.
    - evt.value may be the selected cell, a row dict, or a row list.
    - table_data may be a pandas DataFrame, a dict payload, or a list of rows.

    The previous implementation depended too strongly on table_data. When that
    payload was not populated in the expected shape, the preview silently fell
    back to the first cluster, so every click showed the same images.
    """
    profile = profile or {}
    clusters = _sorted_identity_clusters(profile)
    cluster_ids = [
        str(c.get("cluster_id", "") or "")
        for c in clusters
        if isinstance(c, dict) and str(c.get("cluster_id", "") or "")
    ]

    def _valid(value: Any) -> str:
        s = str(value or "").strip()
        return s if s in cluster_ids else ""

    # 1) Some Gradio versions pass the selected row or cell value.
    value = getattr(evt, "value", None) if evt is not None else None
    if isinstance(value, dict):
        cid = _valid(value.get("cluster_id"))
        if cid:
            return cid
    elif isinstance(value, (list, tuple)) and value:
        cid = _valid(value[0])
        if cid:
            return cid
    else:
        cid = _valid(value)
        if cid:
            return cid

    # 2) Resolve row index from evt.index.
    row_idx: Optional[int] = None
    idx = getattr(evt, "index", None) if evt is not None else None
    try:
        first = idx[0] if isinstance(idx, (tuple, list)) and idx else idx
        # If a future/alternate Gradio build returns the cluster id as row key,
        # accept it directly.
        cid = _valid(first)
        if cid:
            return cid
        if first is not None:
            row_idx = int(first)
    except Exception:
        row_idx = None

    # 3) Prefer the live table payload if available, because the user may have
    # edited role values. The cluster_id column itself should still match.
    records = _table_records_from_gradio(table_data)
    if row_idx is not None and 0 <= row_idx < len(records):
        cid = _valid(records[row_idx].get("cluster_id", ""))
        if cid:
            return cid

    # 4) Robust fallback: use the same row index against the profile cluster
    # order. This keeps selection working even when Gradio does not pass the
    # Dataframe value back in a parseable shape.
    if row_idx is not None and 0 <= row_idx < len(clusters):
        c = clusters[row_idx]
        if isinstance(c, dict):
            cid = _valid(c.get("cluster_id"))
            if cid:
                return cid

    # 5) Final fallback: first cluster, preserving previous startup behavior.
    first = _identity_cluster_by_id(profile)
    return str(first.get("cluster_id", "") or "") if first else ""


IDENTITY_CLUSTER_ROLE_CHOICES = ["core", "variation", "body_reference", "review", "exclude"]


def _identity_cluster_role_for_id(profile: Dict[str, Any], cluster_id: str) -> str:
    c = _identity_cluster_by_id(profile or {}, cluster_id)
    role = str((c or {}).get("role", "variation") or "variation").strip().lower()
    return role if role in IDENTITY_CLUSTER_ROLE_CHOICES else "variation"


def preview_identity_cluster_from_table_ui(
    profile: Dict[str, Any],
    trigger_word: str,
    input_folder: str,
    table_data: Any,
    evt: gr.SelectData,
) -> Tuple[str, List[Tuple[Any, str]], str, Any, Any]:
    """Open the first stable gallery page for a selected cluster."""
    profile = profile or {}
    cluster_id = _identity_cluster_id_from_table_select(profile, table_data, evt)
    role = _identity_cluster_role_for_id(profile, cluster_id)
    return (
        _identity_cluster_preview_markdown(profile, cluster_id, page=1),
        _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cluster_id, page=1),
        cluster_id,
        gr.update(choices=IDENTITY_CLUSTER_ROLE_CHOICES, value=role),
        _cluster_gallery_page_update(profile, cluster_id, 1),
    )


def refresh_identity_cluster_gallery_page_ui(
    profile: Dict[str, Any],
    trigger_word: str,
    input_folder: str,
    cluster_id: str,
    gallery_page: Any,
) -> Tuple[str, List[Tuple[Any, str]], str, str]:
    """Reload one gallery page and clear any stale image selection."""
    profile = profile or {}
    page = _parse_cluster_gallery_page(gallery_page)
    return (
        _identity_cluster_preview_markdown(profile, cluster_id, page=page),
        _identity_cluster_preview_gallery(profile, trigger_word, input_folder, cluster_id, page=page),
        "",
        tr("_Kein Bild ausgewählt._", "_No image selected._"),
    )


def _cluster_table_rows_from_records(records: List[Dict[str, Any]]) -> List[List[Any]]:
    headers = ["cluster_id", "role", "n", "summary", "avg_quality_total", "avg_identity_usefulness"]
    rows: List[List[Any]] = []
    for rec in records:
        rows.append([rec.get(h, "") for h in headers])
    return rows


def apply_identity_cluster_role_to_table_ui(
    table_data: Any,
    selected_cluster_id: str,
    selected_role: str,
) -> List[List[Any]]:
    """Apply the role dropdown value to the currently selected cluster row."""
    records = _table_records_from_gradio(table_data)
    cid = str(selected_cluster_id or "").strip()
    role = str(selected_role or "").strip().lower()
    if not records or not cid or role not in IDENTITY_CLUSTER_ROLE_CHOICES:
        return _cluster_table_rows_from_records(records)
    for rec in records:
        if str(rec.get("cluster_id", "") or "").strip() == cid:
            rec["role"] = role
            break
    return _cluster_table_rows_from_records(records)


def _apply_identity_cluster_role_to_profile(
    profile: Dict[str, Any],
    selected_cluster_id: str,
    selected_role: str,
) -> Tuple[Dict[str, Any], bool]:
    """Update identity cluster roles inside the in-memory profile state.

    This avoids relying on gr.Dataframe cell editing, which is inconsistent
    across Gradio versions. The dropdown becomes the source of truth for the
    selected row and is also merged again during Save.
    """
    profile = profile if isinstance(profile, dict) else {}
    cid = str(selected_cluster_id or "").strip()
    role = str(selected_role or "").strip().lower()
    if not cid or role not in IDENTITY_CLUSTER_ROLE_CHOICES:
        return profile, False

    clusters = profile.get("identity_clusters", []) or []
    if not isinstance(clusters, list):
        return profile, False

    changed = False
    for c in clusters:
        if not isinstance(c, dict):
            continue
        if str(c.get("cluster_id", "") or "").strip() == cid:
            if str(c.get("role", "") or "").strip().lower() != role:
                changed = True
            c["role"] = role
            member_roles = profile.setdefault("identity_cluster_member_roles", {})
            if isinstance(member_roles, dict):
                for mid in c.get("members", []) or []:
                    member_roles[str(mid)] = role
            overrides = profile.setdefault("identity_cluster_role_overrides", {})
            if isinstance(overrides, dict):
                overrides[cid] = role
            break

    profile["identity_clusters"] = clusters
    return profile, changed


def apply_identity_cluster_role_selection_ui(
    profile: Dict[str, Any],
    table_data: Any,
    selected_cluster_id: str,
    selected_role: str,
) -> Tuple[Dict[str, Any], List[List[Any]], str, str]:
    """Apply the role dropdown to the selected cluster and refresh the table.

    The UI previously relied on Dropdown.change updating only the visible
    Dataframe. On some Gradio builds the visual table did not persist the
    changed value. This function updates both the hidden profile state and the
    table rows, and the Save function also receives the selected dropdown value
    as a final override.
    """
    cid = str(selected_cluster_id or "").strip()
    role = str(selected_role or "").strip().lower()
    if cid == AUDITED_REJECT_CLUSTER_ID and role != "exclude":
        return (
            profile if isinstance(profile, dict) else {},
            _identity_clusters_table(profile if isinstance(profile, dict) else {}),
            _identity_cluster_preview_markdown(profile if isinstance(profile, dict) else {}, cid),
            tr(
                "ℹ️ Der auditierte Reject-Bucket bleibt fest auf `exclude`. Löse ein Einzelbild in einen neuen Bucket oder markiere es als Priority.",
                "ℹ️ The audited-reject bucket is fixed to `exclude`. Detach an individual image into a new bucket or mark it Priority.",
            ),
        )
    if not cid:
        return (
            profile if isinstance(profile, dict) else {},
            _cluster_table_rows_from_records(_table_records_from_gradio(table_data)),
            tr("_Kein Cluster ausgewählt._", "_No cluster selected._"),
            tr("⚠️ Erst links eine Cluster-Zeile anklicken.", "⚠️ Click a cluster row on the left first."),
        )
    if role not in IDENTITY_CLUSTER_ROLE_CHOICES:
        return (
            profile if isinstance(profile, dict) else {},
            _cluster_table_rows_from_records(_table_records_from_gradio(table_data)),
            _identity_cluster_preview_markdown(profile if isinstance(profile, dict) else {}, cid),
            tr("⚠️ Ungültige Rolle.", "⚠️ Invalid role."),
        )

    profile, changed = _apply_identity_cluster_role_to_profile(profile if isinstance(profile, dict) else {}, cid, role)

    # Build rows from profile state if available, otherwise fall back to the visible table.
    if isinstance(profile, dict) and profile.get("identity_clusters"):
        rows = _identity_clusters_table(profile)
    else:
        rows = apply_identity_cluster_role_to_table_ui(table_data, cid, role)

    status = tr(
        f"✅ Rolle übernommen: {cid} → {role}. Danach Cluster-Rollen speichern klicken.",
        f"✅ Role applied: {cid} → {role}. Click Save cluster roles afterwards.",
    )
    if not changed:
        status = tr(
            f"ℹ️ Rolle ist bereits gesetzt: {cid} → {role}. Danach Cluster-Rollen speichern klicken.",
            f"ℹ️ Role already set: {cid} → {role}. Click Save cluster roles afterwards.",
        )
    return profile, rows, _identity_cluster_preview_markdown(profile, cid), status


def _identity_clusters_table(profile: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for c in _sorted_identity_clusters(profile):
        rows.append([
            str(c.get("cluster_id", "")),
            str(c.get("role", "variation") or "variation"),
            int(c.get("n", 0) or 0),
            str(c.get("summary", "")),
            str(c.get("avg_quality_total", "")),
            str(c.get("avg_identity_usefulness", "")),
        ])
    return rows

def _identity_clusters_markdown(profile: Dict[str, Any]) -> str:
    clusters = _sorted_identity_clusters(profile)
    if not clusters:
        return tr(
            "_Keine Identity-Cluster im Profil. Starte den Curator mit aktualisierter Version neu oder baue das Profil neu._",
            "_No identity clusters in profile. Restart the curator with the updated version or rebuild the profile._",
        )
    role_counts: Dict[str, int] = {}
    for c in clusters:
        role = str(c.get("role", "variation") or "variation")
        role_counts[role] = role_counts.get(role, 0) + int(c.get("n", 0) or 0)
    warnings = [n for n in profile.get("normalizer_notes", []) or [] if "Identity clustering:" in str(n)]
    priority_count = len({str(x) for x in (profile.get("priority_image_ids", []) or []) if str(x)})
    audited_reject_count = sum(
        int(c.get("n", 0) or 0)
        for c in clusters
        if str(c.get("cluster_kind", "") or "") == "audited_rejects"
    )
    lines = [
        tr("### Identity-/Appearance-Cluster", "### Identity / appearance clusters"),
        "",
        tr(
            "Die Rollen sind **keine Caption-Tokens**. Sie steuern nur Phase 3 Export und Ranking: `core`, `variation`, `body_reference` gehen ins Training; `review` und `exclude` nicht.",
            "Roles are **not caption tokens**. They only control phase-3 export and ranking: `core`, `variation`, `body_reference` go into training; `review` and `exclude` do not.",
        ),
        "",
        "**" + tr("Bildrollen", "Image roles") + ":** " + ", ".join(f"`{k}`={v}" for k, v in sorted(role_counts.items())),
        f"**Priority:** {priority_count}  ",
        f"**{tr('Auditierte Rejects im letzten Bucket', 'Audited rejects in the last bucket')}:** {audited_reject_count}",
    ]
    if warnings:
        lines.append("")
        lines.append("**" + tr("Hinweise", "Notes") + ":**")
        for w in warnings[-8:]:
            lines.append(f"- {str(w).replace('Identity clustering: ', '')}")
    lines.extend([
        "",
        tr(
            "Klicke einen Cluster an und setze die Rolle über das Dropdown unter der Tabelle. Erlaubt sind: `core`, `variation`, `body_reference`, `review`, `exclude`.",
            "Click a cluster and set the role with the dropdown below the table. Allowed values: `core`, `variation`, `body_reference`, `review`, `exclude`.",
        ),
    ])
    return "\n".join(lines)


def save_identity_cluster_roles_ui(trigger_word: str, input_folder: str, table_data: Any, profile_state: Dict[str, Any], selected_cluster_id: str, selected_role: str) -> str:
    path = subject_profile_path_for(input_folder, trigger_word)
    if not os.path.exists(path):
        return tr(f"❌ Kein Profil unter {path}", f"❌ No profile at {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception as e:
        return tr(f"❌ Profil konnte nicht gelesen werden: {e}", f"❌ Could not read profile: {e}")

    # Gradio Dataframe kann je nach Version pandas.DataFrame, dict oder list liefern.
    records: List[Dict[str, Any]] = []
    try:
        if hasattr(table_data, "to_dict"):
            records = table_data.to_dict("records")
        elif isinstance(table_data, dict) and "data" in table_data:
            headers = table_data.get("headers") or ["cluster_id", "role", "n", "summary", "avg_quality_total", "avg_identity_usefulness"]
            for row in table_data.get("data") or []:
                records.append({str(headers[i]): row[i] if i < len(row) else "" for i in range(len(headers))})
        elif isinstance(table_data, list):
            for row in table_data:
                if isinstance(row, dict):
                    records.append(row)
                elif isinstance(row, (list, tuple)):
                    headers = ["cluster_id", "role", "n", "summary", "avg_quality_total", "avg_identity_usefulness"]
                    records.append({headers[i]: row[i] if i < len(row) else "" for i in range(min(len(headers), len(row)))})
    except Exception as e:
        return tr(f"❌ Tabellenformat nicht lesbar: {e}", f"❌ Could not parse table data: {e}")

    role_by_cluster: Dict[str, str] = {}
    invalid: List[str] = []
    for rec in records:
        cid = str(rec.get("cluster_id", "") or "").strip()
        role = str(rec.get("role", "") or "").strip().lower()
        if not cid:
            continue
        if role not in IDENTITY_CLUSTER_ROLE_CHOICES:
            invalid.append(f"{cid}: {role or '<empty>'}")
            continue
        role_by_cluster[cid] = "exclude" if cid == AUDITED_REJECT_CLUSTER_ID else role

    # Merge roles from hidden profile state. This is more reliable than relying
    # only on the visible Dataframe value on all Gradio versions.
    if isinstance(profile_state, dict):
        for c in profile_state.get("identity_clusters", []) or []:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("cluster_id", "") or "").strip()
            role = str(c.get("role", "") or "").strip().lower()
            if cid and role in IDENTITY_CLUSTER_ROLE_CHOICES:
                role_by_cluster[cid] = "exclude" if cid == AUDITED_REJECT_CLUSTER_ID else role

    # Final explicit override from the currently selected dropdown. This makes
    # Save work even if the user changed the dropdown and did not press Apply.
    selected_cid = str(selected_cluster_id or "").strip()
    selected_role_value = str(selected_role or "").strip().lower()
    if selected_cid and selected_role_value in IDENTITY_CLUSTER_ROLE_CHOICES:
        role_by_cluster[selected_cid] = "exclude" if selected_cid == AUDITED_REJECT_CLUSTER_ID else selected_role_value

    if invalid:
        return tr(
            "❌ Ungültige Rollen: " + ", ".join(invalid[:8]) + ". Erlaubt: " + ", ".join(IDENTITY_CLUSTER_ROLE_CHOICES),
            "❌ Invalid roles: " + ", ".join(invalid[:8]) + ". Allowed: " + ", ".join(IDENTITY_CLUSTER_ROLE_CHOICES),
        )

    clusters = profile.get("identity_clusters", []) or []
    member_roles: Dict[str, str] = profile.get("identity_cluster_member_roles", {}) or {}
    changed = 0
    for c in clusters:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("cluster_id", "") or "")
        if cid in role_by_cluster:
            new_role = role_by_cluster[cid]
            if c.get("role") != new_role:
                changed += 1
            c["role"] = new_role
            for mid in c.get("members", []) or []:
                member_roles[str(mid)] = new_role

    profile["identity_clusters"] = clusters
    profile["identity_cluster_member_roles"] = member_roles
    profile["identity_cluster_role_overrides"] = role_by_cluster
    if changed:
        profile["profile_rebuild_required"] = True
        profile["profile_rebuild_reason"] = "Identity cluster roles changed"
    notes = profile.setdefault("normalizer_notes", [])
    if isinstance(notes, list):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        notes.append(f"User-edited identity cluster roles via UI at {stamp}; changed_clusters={changed}.")
        profile["normalizer_notes"] = notes[-30:]

    try:
        core_atomic_write_json(path, profile)
        return tr(
            f"✅ Cluster-Rollen gespeichert ({changed} geändert). Das Identitätsprofil wird vor Phase 3 aus den vorhandenen Audits mit Priority > Core neu konsolidiert.",
            f"✅ Cluster roles saved ({changed} changed). Phase-3 export uses these roles for ranking and in/out selection.",
        )
    except Exception as e:
        return tr(f"❌ Speichern fehlgeschlagen: {e}", f"❌ Save failed: {e}")


def parse_progress(line: str) -> Optional[Tuple[int, int]]:
    # Nur echte Bildstart-Zeilen zaehlen, nicht Heartbeats/Subphasen wie
    # "START [12/80] local_subject_metrics ...". Dadurch bleiben Fortschritt,
    # Sekunden/Bild und ETA stabil.
    m = re.match(r"^\[(\d+)/(\d+)\]", line)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    if seconds < 60:
        return f"{seconds}s"

    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours, mins = divmod(minutes, 60)
    if secs >= 30:
        mins += 1
        if mins == 60:
            hours += 1
            mins = 0
    return f"{hours}h {mins}m"


def parse_openai_usage_delta(line: str) -> Optional[Tuple[int, int, int, int]]:
    m = re.search(
        r"OpenAI usage:\s*req\+=([\d,]+)\s*\|\s*in\+=([\d,]+)\s*\|\s*out\+=([\d,]+)\s*\|\s*total\+=([\d,]+)",
        line,
    )
    if not m:
        return None
    try:
        req = int(m.group(1).replace(",", ""))
        inp = int(m.group(2).replace(",", ""))
        out = int(m.group(3).replace(",", ""))
        total = int(m.group(4).replace(",", ""))
        return req, inp, out, total
    except Exception:
        return None


def format_openai_usage_text(requests_count: int, input_tokens: int, output_tokens: int, total_tokens: int) -> str:
    if requests_count <= 0 and total_tokens <= 0:
        return tr("💰 0 Requests | 0 Tokens", "💰 0 requests | 0 tokens")
    return tr(
        f"💰 {requests_count:,} Requests | {total_tokens:,} Tokens | In {input_tokens:,} | Out {output_tokens:,}",
        f"💰 {requests_count:,} requests | {total_tokens:,} tokens | in {input_tokens:,} | out {output_tokens:,}",
    )


def _terminate_pid_tree(pid: int, proc: Optional[subprocess.Popen] = None, graceful_timeout: float = 2.0) -> bool:
    """Terminate a PID and its descendants. Works even if the Popen global was lost."""
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False

    if proc is not None and proc.poll() is not None:
        return True
    if proc is None and not _pid_is_running(pid):
        return True

    terminated = False
    if sys.platform == "win32":
        try:
            # Do not decode taskkill output. Its return code is sufficient,
            # and Windows may emit bytes that are invalid in cp1252/UTF-8.
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(3.0, graceful_timeout + 1.0),
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            terminated = result.returncode == 0 or not _pid_is_running(pid)
        except Exception:
            terminated = False

        # taskkill can fail in unusual shells or permission contexts.  Keep a
        # direct Popen kill and PowerShell Stop-Process as independent fallbacks.
        if not terminated and proc is not None:
            try:
                proc.kill()
                terminated = True
            except Exception:
                pass
        if not terminated:
            try:
                result = subprocess.run(
                    [
                        "powershell", "-NoProfile", "-NonInteractive",
                        "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                terminated = result.returncode == 0 or not _pid_is_running(pid)
            except Exception:
                pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            terminated = True
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
                terminated = True
            except Exception:
                pass

    deadline = time.time() + max(0.2, graceful_timeout)
    while time.time() < deadline:
        if proc is not None:
            if proc.poll() is not None:
                return True
        elif not _pid_is_running(pid):
            return True
        time.sleep(0.05)

    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    elif proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass

    if proc is not None:
        return proc.poll() is not None
    return not _pid_is_running(pid)


def _terminate_process_tree(proc: subprocess.Popen, graceful_timeout: float = 2.0) -> bool:
    return _terminate_pid_tree(proc.pid, proc=proc, graceful_timeout=graceful_timeout)


def kill_process():
    """Cooperatively request cancellation, then force-stop the registered PID tree."""
    _cancel_requested.set()

    with _active_process_lock:
        proc = _active_process

    state = _read_active_run_state()
    cancel_file = str(state.get("cancel_file") or "")
    marker_written = _touch_cancel_marker(cancel_file)

    pids: List[int] = []
    if proc is not None:
        pids.append(int(proc.pid))
    try:
        state_pid = int(state.get("pid", 0) or 0)
        if state_pid > 0 and state_pid not in pids:
            pids.append(state_pid)
    except Exception:
        pass

    if not pids:
        _cleanup_stale_configs()
        return tr(
            "⏹ Kein aktiver Prozess registriert. Abbruchsignal wurde gesetzt.",
            "⏹ No active process registered. Cancellation marker was set.",
        )

    stopped = False
    for pid in pids:
        matching_proc = proc if proc is not None and int(proc.pid) == pid else None
        stopped = _terminate_pid_tree(pid, proc=matching_proc, graceful_timeout=1.5) or stopped

    if stopped:
        return tr(
            f"⏹ Abgebrochen – Prozessbaum beendet (PID {pids[0]}).",
            f"⏹ Cancelled – process tree stopped (PID {pids[0]}).",
        )
    marker_note_de = "; kooperatives Signal gesetzt" if marker_written else ""
    marker_note_en = "; cooperative marker written" if marker_written else ""
    return tr(
        f"⚠️ Abbruch angefordert, Prozess konnte noch nicht bestätigt beendet werden{marker_note_de}.",
        f"⚠️ Cancellation requested, process termination not yet confirmed{marker_note_en}.",
    )


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

    with _active_process_lock:
        existing = _active_process
        if existing is not None and existing.poll() is None:
            yield "", [], 0.0, tr(
                "⚠️ Es läuft bereits ein Prozess. Bitte zuerst abbrechen oder warten.",
                "⚠️ A process is already running. Cancel it or wait for it to finish.",
            ), format_openai_usage_text(0, 0, 0, 0)
            return

    run_id = uuid.uuid4().hex
    cancel_file = os.path.join(SCRIPT_DIR, f"_cancel_{run_id}.flag")
    try:
        if os.path.exists(cancel_file):
            os.remove(cancel_file)
    except Exception:
        pass

    # Never mutate the shared UI dictionary in-place.  The child receives a
    # unique cooperative cancellation marker for exactly this run.
    config_data = normalize_run_config_payload(config_data)
    config_data["RUN_ID"] = run_id
    config_data["CANCEL_FILE"] = cancel_file

    core_atomic_write_json(config_path, config_data)

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
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        # Gives the runner its own process group so cancellation can also stop
        # descendants such as ffmpeg.
        popen_kwargs["start_new_session"] = True

    _cancel_requested.clear()
    proc = subprocess.Popen(
        [VENV_PYTHON, script_path],
        **popen_kwargs,
    )
    with _active_process_lock:
        _active_process = proc
    _write_active_run_state(proc, run_id, cancel_file, config_path)

    log_lines: List[str] = []
    progress = 0.0
    last_gallery_update = 0
    images: List[Any] = []
    run_started_at = time.time()
    last_progress_idx = 0
    last_progress_total = 0
    seconds_per_item: Optional[float] = None
    eta_seconds: Optional[float] = None
    openai_requests = 0
    openai_input_tokens = 0
    openai_output_tokens = 0
    openai_total_tokens = 0
    openai_usage_text = format_openai_usage_text(0, 0, 0, 0)
    rc: Optional[int] = None

    try:
        stdout = proc.stdout
        if stdout is None:
            raise RuntimeError("Subprocess stdout is unavailable")

        for line in stdout:
            if _cancel_requested.is_set():
                break

            line = line.rstrip("\n\r")
            log_lines.append(line)

            usage_delta = parse_openai_usage_delta(line)
            if usage_delta:
                delta_req, delta_in, delta_out, delta_total = usage_delta
                openai_requests += delta_req
                openai_input_tokens += delta_in
                openai_output_tokens += delta_out
                openai_total_tokens += delta_total
                openai_usage_text = format_openai_usage_text(
                    openai_requests,
                    openai_input_tokens,
                    openai_output_tokens,
                    openai_total_tokens,
                )

            p = parse_progress(line)
            if p:
                idx, total = p
                progress = idx / max(1, total)
                last_progress_idx = idx
                last_progress_total = total

                elapsed = max(0.0, time.time() - run_started_at)
                if idx > 0:
                    seconds_per_item = elapsed / idx
                    remaining_items = max(0, total - idx)
                    eta_seconds = remaining_items * seconds_per_item

            processed_count = sum(1 for l in log_lines if re.match(r"\s*\[\d+/\d+\]", l))
            if image_scan_folder and processed_count - last_gallery_update >= 5:
                images = load_gallery_images(scan_images(image_scan_folder))
                last_gallery_update = processed_count

            log_text = "\n".join(log_lines[-500:])
            if last_progress_total > 0:
                status_de = (
                    f"⏳ Läuft... {last_progress_idx}/{last_progress_total} "
                    f"({int(progress*100)}%)"
                )
                status_en = (
                    f"⏳ Running... {last_progress_idx}/{last_progress_total} "
                    f"({int(progress*100)}%)"
                )
                if seconds_per_item is not None:
                    status_de += f" | {seconds_per_item:.1f} s/Bild"
                    status_en += f" | {seconds_per_item:.1f} s/image"
                if eta_seconds is not None and last_progress_idx < last_progress_total:
                    status_de += f" | Rest ca. {format_duration(eta_seconds)}"
                    status_en += f" | ETA {format_duration(eta_seconds)}"
            else:
                status_de = f"⏳ Läuft... ({int(progress*100)}%)"
                status_en = f"⏳ Running... ({int(progress*100)}%)"

            yield log_text, images, progress, tr(status_de, status_en), openai_usage_text

    except Exception as e:
        if not _cancel_requested.is_set():
            log_lines.append(tr(f"\n⚠️ Fehler: {e}", f"\n⚠️ Error: {e}"))
    finally:
        if _cancel_requested.is_set() and proc.poll() is None:
            _terminate_process_tree(proc)

        try:
            rc = proc.wait(timeout=10)
        except Exception:
            _terminate_process_tree(proc, graceful_timeout=1.0)
            rc = proc.poll()
            if rc is None:
                rc = -1

        with _active_process_lock:
            if _active_process is proc:
                _active_process = None

        _clear_active_run_state(expected_pid=proc.pid)
        try:
            if os.path.exists(cancel_file):
                os.remove(cancel_file)
        except Exception:
            pass
        try:
            if os.path.exists(config_path):
                os.remove(config_path)
        except Exception:
            pass

    if image_scan_folder:
        images = load_gallery_images(scan_images(image_scan_folder))

    log_text = "\n".join(log_lines[-500:])
    total_elapsed = max(0.0, time.time() - run_started_at)
    avg_seconds_per_item = None
    if last_progress_idx > 0:
        avg_seconds_per_item = total_elapsed / last_progress_idx

    was_cancelled = _cancel_requested.is_set()
    if was_cancelled:
        status = tr(
            f"⏹ Abgebrochen nach {format_duration(total_elapsed)}.",
            f"⏹ Cancelled after {format_duration(total_elapsed)}.",
        )
        final_progress = progress
    elif rc == 0:
        status = tr(
            (
                f"✅ Fertig! {last_progress_idx}/{last_progress_total} Bilder | "
                f"Ø {avg_seconds_per_item:.1f} s/Bild | Gesamt {format_duration(total_elapsed)}"
            )
            if avg_seconds_per_item is not None and last_progress_total > 0
            else f"✅ Fertig! ({len(log_lines)} Zeilen)",
            (
                f"✅ Done! {last_progress_idx}/{last_progress_total} images | "
                f"avg {avg_seconds_per_item:.1f} s/image | total {format_duration(total_elapsed)}"
            )
            if avg_seconds_per_item is not None and last_progress_total > 0
            else f"✅ Done! ({len(log_lines)} lines)",
        )
        final_progress = 1.0
    else:
        status = tr(f"❌ Fehlercode {rc}", f"❌ Exit code {rc}")
        final_progress = progress

    # Clear only after the final status has been derived.  A new run also
    # clears the flag before launching, so this cannot leak across runs.
    _cancel_requested.clear()
    yield log_text, images, final_progress, status, openai_usage_text


# ============================================================
# CURATOR LAUNCHER
# ============================================================

CURATOR_INPUT_NAMES = (
    "trigger_word", "input_folder", "target_size", "api_key", "ai_model", "audit_reasoning_effort", "openai_token_limit", "use_trigger_check", "trigger_check_model", "trigger_reasoning_effort",
    "use_review_escalation", "review_escalation_model", "review_escalation_reasoning_effort", "review_escalation_score_min", "review_escalation_score_max",
    "escalate_on_review", "escalate_on_conflict", "escalate_smart_crop", "smart_crop_escalation_delta",
    "ratio_h", "ratio_m", "ratio_f", "keep_score_min", "hard_reject_score", "hard_min_side",
    "use_filesize_filter", "min_filesize_kb", "use_blur_filter", "min_blur_variance", "face_min_blur_variance", "blur_norm_edge",
    "face_min_blur_variance_headshot", "face_min_blur_variance_medium", "face_min_blur_variance_full_body",
    "use_early_phash", "use_early_phash_loop1", "early_phash_threshold_1", "early_phash_keep_per_group_1",
    "use_early_phash_loop2", "early_phash_threshold_2", "early_phash_keep_per_group_2", "subject_sanity", "subject_min_torso",
    "ig_frame_crop", "ig_two_stage_bar", "frame_cleanup_mode", "frame_pause_on_medium", "use_clip", "use_phash", "phash_threshold", "clip_threshold",
    "enable_smart_crop", "crop_min_gain", "crop_padding", "enable_medium_rescue_crop", "medium_rescue_min_gain",
    "use_clustering", "max_outfit", "max_session", "use_diversity", "c_use_canon_representation", "c_canon_representation_target", "c_canon_max_quality_gap",
    "use_pose_diversity", "pose_soft_limit", "pose_penalty_weight", "use_arcface", "arcface_hard", "arcface_soft", "arcface_trim",
    "arcface_min_faces", "arcface_model", "arcface_det_size", "training_target", "caption_options", "variable_feature_mode",
    "krea_caption_model", "krea_caption_reasoning_effort", "use_krea_caption_repair", "krea_caption_repair_model", "krea_caption_repair_reasoning_effort",
    "c_pipeline_mode", "c_profile_normalizer_model", "profile_reasoning_effort", "c_profile_sample_threshold", "c_profile_sample_size",
    "export_review", "export_reject", "export_crop_compare", "controlled_buckets",
)


def _ui_values_dict(values: Tuple[Any, ...]) -> Dict[str, Any]:
    if len(values) != len(CURATOR_INPUT_NAMES):
        raise ValueError(
            f"Curator UI contract mismatch: got {len(values)} values, expected {len(CURATOR_INPUT_NAMES)}."
        )
    return dict(zip(CURATOR_INPUT_NAMES, values))


def build_run_config_from_ui_values(values: Tuple[Any, ...], *, continue_from_profile: bool = False) -> Dict[str, Any]:
    """Single source of truth for both full runs and Phase-3 continuation."""
    v = _ui_values_dict(values)
    all_caption_keys = list(CAPTION_FIELD_CHOICES)
    selected_caption_options = set(v.get("caption_options") or [])
    caption_policy = {key: key in selected_caption_options for key in all_caption_keys}
    target = normalize_training_target(v.get("training_target"))
    config = {
        "TRIGGER_WORD": str(v["trigger_word"] or "").strip(),
        "INPUT_FOLDER": str(v["input_folder"] or "").strip(),
        "TARGET_DATASET_SIZE": int(v["target_size"]),
        "API_KEY": str(v["api_key"] or "").strip(),
        "AI_MODEL": str(v["ai_model"] or "").strip(),
        "AUDIT_REASONING_EFFORT": str(v["audit_reasoning_effort"] or "none"),
        "OPENAI_TOKEN_LIMIT_TOTAL": int(v["openai_token_limit"] or 0),
        "USE_AI_TRIGGERWORD_CHECK": bool(v["use_trigger_check"]),
        "TRIGGER_CHECK_MODEL": str(v["trigger_check_model"] or v["ai_model"] or "").strip(),
        "TRIGGER_CHECK_REASONING_EFFORT": str(v["trigger_reasoning_effort"] or "none"),
        "USE_REVIEW_ESCALATION": bool(v["use_review_escalation"]),
        "REVIEW_ESCALATION_MODEL": str(v["review_escalation_model"] or "").strip(),
        "REVIEW_ESCALATION_REASONING_EFFORT": str(v["review_escalation_reasoning_effort"] or "low"),
        "REVIEW_ESCALATION_SCORE_MIN": int(v["review_escalation_score_min"]),
        "REVIEW_ESCALATION_SCORE_MAX": int(v["review_escalation_score_max"]),
        "ESCALATE_ON_REVIEW_STATUS": bool(v["escalate_on_review"]),
        "ESCALATE_ON_STATUS_CONFLICT": bool(v["escalate_on_conflict"]),
        "ESCALATE_SMART_CROP_CLOSE_CALLS": bool(v["escalate_smart_crop"]),
        "SMART_CROP_ESCALATION_MAX_DELTA": float(v["smart_crop_escalation_delta"]),
        "RATIO_HEADSHOT": round(float(v["ratio_h"]), 2),
        "RATIO_MEDIUM": round(float(v["ratio_m"]), 2),
        "RATIO_FULL_BODY": round(float(v["ratio_f"]), 2),
        "KEEP_SCORE_MIN": int(v["keep_score_min"]),
        "HARD_REJECT_SCORE": int(v["hard_reject_score"]),
        "HARD_MIN_SIDE_PX": int(v["hard_min_side"]),
        "USE_MIN_FILESIZE_FILTER": bool(v["use_filesize_filter"]),
        "HARD_MIN_FILESIZE_KB": int(v["min_filesize_kb"]),
        "USE_BLUR_FILTER": bool(v["use_blur_filter"]),
        "HARD_MIN_BLUR_VARIANCE": float(v["min_blur_variance"]),
        "FACE_MIN_BLUR_VARIANCE": float(v["face_min_blur_variance"]),
        "FACE_MIN_BLUR_VARIANCE_HEADSHOT": float(v["face_min_blur_variance_headshot"]),
        "FACE_MIN_BLUR_VARIANCE_MEDIUM": float(v["face_min_blur_variance_medium"]),
        "FACE_MIN_BLUR_VARIANCE_FULL_BODY": float(v["face_min_blur_variance_full_body"]),
        "BLUR_NORMALIZE_LONG_EDGE": int(v["blur_norm_edge"]),
        "ENABLE_SUBJECT_SANITY_CHECK": bool(v["subject_sanity"]),
        "SUBJECT_MIN_TORSO_LANDMARKS": int(v["subject_min_torso"]),
        "ENABLE_IG_FRAME_CROP": bool(v["ig_frame_crop"]),
        "IG_FRAME_TWO_STAGE_BAR_DETECT": bool(v["ig_two_stage_bar"]),
        "FRAME_CLEANUP_MODE": str(v["frame_cleanup_mode"] or "suggest_only"),
        "FRAME_PAUSE_ON_MEDIUM_REVIEW": bool(v["frame_pause_on_medium"]),
        "USE_EARLY_PHASH_DEDUP": bool(v["use_early_phash"]),
        "USE_EARLY_PHASH_LOOP1": bool(v["use_early_phash_loop1"]),
        "EARLY_PHASH_HAMMING_THRESHOLD_1": int(v["early_phash_threshold_1"]),
        "EARLY_PHASH_KEEP_PER_GROUP_1": int(v["early_phash_keep_per_group_1"]),
        "USE_EARLY_PHASH_LOOP2": bool(v["use_early_phash_loop2"]),
        "EARLY_PHASH_HAMMING_THRESHOLD_2": int(v["early_phash_threshold_2"]),
        "EARLY_PHASH_KEEP_PER_GROUP_2": int(v["early_phash_keep_per_group_2"]),
        "USE_CLIP_DUPLICATE_SCORING": bool(v["use_clip"]),
        "USE_PHASH_DUPLICATE_SCORING": bool(v["use_phash"]),
        "PHASH_HAMMING_THRESHOLD": int(v["phash_threshold"]),
        "CLIP_COSINE_THRESHOLD": float(v["clip_threshold"]),
        "ENABLE_SMART_PRECROP": bool(v["enable_smart_crop"]),
        "SMART_PRECROP_MIN_GAIN": float(v["crop_min_gain"]),
        "SMART_PRECROP_PADDING_FACTOR": float(v["crop_padding"]),
        "ENABLE_MEDIUM_RESCUE_CROP": bool(v["enable_medium_rescue_crop"]),
        "MEDIUM_RESCUE_MIN_GAIN": float(v["medium_rescue_min_gain"]),
        "USE_SESSION_OUTFIT_CLUSTERING": bool(v["use_clustering"]),
        "MAX_PER_OUTFIT_CLUSTER": int(v["max_outfit"]),
        "MAX_PER_SESSION_CLUSTER": int(v["max_session"]),
        "ENABLE_DIVERSITY_PENALTIES": bool(v["use_diversity"]),
        "ENABLE_CANON_REPRESENTATION_BONUS": bool(v["c_use_canon_representation"]),
        "CANON_REPRESENTATION_TARGET": int(v["c_canon_representation_target"]),
        "CANON_REPRESENTATION_MAX_QUALITY_GAP": float(v["c_canon_max_quality_gap"]),
        "ENABLE_POSE_DIVERSITY": bool(v["use_pose_diversity"]),
        "POSE_DIVERSITY_SOFT_LIMIT": int(v["pose_soft_limit"]),
        "POSE_DIVERSITY_PENALTY_WEIGHT": float(v["pose_penalty_weight"]),
        "USE_ARCFACE_IDENTITY_CHECK": bool(v["use_arcface"]),
        "ARCFACE_HARD_THRESHOLD": float(v["arcface_hard"]),
        "ARCFACE_SOFT_THRESHOLD": float(v["arcface_soft"]),
        "ARCFACE_TRIM_FRACTION": float(v["arcface_trim"]),
        "ARCFACE_MIN_FACES_FOR_CENTROID": int(v["arcface_min_faces"]),
        "ARCFACE_MODEL_PACK": str(v["arcface_model"] or "buffalo_l").strip(),
        "ARCFACE_DET_SIZE": int(v["arcface_det_size"]),
        "TRAINING_TARGET": target,
        "CAPTION_PROFILE": caption_profile_for_training_target(target),
        "CAPTION_POLICY": caption_policy,
        "VARIABLE_FEATURE_CAPTION_MODE": str(v["variable_feature_mode"] or "canonical_deviations"),
        "USE_KREA_AI_CAPTIONING": target == "krea2",
        "KREA_CAPTION_MODEL": str(v["krea_caption_model"] or "gpt-5.6-luna").strip(),
        "KREA_CAPTION_REASONING_EFFORT": str(v["krea_caption_reasoning_effort"] or "none"),
        "USE_KREA_CAPTION_REPAIR": bool(v["use_krea_caption_repair"]),
        "KREA_CAPTION_REPAIR_MODEL": str(v["krea_caption_repair_model"] or "gpt-5.6-terra").strip(),
        "KREA_CAPTION_REPAIR_REASONING_EFFORT": str(v["krea_caption_repair_reasoning_effort"] or "low"),
        "PIPELINE_MODE": "profile_then_caption" if continue_from_profile else str(v["c_pipeline_mode"] or "single_pass"),
        "CONTINUE_FROM_PROFILE": bool(continue_from_profile),
        "PROFILE_NORMALIZER_MODEL": str(v["c_profile_normalizer_model"] or "gpt-5.6-terra").strip(),
        "PROFILE_REASONING_EFFORT": str(v["profile_reasoning_effort"] or "low"),
        "PROFILE_SAMPLE_THRESHOLD": int(v["c_profile_sample_threshold"]),
        "PROFILE_SAMPLE_SIZE": int(v["c_profile_sample_size"]),
        "EXPORT_REVIEW_IMAGES": bool(v["export_review"]),
        "EXPORT_REJECT_IMAGES": bool(v["export_reject"]),
        "EXPORT_SMART_CROP_COMPARISON": bool(v["export_crop_compare"]),
        "USE_CONTROLLED_BUCKETS": bool(v["controlled_buckets"]),
        "SEND_TEXT_IMAGES_TO_CAPTION_REMOVE": True,
    }
    return normalize_run_config_payload(config)


def _validate_run_values(values: Tuple[Any, ...]) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Any, ...]]]:
    try:
        v = _ui_values_dict(values)
    except Exception as exc:
        return None, (str(exc), [], 0, tr("❌ Fehler", "❌ Error"), format_openai_usage_text(0, 0, 0, 0))
    trigger_word = str(v.get("trigger_word") or "").strip()
    input_folder = str(v.get("input_folder") or "").strip()
    api_key = str(v.get("api_key") or "").strip()
    if not trigger_word:
        return None, (tr("Bitte ein Triggerwort eingeben.", "Please enter a trigger word."), [], 0, tr("❌ Fehler", "❌ Error"), format_openai_usage_text(0, 0, 0, 0))
    if not os.path.isdir(input_folder):
        return None, (tr(f"Input-Ordner existiert nicht: {input_folder}", f"Input folder does not exist: {input_folder}"), [], 0, tr("❌ Fehler", "❌ Error"), format_openai_usage_text(0, 0, 0, 0))
    if not api_key:
        return None, (tr("Bitte einen OpenAI API Key eingeben.", "Please enter an OpenAI API key."), [], 0, tr("❌ Fehler", "❌ Error"), format_openai_usage_text(0, 0, 0, 0))
    return v, None


def start_curator(*values):
    v, error = _validate_run_values(values)
    if error:
        yield error
        return
    assert v is not None
    workspace = load_project_workspace(v["input_folder"], v["trigger_word"])
    preflight = dict(workspace.get("preflight") or {}) if isinstance(workspace, dict) else {}
    frame_state = dict(workspace.get("frame") or {}) if isinstance(workspace, dict) else {}
    if not preflight.get("completed"):
        yield tr(
            "Die lokalen Vorprüfungen wurden noch nicht abgeschlossen. Bitte zuerst auf der Startseite 'Vorprüfungen starten' ausführen.",
            "Local preflight has not been completed. Run 'Run preflight' on the start page first.",
        ), [], 0, tr("❌ Vorprüfung fehlt", "❌ Preflight missing"), format_openai_usage_text(0, 0, 0, 0)
        return
    if not _workspace_preflight_is_current(v["input_folder"], v["trigger_word"], workspace):
        yield tr(
            "Der Inhalt des Input-Ordners hat sich seit der Vorprüfung geändert. Bitte die Vorprüfung erneut ausführen, bevor Audit oder Rahmenanalyse fortgesetzt werden.",
            "The input folder changed after preflight. Rerun preflight before continuing with frame analysis or audit.",
        ), [], 0, tr("❌ Vorprüfung veraltet", "❌ Preflight stale"), format_openai_usage_text(0, 0, 0, 0)
        return
    if bool(frame_state.get("enabled")) and not bool(frame_state.get("analysis_completed")):
        yield tr(
            "Das Rahmenmodul ist aktiviert, wurde aber noch nicht analysiert. Bitte zuerst Modul 1 'Rahmen' ausführen oder das Rahmenmodul auf der Startseite deaktivieren.",
            "The frame module is enabled but has not been analysed yet. Run module 1 'Frames' first or disable frame processing on the start page.",
        ), [], 0, tr("❌ Rahmenanalyse fehlt", "❌ Frame analysis missing"), format_openai_usage_text(0, 0, 0, 0)
        return
    config = build_run_config_from_ui_values(values, continue_from_profile=False)
    train_dir = os.path.join(output_root_for(v["input_folder"], v["trigger_word"]), "01_train_ready")
    yield from run_script(CURATOR_SCRIPT, CURATOR_CONFIG, config, train_dir)


def start_caption_from_profile(*values):
    """Phase 3: caption/export from an already built and confirmed profile."""
    v, error = _validate_run_values(values)
    if error:
        yield error
        return
    assert v is not None
    stage_path = caption_stage_path_for(v["input_folder"], v["trigger_word"])
    profile_path = subject_profile_path_for(v["input_folder"], v["trigger_word"])
    if not os.path.exists(stage_path):
        yield tr(
            f"_caption_stage.json fehlt. Starte zuerst den Curator im Modus 'Profile then Caption'. Erwartet: {stage_path}",
            f"_caption_stage.json is missing. First run the curator in 'Profile then Caption' mode. Expected: {stage_path}",
        ), [], 0, tr("❌ Fehler", "❌ Error"), format_openai_usage_text(0, 0, 0, 0)
        return
    if not os.path.exists(profile_path):
        yield tr(f"_subject_profile.json fehlt. Erwartet: {profile_path}", f"_subject_profile.json is missing. Expected: {profile_path}"), [], 0, tr("❌ Fehler", "❌ Error"), format_openai_usage_text(0, 0, 0, 0)
        return
    config = build_run_config_from_ui_values(values, continue_from_profile=True)
    train_dir = os.path.join(output_root_for(v["input_folder"], v["trigger_word"]), "01_train_ready")
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
        ), [], 0, tr("❌ Fehler", "❌ Error"), ""
        return
    if not os.path.isfile(reference_image):
        yield tr(
            f"Referenzbild nicht gefunden: {reference_image}",
            f"Reference image not found: {reference_image}",
        ), [], 0, tr("❌ Fehler", "❌ Error"), ""
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

def load_results(input_folder, trigger_word, subfolder, page=1):
    root = output_root_for(input_folder, trigger_word)
    folder_map = {
        "train_ready": "01_train_ready",
        "keep_unused": "02_keep_unused",
        "caption_remove": "03_caption_remove",
        "review": "04_review",
        "reject": "05_reject",
        "manual_review": "06_needs_manual_review",
        "smart_crop_pairs": "08_smart_crop_pairs",
    }
    target = os.path.join(root, folder_map.get(str(subfolder), "01_train_ready"))
    all_paths = scan_images(target, limit=1_000_000)
    page_size = 60
    try:
        page = max(1, int(page or 1))
    except Exception:
        page = 1
    total_pages = max(1, (len(all_paths) + page_size - 1) // page_size)
    page = min(page, total_pages)
    start_index = (page - 1) * page_size
    image_paths = all_paths[start_index:start_index + page_size]

    captions = []
    for img_path in image_paths:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                captions.append(f.read().strip()[:500])
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
        f"📁 {target}\n📷 {len(all_paths)} Bilder · Seite {page}/{total_pages} · angezeigt {len(image_paths)}",
        f"📁 {target}\n📷 {len(all_paths)} images · page {page}/{total_pages} · showing {len(image_paths)}",
    )
    return gallery_data, report, info, gr.update(minimum=1, maximum=total_pages, value=page)



# ============================================================
# LOCAL FRAME REVIEW MODULE
# ============================================================

FRAME_REVIEW_PAGE_SIZE = 18
FRAME_REVIEW_FILTER_CHOICES = [
    ("review", "Mittlere Sicherheit / zu prüfen", "Medium confidence / review"),
    ("high", "Hohe Sicherheit", "High confidence"),
    ("detected", "Alle erkannten Rahmen", "All detected frames"),
    ("decided", "Manuell entschieden", "Manually decided"),
    ("all", "Alle Bilder", "All images"),
]


def _frame_ui_paths(input_folder: str, trigger_word: str) -> Tuple[str, str]:
    output_root = output_root_for(input_folder, trigger_word)
    cache_dir = os.path.join(output_root, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return output_root, cache_dir


def _load_frame_hash_index_ui(cache_dir: str) -> Dict[str, Any]:
    path = os.path.join(cache_dir, "file_hash_index.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") == "v1" and isinstance(payload.get("entries"), dict):
            return payload
    except Exception:
        pass
    return {"schema_version": "v1", "entries": {}, "updated_at": ""}


def _frame_hash_with_index_ui(path: str, payload: Dict[str, Any]) -> Tuple[str, bool]:
    entries = payload.setdefault("entries", {})
    key = os.path.normcase(os.path.abspath(path))
    stat = os.stat(path)
    signature = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    existing = entries.get(key)
    if (
        isinstance(existing, dict)
        and existing.get("size") == signature["size"]
        and existing.get("mtime_ns") == signature["mtime_ns"]
        and isinstance(existing.get("sha1"), str)
    ):
        return str(existing["sha1"]), False
    digest = frame_file_sha1(path)
    entries[key] = {**signature, "sha1": digest}
    return digest, True


def _save_frame_hash_index_ui(cache_dir: str, payload: Dict[str, Any]) -> None:
    payload["schema_version"] = "v1"
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write_json(os.path.join(cache_dir, "file_hash_index.json"), payload)


def _frame_filter_records(state: Any, filter_value: str) -> List[Dict[str, Any]]:
    records = list((state or {}).get("records", []) or []) if isinstance(state, dict) else []
    value = str(filter_value or "review")
    if value == "review":
        return [r for r in records if r.get("confidence_level") == "medium"]
    if value == "high":
        return [r for r in records if r.get("confidence_level") == "high"]
    if value == "detected":
        return [r for r in records if r.get("confidence_level") in {"high", "medium"}]
    if value == "decided":
        return [r for r in records if r.get("user_decision", "auto") != "auto"]
    return records


def _frame_page_choices(count: int) -> List[str]:
    pages = max(1, (max(0, int(count)) + FRAME_REVIEW_PAGE_SIZE - 1) // FRAME_REVIEW_PAGE_SIZE)
    return [str(i) for i in range(1, pages + 1)]


def _frame_selection_location(
    state: Any,
    filter_value: str,
    source_hash: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return filtered records and the selected record's global index."""
    records = _frame_filter_records(state, filter_value)
    selected = str(source_hash or "")
    for index, record in enumerate(records):
        if str(record.get("source_hash", "")) == selected:
            return records, index
    return records, -1


def _frame_viewer_navigation_updates(
    state: Any,
    filter_value: str,
    source_hash: str,
):
    records, index = _frame_selection_location(state, filter_value, source_hash)
    total = len(records)
    if index < 0 or total == 0:
        return (
            tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    return (
        tr(f"**Bild {index + 1} von {total}**", f"**Image {index + 1} of {total}**"),
        gr.update(interactive=index > 0),
        gr.update(interactive=index < total - 1),
    )


def _frame_review_gallery_update(
    state: Any,
    filter_value: str,
    page_value: Any,
    source_hash: str = "",
):
    """Refresh the overview while preserving the active image highlight."""
    gallery = _frame_review_gallery(state, filter_value, page_value)
    records, index = _frame_selection_location(state, filter_value, source_hash)
    try:
        page = max(1, int(page_value or 1))
    except Exception:
        page = 1
    page_start = (page - 1) * FRAME_REVIEW_PAGE_SIZE
    selected_index = index - page_start if page_start <= index < page_start + len(gallery) else None
    return gr.update(value=gallery, selected_index=selected_index)


def _frame_candidate_choices(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = []
    for index, candidate in enumerate(record.get("candidates", []) or []):
        bbox = candidate.get("bbox") or []
        crop_type = str(candidate.get("crop_type", "unknown") or "unknown")
        confidence = float(candidate.get("confidence", 0) or 0)
        sides = ",".join(candidate.get("sides", []) or []) or "-"
        label = f"{index + 1}: {crop_type} · {confidence:.0%} · {sides} · {bbox}"
        choices.append((label, str(index)))
    return choices


def _frame_selected_candidate(record: Dict[str, Any], candidate_value: Any = None) -> Optional[Dict[str, Any]]:
    candidates = list(record.get("candidates", []) or [])
    try:
        index = int(candidate_value if candidate_value not in (None, "") else record.get("selected_candidate_index", 0))
    except Exception:
        index = 0
    if 0 <= index < len(candidates):
        return candidates[index]
    return None


def _frame_candidate_bbox(record: Dict[str, Any], candidate_value: Any = None) -> Optional[List[int]]:
    candidate = _frame_selected_candidate(record, candidate_value)
    bbox = candidate.get("bbox") if candidate else record.get("candidate_bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [int(v) for v in bbox]
    return None



def _frame_option_entries(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all visually selectable outcomes for one frame-review image."""
    entries: List[Dict[str, Any]] = [{
        "key": "original",
        "label": tr("Original behalten", "Keep original"),
        "bbox": None,
        "crop_type": "original",
        "confidence": 1.0,
    }]
    for index, candidate in enumerate(record.get("candidates", []) or []):
        bbox = candidate.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        entries.append({
            "key": f"candidate:{index}",
            "label": tr(f"Variante {index + 1}", f"Variant {index + 1}"),
            "bbox": [int(v) for v in bbox],
            "crop_type": str(candidate.get("crop_type", "unknown") or "unknown"),
            "confidence": float(candidate.get("confidence", 0) or 0),
            "candidate_index": index,
        })
    if str(record.get("user_decision", "")) == "manual" and isinstance(record.get("display_bbox"), list):
        entries.append({
            "key": "manual",
            "label": tr("Manueller Crop", "Manual crop"),
            "bbox": [int(v) for v in record["display_bbox"]],
            "crop_type": "manual",
            "confidence": 1.0,
        })
    return entries


def _frame_current_option_key(record: Dict[str, Any], state: Any = None) -> str:
    decision = str(record.get("user_decision", "auto") or "auto")
    if decision == "keep_original":
        return "original"
    if decision == "manual":
        return "manual"
    if decision == "accept":
        bbox = record.get("display_bbox")
        for entry in _frame_option_entries(record):
            if entry["key"].startswith("candidate:") and entry.get("bbox") == bbox:
                return str(entry["key"])
        return "candidate:0" if record.get("candidates") else "original"

    frame_mode = str((state or {}).get("frame_mode", "suggest_only")) if isinstance(state, dict) else "suggest_only"
    allowed = {str(v) for v in ((state or {}).get("auto_accept_types", []) or [])} if isinstance(state, dict) else set()
    crop_type = str(record.get("crop_type", "unknown") or "unknown")
    level = str(record.get("confidence_level", "low") or "low")
    auto_uses_crop = (
        crop_type in allowed
        and (
            (frame_mode == "auto_high_review_medium" and level == "high")
            or (frame_mode == "auto_high_keep_medium" and level in {"high", "medium"})
        )
    )
    return "candidate:0" if auto_uses_crop and record.get("candidates") else "original"


def _crop_option_preview(record: Dict[str, Any], bbox: Optional[List[int]]) -> Optional[Image.Image]:
    preview = load_gallery_image(str(record.get("source_path", "")), max_size=(1500, 1500))
    if preview is None:
        return None
    if bbox is None:
        return preview
    original_size = record.get("original_size") or []
    if not isinstance(original_size, list) or len(original_size) != 2:
        return preview
    ow, oh = max(1, int(original_size[0])), max(1, int(original_size[1]))
    sx, sy = preview.width / float(ow), preview.height / float(oh)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pb = [
        max(0, int(round(x1 * sx))), max(0, int(round(y1 * sy))),
        min(preview.width, int(round(x2 * sx))), min(preview.height, int(round(y2 * sy))),
    ]
    if pb[2] <= pb[0] or pb[3] <= pb[1]:
        return preview
    result = preview.crop(tuple(pb))
    result.thumbnail((620, 660), Image.Resampling.LANCZOS)
    return result


def _frame_option_gallery_update(record: Optional[Dict[str, Any]], state: Any = None):
    if not record:
        return gr.update(value=[], selected_index=None), gr.update(choices=[], value=None)
    current_key = _frame_current_option_key(record, state)
    values: List[Tuple[Image.Image, str]] = []
    choices: List[Tuple[str, str]] = []
    for index, entry in enumerate(_frame_option_entries(record)):
        image = _crop_option_preview(record, entry.get("bbox"))
        if image is None:
            image = Image.new("RGB", (480, 480), (52, 52, 52))
            drawer = ImageDraw.Draw(image)
            drawer.text((24, 220), tr("Vorschau nicht lesbar", "Preview unavailable"), fill=(235, 235, 235))
        marker = "✓ " if entry["key"] == current_key else ""
        if entry["key"] == "original":
            caption = marker + tr("Original behalten", "Keep original")
        else:
            confidence = float(entry.get("confidence", 0) or 0)
            caption = marker + f"{entry['label']} · {entry.get('crop_type','unknown')} · {confidence:.0%}"
        values.append((image, caption))
        choices.append((caption.replace("✓ ", ""), str(entry["key"])))
    return (
        # The comparison gallery is deliberately display-only. Selection is
        # performed exclusively by the radio control immediately underneath.
        gr.update(value=values, selected_index=None),
        gr.update(choices=choices, value=current_key if any(v == current_key for _l, v in choices) else "original"),
    )


def _frame_option_detail(record: Dict[str, Any], option_key: str) -> Tuple[str, List[int]]:
    if option_key == "original":
        width, height = record.get("original_size") or [0, 0]
        return tr(
            f"**{record.get('filename','')}** – Original wird beibehalten. Die Entscheidung wurde sofort gespeichert.",
            f"**{record.get('filename','')}** – Original is kept. The decision was saved immediately.",
        ), [0, 0, int(width), int(height)]
    for entry in _frame_option_entries(record):
        if entry["key"] == option_key:
            bbox = entry.get("bbox") or [0, 0, *(record.get("original_size") or [0, 0])]
            return tr(
                f"**{record.get('filename','')}** – `{entry.get('label')}` · `{entry.get('crop_type')}` · {float(entry.get('confidence',0)):.1%} · `{bbox}`. Sofort gespeichert.",
                f"**{record.get('filename','')}** – `{entry.get('label')}` · `{entry.get('crop_type')}` · {float(entry.get('confidence',0)):.1%} · `{bbox}`. Saved immediately.",
            ), [int(v) for v in bbox]
    width, height = record.get("original_size") or [0, 0]
    return tr("Unbekannte Auswahl.", "Unknown selection."), [0, 0, int(width), int(height)]


def _frame_review_caption(record: Dict[str, Any]) -> str:
    level = str(record.get("confidence_level", "low")).upper()
    confidence = float(record.get("confidence", 0) or 0)
    decision = str(record.get("user_decision", "auto") or "auto")
    sides = ", ".join(record.get("candidate_sides", []) or []) or "-"
    return f"{record.get('filename', '')}\n{level} {confidence:.0%} | {sides} | {decision}"


def _frame_review_gallery(state: Any, filter_value: str, page_value: Any) -> List[Tuple[Image.Image, str]]:
    """Build the frame-review gallery without exposing external file paths.

    Gradio 5/6 rejects file paths outside the application directory or the
    system temp directory unless they are declared via ``allowed_paths`` at
    launch time. Dataset roots are chosen dynamically after launch, so the UI
    returns bounded in-memory Pillow images instead. Persistent detector and
    preview caches remain in ``curated_<Trigger>/_cache`` as intended.
    """
    records = _frame_filter_records(state, filter_value)
    try:
        page = max(1, int(page_value or 1))
    except Exception:
        page = 1
    start = (page - 1) * FRAME_REVIEW_PAGE_SIZE
    selected = records[start:start + FRAME_REVIEW_PAGE_SIZE]
    gallery: List[Tuple[Image.Image, str]] = []
    cache_dir = str((state or {}).get("cache_dir", "")) if isinstance(state, dict) else ""
    for record in selected:
        caption = _frame_review_caption(record)
        try:
            bbox = record.get("display_bbox") or _frame_candidate_bbox(record) or record.get("candidate_bbox")
            preview_path = build_review_preview(
                record["source_path"], bbox, cache_dir, source_hash=record["source_hash"]
            )
            preview_image = load_gallery_image(preview_path, max_size=(1240, 720))
            if preview_image is None:
                raise ValueError("generated preview could not be decoded")
            gallery.append((preview_image, caption))
        except Exception as exc:
            # Never return the original external path to Gradio. Even the
            # fallback is a bounded Pillow object, so a broken preview cannot
            # trigger another InvalidPathError.
            fallback = load_gallery_image(record.get("source_path", ""), max_size=(1024, 1024))
            if fallback is not None:
                gallery.append((fallback, f"{caption}\nPreview error: {exc}"))
    return gallery


def _frame_review_summary_md(state: Any) -> str:
    if not isinstance(state, dict) or not state.get("records"):
        return tr("Noch keine Rahmenanalyse geladen.", "No frame analysis loaded yet.")
    summary = frame_decision_summary(state.get("records", []))
    decisions = Counter(str(r.get("user_decision", "auto")) for r in state.get("records", []))
    return tr(
        f"**{summary['total']} Bilder lokal geprüft:** {summary['high']} hohe Sicherheit, "
        f"{summary['medium']} mittlere Sicherheit, {summary['low']} ohne belastbaren Rahmenfund.  "
        f"Manuell: {decisions.get('accept',0)} übernommen, {decisions.get('manual',0)} manuell, "
        f"{decisions.get('keep_original',0)} Original behalten. **Keine LLM-Aufrufe.**",
        f"**{summary['total']} images checked locally:** {summary['high']} high confidence, "
        f"{summary['medium']} medium confidence, {summary['low']} without a reliable frame.  "
        f"Manual: {decisions.get('accept',0)} accepted, {decisions.get('manual',0)} manual, "
        f"{decisions.get('keep_original',0)} original kept. **No LLM calls.**",
    )


def scan_frame_review_ui(
    trigger_word: str,
    input_folder: str,
    advanced_types: bool,
    progress=gr.Progress(track_tqdm=False),
):
    if not trigger_word or not str(trigger_word).strip():
        return {}, tr("Bitte zuerst ein Triggerwort eingeben.", "Please enter a trigger word first."), gr.update(choices=["1"], value="1"), [], "", tr("Kein Bild ausgewählt.", "No image selected."), gr.update(value=[], selected_index=None), gr.update(choices=[], value=None), tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"), gr.update(interactive=False), gr.update(interactive=False), 0, 0, 0, 0, tr("❌ Triggerwort fehlt", "❌ Trigger word missing")
    if not input_folder or not os.path.isdir(input_folder):
        return {}, tr("Input-Ordner nicht gefunden.", "Input folder not found."), gr.update(choices=["1"], value="1"), [], "", tr("Kein Bild ausgewählt.", "No image selected."), gr.update(value=[], selected_index=None), gr.update(choices=[], value=None), tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"), gr.update(interactive=False), gr.update(interactive=False), 0, 0, 0, 0, tr("❌ Input fehlt", "❌ Input missing")

    output_root, cache_dir = _frame_ui_paths(input_folder, trigger_word)
    workspace = load_project_workspace(input_folder, trigger_word)
    if not bool((workspace.get("preflight") or {}).get("completed")):
        return {}, tr(
            "Bitte zuerst die pHash-/Datei-Vorprüfung auf der Startseite ausführen.",
            "Please run the pHash/file preflight on the start page first.",
        ), gr.update(choices=["1"], value="1"), [], "", tr("Kein Bild ausgewählt.", "No image selected."), gr.update(value=[], selected_index=None), gr.update(choices=[], value=None), tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"), gr.update(interactive=False), gr.update(interactive=False), 0, 0, 0, 0, tr("❌ Vorprüfung fehlt", "❌ Preflight missing")
    if not _workspace_preflight_is_current(input_folder, trigger_word, workspace):
        return {}, tr(
            "Der Input-Ordner hat sich seit der Vorprüfung geändert. Bitte die Vorprüfung auf der Startseite erneut ausführen.",
            "The input folder changed after preflight. Please rerun preflight on the start page.",
        ), gr.update(choices=["1"], value="1"), [], "", tr("Kein Bild ausgewählt.", "No image selected."), gr.update(value=[], selected_index=None), gr.update(choices=[], value=None), tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"), gr.update(interactive=False), gr.update(interactive=False), 0, 0, 0, 0, tr("❌ Vorprüfung veraltet", "❌ Preflight stale")

    all_paths = scan_frame_source_images(input_folder, output_root)
    early_cache_path = os.path.join(cache_dir, "early_results.json")
    try:
        with open(early_cache_path, "r", encoding="utf-8") as handle:
            early_cache = json.load(handle)
        survivor_paths = {
            os.path.normcase(os.path.abspath(str(path)))
            for path in (early_cache.get("survivor_paths") or [])
            if path
        }
    except Exception:
        survivor_paths = set()
    if not survivor_paths:
        return {}, tr(
            "Die Vorprüfungsdaten fehlen oder sind leer. Bitte die Vorprüfung auf der Startseite erneut ausführen.",
            "Preflight data is missing or empty. Please rerun preflight on the start page.",
        ), gr.update(choices=["1"], value="1"), [], "", tr("Kein Bild ausgewählt.", "No image selected."), gr.update(value=[], selected_index=None), gr.update(choices=[], value=None), tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"), gr.update(interactive=False), gr.update(interactive=False), 0, 0, 0, 0, tr("❌ Vorprüfungsdaten fehlen", "❌ Preflight data missing")
    paths = [
        path for path in all_paths
        if os.path.normcase(os.path.abspath(path)) in survivor_paths
    ]

    user_payload = load_frame_user_decisions(output_root)
    user_map = user_payload.get("decisions", {}) or {}
    hash_index = _load_frame_hash_index_ui(cache_dir)
    hash_index_dirty = False
    workspace_frame = dict(workspace.get("frame") or {})
    settings = SmartFrameDetectorSettings(
        advanced_types=bool(advanced_types),
        auto_accept_types=tuple(str(v) for v in (workspace_frame.get("auto_accept_types") or [])),
    )
    records: List[Dict[str, Any]] = []
    total = max(1, len(paths))
    for index, path in enumerate(paths, start=1):
        progress((index - 1) / total, desc=tr(f"Rahmenanalyse {index}/{len(paths)}", f"Frame analysis {index}/{len(paths)}"))
        try:
            source_hash, hash_changed = _frame_hash_with_index_ui(path, hash_index)
            hash_index_dirty = hash_index_dirty or hash_changed
            analysis = analyze_frame_cleanup(path, cache_dir, source_hash=source_hash, settings=settings, use_cache=True)
            user = user_map.get(source_hash, {}) or {}
            user_decision = str(user.get("decision", "auto") or "auto")
            display_bbox = user.get("bbox") if user_decision in {"manual", "accept"} else analysis.get("candidate_bbox")
            selected_candidate_index = 0
            if user_decision == "accept" and isinstance(user.get("bbox"), list):
                for candidate_index, candidate in enumerate(analysis.get("candidates", []) or []):
                    if candidate.get("bbox") == user.get("bbox"):
                        selected_candidate_index = candidate_index
                        break
            records.append({
                "source_hash": source_hash,
                "source_path": path,
                "filename": os.path.basename(path),
                "original_size": analysis.get("original_size", []),
                "confidence": float(analysis.get("confidence", 0) or 0),
                "confidence_level": analysis.get("confidence_level", "low"),
                "candidate_bbox": analysis.get("candidate_bbox"),
                "candidate_sides": analysis.get("candidate_sides", []),
                "signals": analysis.get("signals", []),
                "layout_class": analysis.get("layout_class", "plain_photo"),
                "layout_flags": analysis.get("layout_flags", []),
                "recommendation": analysis.get("recommendation", "keep_original"),
                "crop_type": analysis.get("crop_type", "unknown"),
                "candidates": list(analysis.get("candidates", []) or []),
                "selected_candidate_index": selected_candidate_index,
                "user_decision": user_decision,
                "display_bbox": display_bbox,
                "cache_hit": bool(analysis.get("cache_hit")),
            })
        except Exception as exc:
            records.append({
                "source_hash": "",
                "source_path": path,
                "filename": os.path.basename(path),
                "original_size": [],
                "confidence": 0.0,
                "confidence_level": "low",
                "candidate_bbox": None,
                "candidate_sides": [],
                "signals": [f"error:{exc}"],
                "recommendation": "keep_original",
                "user_decision": "auto",
                "display_bbox": None,
                "cache_hit": False,
            })
    if hash_index_dirty:
        try:
            _save_frame_hash_index_ui(cache_dir, hash_index)
        except Exception:
            pass
    workspace = dict(workspace or {})
    frame_state = dict(workspace.get("frame") or {})
    frame_state.update({
        "analysis_completed": True,
        "analysis_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis_count": len(records),
        "high_count": sum(1 for r in records if r.get("confidence_level") == "high"),
        "review_count": sum(1 for r in records if r.get("confidence_level") == "medium"),
    })
    workspace["frame"] = frame_state
    workspace.setdefault("preflight", {})
    workspace.update({
        "schema_version": "workspace-v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_folder": os.path.abspath(input_folder),
        "trigger_word": str(trigger_word).strip(),
        "output_root": output_root,
    })
    core_atomic_write_json(os.path.join(output_root, "_project_workspace.json"), workspace)
    progress(1.0, desc=tr("Rahmenanalyse abgeschlossen", "Frame analysis complete"))
    state = {
        "records": records,
        "output_root": output_root,
        "cache_dir": cache_dir,
        "advanced_types": bool(advanced_types),
        "preflight_survivor_count": len(paths),
        "preflight_skipped_count": max(0, len(all_paths) - len(paths)),
        "frame_mode": str(workspace_frame.get("mode", "suggest_only") or "suggest_only"),
        "auto_accept_types": list(workspace_frame.get("auto_accept_types") or []),
    }
    filtered = _frame_filter_records(state, "review")
    pages = _frame_page_choices(len(filtered))
    gallery = _frame_review_gallery(state, "review", "1")
    return (
        state,
        _frame_review_summary_md(state),
        gr.update(choices=pages, value="1"),
        gallery,
        "",
        tr("_Kein Bild ausgewählt._", "_No image selected._"),
        gr.update(value=[], selected_index=None),
        gr.update(choices=[], value=None),
        tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"),
        gr.update(interactive=False),
        gr.update(interactive=False),
        0, 0, 0, 0,
        tr(f"✅ {len(records)} Bilder lokal geprüft", f"✅ {len(records)} images checked locally"),
    )


def refresh_frame_review_page_ui(state: Any, filter_value: str, page_value: Any):
    records = _frame_filter_records(state, filter_value)
    choices = _frame_page_choices(len(records))
    try:
        page = str(min(max(1, int(page_value or 1)), len(choices)))
    except Exception:
        page = "1"
    option_gallery, option_radio = _frame_option_gallery_update(None, state)
    return (
        gr.update(choices=choices, value=page), _frame_review_gallery(state, filter_value, page),
        "", tr("_Kein Bild ausgewählt._", "_No image selected._"),
        option_gallery, option_radio,
        tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"),
        gr.update(interactive=False), gr.update(interactive=False),
        0, 0, 0, 0,
    )


def select_frame_review_image_ui(state: Any, filter_value: str, page_value: Any, evt: gr.SelectData):
    records = _frame_filter_records(state, filter_value)
    try:
        page = max(1, int(page_value or 1))
        index = (page - 1) * FRAME_REVIEW_PAGE_SIZE + int(evt.index)
        record = records[index]
    except Exception:
        empty_gallery, empty_radio = _frame_option_gallery_update(None, state)
        return "", tr("Auswahl konnte nicht aufgelöst werden.", "Could not resolve selection."), empty_gallery, empty_radio, tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"), gr.update(interactive=False), gr.update(interactive=False), 0, 0, 0, 0
    current_key = _frame_current_option_key(record, state)
    detail, bbox = _frame_option_detail(record, current_key)
    detail += tr(
        f"  \\nDetektor: `{record.get('confidence_level')}` ({float(record.get('confidence',0)):.1%}) · Layout `{record.get('layout_class','plain_photo')}` · Vorschläge `{len(record.get('candidates',[]) or [])}`. "
        "Die Vorschaubilder sind nur zum Vergleichen. Wähle direkt darunter genau eine Option; sie wird ohne zusätzlichen Speichern-Schritt übernommen.",
        f"  \\nDetector: `{record.get('confidence_level')}` ({float(record.get('confidence',0)):.1%}) · layout `{record.get('layout_class','plain_photo')}` · suggestions `{len(record.get('candidates',[]) or [])}`. "
        "The previews are comparison-only. Choose exactly one option directly underneath; it is applied without a separate save step.",
    )
    option_gallery, option_radio = _frame_option_gallery_update(record, state)
    position, previous_update, next_update = _frame_viewer_navigation_updates(
        state, filter_value, record.get("source_hash", "")
    )
    return record.get("source_hash", ""), detail, option_gallery, option_radio, position, previous_update, next_update, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def navigate_frame_review_image_ui(
    state: Any,
    filter_value: str,
    source_hash: str,
    direction: int,
):
    """Move the comparison viewer to the previous or next filtered image."""
    records, current_index = _frame_selection_location(state, filter_value, source_hash)
    if not records:
        empty_gallery, empty_radio = _frame_option_gallery_update(None, state)
        return (
            gr.update(choices=["1"], value="1"),
            gr.update(value=[], selected_index=None),
            "",
            tr("_Kein Bild ausgewählt._", "_No image selected._"),
            empty_gallery,
            empty_radio,
            tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"),
            gr.update(interactive=False),
            gr.update(interactive=False),
            0, 0, 0, 0,
        )

    if current_index < 0:
        current_index = 0
    target_index = min(max(0, current_index + int(direction)), len(records) - 1)
    record = records[target_index]
    page = str((target_index // FRAME_REVIEW_PAGE_SIZE) + 1)
    pages = _frame_page_choices(len(records))
    current_key = _frame_current_option_key(record, state)
    detail, bbox = _frame_option_detail(record, current_key)
    detail += tr(
        f"  \\nDetektor: `{record.get('confidence_level')}` ({float(record.get('confidence',0)):.1%}) · Layout `{record.get('layout_class','plain_photo')}` · Vorschläge `{len(record.get('candidates',[]) or [])}`. "
        "Die Vorschaubilder sind nur zum Vergleichen. Wähle direkt darunter genau eine Option.",
        f"  \\nDetector: `{record.get('confidence_level')}` ({float(record.get('confidence',0)):.1%}) · layout `{record.get('layout_class','plain_photo')}` · suggestions `{len(record.get('candidates',[]) or [])}`. "
        "The previews are comparison-only. Choose exactly one option directly underneath.",
    )
    option_gallery, option_radio = _frame_option_gallery_update(record, state)
    position, previous_update, next_update = _frame_viewer_navigation_updates(
        state, filter_value, record.get("source_hash", "")
    )
    return (
        gr.update(choices=pages, value=page),
        _frame_review_gallery_update(state, filter_value, page, record.get("source_hash", "")),
        record.get("source_hash", ""),
        detail,
        option_gallery,
        option_radio,
        position,
        previous_update,
        next_update,
        int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
    )

def select_frame_candidate_ui(state: Any, filter_value: str, page_value: Any, source_hash: str, candidate_value: Any):
    if isinstance(state, dict):
        state = dict(state)
        state["records"] = [dict(r) for r in (state.get("records", []) or [])]
    record = _find_frame_record(state, source_hash)
    if not record:
        return state, _frame_review_gallery(state, filter_value, page_value), tr("Kein Bild ausgewählt.", "No image selected."), 0, 0, 0, 0
    try:
        record["selected_candidate_index"] = int(candidate_value)
    except Exception:
        record["selected_candidate_index"] = 0
    bbox = _frame_candidate_bbox(record, candidate_value) or [0, 0, *(record.get("original_size") or [0, 0])]
    record["display_bbox"] = bbox
    candidate = _frame_selected_candidate(record, candidate_value) or {}
    detail = tr(
        f"**{record.get('filename','')}** – Variante `{int(record['selected_candidate_index']) + 1}` · `{candidate.get('crop_type','unknown')}` · {float(candidate.get('confidence',0)):.1%} · `{bbox}`",
        f"**{record.get('filename','')}** – Variant `{int(record['selected_candidate_index']) + 1}` · `{candidate.get('crop_type','unknown')}` · {float(candidate.get('confidence',0)):.1%} · `{bbox}`",
    )
    return state, _frame_review_gallery(state, filter_value, page_value), detail, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])



def apply_frame_option_ui(
    state: Any,
    filter_value: str,
    page_value: Any,
    source_hash: str,
    option_key: Any,
):
    """Persist one option immediately and refresh both galleries."""
    option_key = str(option_key or "original")
    record = _find_frame_record(state, source_hash)
    if not record:
        empty_gallery, empty_radio = _frame_option_gallery_update(None, state)
        return state, _frame_review_summary_md(state), _frame_review_gallery_update(state, filter_value, page_value), tr("Kein Bild ausgewählt.", "No image selected."), empty_gallery, empty_radio, 0, 0, 0, 0, tr("❌ Keine Auswahl", "❌ No selection")
    if option_key == "original":
        decision, candidate_value = "keep_original", None
    elif option_key == "manual":
        # Existing manual choice remains selected; no re-save is needed.
        detail, bbox = _frame_option_detail(record, option_key)
        option_gallery, option_radio = _frame_option_gallery_update(record, state)
        return state, _frame_review_summary_md(state), _frame_review_gallery_update(state, filter_value, page_value, source_hash), detail, option_gallery, option_radio, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), tr("✅ Manueller Crop bleibt aktiv", "✅ Manual crop remains active")
    elif option_key.startswith("candidate:"):
        decision, candidate_value = "accept", option_key.split(":", 1)[1]
    else:
        decision, candidate_value = "keep_original", None

    updated = set_frame_review_decision_ui(
        state, filter_value, page_value, source_hash, decision, candidate_value,
        0, 0, 0, 0,
    )
    new_state, summary, main_gallery, _message, status = updated
    record = _find_frame_record(new_state, source_hash)
    detail, bbox = _frame_option_detail(record, option_key) if record else (tr("Auswahl gespeichert.", "Selection saved."), [0, 0, 0, 0])
    option_gallery, option_radio = _frame_option_gallery_update(record, new_state)
    return new_state, summary, main_gallery, detail, option_gallery, option_radio, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), status


def restore_frame_auto_ui(state: Any, filter_value: str, page_value: Any, source_hash: str):
    record = _find_frame_record(state, source_hash)
    if not record:
        empty_gallery, empty_radio = _frame_option_gallery_update(None, state)
        return state, _frame_review_summary_md(state), _frame_review_gallery_update(state, filter_value, page_value), tr("Kein Bild ausgewählt.", "No image selected."), empty_gallery, empty_radio, 0, 0, 0, 0, tr("❌ Keine Auswahl", "❌ No selection")
    width, height = record.get("original_size") or [0, 0]
    updated = set_frame_review_decision_ui(state, filter_value, page_value, source_hash, "auto", None, 0, 0, width, height)
    new_state, summary, main_gallery, _message, status = updated
    record = _find_frame_record(new_state, source_hash)
    key = _frame_current_option_key(record or {}, new_state)
    detail, bbox = _frame_option_detail(record or {}, key)
    option_gallery, option_radio = _frame_option_gallery_update(record, new_state)
    return new_state, summary, main_gallery, detail, option_gallery, option_radio, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), status


def _find_frame_record(state: Any, source_hash: str) -> Optional[Dict[str, Any]]:
    if not isinstance(state, dict):
        return None
    for record in state.get("records", []) or []:
        if str(record.get("source_hash", "")) == str(source_hash or ""):
            return record
    return None



def _load_json_dict(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _mark_frame_audit_dependency_stale(output_root: str, record: Dict[str, Any], decision: str, bbox: Any) -> None:
    """Persist that a frame decision changed the image variant used by audit.

    The next normal Curator run reuses every unaffected cache entry. Continue
    from profile is blocked until that targeted re-audit has happened.
    """
    marker = {
        "source_hash": str(record.get("source_hash", "")),
        "source_path": os.path.abspath(str(record.get("source_path", ""))),
        "filename": str(record.get("filename", "")),
        "decision": str(decision),
        "bbox": [int(v) for v in bbox] if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
        "reason": "frame_cleanup_decision_changed",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for filename in ("_subject_profile.json", "_caption_stage.json"):
        path = os.path.join(output_root, filename)
        if not os.path.isfile(path):
            continue
        payload = _load_json_dict(path)
        existing = payload.get("audit_stale_images", [])
        if not isinstance(existing, list):
            existing = []
        key = marker["source_hash"] or marker["source_path"] or marker["filename"]
        by_key = {
            str(item.get("source_hash") or item.get("source_path") or item.get("filename") or ""): item
            for item in existing if isinstance(item, dict)
        }
        by_key[key] = marker
        payload["audit_stale_images"] = list(by_key.values())
        payload["profile_rebuild_required"] = True
        payload["profile_rebuild_reason"] = "Frame decision changed; affected image audit must be refreshed."
        payload["preserve_audit_stale_markers"] = True
        core_atomic_write_json(path, payload)


def set_frame_review_decision_ui(
    state: Any,
    filter_value: str,
    page_value: Any,
    source_hash: str,
    decision: str,
    candidate_value: Any,
    x1: Any, y1: Any, x2: Any, y2: Any,
):
    if isinstance(state, dict):
        state = dict(state)
        state["records"] = [dict(r) for r in (state.get("records", []) or [])]
    record = _find_frame_record(state, source_hash)
    if not record:
        return state, _frame_review_summary_md(state), _frame_review_gallery_update(state, filter_value, page_value), tr("Kein Bild ausgewählt.", "No image selected."), tr("❌ Keine Auswahl", "❌ No selection")
    bbox = None
    if decision == "manual":
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        width, height = record.get("original_size") or frame_image_dimensions(record["source_path"])
        if not (0 <= bbox[0] < bbox[2] <= int(width) and 0 <= bbox[1] < bbox[3] <= int(height)):
            return state, _frame_review_summary_md(state), _frame_review_gallery_update(state, filter_value, page_value, source_hash), tr("Ungültige Crop-Koordinaten.", "Invalid crop coordinates."), tr("❌ Ungültiger Crop", "❌ Invalid crop")
    elif decision == "accept":
        bbox = _frame_candidate_bbox(record, candidate_value)
        if not isinstance(bbox, list) or len(bbox) != 4:
            return state, _frame_review_summary_md(state), _frame_review_gallery_update(state, filter_value, page_value, source_hash), tr("Kein Crop-Vorschlag vorhanden.", "No crop suggestion available."), tr("❌ Kein Vorschlag", "❌ No suggestion")

    old_decision = str(record.get("user_decision", "auto") or "auto")
    old_bbox = record.get("display_bbox") if old_decision in {"accept", "manual"} else None
    save_frame_user_decision(
        state["output_root"], record["source_hash"], record["source_path"], decision, bbox=bbox
    )
    new_effective_bbox = bbox if decision in {"accept", "manual"} else None
    if old_decision != decision or old_bbox != new_effective_bbox:
        _mark_frame_audit_dependency_stale(state["output_root"], record, decision, new_effective_bbox)
    record["user_decision"] = decision
    if decision in {"manual", "accept"}:
        record["display_bbox"] = bbox
        if decision == "accept":
            try:
                record["selected_candidate_index"] = int(candidate_value)
            except Exception:
                pass
    elif decision == "keep_original":
        record["display_bbox"] = None
    else:
        record["display_bbox"] = record.get("candidate_bbox")
    message = {
        "accept": tr("Crop-Vorschlag übernommen.", "Crop suggestion accepted."),
        "manual": tr("Manueller Crop gespeichert.", "Manual crop saved."),
        "keep_original": tr("Original wird beibehalten.", "Original will be kept."),
        "auto": tr("Manuelle Entscheidung entfernt; Automatik gilt wieder.", "Manual decision removed; automatic behavior restored."),
    }.get(decision, decision)
    return state, _frame_review_summary_md(state), _frame_review_gallery_update(state, filter_value, page_value, source_hash), message, f"✅ {message}"


def preview_manual_frame_crop_ui(state: Any, source_hash: str, x1: Any, y1: Any, x2: Any, y2: Any):
    record = _find_frame_record(state, source_hash)
    if not record:
        return None, tr("Kein Bild ausgewählt.", "No image selected.")
    try:
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        width, height = record.get("original_size") or frame_image_dimensions(record["source_path"])
        if not (0 <= bbox[0] < bbox[2] <= int(width) and 0 <= bbox[1] < bbox[3] <= int(height)):
            raise ValueError("bbox outside image")
        preview_path = build_review_preview(
            record["source_path"], bbox, state["cache_dir"], source_hash=record["source_hash"]
        )
        preview_image = load_gallery_image(preview_path, max_size=(1240, 720))
        if preview_image is None:
            raise ValueError("generated preview could not be decoded")
        return preview_image, tr("Manuelle Vorschau aktualisiert.", "Manual preview updated.")
    except Exception as exc:
        return None, tr(f"Ungültige Vorschau: {exc}", f"Invalid preview: {exc}")


def reset_frame_detector_cache_ui(trigger_word: str, input_folder: str):
    if not trigger_word or not os.path.isdir(input_folder):
        return {}, tr("Triggerwort/Input fehlen.", "Trigger word/input missing."), gr.update(choices=["1"], value="1"), [], tr("❌ Eingaben fehlen", "❌ Missing input")
    _, cache_dir = _frame_ui_paths(input_folder, trigger_word)
    count = reset_frame_detector_cache(cache_dir)
    return {}, tr("Erkennungscache geleert. Manuelle Entscheidungen bleiben erhalten.", "Detection cache cleared. Manual decisions were preserved."), gr.update(choices=["1"], value="1"), [], tr(f"🧹 {count} Rahmen-Cachedateien entfernt", f"🧹 Removed {count} frame-cache files")


def reset_frame_manual_decisions_ui(trigger_word: str, input_folder: str):
    if not trigger_word or not os.path.isdir(input_folder):
        return tr("❌ Eingaben fehlen", "❌ Missing input")
    output_root, _ = _frame_ui_paths(input_folder, trigger_word)
    count = reset_frame_user_decisions(output_root)
    return tr(f"🧹 {count} manuelle Rahmenentscheidungen entfernt", f"🧹 Removed {count} manual frame decisions")



# ============================================================
# PROJECT WORKSPACE / PREFLIGHT
# ============================================================

FRAME_AUTO_ACCEPT_TYPE_CHOICES = [
    ("uniform_canvas", "Einfarbige/gleichmäßige Außenfläche", "Uniform outer canvas"),
    ("story_canvas", "Story-Canvas / verschwommene Außenfläche", "Story canvas / blurred outer fill"),
    ("story_bars", "Obere und untere Story-Balken", "Top and bottom story bars"),
    ("app_viewport", "Klarer Foto-Viewport in App-Screenshot", "Clear photo viewport in app screenshot"),
    ("side_canvas", "Beidseitige Canvas-Ränder", "Paired side canvas"),
    ("multi_side_border", "Mehrseitiger rechteckiger Rahmen", "Multi-side rectangular border"),
    ("single_side_border", "Einseitiger Rand", "Single-side border"),
]


def _workspace_is_ready(trigger_word: str, input_folder: str, api_key: str) -> bool:
    return bool(
        str(trigger_word or "").strip()
        and os.path.isdir(str(input_folder or "").strip())
        and (str(api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip())
    )


def _workspace_preflight_is_current(
    input_folder: str,
    trigger_word: str,
    workspace: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True only when the completed preflight matches current source files."""
    if not input_folder or not trigger_word or not os.path.isdir(input_folder):
        return False
    workspace = workspace or load_project_workspace(input_folder, trigger_word)
    preflight = dict(workspace.get("preflight") or {}) if isinstance(workspace, dict) else {}
    expected = str(preflight.get("dataset_fingerprint") or "")
    if not preflight.get("completed") or not expected:
        return False
    try:
        output_root = output_root_for(input_folder, trigger_word)
        images = scan_preflight_images(input_folder, output_root)
        current = preflight_dataset_fingerprint(images, input_folder)
        return current == expected
    except Exception:
        return False


def _workspace_summary_md(input_folder: str, trigger_word: str, workspace: Optional[Dict[str, Any]] = None) -> str:
    workspace = workspace or load_project_workspace(input_folder, trigger_word)
    output_root = output_root_for(input_folder, trigger_word) if input_folder and trigger_word else "—"
    pre = dict(workspace.get("preflight") or {}) if isinstance(workspace, dict) else {}
    frame = dict(workspace.get("frame") or {}) if isinstance(workspace, dict) else {}
    preflight_text = (
        tr(
            f"✅ abgeschlossen: {pre.get('survivor_count', 0)} verbleibend, {pre.get('duplicate_count', 0)} Duplikate, {pre.get('early_reject_count', 0)} frühe Rejects",
            f"✅ complete: {pre.get('survivor_count', 0)} remaining, {pre.get('duplicate_count', 0)} duplicates, {pre.get('early_reject_count', 0)} early rejects",
        )
        if pre.get("completed")
        else tr("⏳ noch nicht durchgeführt", "⏳ not run yet")
    )
    frame_text = tr("aktiv", "enabled") if frame.get("enabled") else tr("übersprungen", "skipped")
    return tr(
        f"### Aktives Dataset\n- **Trigger:** `{trigger_word or '—'}`\n- **Input:** `{input_folder or '—'}`\n- **Output:** `{output_root}`\n- **pHash-Vorprüfung:** {preflight_text}\n- **Rahmenmodul:** {frame_text}",
        f"### Active dataset\n- **Trigger:** `{trigger_word or '—'}`\n- **Input:** `{input_folder or '—'}`\n- **Output:** `{output_root}`\n- **pHash preflight:** {preflight_text}\n- **Frame module:** {frame_text}",
    )


def initialize_workspace_ui(
    trigger_word: str,
    input_folder: str,
    api_key: str,
    frame_enabled: bool,
    frame_mode: str,
    frame_auto_accept_types: List[str],
    frame_pause_on_medium: bool,
    post_frame_duplicate_refresh: bool,
):
    trigger_word = str(trigger_word or "").strip()
    input_folder = str(input_folder or "").strip()
    if not trigger_word:
        return tr("❌ Triggerwort fehlt", "❌ Trigger word missing"), "", False
    if not os.path.isdir(input_folder):
        return tr(f"❌ Input-Ordner nicht gefunden: {input_folder}", f"❌ Input folder not found: {input_folder}"), "", False
    if not (str(api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()):
        return tr("❌ API-Key fehlt", "❌ API key missing"), "", False
    output_root = output_root_for(input_folder, trigger_word)
    os.makedirs(os.path.join(output_root, "_cache"), exist_ok=True)
    workspace = load_project_workspace(input_folder, trigger_word)
    workspace = dict(workspace or {})
    workspace.update({
        "schema_version": "workspace-v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_folder": os.path.abspath(input_folder),
        "trigger_word": trigger_word,
        "output_root": output_root,
        "frame": {
            "enabled": bool(frame_enabled),
            "mode": str(frame_mode or "suggest_only"),
            "auto_accept_types": list(frame_auto_accept_types or []),
            "pause_on_medium": bool(frame_pause_on_medium),
            "post_frame_duplicate_refresh": bool(post_frame_duplicate_refresh),
        },
        "preflight": dict(workspace.get("preflight") or {}),
    })
    core_atomic_write_json(os.path.join(output_root, "_project_workspace.json"), workspace)
    # Persist the project context without rewriting the 90-field runtime contract.
    settings = load_settings()
    settings.update({
        "c_trigger": trigger_word,
        "c_input": input_folder,
        "c_api_key": api_key,
        "c_ig_frame_crop": bool(frame_enabled),
        "c_frame_cleanup_mode": str(frame_mode or "suggest_only"),
        "c_frame_pause_on_medium": bool(frame_pause_on_medium),
        "c_frame_auto_accept_types": list(frame_auto_accept_types or []),
        "c_post_frame_phash_refresh": bool(post_frame_duplicate_refresh),
    })
    core_atomic_write_json(SETTINGS_PATH, settings)
    return tr("✅ Projekt initialisiert. Starte jetzt die lokalen Vorprüfungen.", "✅ Project initialized. Run the local preflight next."), _workspace_summary_md(input_folder, trigger_word, workspace), True


def run_workspace_preflight_ui(
    trigger_word: str,
    input_folder: str,
    api_key: str,
    use_early_phash: bool,
    use_loop1: bool,
    threshold1: int,
    keep1: int,
    use_loop2: bool,
    threshold2: int,
    keep2: int,
    min_side: int,
    use_filesize: bool,
    min_filesize: int,
    frame_enabled: bool,
    frame_mode: str,
    frame_auto_accept_types: List[str],
    frame_pause_on_medium: bool,
    post_frame_duplicate_refresh: bool,
    progress=gr.Progress(track_tqdm=False),
):
    status, _, ready = initialize_workspace_ui(
        trigger_word, input_folder, api_key, frame_enabled, frame_mode,
        frame_auto_accept_types, frame_pause_on_medium, post_frame_duplicate_refresh,
    )
    if not ready:
        return status, "", False
    progress(0.05, desc=tr("Quelldateien prüfen", "Scanning source files"))
    settings = WorkspacePHashSettings(
        enabled=bool(use_early_phash),
        loop1_enabled=bool(use_loop1),
        loop1_threshold=int(threshold1),
        loop1_keep=int(keep1),
        loop2_enabled=bool(use_loop2),
        loop2_threshold=int(threshold2),
        loop2_keep=int(keep2),
        min_side_px=int(min_side),
        use_min_filesize=bool(use_filesize),
        min_filesize_kb=float(min_filesize),
    )
    progress(0.15, desc=tr("pHash-Vorprüfung", "pHash preflight"))
    workspace = run_project_preflight(
        str(input_folder).strip(), str(trigger_word).strip(), settings,
        frame_enabled=bool(frame_enabled),
        frame_mode=str(frame_mode or "suggest_only"),
        frame_auto_accept_types=list(frame_auto_accept_types or []),
        frame_pause_on_medium=bool(frame_pause_on_medium),
        post_frame_duplicate_refresh=bool(post_frame_duplicate_refresh),
    )
    progress(1.0, desc=tr("Vorprüfung abgeschlossen", "Preflight complete"))
    pre = workspace.get("preflight", {}) or {}
    status = tr(
        f"✅ Vorprüfung abgeschlossen: {pre.get('input_count',0)} Dateien, {pre.get('duplicate_count',0)} Duplikate, {pre.get('survivor_count',0)} verbleibend.",
        f"✅ Preflight complete: {pre.get('input_count',0)} files, {pre.get('duplicate_count',0)} duplicates, {pre.get('survivor_count',0)} remaining.",
    )
    return status, _workspace_summary_md(input_folder, trigger_word, workspace), True




def _workspace_sync_payload(
    status: str,
    summary: str,
    ready: bool,
    trigger_word: str,
    input_folder: str,
    api_key: str,
    use_early_phash: bool,
    use_loop1: bool,
    threshold1: int,
    keep1: int,
    use_loop2: bool,
    threshold2: int,
    keep2: int,
    min_side: int,
    use_filesize: bool,
    min_filesize: int,
    frame_enabled: bool,
    frame_advanced: bool,
    frame_mode: str,
    frame_pause: bool,
):
    tab_update = gr.update(interactive=bool(ready))
    return (
        status, summary, bool(ready),
        trigger_word, input_folder, api_key,
        use_early_phash, use_loop1, threshold1, keep1,
        use_loop2, threshold2, keep2,
        min_side, use_filesize, min_filesize,
        frame_enabled, frame_advanced, frame_mode, frame_pause,
        trigger_word, input_folder,
        trigger_word, input_folder,
        tab_update, tab_update, tab_update, tab_update,
    )


def initialize_workspace_and_sync_ui(
    trigger_word: str, input_folder: str, api_key: str,
    use_early_phash: bool, use_loop1: bool, threshold1: int, keep1: int,
    use_loop2: bool, threshold2: int, keep2: int,
    min_side: int, use_filesize: bool, min_filesize: int,
    frame_enabled: bool, frame_advanced: bool, frame_mode: str,
    frame_auto_accept_types: List[str], frame_pause: bool, post_frame_refresh: bool,
):
    status, summary, _ = initialize_workspace_ui(
        trigger_word, input_folder, api_key, frame_enabled, frame_mode,
        frame_auto_accept_types, frame_pause, post_frame_refresh,
    )
    # Initialization alone does not unlock modules. The local preflight is the
    # explicit project gate requested by the workflow.
    return _workspace_sync_payload(
        status, summary, False,
        trigger_word, input_folder, api_key,
        use_early_phash, use_loop1, threshold1, keep1,
        use_loop2, threshold2, keep2,
        min_side, use_filesize, min_filesize,
        frame_enabled, frame_advanced, frame_mode, frame_pause,
    )


def run_workspace_preflight_and_sync_ui(
    trigger_word: str, input_folder: str, api_key: str,
    use_early_phash: bool, use_loop1: bool, threshold1: int, keep1: int,
    use_loop2: bool, threshold2: int, keep2: int,
    min_side: int, use_filesize: bool, min_filesize: int,
    frame_enabled: bool, frame_advanced: bool, frame_mode: str,
    frame_auto_accept_types: List[str], frame_pause: bool, post_frame_refresh: bool,
    progress=gr.Progress(track_tqdm=False),
):
    status, summary, ready = run_workspace_preflight_ui(
        trigger_word, input_folder, api_key,
        use_early_phash, use_loop1, threshold1, keep1,
        use_loop2, threshold2, keep2,
        min_side, use_filesize, min_filesize,
        frame_enabled, frame_mode, frame_auto_accept_types,
        frame_pause, post_frame_refresh, progress,
    )
    return _workspace_sync_payload(
        status, summary, ready,
        trigger_word, input_folder, api_key,
        use_early_phash, use_loop1, threshold1, keep1,
        use_loop2, threshold2, keep2,
        min_side, use_filesize, min_filesize,
        frame_enabled, frame_advanced, frame_mode, frame_pause,
    )


# ============================================================
# GRADIO LAYOUT
# ============================================================

def build_ui() -> gr.Blocks:

    S = load_settings()
    initial_workspace_ready = _workspace_is_ready(S.get("c_trigger", ""), S.get("c_input", ""), S.get("c_api_key", ""))
    initial_workspace = load_project_workspace(S.get("c_input", ""), S.get("c_trigger", "")) if initial_workspace_ready else {}
    initial_preflight_ready = (
        bool((initial_workspace.get("preflight") or {}).get("completed"))
        and _workspace_preflight_is_current(S.get("c_input", ""), S.get("c_trigger", ""), initial_workspace)
    ) if isinstance(initial_workspace, dict) else False

    # Make translations in this UI build consistent.
    global UI_LANG
    UI_LANG = _normalize_lang(S.get("ui_language"))

    def openai_model_dropdown_kwargs() -> Dict[str, Any]:
        """Use custom values when supported so presets stay future-proof."""
        kwargs: Dict[str, Any] = {}
        try:
            dropdown_signature = inspect.signature(gr.Dropdown.__init__)
            if "allow_custom_value" in dropdown_signature.parameters:
                kwargs["allow_custom_value"] = True
        except Exception:
            pass
        return kwargs

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

        gr.Markdown(tr(f"# 🖼️ LoRA Dataset Curator · `{APP_VERSION}`", f"# 🖼️ LoRA Dataset Curator · `{APP_VERSION}`"))
        gr.Markdown(
            tr(
                "Dataset-Aufbereitung und Video-Extraktion für LoRA-Training with [AI Toolkit](https://github.com/ostris/ai-toolkit)",
                "Dataset curation and video extraction for LoRA training with [AI Toolkit](https://github.com/ostris/ai-toolkit)",
            )
        )

        with gr.Tabs(selected="workspace") as main_tabs:

            # ==============================================================
            # TAB 0: PROJECT WORKSPACE / LOCAL PREFLIGHT
            # ==============================================================
            with gr.TabItem(tr("🏠 Start / Projekt", "🏠 Start / Project"), id="workspace") as workspace_tab:
                gr.Markdown(tr(
                    "## Projekt-Workspace\nOrdner, Triggerwort und API-Key gelten für das **gesamte Tool**. "
                    "Die lokale pHash-Vorprüfung läuft zuerst. Danach wird – falls aktiviert – die Rahmenvorbereitung durchgeführt, "
                    "bevor Audit, Profil und Captioning beginnen.",
                    "## Project workspace\nFolder, trigger word and API key apply to the **entire tool**. "
                    "The local pHash preflight runs first. If enabled, frame preparation follows before audit, profile and captioning.",
                ))
                gr.Markdown(tr(
                    "<details open>"
                    "<summary><b>ℹ️ Empfohlener Ablauf und wichtige Hinweise</b></summary>"
                    "\n\n"
                    "1. **Projektangaben festlegen:** Eingabeordner, Triggerwort und API-Key gelten anschließend in allen Modulen.\n"
                    "2. **Lokale Vorprüfung starten:** Dateigröße, Mindestseitenlänge und pHash-Duplikate werden ohne API-Kosten geprüft.\n"
                    "3. **Rahmen vorbereiten:** Falls aktiviert, werden ausschließlich die nach der Vorprüfung verbliebenen Bilder untersucht. Jede Crop-Entscheidung wird sofort gespeichert.\n"
                    "4. **Audit & Auswahl:** Erst danach beginnen kostenpflichtige OpenAI-Aufrufe.\n"
                    "5. **Profil und Captioning:** Krea 2 nutzt standardmäßig den profilbasierten Weg; Single Pass erstellt dasselbe vollständige Profil automatisch im Hintergrund.\n\n"
                    "**Originaldateien werden nie verändert.** Crops, Caches, Profile und Ergebnisse liegen ausschließlich im Unterordner `curated_<Triggerwort>`. "
                    "Ändert sich der Eingabeordner, wird die Vorprüfung als veraltet markiert. Nach einer späteren Crop-Änderung wird nur das betroffene Bild für den Audit als veraltet markiert."
                    "</details>",
                    "<details open>"
                    "<summary><b>ℹ️ Recommended workflow and important notes</b></summary>"
                    "\n\n"
                    "1. **Set project values:** input folder, trigger word and API key then apply to every module.\n"
                    "2. **Run local preflight:** file size, minimum side length and pHash duplicates are checked without API cost.\n"
                    "3. **Prepare frames:** when enabled, only images surviving preflight are analyzed. Every crop choice is saved immediately.\n"
                    "4. **Audit & selection:** paid OpenAI calls start only after this point.\n"
                    "5. **Profile and captioning:** Krea 2 defaults to the profile-based route; Single Pass builds the same complete profile automatically in the background.\n\n"
                    "**Original files are never modified.** Crops, caches, profiles and results remain inside `curated_<TriggerWord>`. "
                    "If the input folder changes, preflight is marked stale. If a crop is changed later, only the affected image is marked stale for audit."
                    "</details>",
                ))
                workspace_ready_state = gr.State(initial_preflight_ready)
                with gr.Row():
                    with gr.Column(scale=3):
                        w_trigger = gr.Textbox(label=tr("Trigger Word", "Trigger word"), value=S["c_trigger"], max_lines=1)
                        w_input = gr.Textbox(label=tr("Input-Ordner", "Input folder"), value=S["c_input"], max_lines=1)
                        w_api_key = gr.Textbox(label=tr("OpenAI API Key", "OpenAI API key"), value=S["c_api_key"], type="password", max_lines=1)
                    with gr.Column(scale=2):
                        workspace_summary = gr.Markdown(_workspace_summary_md(S.get("c_input", ""), S.get("c_trigger", ""), initial_workspace))
                        workspace_status = gr.Textbox(label=tr("Projektstatus", "Project status"), interactive=False, max_lines=3)

                with gr.Accordion(tr("1️⃣ pHash-Duplikat-Vorprüfung", "1️⃣ pHash duplicate preflight"), open=True):
                    gr.Markdown(tr(
                        "Die erste Prüfung läuft vollständig lokal und vor der Rahmenanalyse. Frühe Duplikate werden nur im Projektstatus markiert; die Quelldateien bleiben unverändert.\n\n"
                        "<details><summary><b>ℹ️ Was machen die beiden pHash-Schleifen?</b></summary>\n\n"
                        "**Schleife 1** entfernt praktisch identische Dateien und Re-Uploads. **Schleife 2** kann zusätzlich sehr ähnliche Serienbilder ausdünnen. "
                        "Ein optionaler leichter Refresh nach bestätigten Crops findet anschließend Fälle, bei denen dasselbe Foto einmal mit und einmal ohne Story-Rahmen vorliegt. "
                        "Früh ausgeschiedene Duplikate werden weder an die Rahmenanalyse noch in den großen Profil-Reject-Bucket übernommen."
                        "</details>",
                        "The first check is fully local and runs before frame analysis. Early duplicates are only recorded in project state; source files remain unchanged.\n\n"
                        "<details><summary><b>ℹ️ What do the two pHash loops do?</b></summary>\n\n"
                        "**Loop 1** removes effectively identical files and re-uploads. **Loop 2** can also thin very similar burst/series images. "
                        "An optional light refresh after confirmed crops then catches cases where the same photo exists once with and once without a story frame. "
                        "Early duplicates are sent neither to frame analysis nor to the large profile reject bucket."
                        "</details>",
                    ))
                    w_use_early_phash = gr.Checkbox(label=tr("Frühe pHash-Prüfung aktivieren", "Enable early pHash scan"), value=S["c_use_early_phash"])
                    with gr.Row():
                        with gr.Column():
                            w_use_loop1 = gr.Checkbox(label=tr("Schleife 1 – exakte Duplikate", "Loop 1 – exact duplicates"), value=S["c_use_early_phash_loop1"])
                            w_threshold1 = gr.Slider(label=tr("Hamming-Schwelle 1", "Hamming threshold 1"), minimum=0, maximum=8, step=1, value=S["c_early_phash_thresh_1"])
                            w_keep1 = gr.Slider(label=tr("Pro Gruppe behalten", "Keep per group"), minimum=1, maximum=5, step=1, value=S["c_early_phash_keep_1"])
                        with gr.Column():
                            w_use_loop2 = gr.Checkbox(label=tr("Schleife 2 – Bulk-/Near-Duplikate", "Loop 2 – bulk/near duplicates"), value=S["c_use_early_phash_loop2"])
                            w_threshold2 = gr.Slider(label=tr("Hamming-Schwelle 2", "Hamming threshold 2"), minimum=0, maximum=16, step=1, value=S["c_early_phash_thresh_2"])
                            w_keep2 = gr.Slider(label=tr("Pro Gruppe behalten", "Keep per group"), minimum=1, maximum=5, step=1, value=S["c_early_phash_keep_2"])
                    with gr.Row():
                        w_min_side = gr.Slider(
                            label=tr("Mindestseitenlänge", "Minimum side length"),
                            minimum=128, maximum=2048, step=64, value=S["c_min_side"],
                            info=tr(
                                "Kleinere Bilder werden bereits in der lokalen Vorprüfung aussortiert. Für 1024er Training sind 768 Pixel ein sinnvoller Startwert; für 512er Training entsprechend 512 Pixel.",
                                "Images below this length are removed during local preflight. For 1024 training, 768 px is a sensible starting point; for 512 training, use 512 px accordingly.",
                            ),
                        )
                        w_use_filesize = gr.Checkbox(
                            label=tr("Mindest-Dateigröße prüfen", "Check minimum file size"),
                            value=S["c_use_filesize"],
                        )
                        w_min_filesize = gr.Slider(
                            label=tr("Mindest-Dateigröße (KB)", "Minimum file size (KB)"),
                            minimum=0, maximum=1000, step=10, value=S["c_min_filesize"],
                        )

                with gr.Accordion(tr("2️⃣ Rahmenvorbereitung", "2️⃣ Frame preparation"), open=True):
                    w_frame_enabled = gr.Checkbox(
                        label=tr("Rahmenmodul für dieses Dataset verwenden", "Use frame module for this dataset"),
                        value=S["c_ig_frame_crop"],
                    )
                    w_frame_advanced = gr.Checkbox(
                        label=tr("Erweiterte Story-/App-/Canvas-Typen erkennen", "Detect advanced story/app/canvas types"),
                        value=S["c_ig_two_stage_bar"],
                    )
                    w_frame_mode = gr.Dropdown(
                        label=tr("Automatische Übernahme", "Automatic acceptance"),
                        choices=[
                            (tr("Nur Vorschläge – nichts automatisch", "Suggestions only – never automatic"), "suggest_only"),
                            (tr("Nur hohe Sicherheit automatisch", "Auto-accept high confidence only"), "auto_high_review_medium"),
                            (tr("Hohe und mittlere Sicherheit automatisch", "Auto-accept high and medium confidence"), "auto_high_keep_medium"),
                        ],
                        value=S.get("c_frame_cleanup_mode", "suggest_only"),
                    )
                    w_frame_auto_types = gr.CheckboxGroup(
                        label=tr("Crop-Typen, die automatisch akzeptiert werden dürfen", "Crop types eligible for automatic acceptance"),
                        choices=[(tr(de, en), value) for value, de, en in FRAME_AUTO_ACCEPT_TYPE_CHOICES],
                        value=S.get("c_frame_auto_accept_types", ["uniform_canvas", "story_bars"]),
                        info=tr(
                            "Nur ausgewählte Typen dürfen im Automatikmodus übernommen werden. Collagen, runde Ausschnitte, verschachtelte Screenshots und in das Foto ragende Overlays bleiben immer Review.",
                            "Only selected types may be applied automatically. Collages, circular crops, nested screenshots and overlays intruding into the photo always remain review cases.",
                        ),
                    )
                    with gr.Row():
                        w_frame_pause = gr.Checkbox(
                            label=tr("Vor dem Audit bei ungeklärten Rahmenfällen anhalten", "Pause before audit for unresolved frame cases"),
                            value=S.get("c_frame_pause_on_medium", False),
                        )
                        w_post_frame_phash = gr.Checkbox(
                            label=tr("Nach bestätigten Crops leichten Duplikat-Refresh durchführen", "Run a light duplicate refresh after confirmed crops"),
                            value=S.get("c_post_frame_phash_refresh", True),
                        )

                with gr.Row():
                    w_init_btn = gr.Button(tr("💾 Projekt initialisieren", "💾 Initialize project"), variant="secondary")
                    w_preflight_btn = gr.Button(tr("▶ Vorprüfungen starten", "▶ Run preflight"), variant="primary")

            # ==============================================================
            # TAB 1: FRAME PREPARATION
            # ==============================================================
            with gr.TabItem(tr("1 · 🖼️ Rahmen", "1 · 🖼️ Frames"), id="frames", interactive=initial_preflight_ready) as frame_tab:
                gr.Markdown(tr(
                    "## Rahmenvorschläge prüfen\nDie pHash-Vorprüfung muss zuerst auf der Startseite abgeschlossen sein. "
                    "Bei unsicheren Fällen können mehrere lokale Crop-Varianten gewählt oder das Original beibehalten werden.",
                    "## Review frame suggestions\nThe pHash preflight must be completed on the start page first. "
                    "For uncertain cases, choose among multiple local crop variants or keep the original.",
                ))
                with gr.Row():
                    fr_scan_btn = gr.Button(tr("🔎 Rahmen lokal analysieren", "🔎 Analyze frames locally"), variant="primary", scale=2)
                    fr_reset_cache_btn = gr.Button(tr("♻ Erkennung neu berechnen", "♻ Recompute detection"), variant="secondary", scale=1)
                    fr_reset_manual_btn = gr.Button(tr("↩ Manuelle Entscheidungen löschen", "↩ Clear manual decisions"), variant="secondary", scale=1)
                fr_status = gr.Textbox(label=tr("Status", "Status"), interactive=False, max_lines=1)
                fr_summary = gr.Markdown(tr("Noch keine Rahmenanalyse geladen.", "No frame analysis loaded yet."))
                fr_state = gr.State({})
                fr_selected_hash = gr.State("")
                with gr.Row():
                    fr_filter = gr.Dropdown(
                        label=tr("Anzeige", "View"),
                        choices=[(tr(de, en), value) for value, de, en in FRAME_REVIEW_FILTER_CHOICES],
                        value="review",
                        allow_custom_value=False,
                    )
                    fr_page = gr.Dropdown(label=tr("Seite", "Page"), choices=["1"], value="1", allow_custom_value=False)
                fr_gallery = gr.Gallery(
                    label=tr("Original links – gewählter Crop rechts", "Original left – selected crop right"),
                    columns=2,
                    rows=3,
                    height=720,
                    object_fit="contain",
                )
                fr_selected_info = gr.Markdown(tr("_Kein Bild ausgewählt._", "_No image selected._"))
                gr.Markdown(tr(
                    "### Vergleichsansicht für das markierte Bild\nAlle lokal gefundenen Varianten stehen dauerhaft nebeneinander. **Original behalten** ist immer die erste Option. "
                    "Die Bilder in dieser Ansicht sind nicht anklickbar; die Entscheidung erfolgt ausschließlich direkt darunter. Mit **Vorheriges/Nächstes Bild** kannst du hier durch die aktuelle gefilterte Bilderliste wechseln.",
                    "### Comparison viewer for the selected image\nAll locally found variants remain visible side by side. **Keep original** is always the first option. "
                    "Images in this viewer are not clickable; the decision is made only with the selector directly underneath. Use **Previous/Next image** to move through the current filtered image list here.",
                ))
                with gr.Row(equal_height=True):
                    fr_previous_btn = gr.Button(
                        tr("← Vorheriges Bild", "← Previous image"),
                        variant="primary",
                        interactive=False,
                        scale=1,
                    )
                    fr_viewer_position = gr.Markdown(
                        tr("_Kein Bild im Vergleichsviewer._", "_No image in the comparison viewer._"),
                        elem_classes=["frame-viewer-position"],
                    )
                    fr_next_btn = gr.Button(
                        tr("Nächstes Bild →", "Next image →"),
                        variant="primary",
                        interactive=False,
                        scale=1,
                    )
                fr_option_gallery = gr.Gallery(
                    label=tr("Nicht anklickbare Vergleichsansicht: Original und Crop-Optionen", "Non-clickable comparison viewer: original and crop options"),
                    columns=1,
                    rows=1,
                    height=550,
                    object_fit="contain",
                    allow_preview=False,
                    interactive=False,
                    fit_columns=False,
                    selected_index=None,
                    elem_classes=["frame-comparison-gallery"],
                )
                fr_option_radio = gr.Radio(
                    label=tr("Auswahl für dieses Bild – wird sofort gespeichert", "Choice for this image – saved immediately"),
                    choices=[],
                    value=None,
                    interactive=True,
                    elem_classes=["frame-option-selector"],
                )
                with gr.Row():
                    fr_auto_btn = gr.Button(tr("Automatische Entscheidung wiederherstellen", "Restore automatic decision"), variant="secondary")
                with gr.Accordion(tr("Manuelle Crop-Grenzen", "Manual crop bounds"), open=False):
                    with gr.Row():
                        fr_x1 = gr.Number(label="X1 / links", value=0, precision=0)
                        fr_y1 = gr.Number(label="Y1 / oben", value=0, precision=0)
                        fr_x2 = gr.Number(label="X2 / rechts", value=0, precision=0)
                        fr_y2 = gr.Number(label="Y2 / unten", value=0, precision=0)
                    with gr.Row():
                        fr_preview_manual_btn = gr.Button(tr("Vorschau aktualisieren", "Update preview"), variant="secondary")
                        fr_accept_manual_btn = gr.Button(tr("Manuellen Crop übernehmen", "Accept manual crop"), variant="primary")
                    fr_manual_preview = gr.Image(label=tr("Manuelle Original/Crop-Vorschau", "Manual original/crop preview"), interactive=False, height=480)
                    fr_manual_status = gr.Markdown("")

            # ==============================================================
            # TAB 2: AUDIT AND IMAGE SELECTION
            # ==============================================================
            with gr.TabItem(tr("2 · 📸 Audit & Auswahl", "2 · 📸 Audit & Selection"), id="audit", interactive=initial_preflight_ready) as audit_tab:

                c_training_target = gr.Dropdown(
                    label=tr("Trainingsziel / Basismodell", "Training target / base model"),
                    choices=training_target_choices(),
                    value=normalize_training_target(S.get("c_training_target")),
                    info=tr(
                        "Legt die Prompt-Familie, Caption-Engine und empfohlenen Standardwerte fest. Änderungen an einzelnen Caption-Regeln ändern dieses Trainingsziel nicht.",
                        "Selects the prompt family, caption engine and recommended defaults. Editing individual caption rules never changes this training target.",
                    ),
                )
                gr.Markdown(tr(
                    "**Pipeline-Auswahl:** ERNIE und Z-Image verwenden strukturierte lokale Captions; Krea 2 verwendet nach der Auswahl natürliche GPT-Captions.",
                    "**Pipeline selection:** ERNIE and Z-Image use structured local captions; Krea 2 uses natural GPT captions after selection.",
                ))

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(tr("### Basis-Einstellungen", "### Basic Settings"))
                        c_trigger = gr.Textbox(
                            label=tr("Trigger Word", "Trigger Word"),
                            value=S["c_trigger"],
                            visible=False,
                            info=tr(
                                "Eindeutiges Wort, das im Training die Person identifiziert. Sollte bei bestimmten Modellen kein realer Name oder Alltagsbegriff sein.",
                                "Unique word that identifies the subject during training. For certain models this should not be a real name or common word.",
                            ),
                            max_lines=1,
                        )
                        c_input = gr.Textbox(
                            label=tr("Input-Ordner (Bilder)", "Input folder (images)"),
                            value=S["c_input"],
                            visible=False,
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
                                "Wie viele Bilder das finale Training-Set haben soll. Qualität geht vor Füllmaterial. Das Trainingsziel Krea 2 setzt 20 als Empfehlung; 12 hochwertige Bilder gelten als kompaktes Minimum.",
                                "How many images the final training set should contain. Quality over filler images. The Krea 2 training target recommends 20; 12 high-quality images are a compact minimum.",
                            ),
                        )
                        c_api_key = gr.Textbox(
                            label=tr("OpenAI API Key", "OpenAI API Key"),
                            value=S["c_api_key"],
                            visible=False,
                            type="password",
                            info=tr(
                                "Wird für die Bildanalyse benötigt. Kann auch als Umgebungsvariable OPENAI_API_KEY gesetzt werden.",
                                "Required for image analysis. Can also be set via environment variable OPENAI_API_KEY.",
                            ),
                            max_lines=1,
                        )
                        c_model = gr.Dropdown(
                            label=tr("Primäres AI-Modell", "Primary AI model"),
                            choices=OPENAI_MODEL_PRESET_CHOICES,
                            value=S["c_model"],
                            info=tr(
                                "Hauptmodell für den ersten Audit-Durchlauf. Für Krea 2 empfohlen: `gpt-5.6-luna` ohne Reasoning. Schwierige Grenzfälle können optional an `gpt-5.6-terra` eskaliert werden. Eigene Modellnamen können bei unterstützter Gradio-Version trotzdem eingetragen werden.",
                                "Main model for the first audit pass. Recommended for Krea 2: `gpt-5.6-luna` without reasoning. Difficult borderline cases can optionally escalate to `gpt-5.6-terra`. On supported Gradio versions, you can still enter custom model names.",
                            ),
                            **openai_model_dropdown_kwargs(),
                        )
                        c_audit_reasoning_effort = gr.Dropdown(
                            label=tr("Reasoning Effort – Bildaudit", "Reasoning effort – image audit"),
                            choices=REASONING_EFFORT_CHOICES,
                            value=S["c_audit_reasoning_effort"],
                            info=tr(
                                "Reasoning-Aufwand für jeden normalen Bildaudit-Call. Für Luna und klar strukturierte Extraktion ist `none` der empfohlene schnelle Standard; `low` kann bei schwierigen visuellen Grenzfällen helfen.",
                                "Reasoning effort for every regular image-audit call. For Luna and structured extraction, `none` is the recommended fast default; `low` may help with difficult visual edge cases.",
                            ),
                        )
                        c_openai_token_limit = gr.Number(
                            label=tr("OpenAI Token-Limit pro Lauf", "OpenAI token limit per run"),
                            value=S["c_openai_token_limit"],
                            precision=0,
                            info=tr(
                                "0 = kein Limit. Wenn das Limit erreicht ist, stoppt der Curator vor weiteren OpenAI-API-Calls. Tipp: bei 2.500.000 Tageslimit lieber etwas Reserve lassen, z. B. 2.400.000.",
                                "0 = no limit. Once the limit is reached, the curator stops before any further OpenAI API calls. Tip: if your daily cap is 2,500,000, leave some headroom, e.g. 2,400,000."
                            ),
                        )
                        c_use_trigger_check = gr.Checkbox(
                            label=tr("Trigger-Check aktivieren", "Enable trigger check"),
                            value=S["c_use_trigger_check"],
                            info=tr(
                                "Prüft das Triggerwort per KI auf Kollisionen oder problematische Ähnlichkeiten. Wenn deaktiviert, wird die Prüfung komplett übersprungen.",
                                "Checks the trigger word via AI for collisions or problematic similarities. If disabled, the check is skipped entirely.",
                            ),
                        )
                        c_trigger_model = gr.Dropdown(
                            label=tr("Trigger-Check-Modell", "Trigger-check model"),
                            choices=[""] + OPENAI_MODEL_PRESET_CHOICES,
                            value=S["c_trigger_model"],
                            info=tr(
                                "Optional separates Modell für die Triggerwort-Prüfung. Leer = primäres Modell verwenden. Wird nur genutzt, wenn der Trigger-Check aktiviert ist.",
                                "Optional separate model for trigger-word checks. Empty = use primary model. Only used when trigger check is enabled.",
                            ),
                            **openai_model_dropdown_kwargs(),
                        )
                        c_trigger_reasoning_effort = gr.Dropdown(
                            label=tr("Reasoning Effort – Trigger-Check", "Reasoning effort – trigger check"),
                            choices=REASONING_EFFORT_CHOICES,
                            value=S["c_trigger_reasoning_effort"],
                            info=tr(
                                "Wird nur verwendet, wenn der Trigger-Check aktiv ist. `none` reicht normalerweise aus.",
                                "Used only when trigger checking is enabled. `none` is normally sufficient.",
                            ),
                        )

                with gr.Accordion(tr("🧠 Modellstrategie & Eskalation", "🧠 Model strategy & escalation"), open=False):
                    gr.Markdown(tr(
                        "<details>"
                        "<summary><b>ℹ️ Was bedeutet Eskalation und wann brauche ich das?</b></summary>"
                        "\n\n"
                        "Standardmäßig bewertet ein einziges Modell (`gpt-5.6-luna` empfohlen) "
                        "alle Bilder. Frühere Nano-Modelle haben sich bei schwierigen visuellen Grenzfällen als zu "
                        "ungenau für die Erkennung von Filter-Hauttextur und extremen "
                        "Kamerawinkeln erwiesen – diese Bilder rutschten regelmäßig fälschlich "
                        "in 'train_ready'. Bei **Grenzfällen** – also Bildern, die das "
                        "Hauptmodell nicht klar als "
                        "'gut genug' oder 'rauswerfen' einordnen kann – kann der Curator diese "
                        "Bilder optional an ein **stärkeres zweites Modell** weiterleiten "
                        "(z. B. `gpt-5.6-terra` oder `gpt-5.6-sol`).\n\n"
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
                        "By default, a single model (`gpt-5.6-luna` recommended) scores all "
                        "images. Earlier nano-class models proved too inaccurate at "
                        "spotting filter-smoothed skin and extreme camera angles - those "
                        "images regularly slipped into 'train_ready' incorrectly. For "
                        "**borderline cases** - "
                        "images the main model can't clearly classify as 'keep' or 'reject' - "
                        "the curator can optionally forward these images to a **stronger second "
                        "model** (e.g. `gpt-5.6-terra` or `gpt-5.6-sol`).\n\n"
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
                            "Empfohlen: zunächst aus. `gpt-5.6-luna` übernimmt den Routine-Audit. "
                            "Aktiviere die Eskalation, wenn schwierige Grenzfälle gezielt durch "
                            "`gpt-5.6-terra` oder `gpt-5.6-sol` nachgeprüft werden sollen. Zusätzliche "
                            "Kosten entstehen nur für tatsächlich eskalierte Bilder.",
                            "Recommended: off initially. `gpt-5.6-luna` handles routine audits. "
                            "Enable escalation when difficult borderline cases should be rechecked "
                            "by `gpt-5.6-terra` or `gpt-5.6-sol`. Extra cost applies only to images "
                            "that are actually escalated.",
                        ),
                    )
                    with gr.Row():
                        c_review_escalation_model = gr.Dropdown(
                            label=tr("Eskalationsmodell", "Escalation model"),
                            choices=[""] + OPENAI_MODEL_PRESET_CHOICES,
                            value=S["c_review_escalation_model"],
                            info=tr(
                                "Stärkeres Modell für die Eskalation. Leer = Eskalation effektiv aus, auch wenn der Schalter oben an ist. Für Luna als Hauptmodell empfohlen: `gpt-5.6-terra`.",
                                "Stronger model for escalation. Empty = escalation effectively off, even if the switch above is on. Recommended with Luna as the main model: `gpt-5.6-terra`.",
                            ),
                            **openai_model_dropdown_kwargs(),
                        )
                        c_review_escalation_reasoning_effort = gr.Dropdown(
                            label=tr("Reasoning Effort – Eskalation", "Reasoning effort – escalation"),
                            choices=REASONING_EFFORT_CHOICES,
                            value=S["c_review_escalation_reasoning_effort"],
                            info=tr(
                                "Reasoning nur für tatsächlich eskalierte Grenzfälle. `low` ist der empfohlene Ausgangspunkt mit Terra; höhere Stufen erhöhen Laufzeit und Tokenverbrauch.",
                                "Reasoning only for cases that are actually escalated. `low` is the recommended starting point with Terra; higher levels increase latency and token usage.",
                            ),
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
                        "**direkt verworfen** und landet in `05_reject` (wenn Reject-Export aktiv ist). Keine zweite Chance.\n\n"
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
                        "**immediately rejected** and lands in `05_reject` (if reject export is enabled). No second chance.\n\n"
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
                            label=tr("Stufe 2: Min. Schärfe (Gesicht, Fallback)", "Stage 2: min sharpness (face, fallback)"),
                            minimum=10,
                            maximum=200,
                            step=5,
                            value=S["c_face_min_blur"],
                            info=tr(
                                "Globaler Fallback-Wert. Wird nur verwendet, wenn die shot-type-spezifischen Werte unten alle 0 sind. Empfohlen: 45.",
                                "Global fallback value. Only used when all shot-type-specific values below are 0. Recommended: 45.",
                            ),
                        )
                    with gr.Accordion(
                        tr(
                            "Schärfe-Schwellen pro Shot-Typ (empfohlen)",
                            "Sharpness thresholds per shot type (recommended)",
                        ),
                        open=True,
                    ):
                        gr.Markdown(tr(
                            "Bei Closeups (Headshots) ist die Gesichts-Bbox sehr groß. "
                            "Glatte Wangen-/Stirnflächen drücken die Schärfemessung statistisch nach unten, "
                            "selbst wenn das Bild scharf ist. Daher braucht headshot eine niedrigere Schwelle "
                            "als full_body (kleines Gesicht, hohe Detail-Dichte). "
                            "Werte 0 deaktivieren den shot-type-spezifischen Pfad und nutzen den globalen Fallback oben.",
                            "Closeups (headshots) have a very large face bbox. Smooth cheek/forehead areas "
                            "statistically drag the sharpness measurement down even when the image is sharp. "
                            "So headshot needs a lower threshold than full_body (small face, high detail density). "
                            "Values of 0 disable the shot-type-specific path and fall back to the global value above.",
                        ))
                        with gr.Row():
                            c_face_min_blur_headshot = gr.Slider(
                                label=tr("Headshot", "Headshot"),
                                minimum=0,
                                maximum=200,
                                step=5,
                                value=S["c_face_min_blur_headshot"],
                                info=tr(
                                    "Empfohlen: 25. Closeups vertragen weniger Variance.",
                                    "Recommended: 25. Closeups tolerate less variance.",
                                ),
                            )
                            c_face_min_blur_medium = gr.Slider(
                                label=tr("Medium", "Medium"),
                                minimum=0,
                                maximum=200,
                                step=5,
                                value=S["c_face_min_blur_medium"],
                                info=tr(
                                    "Empfohlen: 35. Mittlere Schwelle für Halbfiguren.",
                                    "Recommended: 35. Middle threshold for medium shots.",
                                ),
                            )
                            c_face_min_blur_full_body = gr.Slider(
                                label=tr("Full-Body", "Full-body"),
                                minimum=0,
                                maximum=200,
                                step=5,
                                value=S["c_face_min_blur_full_body"],
                                info=tr(
                                    "Empfohlen: 45. Kleines Gesicht im Frame braucht höhere Schwelle.",
                                    "Recommended: 45. Small face in frame needs higher threshold.",
                                ),
                            )
                    with gr.Row():
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
                    with gr.Group(visible=False):
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

                with gr.Accordion(tr("🖼️ Smart-Rahmenbereinigung", "🖼️ Smart frame cleanup"), open=False, visible=False):
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
                            label=tr("Lokale Rahmenbereinigung aktivieren", "Enable local frame cleanup"),
                            value=S["c_ig_frame_crop"],
                            info=tr(
                                "Hauptschalter für die Rand-Erkennung und das Wegschneiden. Empfohlen: an, sobald deine Bilder aus Social Media stammen. Bei reinen DSLR-/Studio-Fotos kannst du es ausschalten.",
                                "Main switch for border detection and cropping. Recommended: on as soon as your images come from social media. For pure DSLR/studio photos you can turn it off.",
                            ),
                        )
                        c_ig_two_stage_bar = gr.Checkbox(
                            label=tr("Erweiterte Rahmentypen erkennen", "Detect advanced frame types"),
                            value=S["c_ig_two_stage_bar"],
                            info=tr(
                                "Erkennt zusätzlich asymmetrische Canvas-Ränder, Status-/Navigationsleisten, Polaroid-Unterkanten und verlaufende Social-Media-Flächen. Die vier Seiten werden unabhängig geprüft.",
                                "Also detects asymmetric canvas borders, status/navigation bars, Polaroid-style bottom borders and gradient social-media canvases. All four sides are analysed independently.",
                            ),
                        )
                    c_frame_cleanup_mode = gr.Dropdown(
                        label=tr("Verhalten bei Rahmenfunden", "Frame-cleanup behavior"),
                        choices=[
                            (tr("Hohe Sicherheit automatisch; mittlere Fälle prüfen", "Auto-apply high confidence; review medium cases"), "auto_high_review_medium"),
                            (tr("Hohe und mittlere Sicherheit automatisch anwenden", "Auto-apply high and medium confidence"), "auto_high_keep_medium"),
                            (tr("Nur Vorschläge; nichts automatisch beschneiden", "Suggestions only; never auto-crop"), "suggest_only"),
                        ],
                        value=S.get("c_frame_cleanup_mode", "suggest_only"),
                        allow_custom_value=False,
                        info=tr(
                            "Sicherer Standard: Nur Vorschläge. Nach bestandenem Regressionstest kann der automatische Modus bewusst aktiviert werden. Mittlere Fälle erscheinen im Tab Rahmenprüfung. Es entstehen keine LLM-Aufrufe.",
                            "Safe default: suggestions only. Automatic mode can be enabled deliberately after regression validation. Medium cases appear in the Frame Review tab. No LLM calls are made.",
                        ),
                    )
                    c_frame_pause_on_medium = gr.Checkbox(
                        label=tr("Vor dem Audit bei mittleren Rahmenfällen anhalten", "Pause before audit for medium-confidence frame cases"),
                        value=S.get("c_frame_pause_on_medium", False),
                        info=tr(
                            "Optional. Prüft die lokalen Rahmen-Caches zuerst. Gibt es ungeklärte mittlere Fälle, endet der Lauf vor kostenpflichtigen Audits. Nach der Entscheidung im Tab Rahmenprüfung erneut starten; unveränderte Analysen kommen aus dem Cache.",
                            "Optional. Checks local frame caches first. If unresolved medium cases exist, the run stops before paid audits. Review them in the Frame Review tab and restart; unchanged analysis is loaded from cache.",
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
                    gr.Markdown(tr(
                        "**Getrennte Rettungsmechanik:** Ein schwaches Full-Body-Bild kann zusätzlich als Medium-Crop geprüft werden. Das Original bleibt ein Full Body; der Crop ist ein eigener Kandidat.",
                        "**Separate rescue mechanism:** A weak full-body image can additionally be tested as a medium crop. The original remains full body; the crop is a separate candidate.",
                    ))
                    with gr.Row():
                        c_medium_rescue_crop = gr.Checkbox(
                            label=tr("Medium-Rettungs-Crop aktivieren", "Enable medium rescue crop"),
                            value=S["c_medium_rescue_crop"],
                            info=tr(
                                "Empfohlen: an. Versucht bei schwachen Ganzkörperbildern Gesicht, Schultern, Torso und möglichst Hüfte als brauchbaren Medium Shot zu retten. Verändert das Original nicht.",
                                "Recommended: on. Tries to rescue face, shoulders, torso and preferably hips from weak full-body images as a usable medium shot. Does not alter the original.",
                            ),
                        )
                        c_medium_rescue_gain = gr.Slider(
                            label=tr("Mindestvorsprung des Medium-Crops", "Min medium-crop score gain"),
                            minimum=0,
                            maximum=20,
                            step=1,
                            value=S["c_medium_rescue_gain"],
                            info=tr(
                                "Wie viele Punkte besser der Medium-Rettungs-Crop sein muss. Empfohlen: 4, da er zusätzlich eine fehlende Shot-Kategorie abdecken kann.",
                                "How many points better the medium rescue crop must score. Recommended: 4 because it may also fill a missing shot category.",
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

                    gr.Markdown(tr(
                        "**Weiche Canon-Repräsentation:** Nach Bestätigung des Subject Profiles erhält die gewählte kanonische Haarfarbe einen abnehmenden Auswahlbonus. Die Headshot-/Medium-/Full-Body-Quoten bleiben unverändert. Review- und Reject-Bilder werden durch den Canon-Bonus niemals automatisch hochgesetzt; eine ausdrückliche Priority-Markierung bleibt davon unberührt.",
                        "**Soft canon representation:** After the Subject Profile is confirmed, the selected canonical hair color receives a diminishing selection bonus. Headshot/medium/full-body quotas remain unchanged. Review and reject images are never promoted automatically by the canon bonus; an explicit Priority override remains unaffected.",
                    ))
                    c_use_canon_representation = gr.Checkbox(
                        label=tr("Canon-Repräsentation bei der Auswahl fördern", "Promote canon representation during selection"),
                        value=S["c_use_canon_representation"],
                        info=tr(
                            "Wirkt erst beim Captioning aus dem bestätigten Profil. Der Bonus gilt nur für bereits geeignete Keep-Kandidaten innerhalb der jeweiligen Shot-Kategorie.",
                            "Takes effect when continuing from the confirmed profile. The bonus only applies to already eligible keep candidates within the current shot category.",
                        ),
                    )
                    with gr.Row():
                        c_canon_representation_target = gr.Slider(
                            label=tr("Weiches Canon-Ziel", "Soft canon target"),
                            minimum=0,
                            maximum=5,
                            step=1,
                            value=S["c_canon_representation_target"],
                            info=tr(
                                "Gewünschte Mindestanzahl guter Bilder mit der bestätigten Canon-Haarfarbe. Standard bei 20 Bildern: 3. Kein hartes Minimum.",
                                "Desired minimum number of good images with the confirmed canonical hair color. Default for 20 images: 3. Not a hard minimum.",
                            ),
                        )
                        c_canon_max_quality_gap = gr.Slider(
                            label=tr("Max. Qualitätsabstand für Canon-Bonus", "Max quality gap for canon bonus"),
                            minimum=0,
                            maximum=15,
                            step=0.5,
                            value=S["c_canon_max_quality_gap"],
                            info=tr(
                                "Ein Canon-Kandidat erhält nur dann den Bonus, wenn er höchstens so viele Quality-Total-Punkte hinter der besten Alternative liegt. Standard: 5.",
                                "A canon candidate only receives the bonus if it is no more than this many quality-total points behind the best alternative. Default: 5.",
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

                with gr.Accordion(tr("🧬 Subject Profile / Pipeline-Modus", "🧬 Subject Profile / pipeline mode"), open=False):
                    gr.Markdown(tr(
                        "Phase 2 baut nach dem Bild-Audit ein zentrales Profil des Models. "
                        "Reject- und Review-Bilder werden dafür ausgeschlossen. Bei großen Datasets "
                        "wird ein stratifiziertes Sample an den Normalizer geschickt und die Regeln "
                        "werden lokal auf alle verwertbaren Bilder angewendet. Marker wie Brille, "
                        "Tattoos und Piercings bleiben strikt force-only-when-visible.\n\n"
                        "👉 **Profil bearbeiten** und das anschließende Captioning aus dem Profil "
                        "starten findest du im eigenen Tab `🧬 Subject Profile`.",
                        "Phase 2 builds a central subject profile after image audit. Reject and review "
                        "images are excluded. For large datasets, a stratified sample is sent to the "
                        "normalizer and the rules are applied locally to all usable images. Markers "
                        "such as glasses, tattoos and piercings remain strictly force-only-when-visible.\n\n"
                        "👉 **Editing the profile** and starting captioning from the profile is done "
                        "in the dedicated `🧬 Subject Profile` tab.",
                    ))
                    c_pipeline_mode = gr.Dropdown(
                        label=tr("Pipeline-Modus", "Pipeline mode"),
                        choices=[
                            (tr("Single Pass – Profil automatisch nutzen", "Single pass – use profile automatically"), "single_pass"),
                            (tr("Profile then Caption – UI-Gate (Profil-Tab nutzen)", "Profile then caption – UI gate (use Profile tab)"), "profile_then_caption"),
                        ],
                        value=S["c_pipeline_mode"],
                        info=tr(
                            "Single Pass nutzt das Profil automatisch. Profile then Caption pausiert nach dem Profil-Build, du gehst dann in den Profil-Tab und startest Captioning dort separat.",
                            "Single pass uses the profile automatically. Profile then Caption pauses after the profile build; switch to the Profile tab to edit and start captioning separately.",
                        ),
                    )
                    c_profile_normalizer_model = gr.Dropdown(
                        label=tr("Profile-Normalizer-Modell", "Profile normalizer model"),
                        choices=OPENAI_MODEL_PRESET_CHOICES,
                        value=S["c_profile_normalizer_model"],
                        info=tr(
                            "Modell für den einen zusätzlichen Profil-Call pro Lauf. Für Krea 2 empfohlen: gpt-5.6-terra mit niedrigem Reasoning.",
                            "Model for the single additional profile call per run. Recommended for Krea 2: gpt-5.6-terra with low reasoning.",
                        ),
                        **openai_model_dropdown_kwargs(),
                    )
                    c_profile_reasoning_effort = gr.Dropdown(
                        label=tr("Reasoning Effort – Subject Profile", "Reasoning effort – subject profile"),
                        choices=REASONING_EFFORT_CHOICES,
                        value=S["c_profile_reasoning_effort"],
                        info=tr(
                            "Der Profil-Call muss stabile und variable Merkmale über viele Bilder abgleichen. `low` ist der empfohlene Standard; `medium` kann bei widersprüchlichen Datensätzen sinnvoll sein.",
                            "The profile call reconciles stable and variable traits across many images. `low` is the recommended default; `medium` can help with contradictory datasets.",
                        ),
                    )
                    with gr.Row():
                        c_profile_sample_threshold = gr.Slider(
                            label=tr("Sampling ab N Bildern", "Sample when above N images"),
                            minimum=20,
                            maximum=2000,
                            step=10,
                            value=S["c_profile_sample_threshold"],
                            info=tr(
                                "Bis zu dieser Anzahl gehen alle verwertbaren Bilder in den Profil-Normalizer. Darüber wird gesampelt.",
                                "Up to this number, all usable images go into the profile normalizer. Above this, stratified sampling is used.",
                            ),
                        )
                        c_profile_sample_size = gr.Slider(
                            label=tr("Profil-Sample-Größe", "Profile sample size"),
                            minimum=20,
                            maximum=300,
                            step=5,
                            value=S["c_profile_sample_size"],
                            info=tr(
                                "Wie viele Bilder bei großen Datasets maximal für den Normalizer ausgewählt werden. Empfehlung: 80.",
                                "Maximum number of images selected for the normalizer on large datasets. Recommended: 80.",
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
                        "**Trainingsziel und Caption-Policy:**\n\n"
                        "Das oben gewählte Trainingsziel bestimmt Workflow, Promptfamilie und Caption-Engine. "
                        "Die Einzelfelder hier steuern nur, welche Merkmale diese Engine verwenden darf.\n\n"
                        "**Aktive Caption-Felder:**\n\n"
                        "Welche Eigenschaften gehen in die Beschreibung? Die "
                        "Antwort hängt am Basismodell:\n\n"
                        "**ERNIE** – alle Felder einschließen. ERNIE-Image hat im "
                        "Default einen asiatischen Bias, der durch redundante Anker "
                        "(blonde hair, blue eyes, fair skin) ausgeglichen wird. "
                        "Auch permanente Eigenschaften gehören in jede Caption.\n\n"
                        "**Z-Image Base** – nur veränderliche Felder. Z-Image hat "
                        "ein starkes Sprachverständnis und kennt Standardkonzepte "
                        "schon. Permanente Identitätsmerkmale (Hautton, Augenfarbe, "
                        "Körperbau, konstante Frisur) werden weggelassen, damit das "
                        "Trigger-Wort die Person-Identität sauber absorbiert. Nur "
                        "variable Sachen (Kleidung, Pose, Hintergrund, Brille wenn "
                        "wechselnd, Hair-when-variable, variable Piercings/Ohrschmuck, Make-up, Bildstil) "
                        "kommen rein. Vorteil: bessere Steuerbarkeit bei Inferenz "
                        "('Kathi mit roten Haaren' funktioniert sauber, weil der "
                        "Trigger 'blonde' nicht in der Caption mitfährt).\n\n"
                        "**Regel für variable Identitätsmerkmale:** Das Subject Profile "
                        "normalisiert Haarfarbe/-form, Augenfarbe, Bart und Brille und "
                        "legt eine kanonische Baseline fest. Im Standardmodus wird nur "
                        "eine Abweichung captioniert (z. B. 'red hair' bei canon blond). "
                        "Alternativ kann bei echter Variation jeder sichtbare Wert "
                        "captioniert werden. Brillenbegriffe werden dabei auf eine "
                        "kanonische Beschreibung vereinheitlicht.\n\n"
                        "**Wenn du unsicher bist:** ERNIE ist der robustere Default, "
                        "Z-Image Base ist die saubere Wahl wenn du gezielt auf "
                        "Z-Image_Base trainierst und maximale Inferenz-Flexibilität "
                        "willst."
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
                        "**Training target and caption policy:**\n\n"
                        "The training target selected above chooses the workflow, prompt family and caption engine. "
                        "The fields here only control which facts that engine may use.\n\n"
                        "**Active caption fields:**\n\n"
                        "Which attributes go into the description? The answer "
                        "depends on the base model:\n\n"
                        "**ERNIE** – include all fields. ERNIE-Image has an "
                        "Asia-leaning default bias that's compensated by redundant "
                        "anchors (blonde hair, blue eyes, fair skin). Even "
                        "persistent attributes belong in every caption.\n\n"
                        "**Z-Image Base** – only variable fields. Z-Image has "
                        "strong language understanding and already knows standard "
                        "concepts. Persistent identity features (skin tone, eye "
                        "color, body build, consistent hair) are omitted so the "
                        "trigger word cleanly absorbs the person identity. Only "
                        "variable things (clothing, pose, background, glasses if "
                        "they vary, hair-when-variable, variable piercings or ear jewelry, makeup, visual "
                        "style) go in. Benefit: better inference steerability "
                        "('Kathi with red hair' works cleanly because the trigger "
                        "doesn't drag 'blonde' along in every caption).\n\n"
                        "**Variable identity feature rule:** The Subject Profile "
                        "normalizes hair color/form, eye color, beard and glasses and "
                        "stores a canonical baseline. In the default mode only a "
                        "deviation is captioned (for example 'red hair' when blonde "
                        "is canonical). Alternatively, every visible value can be "
                        "captioned once genuine variation is detected. Glasses terms "
                        "are consolidated into one canonical description.\n\n"
                        "**Krea 2** – natural-language captions are generated after selection. "
                        "Stable identity and body traits such as tattoos are stored in the subject profile "
                        "and omitted from captions; scene-specific details remain captioned. Selecting this "
                        "training target also applies the recommended 20-image, 40/35/25 distribution and Luna/Terra "
                        "model defaults.\n\n"
                        "**If unsure:** Choose the training target matching the base model."
                        "</details>",
                    ))
                    gr.Markdown(tr(
                        "Das Trainingsziel wird oben gewählt. Die folgenden Felder passen nur die Caption-Policy an; die gewählte Prompt-Familie und Caption-Engine bleiben aktiv.",
                        "Choose the training target at the top. The fields below only customize caption policy; the selected prompt family and caption engine remain active.",
                    ))
                    c_captions = gr.CheckboxGroup(
                        label=tr("Aktive Caption-Felder", "Active caption fields"),
                        choices=CAPTION_FIELD_CHOICES,
                        value=S["c_captions"],
                        info=tr(
                            "Welche Merkmale in die Trainings-Captions aufgenommen "
                            "werden. Empfehlung pro Basismodell: bei ERNIE "
                            "alle Felder einschließen, bei Z-Image Base nur "
                            "variable Felder (Kleidung, Pose, Hintergrund, "
                            "Brille-wenn-variabel, Hair-when-variable, variable Piercings/"
                            "Ohrschmuck, Make-up, Visual-Style). Permanente Merkmale "
                            "(Hautton, Augenfarbe, Körperbau, konstante Haare) "
                            "lässt man bei Z-Image Base weg, damit das Trigger-Wort "
                            "die Identität sauber absorbiert. Im Zweifel auf das "
                            "Preset oben verlassen.",
                            "Which attributes go into the training captions. "
                            "Recommendation per base model: for ERNIE include "
                            "all fields; for Z-Image Base only variable fields "
                            "(clothing, pose, background, glasses-when-variable, "
                            "hair-when-variable, variable piercings/ear jewelry, makeup, "
                            "visual style). Persistent features (skin tone, eye "
                            "color, body build, constant hair) are omitted with "
                            "Z-Image Base so the trigger word cleanly absorbs "
                            "identity. When in doubt trust the preset above.",
                        ),
                    )
                    c_caption_policy_status = gr.Markdown(
                        caption_policy_adjustment_note(S.get("c_training_target"), S.get("c_captions"))
                    )
                    c_variable_feature_mode = gr.Dropdown(
                        label=tr("Regel für variable Identitätsmerkmale", "Variable identity feature rule"),
                        choices=[
                            (
                                tr("Nur Abweichungen von der kanonischen Erscheinung", "Only deviations from the canonical appearance"),
                                "canonical_deviations",
                            ),
                            (
                                tr("Bei echter Variation jeden sichtbaren Wert captionieren", "Caption every visible value when genuine variation exists"),
                                "all_visible_when_variable",
                            ),
                        ],
                        value=S.get("c_variable_feature_mode", "canonical_deviations"),
                        info=tr(
                            "Standard: Die im Subject Profile festgelegte Grunderscheinung gehört zum Triggerwort; nur Abweichungen werden genannt. Alternative: Sobald echte Variation erkannt ist, wird der jeweilige sichtbare Wert in allen Bildern genannt. Gilt für Haarfarbe/-form, Augenfarbe, Bart und Brillenstatus.",
                            "Default: the canonical appearance stored in the Subject Profile belongs to the trigger; only deviations are captioned. Alternative: once genuine variation is detected, every visible value is captioned. Applies to hair color/form, eye color, beard and glasses state.",
                        ),
                    )
                    c_krea_caption_model = gr.Dropdown(
                        label=tr("Krea-2-Caption-Modell", "Krea 2 caption model"),
                        choices=OPENAI_MODEL_PRESET_CHOICES,
                        value=S["c_krea_caption_model"],
                        info=tr(
                            "Nur beim Trainingsziel `Krea 2`: Erstellt nach der finalen Bildauswahl natürliche englische Captions aus Originalbild, Audit und bestätigtem Subject Profile. Empfehlung: gpt-5.6-luna.",
                            "Only for the `Krea 2` training target: creates natural English captions after final image selection using the original image, audit and confirmed subject profile. Recommended: gpt-5.6-luna.",
                        ),
                        **openai_model_dropdown_kwargs(),
                    )
                    c_krea_caption_reasoning_effort = gr.Dropdown(
                        label=tr("Reasoning Effort – Krea-Caption", "Reasoning effort – Krea caption"),
                        choices=REASONING_EFFORT_CHOICES,
                        value=S["c_krea_caption_reasoning_effort"],
                        info=tr(
                            "Reasoning für die finale natürliche Caption jedes ausgewählten Bildes. `none` ist der empfohlene Standard; `low` kann bei komplexen Szenen getestet werden.",
                            "Reasoning for the final natural caption of each selected image. `none` is the recommended default; `low` can be tested for complex scenes.",
                        ),
                    )
                    c_use_krea_caption_repair = gr.Checkbox(
                        label=tr("Automatischen Caption-Reparaturversuch verwenden", "Use automatic caption repair attempt"),
                        value=S["c_use_krea_caption_repair"],
                        info=tr(
                            "Wenn die erste Krea-Caption leer ist oder gegen Profil-/Caption-Regeln verstößt, wird nur dieses Bild einmal mit dem Reparaturmodell neu captioniert. Audit, Profil und Auswahl werden nicht wiederholt.",
                            "If the first Krea caption is empty or violates profile/caption rules, only that image is captioned once more with the repair model. Audit, profile and selection are not repeated.",
                        ),
                    )
                    with gr.Row():
                        c_krea_caption_repair_model = gr.Dropdown(
                            label=tr("Krea-2-Caption-Reparaturmodell", "Krea 2 caption repair model"),
                            choices=OPENAI_MODEL_PRESET_CHOICES,
                            value=S["c_krea_caption_repair_model"],
                            info=tr(
                                "Wird ausschließlich nach einem fehlgeschlagenen oder regelwidrigen ersten Caption-Versuch aufgerufen. Empfehlung: gpt-5.6-terra.",
                                "Called only after a failed or policy-invalid first caption attempt. Recommended: gpt-5.6-terra.",
                            ),
                            **openai_model_dropdown_kwargs(),
                        )
                        c_krea_caption_repair_reasoning_effort = gr.Dropdown(
                            label=tr("Reasoning Effort – Caption-Reparatur", "Reasoning effort – caption repair"),
                            choices=REASONING_EFFORT_CHOICES,
                            value=S["c_krea_caption_repair_reasoning_effort"],
                            info=tr(
                                "Der Reparaturversuch erhält die konkrete Validierungsfehlermeldung und die erste Caption. `low` ist der empfohlene Standard.",
                                "The repair attempt receives the exact validation error and first caption. `low` is the recommended default.",
                            ),
                        )
                    c_target_defaults_event = c_training_target.change(
                        fn=apply_training_target_defaults,
                        inputs=[
                            c_training_target, c_captions, c_target,
                            c_ratio_h, c_ratio_m, c_ratio_f,
                            c_model, c_audit_reasoning_effort,
                            c_trigger_reasoning_effort, c_review_escalation_reasoning_effort,
                            c_profile_normalizer_model, c_profile_reasoning_effort,
                            c_krea_caption_model, c_krea_caption_reasoning_effort,
                            c_pipeline_mode,
                        ],
                        outputs=[
                            c_captions, c_target,
                            c_ratio_h, c_ratio_m, c_ratio_f,
                            c_model, c_audit_reasoning_effort,
                            c_trigger_reasoning_effort, c_review_escalation_reasoning_effort,
                            c_profile_normalizer_model, c_profile_reasoning_effort,
                            c_krea_caption_model, c_krea_caption_reasoning_effort,
                            c_pipeline_mode,
                        ],
                    )

                    c_target_defaults_event.then(
                        fn=caption_policy_adjustment_note,
                        inputs=[c_training_target, c_captions],
                        outputs=[c_caption_policy_status],
                    )
                    c_captions.change(
                        fn=caption_policy_adjustment_note,
                        inputs=[c_training_target, c_captions],
                        outputs=[c_caption_policy_status],
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
                        "**05_reject** (optional): Verworfene Bilder mit Begründung in "
                        "der Caption-Datei (z. B. 'rejected: face_blur_too_low'). Nützlich "
                        "zum Debuggen, ob deine Schwellen zu streng sind. Bei großen "
                        "Datasets kann das viele MB werden.\n\n"
                        "**04_review** (optional): Bilder die das Hauptmodell "
                        "als 'review' markiert hat – Grenzfälle, die du visuell prüfen "
                        "kannst. Wenn die Eskalation an ist, sind das die Bilder, die "
                        "zusätzlich vom stärkeren Modell entschieden wurden.\n\n"
                        "**06_needs_manual_review** (immer, wenn nötig): Sammelt alles, "
                        "was menschliche Prüfung braucht – z. B. technische Audit-Ausfälle (Präfix "
                        "`AUDITTECH_`) und Hard-Flag-Identitätsmismatches (Präfix `IDCHECK_`).\n\n"
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
                        "**05_reject** (optional): Rejected images with reason in the "
                        "caption file (e.g. 'rejected: face_blur_too_low'). Useful for "
                        "debugging whether your thresholds are too strict. For large "
                        "datasets this can grow to many MB.\n\n"
                        "**04_review** (optional): Images the main model "
                        "flagged as 'review' – borderline cases for visual inspection. "
                        "When escalation is on, these are the images additionally "
                        "decided by the stronger model.\n\n"
                        "**06_needs_manual_review** (always, when needed): Collects "
                        "everything that needs human review – e.g. technical audit failures (prefix "
                        "`AUDITTECH_`) and hard-flagged identity mismatches (prefix "
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
                                "Speichert Bilder im Grenzbereich in `04_review` zur manuellen Sichtung. Empfohlen: an, besonders wenn dir der Curator gerade neue Schwellen lernt.",
                                "Saves borderline images to `04_review` for manual review. Recommended: on, especially while you're tuning the curator's thresholds.",
                            ),
                        )
                        c_exp_reject = gr.Checkbox(
                            label=tr("Verworfene Bilder exportieren", "Export rejected images"),
                            value=S["c_exp_reject"],
                            info=tr(
                                "Speichert verworfene Bilder mit Reject-Grund in `05_reject`. Empfohlen: an für die ersten Läufe (zum Debuggen). Bei großen Produktions-Datasets kannst du es ausschalten, um Platz zu sparen.",
                                "Saves rejected images with reject reason to `05_reject`. Recommended: on for initial runs (for debugging). For large production datasets you can turn it off to save space.",
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
                    c_controlled_buckets = gr.Checkbox(
                        label=tr("Kontrollierte Buckets verwenden", "Use controlled buckets"),
                        value=S["c_controlled_buckets"],
                        info=tr(
                            "Standard: aus. Aus behält die natürliche Komposition der ausgewählten Bilder bei. An normalisiert erst beim finalen Export: Headshots 1024×1024, Medium und Full Body 832×1216. IG-Rahmenentfernung und Rettungs-Crops bleiben davon unabhängig.",
                            "Default: off. Off preserves the natural composition of selected images. On normalizes only during final export: headshots 1024×1024, medium and full body 832×1216. IG-frame removal and rescue crops remain independent.",
                        ),
                    )

                # ── Aktionen ──
                gr.Markdown("---")
                with gr.Row():
                    c_start_btn = gr.Button(tr("▶ Curator starten", "▶ Start curator"), variant="primary", scale=3)
                    c_stop_btn = gr.Button(tr("⏹ Abbrechen", "⏹ Cancel"), variant="stop", scale=1)
                    c_save_btn = gr.Button(tr("💾 Einstellungen speichern", "💾 Save settings"), variant="secondary", scale=2)

                with gr.Row():
                    c_status = gr.Textbox(label=tr("Status", "Status"), interactive=False, max_lines=1, scale=2)
                    c_openai_usage = gr.Textbox(label=tr("OpenAI Tokens", "OpenAI tokens"), interactive=False, max_lines=1, value=tr("💰 0 Requests | 0 Tokens", "💰 0 requests | 0 tokens"), scale=2)
                c_progress = gr.Slider(label=tr("Fortschritt", "Progress"), minimum=0, maximum=1, step=0.01, value=0, interactive=False)

                with gr.Row():
                    with gr.Column(scale=3):
                        c_log = gr.Textbox(label=tr("Live-Log", "Live log"), lines=18, max_lines=18, interactive=False, elem_classes=["log-box"])
                    with gr.Column(scale=2):
                        c_gallery = gr.Gallery(label=tr("Train-Ready Vorschau", "Train-ready preview"), columns=3, rows=3, height=380, object_fit="cover")

                # Alle Curator-Inputs als Liste (fuer Save und Start)
                curator_inputs = [
                    c_trigger, c_input, c_target, c_api_key, c_model, c_audit_reasoning_effort, c_openai_token_limit, c_use_trigger_check, c_trigger_model, c_trigger_reasoning_effort,
                    c_use_review_escalation, c_review_escalation_model, c_review_escalation_reasoning_effort,
                    c_review_escalation_score_min, c_review_escalation_score_max,
                    c_escalate_on_review, c_escalate_on_conflict, c_escalate_smart_crop, c_smart_crop_escalation_delta,
                    c_ratio_h, c_ratio_m, c_ratio_f,
                    c_keep_min, c_reject, c_min_side,
                    c_use_filesize, c_min_filesize,
                    c_use_blur, c_min_blur, c_face_min_blur, c_blur_norm_edge,
                    c_face_min_blur_headshot, c_face_min_blur_medium, c_face_min_blur_full_body,
                    c_use_early_phash,
                    c_use_early_phash_loop1, c_early_phash_thresh_1, c_early_phash_keep_1,
                    c_use_early_phash_loop2, c_early_phash_thresh_2, c_early_phash_keep_2,
                    c_subject_sanity, c_subject_min_torso,
                    c_ig_frame_crop, c_ig_two_stage_bar, c_frame_cleanup_mode, c_frame_pause_on_medium,
                    c_use_clip, c_use_phash, c_phash_thresh, c_clip_thresh,
                    c_smart_crop, c_crop_gain, c_crop_pad,
                    c_medium_rescue_crop, c_medium_rescue_gain,
                    c_use_cluster, c_max_outfit, c_max_session, c_use_diversity,
                    c_use_canon_representation, c_canon_representation_target, c_canon_max_quality_gap,
                    c_use_pose_diversity, c_pose_soft_limit, c_pose_penalty_weight,
                    c_use_arcface, c_arcface_hard, c_arcface_soft, c_arcface_trim,
                    c_arcface_min_faces, c_arcface_model, c_arcface_det_size,
                    c_training_target,
                    c_captions, c_variable_feature_mode, c_krea_caption_model, c_krea_caption_reasoning_effort,
                    c_use_krea_caption_repair, c_krea_caption_repair_model, c_krea_caption_repair_reasoning_effort,
                    c_pipeline_mode, c_profile_normalizer_model, c_profile_reasoning_effort,
                    c_profile_sample_threshold, c_profile_sample_size,
                    c_exp_review, c_exp_reject, c_exp_compare, c_controlled_buckets,
                ]

                c_run_event = c_start_btn.click(
                    fn=start_curator,
                    inputs=curator_inputs,
                    outputs=[c_log, c_gallery, c_progress, c_status, c_openai_usage],
                    concurrency_id="dataset_curator_process",
                    concurrency_limit=1,
                )
                c_stop_btn.click(
                    fn=kill_process,
                    outputs=[c_status],
                    queue=False,
                    show_progress="hidden",
                )

            with gr.TabItem(tr("3 · 🧬 Subject Profile", "3 · 🧬 Subject Profile"), id="profile", interactive=initial_preflight_ready) as profile_tab:
                gr.Markdown(tr(
                    "### Profil bearbeiten und Captioning starten\n\n"
                    "Workflow:\n"
                    "1. Im Curator-Tab `Pipeline-Modus = Profile then Caption` setzen und einen Lauf starten.\n"
                    "2. Nach der Pause hier `Profil laden` klicken.\n"
                    "3. Stable Identity prüfen (Dropdowns mit Confidence-Anzeige), bei Bedarf korrigieren. Speziell **Body Build** prüfen — bei Headshot-Dominanz wird er auf leer gesetzt, das ist Absicht.\n"
                    "4. Variable Traits per Bucket-Sicht prüfen, Ausreißer per `Re-Bucket all` korrigieren.\n"
                    "5. `Profil speichern` klicken.\n"
                    "6. `Captioning aus bestätigtem Profil starten` — es läuft kein neues Audit, nur Export + Caption-Build.\n",
                    "### Edit profile and start captioning\n\n"
                    "Workflow:\n"
                    "1. In the Curator tab set `Pipeline mode = Profile then Caption` and start a run.\n"
                    "2. After the pause click `Load profile` here.\n"
                    "3. Review stable identity (dropdowns with confidence indicators), correct if needed. Pay attention to **Body Build** — it is forced empty on headshot-dominated sets, by design.\n"
                    "4. Review variable traits in the bucket view; fix outliers via `Re-bucket all`.\n"
                    "5. Click `Save profile`.\n"
                    "6. `Start captioning from confirmed profile` — no new audit runs, only export + caption build.\n",
                ))

                p_state = gr.State({})
                p_raw_json = gr.State("")

                with gr.Row():
                    p_trigger = gr.Textbox(
                        label=tr("Trigger Word", "Trigger Word"),
                        value=S["c_trigger"],
                        visible=False,
                        max_lines=1,
                        scale=2,
                    )
                    p_input = gr.Textbox(
                        label=tr("Input-Ordner", "Input folder"),
                        value=S["c_input"],
                        visible=False,
                        max_lines=1,
                        scale=3,
                    )
                    p_load_btn = gr.Button(tr("📂 Profil laden", "📂 Load profile"), variant="secondary", scale=1)

                p_status = gr.Textbox(
                    label=tr("Status", "Status"),
                    value=tr("Noch kein Profil geladen.", "No profile loaded yet."),
                    interactive=False,
                    max_lines=2,
                )

                with gr.Tabs():
                    # ----- Subtab: Stable Identity -----
                    with gr.TabItem(tr("👤 Stable Identity", "👤 Stable identity")):
                        gr.Markdown(tr(
                            "Diese Werte definieren die **kanonische Identität**. Je Caption-Regel werden sie entweder vom Triggerwort getragen oder ausdrücklich genannt. Confidence-Anzeige hilft bei der Beurteilung. Body Build ist bei Headshot-Sets bewusst leer.",
                            "These values define the **canonical identity**. Depending on the caption rule, they are either carried by the trigger token or explicitly described. Confidence indicators help judge reliability. Body build is intentionally empty on headshot-dominated sets.",
                        ))
                        with gr.Row():
                            p_gender = gr.Dropdown(
                                label=tr("Gender", "Gender"),
                                choices=[""] + PROFILE_VOCAB_GENDER,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_gender_info = gr.Markdown("—")
                        with gr.Row():
                            p_skin = gr.Dropdown(
                                label=tr("Skin Tone", "Skin tone"),
                                choices=[""] + PROFILE_VOCAB_SKIN,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_skin_info = gr.Markdown("—")
                        with gr.Row():
                            p_eyes = gr.Dropdown(
                                label=tr("Eye Color", "Eye color"),
                                choices=[""] + PROFILE_VOCAB_EYES,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_eyes_info = gr.Markdown("—")
                        with gr.Row():
                            p_hair_texture = gr.Dropdown(
                                label=tr("Hair Texture", "Hair texture"),
                                choices=[""] + PROFILE_VOCAB_HAIR_TEXTURE,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_hair_texture_info = gr.Markdown("—")
                        with gr.Row():
                            p_body = gr.Dropdown(
                                label=tr("Körperbau", "Body build"),
                                choices=[""] + PROFILE_VOCAB_BODY,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_body_info = gr.Markdown("—")
                        with gr.Row():
                            p_body_height = gr.Dropdown(
                                label=tr("Größeneindruck", "Body height impression"),
                                choices=[""] + PROFILE_VOCAB_BODY_HEIGHT,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_body_height_info = gr.Markdown("—")

                        gr.Markdown(tr("**Kanonische variable Merkmale**", "**Canonical variable features**"))
                        gr.Markdown(tr(
                            "Diese Baselines steuern die Regel `nur Abweichungen`: Der kanonische Wert bleibt am Triggerwort, abweichende Werte werden captioniert.",
                            "These baselines drive the `deviations only` rule: the canonical value stays attached to the trigger token and deviations are captioned.",
                        ))
                        with gr.Row():
                            p_hair_color_baseline = gr.Dropdown(
                                label=tr("Kanonische Haarfarbe", "Canonical hair color"),
                                choices=[""] + PROFILE_VOCAB_HAIR_COLOR,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_beard_pattern = gr.Dropdown(
                                label=tr("Kanonischer Bartzustand", "Canonical beard state"),
                                choices=[""] + PROFILE_VOCAB_BEARD_PATTERN,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            p_beard_color = gr.Dropdown(
                                label=tr("Kanonische Bartfarbe", "Canonical beard color"),
                                choices=PROFILE_VOCAB_BEARD_COLOR,
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                            )

                        gr.Markdown(tr("**Brille (Identity Marker)**", "**Glasses (identity marker)**"))
                        with gr.Row():
                            p_glasses_regular = gr.Checkbox(
                                label=tr("Trägt regelmäßig Brille", "Wears glasses regularly"),
                                value=False,
                                interactive=True,
                                scale=1,
                            )
                            p_glasses_desc = gr.Textbox(
                                label=tr("Kanonische Beschreibung", "Canonical description"),
                                value="",
                                max_lines=1,
                                interactive=True,
                                scale=3,
                            )
                        gr.Markdown(tr(
                            "_Force-only-when-visible bleibt aktiv: Brille wird nur in Captions gesetzt, "
                            "wenn sie im Bild sichtbar ist._",
                            "_Force-only-when-visible stays active: glasses are only captioned when visible in the image._",
                        ))

                    # ----- Subtab: Variable Traits -----
                    with gr.TabItem(tr("🎨 Variable Traits", "🎨 Variable traits")):
                        gr.Markdown(tr(
                            "Per-Image-Tokens und sichtbarkeitsabhängige Merkmale wie Sommersprossen. "
                            "Bei Ausreißern (z.B. Lichtartefakte als 'red' klassifiziert) kannst du Bilder "
                            "eines Buckets bequem auf einen anderen Wert umbuchen.",
                            "Per-image tokens and visibility-dependent traits like freckles. For outliers "
                            "(e.g. lighting artifacts classified as 'red'), use Re-bucket to move all images "
                            "of a bucket to another value.",
                        ))

                        gr.Markdown(tr("**Sommersprossen (flexibler Marker)**", "**Freckles (flexible marker)**"))
                        with gr.Row():
                            p_freckles_present = gr.Checkbox(
                                label=tr("Hat regelmäßig sichtbare Sommersprossen", "Has regularly visible freckles"),
                                value=False,
                                interactive=True,
                                scale=1,
                            )
                            p_freckles_desc = gr.Textbox(
                                label=tr("Kanonische Beschreibung", "Canonical description"),
                                value="",
                                max_lines=1,
                                interactive=True,
                                scale=3,
                            )
                        gr.Markdown(tr(
                            "_Flexibler Sichtbarkeits-Marker: Sommersprossen werden nur in Captions gesetzt, wenn sie im Bild sichtbar sind._",
                            "_Flexible visibility marker: freckles are only captioned when they are visible in the image._",
                        ))

                        with gr.Row():
                            with gr.Column():
                                p_hair_color_md = gr.Markdown("_kein Profil geladen_")
                                with gr.Row():
                                    p_color_from = gr.Dropdown(
                                        label=tr("Quelle", "From"),
                                        choices=[""],
                                        value="",
                                        allow_custom_value=True,
                                        scale=2,
                                    )
                                    p_color_to = gr.Dropdown(
                                        label=tr("Ziel", "To"),
                                        choices=PROFILE_VOCAB_HAIR_COLOR,
                                        value=PROFILE_VOCAB_HAIR_COLOR[0],
                                        allow_custom_value=True,
                                        scale=2,
                                    )
                                    p_color_rebucket_btn = gr.Button(tr("Umbuchen", "Re-bucket"), scale=1)
                            with gr.Column():
                                p_hair_form_md = gr.Markdown("_kein Profil geladen_")
                                with gr.Row():
                                    p_form_from = gr.Dropdown(
                                        label=tr("Quelle", "From"),
                                        choices=[""],
                                        value="",
                                        allow_custom_value=True,
                                        scale=2,
                                    )
                                    p_form_to = gr.Dropdown(
                                        label=tr("Ziel", "To"),
                                        choices=PROFILE_VOCAB_HAIR_FORM,
                                        value=PROFILE_VOCAB_HAIR_FORM[0],
                                        allow_custom_value=True,
                                        scale=2,
                                    )
                                    p_form_rebucket_btn = gr.Button(tr("Umbuchen", "Re-bucket"), scale=1)
                            with gr.Column():
                                p_makeup_md = gr.Markdown("_kein Profil geladen_")
                                with gr.Row():
                                    p_makeup_from = gr.Dropdown(
                                        label=tr("Quelle", "From"),
                                        choices=[""],
                                        value="",
                                        allow_custom_value=True,
                                        scale=2,
                                    )
                                    p_makeup_to = gr.Dropdown(
                                        label=tr("Ziel", "To"),
                                        choices=PROFILE_VOCAB_MAKEUP,
                                        value=PROFILE_VOCAB_MAKEUP[0],
                                        allow_custom_value=True,
                                        scale=2,
                                    )
                                    p_makeup_rebucket_btn = gr.Button(tr("Umbuchen", "Re-bucket"), scale=1)

                    # ----- Subtab: Inventur & Notizen -----
                    with gr.TabItem(tr("🏷️ Tattoos & Piercings", "🏷️ Tattoos & piercings")):
                        gr.Markdown(tr(
                            "Inventur aller jemals gesehenen Tattoos und Piercings. In Captions werden "
                            "nur die im jeweiligen Bild sichtbaren Marker erwähnt (Force-only-when-visible).",
                            "Inventory of all ever-seen tattoos and piercings. In captions only the markers "
                            "actually visible in each image are mentioned (force-only-when-visible).",
                        ))
                        with gr.Row():
                            with gr.Column():
                                p_tattoo_md = gr.Markdown("_kein Profil geladen_")
                            with gr.Column(scale=2):
                                p_piercing_md = gr.Dataframe(
                                    headers=["location", "canonical_description", "frequency", "category", "role"],
                                    column_count=(5, "fixed"),
                                    datatype=["str", "str", "str", "str", "str"],
                                    value=[],
                                    interactive=True,
                                    row_count=(0, "dynamic"),
                                    label=tr("Piercing- und Ohrschmuck-Inventar", "Piercing and ear-jewelry inventory"),
                                )
                                gr.Markdown(tr(
                                    "Rollen: `canonical` = Grunderscheinung · `variable` = sichtbare Abweichung · `accessory` = austauschbarer Ohrschmuck · `ignore` = nie captionieren.",
                                    "Roles: `canonical` = baseline appearance · `variable` = visible variation · `accessory` = swappable ear jewelry · `ignore` = never caption.",
                                ))

                    # ----- Subtab: Identity Clustering -----
                    with gr.TabItem(tr("🧩 Identity Clustering", "🧩 Identity clustering")):
                        p_cluster_md = gr.Markdown("_kein Profil geladen_")
                        gr.Markdown(tr(
                            "`core` gibt einen kleinen Ranking-Boost. `variation` und `body_reference` bleiben Trainingskandidaten. `review` und `exclude` gehen nicht in `01_train_ready`. Vollständig auditierte Rejects stehen immer im letzten Bucket, beginnen als `exclude`, und zeigen den Reject-Grund direkt am Bild. Ein einzelnes Bild kann aus seinem Bucket gelöst oder als Priority zwingend in Train Ready übernommen werden.",
                            "`core` gives a small ranking boost. `variation` and `body_reference` remain training candidates. `review` and `exclude` do not go to `01_train_ready`. Fully audited rejects always appear in the last bucket, start as `exclude`, and show the reject reason on each image. An individual image can be detached from its bucket or forced into Train Ready as Priority.",
                        ))
                        with gr.Row():
                            with gr.Column(scale=3):
                                p_cluster_table = gr.Dataframe(
                                    headers=["cluster_id", "role", "n", "summary", "avg_quality_total", "avg_identity_usefulness"],
                                    column_count=(6, "fixed"),
                                    datatype=["str", "str", "number", "str", "str", "str"],
                                    value=[],
                                    interactive=True,
                                    row_count=(0, "dynamic"),
                                    label=tr("Cluster-Kategorien / Rollen", "Cluster categories / roles"),
                                )
                                p_selected_cluster_id = gr.State("")
                                p_cluster_role_editor = gr.Dropdown(
                                    label=tr("Rolle für ausgewählten Cluster", "Role for selected cluster"),
                                    choices=IDENTITY_CLUSTER_ROLE_CHOICES,
                                    value="variation",
                                    interactive=True,
                                    allow_custom_value=False,
                                    info=tr(
                                        "Cluster links anklicken, dann hier die Rolle per Dropdown ändern. Speichern übernimmt immer auch die aktuell gewählte Rolle.",
                                        "Click a cluster on the left, then change its role here. Save always applies the currently selected role too.",
                                    ),
                                )
                                with gr.Row():
                                    p_apply_cluster_role_btn = gr.Button(tr("↪ Rolle übernehmen", "↪ Apply role"), variant="secondary", scale=1)
                                    p_save_clusters_btn = gr.Button(tr("💾 Cluster-Rollen speichern", "💾 Save cluster roles"), variant="secondary", scale=1)
                            with gr.Column(scale=2):
                                p_cluster_preview_md = gr.Markdown(tr("_Kein Cluster ausgewählt._", "_No cluster selected._"))
                                p_cluster_gallery_page = gr.Dropdown(
                                    label=tr("Galerieseite", "Gallery page"),
                                    choices=[("1 / 1", "1")],
                                    value="1",
                                    interactive=False,
                                    allow_custom_value=False,
                                    info=tr(
                                        "Große Buckets werden seitenweise geladen, damit Auswahl und Detach stabil bleiben.",
                                        "Large buckets are loaded page by page so selection and detach remain stable.",
                                    ),
                                )
                                p_cluster_gallery = gr.Gallery(
                                    label=tr("Cluster-Vorschau – Bild anklicken für Einzelaktionen", "Cluster preview – click an image for individual actions"),
                                    columns=3,
                                    rows=4,
                                    height=620,
                                    object_fit="cover",
                                )
                                p_selected_cluster_image_id = gr.State("")
                                p_selected_cluster_image_info = gr.Markdown(tr("_Kein Bild ausgewählt._", "_No image selected._"))
                                with gr.Row():
                                    p_detach_cluster_image_btn = gr.Button(tr("↗ Bild aus Bucket lösen", "↗ Detach image from bucket"), variant="secondary", scale=2)
                                    p_priority_image_btn = gr.Button(tr("⭐ Als Priority markieren", "⭐ Mark as Priority"), variant="primary", scale=2)
                                    p_unpriority_image_btn = gr.Button(tr("☆ Priority entfernen", "☆ Remove Priority"), variant="secondary", scale=2)
                                p_priority_hazard_confirm = gr.Checkbox(
                                    label=tr("Problematisches Priority-Bild trotzdem bestätigen", "Confirm problematic Priority image anyway"),
                                    value=False,
                                    info=tr(
                                        "Nur nötig bei Duplikat-, Mehrpersonen-, ArcFace-Hard- oder technischen Warnungen.",
                                        "Only needed for duplicate, multi-person, ArcFace-hard or technical warnings.",
                                    ),
                                )
                                gr.Markdown(tr(
                                    "**Priority** überschreibt Qualitätswert, ArcFace, Bucket-Rolle, Quoten, Dubletten- und Caption-Remove-Entscheidungen. Problematische Fälle verlangen vorab die Bestätigung oben.",
                                    "**Priority** overrides quality, ArcFace, bucket role, quotas, duplicate and caption-remove decisions. Problematic cases require the confirmation above first.",
                                ))

                    # ----- Subtab: Diagnostics -----
                    with gr.TabItem(tr("🔬 Diagnostik & Raw JSON", "🔬 Diagnostics & raw JSON")):
                        p_notes_md = gr.Markdown("_kein Profil geladen_")
                        p_raw_view = gr.Textbox(
                            label=tr("_subject_profile.json (Read-only Backup)", "_subject_profile.json (read-only backup)"),
                            value="",
                            lines=18,
                            max_lines=28,
                            interactive=False,
                            elem_classes=["log-box"],
                            info=tr(
                                "Snapshot beim Laden. Wird beim Reset zurückgespielt.",
                                "Snapshot at load time. Used for reset.",
                            ),
                        )

                # Action bar
                with gr.Row():
                    p_save_btn = gr.Button(tr("💾 Profil speichern", "💾 Save profile"), variant="secondary", scale=1)
                    p_reset_btn = gr.Button(tr("↩️ Reset auf Backup", "↩️ Reset to backup"), variant="secondary", scale=1)
                    p_continue_btn = gr.Button(
                        tr("▶ Captioning aus bestätigtem Profil starten",
                           "▶ Start captioning from confirmed profile"),
                        variant="primary",
                        scale=2,
                    )
                    p_stop_btn = gr.Button(tr("⏹ Abbrechen", "⏹ Cancel"), variant="stop", scale=1)

                # Live log + gallery für Phase 3 Continue-Run
                with gr.Row():
                    p_status_run = gr.Textbox(label=tr("Run-Status", "Run status"), interactive=False, max_lines=1, scale=2)
                    p_openai_usage = gr.Textbox(label=tr("OpenAI Tokens", "OpenAI tokens"), interactive=False, max_lines=1, value=tr("💰 0 Requests | 0 Tokens", "💰 0 requests | 0 tokens"), scale=2)
                p_progress = gr.Slider(label=tr("Fortschritt", "Progress"), minimum=0, maximum=1, step=0.01, value=0, interactive=False)
                with gr.Row():
                    with gr.Column(scale=3):
                        p_log = gr.Textbox(label=tr("Live-Log", "Live log"), lines=14, max_lines=14, interactive=False, elem_classes=["log-box"])
                    with gr.Column(scale=2):
                        p_gallery = gr.Gallery(label=tr("Train-Ready Vorschau", "Train-ready preview"), columns=3, rows=3, height=340, object_fit="cover")

                # ---- Wiring ----
                p_load_outputs = [
                    p_state, p_raw_json,
                    p_gender, p_skin, p_eyes, p_hair_texture, p_body, p_body_height,
                    p_gender_info, p_skin_info, p_eyes_info, p_hair_texture_info, p_body_info, p_body_height_info,
                    p_hair_color_baseline, p_beard_pattern, p_beard_color,
                    p_glasses_regular, p_glasses_desc,
                    p_freckles_present, p_freckles_desc,
                    p_hair_color_md, p_hair_form_md, p_makeup_md,
                    p_tattoo_md, p_piercing_md, p_notes_md,
                    p_color_from, p_form_from, p_makeup_from,
                    p_color_to, p_form_to, p_makeup_to,
                    p_cluster_md, p_cluster_table, p_cluster_preview_md, p_cluster_gallery,
                    p_selected_cluster_id, p_cluster_role_editor,
                    p_status,
                ]
                p_load_btn.click(
                    fn=load_profile_for_editor,
                    inputs=[p_trigger, p_input],
                    outputs=p_load_outputs,
                ).then(
                    fn=lambda: ("", tr("_Kein Bild ausgewählt._", "_No image selected._")),
                    outputs=[p_selected_cluster_image_id, p_selected_cluster_image_info],
                )

                # Snapshot p_raw_json into the read-only diagnostics view
                p_raw_json.change(
                    fn=lambda s: s,
                    inputs=[p_raw_json],
                    outputs=[p_raw_view],
                )

                p_cluster_table.select(
                    fn=preview_identity_cluster_from_table_ui,
                    inputs=[p_state, p_trigger, p_input, p_cluster_table],
                    outputs=[p_cluster_preview_md, p_cluster_gallery, p_selected_cluster_id, p_cluster_role_editor, p_cluster_gallery_page],
                ).then(
                    fn=lambda: ("", tr("_Kein Bild ausgewählt._", "_No image selected._")),
                    outputs=[p_selected_cluster_image_id, p_selected_cluster_image_info],
                )

                p_cluster_gallery.select(
                    fn=select_identity_cluster_image_ui,
                    inputs=[p_state, p_trigger, p_input, p_selected_cluster_id, p_cluster_gallery_page],
                    outputs=[p_selected_cluster_image_id, p_selected_cluster_image_info],
                )

                p_cluster_gallery_page.input(
                    fn=refresh_identity_cluster_gallery_page_ui,
                    inputs=[p_state, p_trigger, p_input, p_selected_cluster_id, p_cluster_gallery_page],
                    outputs=[p_cluster_preview_md, p_cluster_gallery, p_selected_cluster_image_id, p_selected_cluster_image_info],
                    show_progress="hidden",
                )

                p_detach_cluster_image_btn.click(
                    fn=detach_selected_image_from_cluster_ui,
                    inputs=[p_trigger, p_input, p_state, p_selected_cluster_id, p_selected_cluster_image_id, p_cluster_gallery_page],
                    outputs=[
                        p_state, p_raw_json, p_cluster_md, p_cluster_table, p_cluster_preview_md, p_cluster_gallery,
                        p_selected_cluster_id, p_cluster_role_editor, p_cluster_gallery_page, p_selected_cluster_image_id,
                        p_selected_cluster_image_info, p_status,
                    ],
                ).then(
                    fn=refresh_identity_cluster_panel_ui,
                    inputs=[p_trigger, p_input, p_state, p_selected_cluster_id],
                    outputs=[
                        p_state, p_raw_json, p_cluster_md, p_cluster_table, p_cluster_preview_md, p_cluster_gallery,
                        p_selected_cluster_id, p_cluster_role_editor, p_cluster_gallery_page, p_selected_cluster_image_id,
                        p_selected_cluster_image_info,
                    ],
                    show_progress="hidden",
                )

                p_priority_image_btn.click(
                    fn=lambda t, i, p, c, img, pg, confirm: set_selected_image_priority_ui(t, i, p, c, img, pg, True, confirm),
                    inputs=[p_trigger, p_input, p_state, p_selected_cluster_id, p_selected_cluster_image_id, p_cluster_gallery_page, p_priority_hazard_confirm],
                    outputs=[p_state, p_raw_json, p_cluster_preview_md, p_cluster_gallery, p_selected_cluster_image_info, p_status],
                )
                p_unpriority_image_btn.click(
                    fn=lambda t, i, p, c, img, pg: set_selected_image_priority_ui(t, i, p, c, img, pg, False, False),
                    inputs=[p_trigger, p_input, p_state, p_selected_cluster_id, p_selected_cluster_image_id, p_cluster_gallery_page],
                    outputs=[p_state, p_raw_json, p_cluster_preview_md, p_cluster_gallery, p_selected_cluster_image_info, p_status],
                )

                p_cluster_role_editor.change(
                    fn=apply_identity_cluster_role_selection_ui,
                    inputs=[p_state, p_cluster_table, p_selected_cluster_id, p_cluster_role_editor],
                    outputs=[p_state, p_cluster_table, p_cluster_preview_md, p_status],
                )
                p_apply_cluster_role_btn.click(
                    fn=apply_identity_cluster_role_selection_ui,
                    inputs=[p_state, p_cluster_table, p_selected_cluster_id, p_cluster_role_editor],
                    outputs=[p_state, p_cluster_table, p_cluster_preview_md, p_status],
                )

                p_save_btn.click(
                    fn=save_profile_from_editor,
                    inputs=[
                        p_trigger, p_input, p_raw_json,
                        p_gender, p_skin, p_eyes, p_hair_texture, p_body, p_body_height,
                        p_hair_color_baseline, p_beard_pattern, p_beard_color,
                        p_glasses_regular, p_glasses_desc,
                        p_freckles_present, p_freckles_desc,
                        p_piercing_md,
                    ],
                    outputs=[p_status],
                )

                p_save_clusters_btn.click(
                    fn=save_identity_cluster_roles_ui,
                    inputs=[p_trigger, p_input, p_cluster_table, p_state, p_selected_cluster_id, p_cluster_role_editor],
                    outputs=[p_status],
                ).then(
                    fn=load_profile_for_editor,
                    inputs=[p_trigger, p_input],
                    outputs=p_load_outputs,
                )

                p_reset_btn.click(
                    fn=reset_profile_from_backup,
                    inputs=[p_trigger, p_input, p_raw_json],
                    outputs=[p_status],
                ).then(
                    fn=load_profile_for_editor,
                    inputs=[p_trigger, p_input],
                    outputs=p_load_outputs,
                )

                # Re-bucket buttons
                p_color_rebucket_btn.click(
                    fn=lambda t, i, fr, to: rebucket_per_image_field(t, i, "hair_color_base", fr, to),
                    inputs=[p_trigger, p_input, p_color_from, p_color_to],
                    outputs=[p_status],
                ).then(
                    fn=load_profile_for_editor,
                    inputs=[p_trigger, p_input],
                    outputs=p_load_outputs,
                )
                p_form_rebucket_btn.click(
                    fn=lambda t, i, fr, to: rebucket_per_image_field(t, i, "hair_form", fr, to),
                    inputs=[p_trigger, p_input, p_form_from, p_form_to],
                    outputs=[p_status],
                ).then(
                    fn=load_profile_for_editor,
                    inputs=[p_trigger, p_input],
                    outputs=p_load_outputs,
                )
                p_makeup_rebucket_btn.click(
                    fn=lambda t, i, fr, to: rebucket_per_image_field(t, i, "makeup_intensity", fr, to),
                    inputs=[p_trigger, p_input, p_makeup_from, p_makeup_to],
                    outputs=[p_status],
                ).then(
                    fn=load_profile_for_editor,
                    inputs=[p_trigger, p_input],
                    outputs=p_load_outputs,
                )

                # Continue-Button: nutzt die bestehenden Curator-Inputs (shared state across tabs).
                # Die Reihenfolge muss exakt zu start_caption_from_profile passen.
                # p_trigger / p_input ueberschreiben c_trigger / c_input wenn der User sie hier
                # geaendert hat - daher syncen wir sie zuerst zurueck in den Curator-Tab.
                def _sync_trigger_input(trigger_value, input_value):
                    return trigger_value, input_value

                p_sync_event = p_continue_btn.click(
                    fn=_sync_trigger_input,
                    inputs=[p_trigger, p_input],
                    outputs=[c_trigger, c_input],
                )
                p_run_event = p_sync_event.then(
                    fn=start_caption_from_profile,
                    inputs=curator_inputs,  # dieselbe Liste wie der ehemalige Continue-Button im Curator-Tab
                    outputs=[p_log, p_gallery, p_progress, p_status_run, p_openai_usage],
                    concurrency_id="dataset_curator_process",
                    concurrency_limit=1,
                )
                p_stop_btn.click(
                    fn=kill_process,
                    outputs=[p_status_run],
                    queue=False,
                    show_progress="hidden",
                )

            # ==============================================================
            # TAB 4: ERGEBNISSE
            # ==============================================================
            with gr.TabItem(tr("4 · 📊 Ergebnisse", "4 · 📊 Results"), id="results", interactive=initial_preflight_ready) as results_tab:
                gr.Markdown(tr("### Datensatz durchsuchen", "### Browse dataset"))
                gr.Markdown(tr(
                    "Lade Ergebnisse eines früheren Curator-Laufs. Bilder werden mit Captions angezeigt.",
                    "Load results from a previous curator run. Images are shown with captions.",
                ))

                with gr.Row():
                    r_trigger = gr.Textbox(
                        label=tr("Trigger Word", "Trigger Word"),
                        value=S["c_trigger"],
                        visible=False,
                        info=tr("Triggerwort des Laufs.", "Trigger word of the run."),
                        max_lines=1,
                        scale=2,
                    )
                    r_input = gr.Textbox(
                        label=tr("Input-Ordner", "Input folder"),
                        value=S["c_input"],
                        visible=False,
                        info=tr("Der Original-Input-Ordner.", "Original input folder."),
                        max_lines=1,
                        scale=3,
                    )
                    r_subfolder = gr.Dropdown(
                        label=tr("Kategorie", "Category"),
                        choices=[
                            (tr("Train Ready", "Train Ready"), "train_ready"),
                            (tr("Keep Unused", "Keep Unused"), "keep_unused"),
                            (tr("Caption Remove", "Caption Remove"), "caption_remove"),
                            (tr("Review", "Review"), "review"),
                            (tr("Reject", "Reject"), "reject"),
                            (tr("Manuelle Prüfung", "Manual review"), "manual_review"),
                            (tr("Smart Crop Paare", "Smart crop pairs"), "smart_crop_pairs"),
                        ],
                        value="train_ready",
                        info=tr("Welche Ergebnis-Kategorie.", "Which result category to load."),
                        scale=2,
                    )
                    r_page = gr.Number(label=tr("Seite", "Page"), value=1, minimum=1, maximum=1, step=1, precision=0, scale=1)
                    r_load_btn = gr.Button(tr("🔄 Laden", "🔄 Load"), variant="primary", scale=1)

                r_info = gr.Textbox(label=tr("Info", "Info"), interactive=False, max_lines=2)
                r_gallery = gr.Gallery(label=tr("Bilder (mit Captions)", "Images (with captions)"), columns=4, rows=3, height=420, object_fit="cover")
                r_report = gr.Markdown(label=tr("Report", "Report"))

                r_load_btn.click(fn=load_results, inputs=[r_input, r_trigger, r_subfolder, r_page], outputs=[r_gallery, r_report, r_info, r_page])

            # ==============================================================
            # SEPARATE TOOL: VIDEO PROCESSOR
            # ==============================================================
            with gr.TabItem(tr("🛠️ Video Processor", "🛠️ Video Processor"), id="video") as video_tab:

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

                with gr.Row():
                    v_status = gr.Textbox(label=tr("Status", "Status"), interactive=False, max_lines=1, scale=2)
                    v_openai_usage = gr.Textbox(label=tr("Lokales Hilfstool", "Local helper"), interactive=False, max_lines=1, value=tr("Keine OpenAI-Aufrufe", "No OpenAI calls"), scale=2)
                v_progress = gr.Slider(label=tr("Fortschritt", "Progress"), minimum=0, maximum=1, step=0.01, value=0, interactive=False)

                with gr.Row():
                    with gr.Column(scale=3):
                        v_log = gr.Textbox(label=tr("Live-Log", "Live log"), lines=15, max_lines=15, interactive=False, elem_classes=["log-box"])
                    with gr.Column(scale=2):
                        v_gallery = gr.Gallery(label=tr("Extrahierte Frames", "Extracted frames"), columns=3, rows=3, height=340, object_fit="cover")

                video_inputs = [v_source, v_target, v_ref, v_fpm, v_fps, v_sim, v_sharp]
                v_run_event = v_start_btn.click(
                    fn=start_video,
                    inputs=video_inputs,
                    outputs=[v_log, v_gallery, v_progress, v_status, v_openai_usage],
                    concurrency_id="dataset_curator_process",
                    concurrency_limit=1,
                )
                v_stop_btn.click(
                    fn=kill_process,
                    outputs=[v_status],
                    queue=False,
                    show_progress="hidden",
                )

            # ── Workspace gate / shared project context ──
            workspace_sync_outputs = [
                workspace_status, workspace_summary, workspace_ready_state,
                c_trigger, c_input, c_api_key,
                c_use_early_phash, c_use_early_phash_loop1, c_early_phash_thresh_1, c_early_phash_keep_1,
                c_use_early_phash_loop2, c_early_phash_thresh_2, c_early_phash_keep_2,
                c_min_side, c_use_filesize, c_min_filesize,
                c_ig_frame_crop, c_ig_two_stage_bar, c_frame_cleanup_mode, c_frame_pause_on_medium,
                p_trigger, p_input, r_trigger, r_input,
                frame_tab, audit_tab, profile_tab, results_tab,
            ]
            workspace_common_inputs = [
                w_trigger, w_input, w_api_key,
                w_use_early_phash, w_use_loop1, w_threshold1, w_keep1,
                w_use_loop2, w_threshold2, w_keep2,
                w_min_side, w_use_filesize, w_min_filesize,
                w_frame_enabled, w_frame_advanced, w_frame_mode,
                w_frame_auto_types, w_frame_pause, w_post_frame_phash,
            ]
            w_init_btn.click(
                fn=initialize_workspace_and_sync_ui,
                inputs=workspace_common_inputs,
                outputs=workspace_sync_outputs,
                show_progress="hidden",
            )
            w_preflight_btn.click(
                fn=run_workspace_preflight_and_sync_ui,
                inputs=[
                    w_trigger, w_input, w_api_key,
                    w_use_early_phash, w_use_loop1, w_threshold1, w_keep1,
                    w_use_loop2, w_threshold2, w_keep2,
                    w_min_side, w_use_filesize, w_min_filesize,
                    w_frame_enabled, w_frame_advanced, w_frame_mode,
                    w_frame_auto_types, w_frame_pause, w_post_frame_phash,
                ],
                outputs=workspace_sync_outputs,
                concurrency_id="workspace_preflight",
                concurrency_limit=1,
            )

            # ── Frame module callbacks with multiple candidate selection ──
            fr_scan_btn.click(
                fn=scan_frame_review_ui,
                inputs=[w_trigger, w_input, w_frame_advanced],
                outputs=[fr_state, fr_summary, fr_page, fr_gallery, fr_selected_hash, fr_selected_info, fr_option_gallery, fr_option_radio, fr_viewer_position, fr_previous_btn, fr_next_btn, fr_x1, fr_y1, fr_x2, fr_y2, fr_status],
                concurrency_id="frame_review_local",
            )
            fr_filter.change(
                fn=refresh_frame_review_page_ui,
                inputs=[fr_state, fr_filter, fr_page],
                outputs=[fr_page, fr_gallery, fr_selected_hash, fr_selected_info, fr_option_gallery, fr_option_radio, fr_viewer_position, fr_previous_btn, fr_next_btn, fr_x1, fr_y1, fr_x2, fr_y2],
                show_progress="hidden",
            )
            fr_page.input(
                fn=refresh_frame_review_page_ui,
                inputs=[fr_state, fr_filter, fr_page],
                outputs=[fr_page, fr_gallery, fr_selected_hash, fr_selected_info, fr_option_gallery, fr_option_radio, fr_viewer_position, fr_previous_btn, fr_next_btn, fr_x1, fr_y1, fr_x2, fr_y2],
                show_progress="hidden",
            )
            fr_gallery.select(
                fn=select_frame_review_image_ui,
                inputs=[fr_state, fr_filter, fr_page],
                outputs=[fr_selected_hash, fr_selected_info, fr_option_gallery, fr_option_radio, fr_viewer_position, fr_previous_btn, fr_next_btn, fr_x1, fr_y1, fr_x2, fr_y2],
            )
            fr_previous_btn.click(
                fn=lambda st, flt, sid: navigate_frame_review_image_ui(st, flt, sid, -1),
                inputs=[fr_state, fr_filter, fr_selected_hash],
                outputs=[fr_page, fr_gallery, fr_selected_hash, fr_selected_info, fr_option_gallery, fr_option_radio, fr_viewer_position, fr_previous_btn, fr_next_btn, fr_x1, fr_y1, fr_x2, fr_y2],
                show_progress="hidden",
            )
            fr_next_btn.click(
                fn=lambda st, flt, sid: navigate_frame_review_image_ui(st, flt, sid, 1),
                inputs=[fr_state, fr_filter, fr_selected_hash],
                outputs=[fr_page, fr_gallery, fr_selected_hash, fr_selected_info, fr_option_gallery, fr_option_radio, fr_viewer_position, fr_previous_btn, fr_next_btn, fr_x1, fr_y1, fr_x2, fr_y2],
                show_progress="hidden",
            )
            fr_option_radio.input(
                fn=apply_frame_option_ui,
                inputs=[fr_state, fr_filter, fr_page, fr_selected_hash, fr_option_radio],
                outputs=[fr_state, fr_summary, fr_gallery, fr_selected_info, fr_option_gallery, fr_option_radio, fr_x1, fr_y1, fr_x2, fr_y2, fr_status],
                show_progress="hidden",
            )
            fr_auto_btn.click(
                fn=restore_frame_auto_ui,
                inputs=[fr_state, fr_filter, fr_page, fr_selected_hash],
                outputs=[fr_state, fr_summary, fr_gallery, fr_selected_info, fr_option_gallery, fr_option_radio, fr_x1, fr_y1, fr_x2, fr_y2, fr_status],
                show_progress="hidden",
            )
            fr_preview_manual_btn.click(
                fn=preview_manual_frame_crop_ui,
                inputs=[fr_state, fr_selected_hash, fr_x1, fr_y1, fr_x2, fr_y2],
                outputs=[fr_manual_preview, fr_manual_status],
            )
            fr_accept_manual_btn.click(
                fn=lambda st, flt, pg, sid, a, b, c, d: set_frame_review_decision_ui(st, flt, pg, sid, "manual", None, a, b, c, d),
                inputs=[fr_state, fr_filter, fr_page, fr_selected_hash, fr_x1, fr_y1, fr_x2, fr_y2],
                outputs=[fr_state, fr_summary, fr_gallery, fr_selected_info, fr_status],
            ).then(
                fn=lambda st, sid: _frame_option_gallery_update(_find_frame_record(st, sid), st),
                inputs=[fr_state, fr_selected_hash],
                outputs=[fr_option_gallery, fr_option_radio],
                show_progress="hidden",
            )
            fr_reset_cache_btn.click(
                fn=reset_frame_detector_cache_ui,
                inputs=[w_trigger, w_input],
                outputs=[fr_state, fr_summary, fr_page, fr_gallery, fr_status],
            )
            fr_reset_manual_btn.click(
                fn=reset_frame_manual_decisions_ui,
                inputs=[w_trigger, w_input],
                outputs=[fr_status],
            )

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
