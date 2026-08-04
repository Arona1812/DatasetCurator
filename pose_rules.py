#!/usr/bin/env python3
"""Structured pose and edit-provenance decisions for person-LoRA curation."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Tuple

POSTURES = {
    "standing", "seated_upright", "seated_leaning", "crouching", "kneeling",
    "reclining", "lying_side", "lying_back", "lying_front", "acrobatic", "ambiguous",
}
BODY_AXES = {"upright", "diagonal", "horizontal"}
POSE_NATURALNESS = {"natural", "stylized", "contorted"}
PERSPECTIVE_STRENGTH = {"normal", "strong", "extreme"}
IMAGE_ROTATIONS = {"normal", "rotated_90", "rotated_180"}
EDIT_STATUSES = {
    "unknown", "unedited", "background_or_bystanders_edited", "target_retouched",
    "target_reconstructed", "fully_synthetic",
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _choice(value: Any, allowed: Iterable[str], fallback: str) -> str:
    text = _norm(value).replace("-", "_").replace(" ", "_")
    return text if text in set(allowed) else fallback


def derive_structured_pose(item: Dict[str, Any]) -> Dict[str, str]:
    """Return structured fields, conservatively deriving missing legacy data."""
    posture = _choice(item.get("posture"), POSTURES, "ambiguous")
    axis = _choice(item.get("body_axis"), BODY_AXES, "upright")
    naturalness = _choice(item.get("pose_naturalness"), POSE_NATURALNESS, "natural")
    perspective = _choice(item.get("perspective_strength"), PERSPECTIVE_STRENGTH, "normal")
    rotation = _choice(item.get("image_rotation"), IMAGE_ROTATIONS, "normal")

    text = " ".join(_norm(item.get(k)) for k in (
        "pose_description", "action_description", "composition_description", "short_reason"
    ))
    face_orientation = _norm(item.get("face_orientation_in_frame"))
    old_perspective = _norm(item.get("perspective_distortion"))
    camera_angle = _norm(item.get("camera_angle"))

    if posture == "ambiguous":
        if any(k in text for k in ("lying on back", "lying face up", "supine")):
            posture = "lying_back"
        elif any(k in text for k in ("lying on stomach", "lying face down", "prone")):
            posture = "lying_front"
        elif any(k in text for k in ("lying on side", "side-lying", "lying sideways")):
            posture = "lying_side"
        elif any(k in text for k in ("lying", "reclining", "reclined", "lounging on")):
            posture = "reclining"
        elif any(k in text for k in ("all fours", "hands and knees", "acrobatic", "contorted")):
            posture = "acrobatic"
        elif "kneeling" in text:
            posture = "kneeling"
        elif any(k in text for k in ("crouching", "crouched", "squatting")):
            posture = "crouching"
        elif any(k in text for k in ("seated", "sitting")):
            posture = "seated_leaning" if any(k in text for k in ("leaning", "slouched", "tilted back")) else "seated_upright"
        elif any(k in text for k in ("standing", "stands", "upright")):
            posture = "standing"

    if axis == "upright":
        if posture in {"lying_side", "lying_back", "lying_front", "reclining"}:
            axis = "horizontal"
        elif posture in {"acrobatic"} or face_orientation in {"sideways", "inverted"}:
            axis = "diagonal"

    if naturalness == "natural":
        if posture == "acrobatic" or any(k in text for k in ("contorted", "extreme arch", "unnatural twist")):
            naturalness = "contorted"
        elif any(k in text for k in ("dramatic pose", "stylized pose", "pin-up", "foreshortened")):
            naturalness = "stylized"

    if perspective == "normal":
        if old_perspective == "strong" or camera_angle in {"overhead", "low_angle", "high_angle"}:
            perspective = "strong"
        if any(k in text for k in ("extreme perspective", "extreme foreshortening", "fisheye")):
            perspective = "extreme"

    if rotation == "normal":
        if face_orientation == "sideways":
            rotation = "rotated_90"
        elif face_orientation == "inverted":
            rotation = "rotated_180"

    return {
        "posture": posture,
        "body_axis": axis,
        "pose_naturalness": naturalness,
        "perspective_strength": perspective,
        "image_rotation": rotation,
    }


def apply_structured_pose(item: Dict[str, Any]) -> Dict[str, str]:
    result = derive_structured_pose(item)
    item.update(result)
    return result


def pose_suitability_decision(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return keep/review based solely on training suitability.

    Clothing, nudity and sexualized presentation are intentionally ignored.
    """
    pose = apply_structured_pose(item)
    reasons: List[str] = []
    if pose["posture"] in {"reclining", "lying_side", "lying_back", "lying_front", "acrobatic"}:
        reasons.append(f"posture={pose['posture']}")
    if pose["body_axis"] == "horizontal":
        reasons.append("body_axis=horizontal")
    if pose["pose_naturalness"] == "contorted":
        reasons.append("pose_naturalness=contorted")
    if pose["perspective_strength"] == "extreme":
        reasons.append("perspective_strength=extreme")
    # A rotated image is not rejected; it is a local correction candidate.
    return ("review", reasons) if reasons else ("keep", [])


def infer_edit_provenance(item: Dict[str, Any]) -> Dict[str, Any]:
    explicit = _choice(item.get("edit_status"), EDIT_STATUSES, "unknown")
    evidence = list(item.get("edit_evidence") or []) if isinstance(item.get("edit_evidence"), list) else []
    filename = os.path.basename(str(item.get("source_original_path") or item.get("original_path") or item.get("original_filename") or ""))
    low = filename.lower()

    if explicit == "unknown":
        # Filenames are evidence, never an automatic rejection.
        if any(token in low for token in ("flux2-klein", "remove", "cleanup", "generative_fill")):
            explicit = "background_or_bystanders_edited"
            evidence.append(f"filename:{filename}")
        elif any(token in low for token in ("fully_synthetic", "synthetic_target")):
            explicit = "fully_synthetic"
            evidence.append(f"filename:{filename}")

    return {
        "edit_status": explicit,
        "edit_evidence": list(dict.fromkeys(str(x) for x in evidence if str(x))),
    }


def edit_suitability_decision(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    provenance = infer_edit_provenance(item)
    item.update(provenance)
    status = provenance["edit_status"]
    if status in {"fully_synthetic", "target_reconstructed"}:
        return "review", [f"{status}_target"]
    if status in {"background_or_bystanders_edited", "target_retouched"}:
        item.setdefault("status_notes", []).append(f"edit_warning:{status}")
        return "keep", [f"warning:{status}"]
    return "keep", []
