"""Consistent shot types from the audit answer and the measured face size.

The audit prompt names headshot | medium | full_body but never defines the
borders, so the model shifts whole groups: chest-up selfies become medium and
three-quarter shots become full_body, which empties the headshot quota. The
face height relative to the image height is a stable framing measure and
works on cached audits without a new API call.

Face height (face box, roughly chin to hairline) per framing:
  shoulders/chest up ~0.40-0.55, waist up ~0.25-0.30,
  knees up ~0.14-0.16, head to feet <= ~0.12.
"""
from typing import Any, Dict, Optional

SHOT_TYPES = ("headshot", "medium", "full_body")

# Chest-up framing or closer. Waist-up shots stay medium.
HEADSHOT_MIN_FACE_HEIGHT = 0.33
# A standing or seated whole body cannot show the face this large.
FULL_BODY_MAX_FACE_HEIGHT = 0.18


def face_height_ratio(row: Dict[str, Any]) -> Optional[float]:
    """Face height relative to the image height, or None when unknown."""
    box = row.get("face_bbox_ai")
    if row.get("face_visible") is not False and isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            height = float(box[3])
        except (TypeError, ValueError):
            height = 0.0
        if 0.01 <= height <= 1.0:
            return height
    box = row.get("main_face_bbox")
    try:
        image_height = float(row.get("height") or 0)
        if isinstance(box, (list, tuple)) and len(box) == 4 and image_height > 0:
            height = float(box[3]) / image_height
            if 0.01 <= height <= 1.0:
                return height
    except (TypeError, ValueError):
        pass
    return None


def audit_shot_type(row: Dict[str, Any]) -> str:
    """Shot type as the audit returned it, also after apply_shot_type ran."""
    current = str(row.get("shot_type") or "").strip().lower()
    if row.get("audit_shot_type") and current == row.get("shot_type_reconciled"):
        return str(row["audit_shot_type"]).strip().lower()
    return current


def reconcile_shot_type(row: Dict[str, Any]) -> str:
    """Return the shot type that matches frame subtype and face size.

    Rows without a usable face measurement keep the audit value, except that a
    three-quarter subtype is always medium and a close-up always headshot.
    """
    shot = audit_shot_type(row)
    if shot not in SHOT_TYPES:
        return shot
    subtype = str(row.get("frame_subtype") or "").strip().lower()
    face = face_height_ratio(row)

    if subtype == "close_up" or (face is not None and face >= HEADSHOT_MIN_FACE_HEIGHT):
        return "headshot"
    if shot == "headshot":
        return shot
    if subtype == "three_quarter_body":
        return "medium"
    if shot == "full_body" and face is not None and face >= FULL_BODY_MAX_FACE_HEIGHT:
        return "medium"
    return shot


def apply_shot_type(row: Dict[str, Any], from_audit: bool = False) -> Dict[str, Any]:
    """Set the reconciled shot type on the row; the audit value stays in audit_shot_type.

    ``from_audit`` marks that shot_type was just written by an audit answer, so
    an earlier reconciliation on the same row no longer applies.
    """
    if from_audit:
        row.pop("audit_shot_type", None)
        row.pop("shot_type_reconciled", None)
    original = audit_shot_type(row)
    shot = reconcile_shot_type(row)
    if shot == original:
        row.pop("audit_shot_type", None)
        row.pop("shot_type_reconciled", None)
        row["shot_type"] = original
        return row
    row["audit_shot_type"] = original
    row["shot_type_reconciled"] = shot
    row["shot_type"] = shot
    notes = row.setdefault("status_notes", [])
    note = f"shot_type_{original}_to_{shot}"
    if isinstance(notes, list) and note not in notes:
        notes.append(note)
    return row
