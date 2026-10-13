"""Conservative cross-check of free initial captions against image audits.

Claims are *not* promoted into audit facts. Only an agreement can be passed to
the profile normalizer; conflicts and missing audit observations are review cues.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

VERSION = "caption-profile-evidence-v1"

_HAIR_RE = re.compile(
    r"\b(?P<color>platinum(?:[ -]blond(?:e)?)?|strawberry[ -]blond(?:e)?|"
    r"dark[ -]blond(?:e)?|light[ -]blond(?:e)?|blond(?:e)?|dark[ -]brown|"
    r"light[ -]brown|brown|brunette|black|auburn|copper|red|silver|"
    r"gray|grey|white)\s+(?:(?:long|short|wavy|curly|straight|thick|"
    r"shoulder[ -]length)\s+){0,2}hair\b", re.I)
_EYE_RE = re.compile(r"\b(?P<color>blue[ -]green|gray[ -]blue|grey[ -]blue|"
                     r"dark[ -]brown|blue|green|hazel|brown|gray|grey|amber)\s+eyes\b", re.I)
_GLASSES_RE = re.compile(r"\b(?:wearing|wears|with|in)\s+(?:(?:round|square|"
                         r"black|clear|wire[ -]frame|large|small|dark|tinted|"
                         r"thin|thick|prescription)\s+){0,3}"
                         r"(?:eyeglasses|sunglasses|glasses|spectacles)\b", re.I)
_ACCESSORY_RE = {
    "nose_piercing": re.compile(r"\b(?:nose|nostril|septum)\s+(?:ring|stud|piercing)\b|"
                                 r"\b(?:piercing|stud|ring)\s+(?:in|on)\s+(?:the|her|his)?\s*"
                                 r"(?:nose|nostril|septum)\b", re.I),
    "navel_piercing": re.compile(r"\b(?:navel|belly[ -]button)\s+(?:ring|stud|piercing)\b", re.I),
    "ear_jewelry": re.compile(r"\b(?:earrings?|ear studs?|ear[ -]jewell?ery)\b", re.I),
    "tattoo": re.compile(r"\btattoos?\b", re.I),
}


def _unnegated_matches(pattern, text):
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - 32):match.start()]
        # An absent trait is not a visible claim. Captioners use varied prose;
        # avoid attempting to infer absence from anything else.
        if re.search(r"\b(?:no|without|not|lacks?)\s+(?:\w+\s+){0,3}$", before, re.I):
            continue
        yield match


def _hair_family(value):
    text = str(value or "").casefold().replace("-", " ").replace("_", " ")
    # Highlights describe a modifier, not the base hair color. This matters
    # when a structured audit carries a prose description rather than a token.
    text = re.sub(r"\b(?:\w+\s+){1,3}highlights?\b", " ", text)
    if "strawberry" in text or any(k in text for k in ("auburn", "copper", "red")):
        return "red_auburn"
    if "blond" in text or "platinum" in text:
        return "blonde"
    if "brown" in text or "brunette" in text:
        return "brown"
    if "black" in text:
        return "black"
    if any(k in text for k in ("silver", "gray", "grey", "white")):
        return "gray_white"
    return ""


def _eye_family(value):
    text = str(value or "").casefold().replace("-", "_").replace(" ", "_")
    text = text.replace("grey", "gray")
    if text in {"blue", "gray_blue", "blue_green"}:
        return "blue_family"
    if text in {"brown", "dark_brown"}:
        return "brown_family"
    return text if text in {"green", "hazel", "gray", "amber"} else ""


def visible_claims(caption):
    """Extract only narrow positive claims; omit ambiguity and generic jewelry."""
    text = str(caption or "")[:10000]
    claims = {}
    for name, pattern, normalizer in (("hair_color", _HAIR_RE, _hair_family),
                                      ("eye_color", _EYE_RE, _eye_family)):
        values = {normalizer(m.group("color")) for m in _unnegated_matches(pattern, text)}
        values.discard("")
        if len(values) == 1:
            claims[name] = values.pop()
    if any(_unnegated_matches(_GLASSES_RE, text)):
        claims["glasses"] = "present"
    for name, pattern in _ACCESSORY_RE.items():
        if any(_unnegated_matches(pattern, text)):
            claims[name] = "present"
    return claims


def compare_caption(row, *, audit_hair="", audit_eye="", eye_reliable=False,
                    glasses_state="uncertain"):
    record = row.get("initial_caption") or {}
    if not isinstance(record, dict) or not str(record.get("caption") or "").strip():
        return None
    settings = record.get("settings") or {}
    engine = str(settings.get("engine") or "")
    caption = str(record["caption"])
    provenance = {"version": VERSION, "engine": engine,
                  "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest()}
    if int(row.get("face_count_local") or 0) > 1:
        return {**provenance, "skipped": "multiple_faces", "agreements": {},
                "conflicts": {}, "unverified": {}}
    claims = visible_claims(caption)
    audit = {
        "hair_color": _hair_family(audit_hair),
        "eye_color": _eye_family(audit_eye) if eye_reliable else "",
        "glasses": glasses_state if glasses_state in {"present", "absent"} else "",
        "nose_piercing": "present" if any(str(x.get("location", "")).startswith("nose") for x in row.get("piercing_inventory_now") or [] if isinstance(x, dict)) else "",
        "navel_piercing": "present" if any(str(x.get("location", "")) == "navel" for x in row.get("piercing_inventory_now") or [] if isinstance(x, dict)) else "",
        "ear_jewelry": "present" if any(str(x.get("location", "")).startswith("ear") for x in row.get("piercing_inventory_now") or [] if isinstance(x, dict)) else "",
        "tattoo": "present" if row.get("tattoos_visible") or bool(row.get("tattoo_inventory_now")) else "",
    }
    result = {**provenance, "agreements": {}, "conflicts": {}, "unverified": {}}
    for field, claimed in claims.items():
        observed = audit[field]
        if not observed:
            result["unverified"][field] = claimed
        elif observed == claimed:
            result["agreements"][field] = claimed
        else:
            result["conflicts"][field] = {"caption": claimed, "audit": observed}
    return result


def profile_summary(rows, compare):
    agreements, conflicts, unverified = Counter(), Counter(), Counter()
    review = []
    providers = Counter()
    for row in rows:
        evidence = compare(row)
        if not evidence:
            continue
        providers[evidence["engine"]] += 1
        agreements.update(evidence["agreements"].keys())
        conflicts.update(evidence["conflicts"].keys())
        unverified.update(evidence["unverified"].keys())
        if evidence["conflicts"] or evidence["unverified"]:
            review.append({"image_id": row.get("profile_image_id") or row.get("original_filename", ""),
                           "filename": row.get("original_filename", ""),
                           "conflicts": evidence["conflicts"],
                           "unverified": evidence["unverified"]})
    review.sort(key=lambda item: (-len(item["conflicts"]), -len(item["unverified"]),
                                  str(item["filename"]).casefold()))
    return {"version": VERSION, "providers": dict(providers),
            "agreements": dict(agreements), "conflicts": dict(conflicts),
            "unverified": dict(unverified), "review_examples": review[:30]}
