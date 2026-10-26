"""Prompt, rule check and cache key for the Luna caption writer.

The caption rules and the confirmed subject profile have already been turned
into caption facts (``collect_caption_facts`` in the pipeline). The writer only
phrases those facts as fluent sentences for the chosen training target; this
module never decides what a caption contains.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from caption_grammar import caption_opening
from curator_core import stable_hash

PROMPT_VERSION = "caption-writer-v1-rules-profile-image"

# Detailed natural prose for most targets; FLUX.2 starts from a compact form.
TARGET_STYLE = {
    "flux2": ("25-60", "compact natural prose: subject, clothing, pose, setting and light in two or three sentences"),
    "flux2_klein_4b": ("25-60", "compact natural prose: subject, clothing, pose, setting and light in two or three sentences"),
    "flux2_klein_9b": ("25-60", "compact natural prose: subject, clothing, pose, setting and light in two or three sentences"),
}
DEFAULT_STYLE = ("40-110", "detailed, objective natural prose in complete sentences")

_SCENE_LABELS = {
    "expression": "facial expression",
    "gaze": "gaze direction",
    "lighting": "lighting",
    "background": "background and surroundings",
}
_PIERCING_WORDS = r"piercing|nose ring|septum|earring|ear gauge|gauge|nose stud|labret|tongue ring|navel ring"
_MAKEUP_WORDS = r"makeup|make-up|eyeliner|lipstick|mascara|eyeshadow|eye shadow|lip gloss|blush"
_FORBIDDEN_FORMS = (
    (r"\bthis (?:image|photo|picture)\b|\bthe (?:image|photo|picture) shows\b", "meta wording about the image"),
    (r"[#*_`]|^\s*-|\n", "markdown, list or line break"),
    (r"\b(?:tags?|caption)\s*:", "label instead of prose"),
)


def writer_request(facts: Dict[str, Any], target_label: str, reference: str) -> Dict[str, Any]:
    """Instructions and text payload for one caption; the image is attached by the caller."""
    length, style = TARGET_STYLE.get(facts.get("training_target", ""), DEFAULT_STYLE)
    rules = facts.get("scene_rules", {})
    describe = [label for key, label in _SCENE_LABELS.items() if rules.get(key)]
    leave_out = [label for key, label in _SCENE_LABELS.items() if not rules.get(key)]
    must_include = [entry["phrase"] for entry in facts.get("identity", [])]
    omitted = facts.get("omitted", {})
    never = ["skin tone", "ethnicity", "age", "face shape", "body proportions", "name or relationship"]
    if omitted.get("body_build"):
        never.append("body build")
    if omitted.get("piercings"):
        never.append("piercings and ear jewelry")
    if omitted.get("makeup"):
        never.append("makeup")
    if omitted.get("tattoos"):
        never.append("tattoos")
    states = facts.get("feature_states", {})
    for name, label in (("hair", "hair color, length or style (a pose may still say 'her hair')"), ("eye", "eye color"), ("beard", "beard"), ("glasses", "glasses")):
        if not (states.get(name) or {}).get("must_caption"):
            never.append(label)
    pronoun = facts.get("pronoun", "They")
    instructions = f"""
You write the final training caption for an image LoRA of one person. Target model: {target_label}.
Write {style}, {length} words, present tense, in English.

Start with exactly this text: {caption_opening(facts)}
After that opening, refer to the person only as "{pronoun}" (never as "the woman", "the man" or "the person").

Mandatory phrases: include every phrase in must_include word for word, woven into natural sentences.
Never write them as a comma-separated list after the opening.

The person profile already decided everything about identity. Never mention: {", ".join(never)}.
Describe: clothing, pose and action{", " + ", ".join(describe) if describe else ""}, camera framing and visible style.
{"Do not describe: " + ", ".join(leave_out) + "." if leave_out else ""}

Describe only what is visible in the image. scene_notes come from an audit of this image; prefer them when
they match the image. reference_description is an earlier free description of the same image: use it only
for scene details you can confirm in the image and ignore any identity traits in it.
No tags, labels, markdown, line breaks, hedging, quality remarks or phrases like "This image shows".
""".strip()
    payload = {
        "must_include": must_include,
        "scene_notes": {key: value for key, value in facts.get("scene", {}).items() if value},
        "reference_description": str(reference or "")[:3000],
    }
    return {"instructions": instructions, "payload": payload}


def rule_errors(caption: str, facts: Dict[str, Any]) -> List[str]:
    """Rule violations of a written caption that are not covered by the profile feature check."""
    text = re.sub(r"\s+", " ", str(caption or "")).strip()
    errors: List[str] = []
    if not text:
        return ["empty caption"]
    if not text.lower().startswith(caption_opening(facts).lower()):
        errors.append(f"caption must start with: {caption_opening(facts)}")
    lower = text.lower()
    for entry in facts.get("identity", []):
        if entry.get("kind") in {"makeup", "accessory", "freckles", "tattoo", "body_build"}:
            if re.sub(r"[\s-]+", " ", entry["phrase"].lower()) not in re.sub(r"[\s-]+", " ", lower):
                errors.append(f"missing required phrase: {entry['phrase']}")
    omitted = facts.get("omitted", {})
    if omitted.get("piercings") and re.search(rf"\b(?:{_PIERCING_WORDS})", lower):
        errors.append("piercings must be omitted by the caption rules")
    if omitted.get("makeup") and re.search(rf"\b(?:{_MAKEUP_WORDS})\b", lower):
        errors.append("makeup must be omitted by the caption rules")
    tone = str(omitted.get("skin_tone") or "").strip().lower()
    if tone and re.search(rf"\b{re.escape(tone)}[- ]skin", lower):
        errors.append("skin tone must be omitted")
    build = str(omitted.get("body_build") or "").strip().lower().replace("_", " ")
    if build and re.search(rf"\b{re.escape(build)}\b(?:[- ]build| figure| body)", lower):
        errors.append("body build must be omitted")
    for pattern, reason in _FORBIDDEN_FORMS:
        if re.search(pattern, text, re.I):
            errors.append(reason)
    words = len(text.split())
    if words < 12 or words > 200:
        errors.append(f"length {words} words is outside 12-200")
    return errors


def cache_key(request: Dict[str, Any], image_key: str, models: Dict[str, Any]) -> str:
    """Key over exactly what the model sees (instructions, payload, image) and the models.

    Internal fields of the caption facts that never reach the model do not
    invalidate stored captions; any change to rules, profile phrases, scene
    notes or the reference text does.
    """
    return stable_hash({
        "version": PROMPT_VERSION,
        "image": image_key,
        "instructions": request.get("instructions", ""),
        "payload": request.get("payload", {}),
        "models": models,
    })
