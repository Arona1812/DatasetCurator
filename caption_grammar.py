"""Grammar helpers that turn audited caption parts into natural English sentences.

The caption rules and the confirmed subject profile decide WHAT a caption
contains. These helpers only decide HOW it reads: whole sentences instead of
tag lists, a verb form that fits the subject, one article, consistent
pronouns and capitalised sentence starts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Nouns that take no article in a list ("dramatic eyeliner", "freckles").
_MASS_NOUNS = {
    "eyeliner", "makeup", "make-up", "mascara", "lipstick", "eyeshadow", "blush", "gloss",
    "glitter", "hair", "stubble", "foundation", "bronzer", "contour", "highlighter", "jewelry",
    "jewellery", "clothing", "light", "lighting", "daylight", "sunlight",
}
_ARTICLES = {"a", "an", "the", "her", "his", "their", "its", "my", "one", "two", "both", "several", "some"}

# Third-person verbs that can open a pose description ("holds a phone").
_FINITE_POSE_VERBS = {
    "holds", "stands", "sits", "leans", "rests", "faces", "looks", "poses", "kneels", "lies",
    "crouches", "smiles", "raises", "tilts", "touches", "places", "keeps", "extends", "gazes",
    "walks", "turns", "crosses", "grips", "hugs", "covers", "reaches", "squats", "bends", "perches",
    "cradles", "presses", "adjusts", "plays", "takes", "uses", "carries", "points", "waves", "sips",
    "drinks", "reclines", "lounges", "lifts", "pulls", "runs", "glances", "stares", "peers",
    "frames", "shields", "clutches", "embraces", "folds", "hides", "wraps", "braces", "balances",
    "steps", "strikes", "angles", "supports", "props", "puts", "laughs", "grins", "bites", "blows",
}
# Words that open a pose which needs "is" ("seated", "close to the camera").
_STATE_STARTERS = {
    "close", "near", "in", "on", "at", "against", "beside", "behind", "under", "inside", "outside",
    "upright", "alone", "partially", "slightly", "half", "mid-stride", "barefoot", "centered",
    "centred", "seated",
}
_FRAMING_WORDS = {"portrait", "selfie", "shot", "view", "photo", "close-up", "closeup", "headshot", "frame"}
_BODY_PART_STARTERS = {
    "one", "both", "her", "his", "their", "the", "a", "an", "arms", "hands", "legs", "head", "body",
    "torso", "shoulders", "hand", "arm", "leg", "knees", "feet", "fingers", "chin",
}
_EXPRESSION_NOUNS = {
    "expression", "look", "smile", "smirk", "frown", "grin", "pout", "stare", "gaze", "glance",
    "face", "demeanor", "mood", "lips", "mouth", "teeth", "brows", "eyebrows", "laugh",
}
_PLURAL_EXPRESSION_NOUNS = {"lips", "teeth", "brows", "eyebrows"}
_FINITE_EXPRESSION_VERBS = {"smiles", "grins", "frowns", "pouts", "laughs", "smirks", "beams"}
_GAZE_ING = {
    "looks": "looking", "gazes": "gazing", "faces": "facing", "glances": "glancing",
    "stares": "staring", "peers": "peering", "watches": "watching", "directed": "looking",
    "directs": "looking", "turned": "looking",
}
_GAZE_VERBS = {"looking", "gazing", "staring", "glancing", "facing", "watching", "peering", "with"}
_GAZE_NEEDS_VERB = {
    "toward", "towards", "at", "into", "upon", "across", "past", "through", "above", "below",
    "behind", "ahead", "directly", "straight", "off-camera", "off", "over", "back", "out",
    "downward", "upward", "sideways", "leftward", "rightward", "forward", "backward", "away",
    "down", "up", "left", "right", "outward", "inward", "slightly", "just", "camera",
}
_BACKGROUND_ADJECTIVES = {
    "plain", "light", "dark", "gray", "grey", "white", "black", "beige", "neutral", "blurred",
    "blurry", "pale", "bright", "dim", "and", "soft", "muted", "solid", "simple", "warm", "cool",
    "cream", "off-white", "uniform", "smooth", "out-of-focus", "textured", "bare",
}
_HAS_KINDS = ("body_build", "hair", "eye", "beard", "freckles", "piercing", "tattoo")
_WEARS_KINDS = ("glasses", "makeup", "accessory")


def _words(text: str) -> List[str]:
    return [w.strip(",.;:").lower() for w in str(text or "").split()]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def with_article(phrase: str) -> str:
    """Prefix ``a``/``an`` unless the phrase is plural, uncountable or already determined."""
    text = str(phrase or "").strip()
    words = _words(text)
    if not words or words[0] in _ARTICLES:
        return text
    head = re.split(r"\s+(?:with|over|and|on|in|at|from)\s+|,", text.lower())[0].split()
    noun = head[-1] if head else words[-1]
    if noun in _MASS_NOUNS or (noun.endswith("s") and not noun.endswith(("ss", "us"))):
        return text
    article = "an" if re.match(r"[aeiou]", words[0]) and not words[0].startswith(("uni", "use", "one")) else "a"
    return f"{article} {text}"


def join_natural(items: Iterable[str]) -> str:
    """'a', 'b', 'c' -> 'a, b and c'."""
    values = [str(v).strip() for v in items if str(v or "").strip()]
    if len(values) <= 1:
        return "".join(values)
    return ", ".join(values[:-1]) + " and " + values[-1]


def fix_pronouns(text: str, pronoun: str) -> str:
    """Use the caption's pronoun instead of 'their'/'them' or 'the woman's'."""
    if pronoun not in {"She", "He"}:
        return text
    possessive, objective, reflexive = ("her", "her", "herself") if pronoun == "She" else ("his", "him", "himself")
    text = re.sub(r"\bthe (?:woman|man|person|subject|model)'s\b", possessive, text, flags=re.I)
    text = re.sub(r"\btheir\b", possessive, text)
    text = re.sub(r"\bTheir\b", _upper_first(possessive), text)
    text = re.sub(r"\bthemselves\b", reflexive, text)
    return re.sub(r"\bthem\b", objective, text)


def identity_sentence(pronoun: str, identity: Sequence[Dict[str, str]]) -> str:
    """'She has a nose ring and wears dramatic eyeliner.' from tagged identity phrases."""
    has = [with_article(e["phrase"]) for e in identity if e.get("kind") in _HAS_KINDS]
    wears = [with_article(e["phrase"]) for e in identity if e.get("kind") in _WEARS_KINDS]
    if not has and not wears:
        return ""
    plural = pronoun not in {"She", "He"}
    parts = []
    if has:
        parts.append(("have " if plural else "has ") + join_natural(has))
    if wears:
        parts.append(("wear " if plural else "wears ") + join_natural(wears))
    joiner = ", and " if len(has) > 1 and wears else " and "
    return f"{pronoun} {joiner.join(parts)}."


def sentence_case(text: str) -> str:
    """Capitalise the first letter of every sentence and end with a period."""
    text = re.sub(r"\s+", " ", str(text or "")).strip().rstrip(";,")
    if not text:
        return ""
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), _upper_first(text))
    return text if text.endswith((".", "!", "?")) else text + "."


def descriptive_sentence(text: str, pronoun: str, background: bool = False) -> str:
    """Lighting or background text as complete sentences."""
    value = fix_pronouns(str(text or "").strip().rstrip("."), pronoun)
    if background and value and len(_words(value)) <= 5 and set(_words(value)) <= _BACKGROUND_ADJECTIVES:
        value = f"the background is {_lower_first(value)}"
    return sentence_case(value)


def _strip_subject(text: str) -> Tuple[str, str]:
    """Remove a leading subject; returns the rest and 'is' or 'finite' when known."""
    match = re.match(r"^(?:she|he|they|the (?:woman|man|person|subject))\s+(?:is|are)\s+(.*)$", text, re.I)
    if match:
        return match.group(1), "is"
    match = re.match(r"^(?:she|he|they|the (?:woman|man|person|subject))\s+(.*)$", text, re.I)
    if match:
        return match.group(1), "finite"
    return text, ""


def _pose_clause(pose: str, pronoun: str) -> Tuple[str, str]:
    """Classify a pose: ('is', text), ('finite', text), ('sentence', text) or ('', '')."""
    text = fix_pronouns(re.sub(r"\s+", " ", str(pose or "")).strip().rstrip(".,;"), pronoun)
    if not text:
        return "", ""
    text, known = _strip_subject(_lower_first(text) if text[:1].isupper() and not text[:2].isupper() else text)
    words = _words(text)
    if known:
        return known, text
    first = words[0]
    if first in _FINITE_POSE_VERBS:
        return "finite", text
    if any(w in _FRAMING_WORDS for w in words[:3]):
        return "sentence", text
    if first.endswith("ing") or first.endswith("ed") or first in _STATE_STARTERS:
        return "is", text
    if first == "with":
        return "finite", "poses " + text
    if first in _BODY_PART_STARTERS:
        if re.search(r"\b(?:is|are|rests?|hangs?|holds?)\b", text):
            return "sentence", text
        return "is", "posed with " + text
    return "sentence", text


def expression_phrase(expression: str) -> Tuple[str, str]:
    """('with', 'a neutral expression and closed lips') or ('clause', 'smiles broadly')."""
    text = re.sub(r"\s+", " ", str(expression or "")).strip().rstrip(".,;")
    if not text:
        return "", ""
    text = re.sub(r"^(?:she|he|they)\s+(?:is\s+|has\s+)?", "", _lower_first(text), flags=re.I)
    text = re.sub(r"^with\s+", "", text)
    if _words(text)[0] in _FINITE_EXPRESSION_VERBS:
        return "clause", text
    text = re.sub(r"\s*(?:,\s*)?(?:and\s+)?with\s+", " and ", text)
    adjectives: List[str] = []
    nouns: List[str] = []
    pieces: List[str] = []
    for chunk in [c.strip() for c in text.split(" and ") if c.strip()]:
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        # "soft, closed-mouth smile" is one noun phrase; "pensive, lips pursed" are two.
        if len(parts) > 1 and _words(parts[-1])[-1] in _EXPRESSION_NOUNS and not any(
                w in _EXPRESSION_NOUNS for p in parts[:-1] for w in _words(p)):
            pieces.append(chunk)
        else:
            pieces.extend(parts)
    for chunk in pieces:
        chunk = re.sub(r"^(?:an?|the)\s+", "", chunk)
        chunk = re.sub(r"^(lips|mouth)\s+(pursed|closed|parted|open)$", r"\2 \1", chunk)
        words = _words(chunk)
        if any(w in _EXPRESSION_NOUNS for w in words):
            noun = next(w for w in reversed(words) if w in _EXPRESSION_NOUNS)
            nouns.append(chunk if noun in _PLURAL_EXPRESSION_NOUNS else with_article(chunk))
        elif chunk and len(words) <= 4:
            adjectives.append(chunk)
    parts = []
    if adjectives:
        parts.append(with_article(f"{join_natural(adjectives)} expression"))
    parts.extend(nouns)
    return ("with", join_natural(parts)) if parts else ("", "")


def gaze_phrase(gaze: str, pose_text: str = "") -> str:
    """Participle gaze phrase ('looking toward the camera'); empty when the pose already says it."""
    text = re.sub(r"\s+", " ", str(gaze or "")).strip().rstrip(".,;")
    if not text:
        return ""
    text, _known = _strip_subject(_lower_first(text))
    words = text.split()
    first = words[0].lower()
    if first in _GAZE_ING:
        text = " ".join([_GAZE_ING[first]] + words[1:])
    elif first not in _GAZE_VERBS and (first in _GAZE_NEEDS_VERB or first.startswith("off")):
        text = "looking " + text
    target = re.sub(r"^(?:looking|gazing|staring|glancing|facing|watching|peering)\s+(?:directly\s+|straight\s+)?(?:at|toward|towards|into)?\s*", "", text)
    pose_lower = pose_text.lower()
    if target and target in pose_lower and re.search(r"\b(?:look|fac|gaz|toward)", pose_lower):
        return ""
    return text


def pose_sentences(pronoun: str, pose: str, expression: str, gaze: str) -> List[str]:
    """One or two sentences for pose, expression and gaze with a matching verb form."""
    be = "is" if pronoun in {"She", "He"} else "are"
    kind, clause = _pose_clause(pose, pronoun)
    expr_kind, expr = expression_phrase(expression)
    eyes_closed = bool(re.search(r"\beyes closed\b", f"{expression} {gaze}", re.I)) and "eyes closed" not in clause.lower()
    look = "" if re.search(r"\beyes closed\b", gaze, re.I) else gaze_phrase(fix_pronouns(gaze, pronoun), clause)
    if re.search(r"\beyes closed\b", expression, re.I):
        expr_kind, expr = "", ""
    tail = []
    if expr_kind == "with":
        tail.append("with " + expr)
    if eyes_closed:
        tail.append("with eyes closed")
    if look:
        tail.append(look)
    sentences: List[str] = []
    if kind in {"is", "finite"}:
        verb = f"{pronoun} {be} " if kind == "is" else f"{pronoun} "
        sentences.append(sentence_case(verb + ", ".join([clause] + tail)))
    else:
        if kind == "sentence":
            sentences.append(sentence_case(clause))
        if look:
            rest = [t for t in tail if t != look]
            sentences.append(sentence_case(f"{pronoun} {be} " + ", ".join([look] + rest)))
        elif tail:
            sentences.append(sentence_case(f"{pronoun} {'has' if be == 'is' else 'have'} " + " and ".join(t[5:] if t.startswith("with ") else t for t in tail)))
    if expr_kind == "clause":
        sentences.append(sentence_case(f"{pronoun} {fix_pronouns(expr, pronoun)}" if be == "is" else f"{pronoun} {expr}"))
    return sentences


def caption_opening(facts: Dict[str, Any]) -> str:
    """Return the exact start every caption must have (trigger and photo type)."""
    style = facts["photo_type"]
    if facts.get("visual_style"):
        style = f"{facts['visual_style']} {style}"
    if facts["caption_profile"] == "krea2_character":
        opening = f"{facts['trigger']}, {with_article(style)}"
    else:
        opening = f"{with_article(style)[:1].upper()}{with_article(style)[1:]} of {facts['trigger']}"
    if facts.get("gender_class"):
        opening += f", {with_article(facts['gender_class'])}"
    if facts.get("anchor"):
        opening += " with " + join_natural(facts["anchor"])
    return opening
