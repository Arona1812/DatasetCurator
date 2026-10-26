"""Shared model defaults and Responses request compatibility (2026-10-11).

Sources: https://developers.openai.com/api/docs/guides/latest-model
         https://developers.openai.com/api/docs/models/chat-latest
Keep explicit saved model choices; upgrades are applied through UI presets.
"""
from copy import deepcopy
from typing import Any, Dict, Optional

ROUTINE_MODEL = "gpt-6-luna"
FRAME_VISION_MODEL = ROUTINE_MODEL
PROFILE_MODEL = "gpt-6.1-sol"
CHATGPT_MODEL = "chat-latest"
FLAGSHIP_MODEL = "gpt-6-astra"
# Only current models are offered. Custom names stay possible in the UI
# dropdowns; saved GPT-5.x presets are migrated by curator_core.
MODEL_CHOICES = [
    ROUTINE_MODEL, PROFILE_MODEL, "gpt-6-sol", FLAGSHIP_MODEL, CHATGPT_MODEL,
]


def normalize_reasoning(model: str, effort: Optional[str]) -> Optional[str]:
    name = str(model or "").strip().lower()
    value = str(effort or "").strip().lower()
    # The rolling ChatGPT alias does not expose a stable reasoning contract.
    if name == CHATGPT_MODEL or value in {"", "auto", "default"}:
        return None
    if value == "minimal":
        value = "low"
    if value not in {"none", "low", "medium", "high", "xhigh", "max"}:
        value = "none"
    if name.startswith(("gpt-6-astra", "gpt-6.1-sol")) and value == "none":
        value = "low"
    if value == "max" and not name.startswith(("gpt-5.6", "gpt-6")):
        value = "xhigh"
    return value


def prepare_response_payload(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy; retries and callers retain their original request."""
    request = deepcopy(payload)
    request["model"] = model
    effort = normalize_reasoning(model, request.pop("_reasoning_effort", None))
    if effort is not None:
        request["reasoning"] = {"effort": effort}
    elif model == CHATGPT_MODEL:
        request.pop("reasoning", None)
    effective_effort = (request.get("reasoning") or {}).get("effort")
    name = str(model).lower()
    if name == CHATGPT_MODEL or (
        name.startswith("gpt-6") and effective_effort != "none"
    ) or (
        name.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.6")) and effective_effort
    ):
        for key in ("temperature", "top_p", "top_logprobs"):
            request.pop(key, None)
        if isinstance(request.get("include"), list):
            request["include"] = [v for v in request["include"] if v != "message.output_text.logprobs"]
    return request


def model_preset(name: str) -> tuple:
    """Audit, trigger, escalation, profile, caption, repair model/effort pairs."""
    if name == "chatgpt":
        return (CHATGPT_MODEL, "none", ROUTINE_MODEL, "none", PROFILE_MODEL, "low",
                PROFILE_MODEL, "low", CHATGPT_MODEL, "none", PROFILE_MODEL, "low")
    if name == "quality":
        return ("gpt-6-astra", "low", ROUTINE_MODEL, "none", "gpt-6-astra", "medium",
                "gpt-6-astra", "low", PROFILE_MODEL, "low", "gpt-6-astra", "low")
    return (ROUTINE_MODEL, "none", ROUTINE_MODEL, "none", PROFILE_MODEL, "low",
            PROFILE_MODEL, "low", ROUTINE_MODEL, "none", PROFILE_MODEL, "low")
