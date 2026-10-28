"""Deterministic normalization for the enrolled confirmation response."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class ConfirmationNormalizationContext:
    """Trusted presentation context captured by a per-request hook factory."""

    language: str
    fallback_message: str

    def __post_init__(self) -> None:
        if self.language not in {"en", "zh"}:
            raise ValueError("Confirmation normalization language must be 'en' or 'zh'.")
        if not isinstance(self.fallback_message, str) or not self.fallback_message.strip():
            raise ValueError("Confirmation normalization fallback message must be nonempty.")
        object.__setattr__(self, "fallback_message", self.fallback_message.strip())


def is_meaningful_patch_value(value: Any) -> bool:
    """Match the existing recursive meaning check used by confirmation actions."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() != "any"
    if isinstance(value, (list, tuple, set)):
        return any(is_meaningful_patch_value(item) for item in value)
    if isinstance(value, dict):
        return any(is_meaningful_patch_value(item) for item in value.values())
    return True


def _slugify_quick_action(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:64] or "option"


def _allocate_unique_action_id(
    base_id: str,
    *,
    seen_ids: set[str],
    initial_suffix: int,
) -> str:
    """Return a deterministic unique ID within the registered 66-char limit."""

    if base_id not in seen_ids:
        return base_id

    suffix_number = initial_suffix
    while True:
        suffix = f"_{suffix_number}"
        candidate = f"{base_id[: 66 - len(suffix)]}{suffix}"
        if candidate not in seen_ids:
            return candidate
        suffix_number += 1


def _clean_quick_actions(
    raw_actions: Any,
    allowed_preference_patch_keys: Iterable[str],
) -> List[Dict[str, Any]]:
    if not isinstance(raw_actions, list):
        return []

    allowed_keys = set(allowed_preference_patch_keys)
    cleaned: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_labels = set()
    patch_keys = set()

    for index, raw in enumerate(raw_actions[:6]):
        if not isinstance(raw, dict):
            continue

        label = str(raw.get("label") or raw.get("value") or "").strip()[:40]
        if not label:
            continue

        patch = raw.get("preference_patch")
        if not isinstance(patch, dict):
            continue
        filtered_patch = {
            str(key): value
            for key, value in patch.items()
            if str(key) in allowed_keys and is_meaningful_patch_value(value)
        }
        if len(filtered_patch) != 1:
            continue

        patch_key = next(iter(filtered_patch))
        action_id = _slugify_quick_action(
            raw.get("id") or f"{patch_key}_{raw.get('value') or label or index}"
        )
        action_id = _allocate_unique_action_id(
            action_id,
            seen_ids=seen_ids,
            initial_suffix=index + 1,
        )

        label_key = label.casefold()
        if label_key in seen_labels:
            continue

        value = raw.get("value")
        if value is None or value == "":
            value = next(iter(filtered_patch.values()))

        action: Dict[str, Any] = {
            "id": action_id,
            "label": label,
            "value": str(value),
            "preference_patch": filtered_patch,
        }
        message = raw.get("message")
        if isinstance(message, str) and message.strip():
            action["message"] = message.strip()[:120]

        cleaned.append(action)
        seen_ids.add(action_id)
        seen_labels.add(label_key)
        patch_keys.add(patch_key)

    if len(cleaned) < 2 or len(cleaned) > 4:
        return []
    if len(patch_keys) != 1:
        return []
    return cleaned


def _join_choice_labels(labels: List[str], language: str) -> str:
    clean = [str(label).strip() for label in labels if str(label).strip()]
    if not clean:
        return ""
    if language == "zh":
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]}还是{clean[1]}"
        return f"{'、'.join(clean[:-1])}还是{clean[-1]}"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} or {clean[1]}"
    return f"{', '.join(clean[:-1])}, or {clean[-1]}"


def _ensure_message_mentions_quick_actions(
    message: str,
    quick_actions: List[Dict[str, Any]],
    language: str,
) -> str:
    labels = [
        str(action.get("label") or "").strip()
        for action in quick_actions
        if action.get("label")
    ]
    if not labels:
        return message
    lowered = message.casefold()
    if any(label.casefold() in lowered for label in labels):
        return message
    choices = _join_choice_labels(labels, language)
    if not choices:
        return message
    suffix = f"您更偏向{choices}呢？" if language == "zh" else f"Would you prefer {choices}?"
    return f"{message.rstrip()} {suffix}".strip()


def normalize_confirmation_payload(
    parsed_payload: Any,
    *,
    allowed_preference_patch_keys: Iterable[str],
    context: ConfirmationNormalizationContext,
) -> Dict[str, Any]:
    """Normalize an already parsed JSON candidate into the releasable shape."""

    source: Mapping[str, Any] = parsed_payload if isinstance(parsed_payload, dict) else {}
    message = source.get("message")
    if not isinstance(message, str) or not message.strip():
        message = context.fallback_message

    normalized: Dict[str, Any] = {"message": message.strip()}
    quick_actions = _clean_quick_actions(
        source.get("quick_actions"),
        allowed_preference_patch_keys,
    )
    if quick_actions:
        normalized["quick_actions"] = quick_actions
        normalized["message"] = _ensure_message_mentions_quick_actions(
            normalized["message"],
            quick_actions,
            context.language,
        )
    return normalized


__all__ = [
    "ConfirmationNormalizationContext",
    "is_meaningful_patch_value",
    "normalize_confirmation_payload",
]
