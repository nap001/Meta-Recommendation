"""WP5-B semantic validators registered in Member A's executable registry."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from llm_gateway.constants import SemanticValidatorId
from llm_gateway.validation.normalization import is_meaningful_patch_value
from llm_gateway.validators import ValidatorRegistry

_MAX_SEMANTIC_ERRORS = 16


def _append_bounded(errors: List[str], message: str) -> None:
    if len(errors) < _MAX_SEMANTIC_ERRORS:
        errors.append(message)


def validate_confirmation_payload_semantics(
    output: Any,
    inputs: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate deterministic invariants that JSON Schema cannot express."""

    errors: List[str] = []
    if not isinstance(output, dict):
        return False, ["Confirmation payload must be an object."]

    if not isinstance(inputs, dict):
        return False, ["Semantic validator inputs must be an object."]
    allowed_input = inputs.get("allowed_preference_patch_keys")
    if (
        not isinstance(allowed_input, list)
        or not allowed_input
        or not all(isinstance(item, str) for item in allowed_input)
    ):
        return False, ["Allowed preference patch keys are unavailable."]
    allowed_keys = set(allowed_input)

    quick_actions = output.get("quick_actions")
    if quick_actions is None:
        return True, []
    if not isinstance(quick_actions, list):
        return False, ["Quick actions must be an array."]

    seen_ids = set()
    seen_labels = set()
    patch_dimensions = set()
    labels: List[str] = []

    for index, action in enumerate(quick_actions):
        if not isinstance(action, dict):
            _append_bounded(errors, f"Quick action {index} must be an object.")
            continue

        patch = action.get("preference_patch")
        if not isinstance(patch, dict) or len(patch) != 1:
            _append_bounded(errors, f"Quick action {index} must patch exactly one preference key.")
        else:
            patch_key, patch_value = next(iter(patch.items()))
            if patch_key not in allowed_keys:
                _append_bounded(errors, f"Quick action {index} uses a disallowed preference key.")
            if not is_meaningful_patch_value(patch_value):
                _append_bounded(errors, f"Quick action {index} has a non-meaningful patch value.")
            patch_dimensions.add(patch_key)

        action_id = action.get("id")
        if not isinstance(action_id, str) or not action_id:
            _append_bounded(errors, f"Quick action {index} has an invalid ID.")
        elif action_id in seen_ids:
            _append_bounded(errors, "Quick-action IDs must be unique.")
        else:
            seen_ids.add(action_id)

        label = action.get("label")
        if not isinstance(label, str) or not label:
            _append_bounded(errors, f"Quick action {index} has an invalid label.")
        else:
            label_key = label.casefold()
            if label_key in seen_labels:
                _append_bounded(errors, "Quick-action labels must be unique case-insensitively.")
            else:
                seen_labels.add(label_key)
            labels.append(label)

    if len(patch_dimensions) > 1:
        _append_bounded(errors, "All quick actions must patch the same preference dimension.")

    message = output.get("message")
    if labels and (
        not isinstance(message, str)
        or not any(label.casefold() in message.casefold() for label in labels)
    ):
        _append_bounded(errors, "The confirmation message must mention at least one quick-action label.")

    return len(errors) == 0, errors


def register_wp5_semantic_validators(registry: ValidatorRegistry) -> None:
    """Explicitly install complete WP5-B predicates in an existing registry."""

    registry.register_validator(
        SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
        validate_confirmation_payload_semantics,
    )


def is_complete_confirmation_validator_registered(registry: ValidatorRegistry) -> bool:
    """Return whether the registry contains this exact reviewed implementation."""

    return (
        registry.get_validator(SemanticValidatorId.CONFIRMATION_PAYLOAD.value)
        is validate_confirmation_payload_semantics
    )


__all__ = [
    "is_complete_confirmation_validator_registered",
    "register_wp5_semantic_validators",
    "validate_confirmation_payload_semantics",
]
