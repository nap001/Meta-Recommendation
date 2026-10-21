"""Unit tests for the complete confirmation semantic validator."""

from __future__ import annotations

import pytest

from llm_gateway.constants import SemanticValidatorId
from llm_gateway.validation import (
    is_complete_confirmation_validator_registered,
    register_wp5_semantic_validators,
    validate_confirmation_payload_semantics,
)
from llm_gateway.validators import ValidatorRegistry

pytestmark = pytest.mark.backend_unit


def _action(action_id: str, label: str, value: object, key: str = "use_case") -> dict:
    return {
        "id": action_id,
        "label": label,
        "value": str(value),
        "preference_patch": {key: value},
    }


def _payload() -> dict:
    return {
        "message": "Would you prefer Work or Gaming?",
        "quick_actions": [
            _action("work", "Work", "work"),
            _action("gaming", "Gaming", "gaming"),
        ],
    }


def _validate(payload: dict, allowed: list[str] | None = None) -> tuple[bool, list[str]]:
    return validate_confirmation_payload_semantics(
        payload,
        {"allowed_preference_patch_keys": allowed or ["use_case"]},
    )


def test_explicit_registration_replaces_placeholder_only_when_called() -> None:
    registry = ValidatorRegistry()
    original = registry.get_validator(SemanticValidatorId.CONFIRMATION_PAYLOAD.value)
    assert original is not validate_confirmation_payload_semantics
    assert not is_complete_confirmation_validator_registered(registry)

    register_wp5_semantic_validators(registry)

    assert registry.get_validator(SemanticValidatorId.CONFIRMATION_PAYLOAD.value) is validate_confirmation_payload_semantics
    assert is_complete_confirmation_validator_registered(registry)
    assert registry.get_validator(SemanticValidatorId.NONE.value) is not None


def test_valid_payload_and_payload_without_actions_pass() -> None:
    assert _validate(_payload()) == (True, [])
    assert _validate({"message": "Is that correct?"}) == (True, [])


def test_each_action_must_have_exactly_one_allowed_patch_key() -> None:
    payload = _payload()
    payload["quick_actions"][0]["preference_patch"] = {"use_case": "work", "mood": "calm"}
    ok, errors = _validate(payload, ["use_case", "mood"])
    assert not ok
    assert any("exactly one" in error for error in errors)

    payload = _payload()
    payload["quick_actions"][0]["preference_patch"] = {"unknown": "work"}
    ok, errors = _validate(payload)
    assert not ok
    assert any("disallowed" in error for error in errors)


@pytest.mark.parametrize("value", [None, "", "  ", "any", [], {"nested": []}])
def test_patch_values_must_be_meaningful(value: object) -> None:
    payload = _payload()
    payload["quick_actions"][0]["preference_patch"] = {"use_case": value}
    ok, errors = _validate(payload)
    assert not ok
    assert any("non-meaningful" in error for error in errors)


@pytest.mark.parametrize("value", [0, False, ["", "work"], {"nested": "work"}])
def test_existing_meaningful_value_semantics_are_preserved(value: object) -> None:
    payload = _payload()
    payload["quick_actions"][0]["preference_patch"] = {"use_case": value}
    assert _validate(payload)[0]


def test_all_actions_must_patch_the_same_dimension() -> None:
    payload = _payload()
    payload["quick_actions"][1]["preference_patch"] = {"mood": "energetic"}
    ok, errors = _validate(payload, ["use_case", "mood"])
    assert not ok
    assert any("same preference dimension" in error for error in errors)


def test_action_ids_and_casefolded_labels_must_be_unique() -> None:
    payload = _payload()
    payload["quick_actions"][1]["id"] = "work"
    payload["quick_actions"][1]["label"] = "WORK"
    ok, errors = _validate(payload)
    assert not ok
    assert any("IDs must be unique" in error for error in errors)
    assert any("labels must be unique" in error for error in errors)


def test_message_must_mention_at_least_one_label() -> None:
    payload = _payload()
    payload["message"] = "Please choose an option."
    ok, errors = _validate(payload)
    assert not ok
    assert any("mention at least one" in error for error in errors)

    payload["message"] = "Would WORK suit you?"
    assert _validate(payload)[0]


def test_missing_validator_inputs_fail_closed() -> None:
    ok, errors = validate_confirmation_payload_semantics(_payload(), {})
    assert not ok
    assert errors == ["Allowed preference patch keys are unavailable."]


@pytest.mark.parametrize(
    "inputs",
    [
        None,
        {"allowed_preference_patch_keys": []},
        {"allowed_preference_patch_keys": [["use_case"]]},
        {"allowed_preference_patch_keys": [{"key": "use_case"}]},
    ],
)
def test_malformed_validator_inputs_fail_closed_without_type_error(inputs: object) -> None:
    ok, errors = validate_confirmation_payload_semantics(_payload(), inputs)  # type: ignore[arg-type]
    assert not ok
    assert errors


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("id", [], "invalid ID"),
        ("id", {}, "invalid ID"),
        ("label", [], "invalid label"),
        ("label", {}, "invalid label"),
    ],
)
def test_unhashable_action_identity_fields_fail_closed(
    field: str,
    value: object,
    expected_error: str,
) -> None:
    payload = _payload()
    payload["quick_actions"][0][field] = value

    ok, errors = _validate(payload)

    assert not ok
    assert any(expected_error in error for error in errors)
