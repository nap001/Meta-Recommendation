"""Deterministic tests for the WP5-B immutable Draft-7 schema registry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from jsonschema import Draft7Validator

from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import ResponseKind
from llm_gateway.validation import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    CONVERSATION_SUMMARY_SCHEMA_ID,
    registered_response_schema_ids,
    registered_validator_ids,
    resolve_response_schema,
    resolve_validator_input_schema,
)

pytestmark = pytest.mark.backend_unit


def _confirmation_response_validator() -> Draft7Validator:
    entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert entry is not None
    schema = entry.schema()
    assert schema is not None
    return Draft7Validator(schema)


def _quick_action(
    index: int,
    *,
    action_id: str | None = None,
    label: str | None = None,
    patch_key: str = "use_case",
    message: str | None = None,
) -> dict:
    action = {
        "id": action_id or f"option_{index}",
        "label": label or f"Option {index}",
        "value": f"value_{index}",
        "preference_patch": {patch_key: f"value_{index}"},
    }
    if message is not None:
        action["message"] = message
    return action


def _confirmation_payload(*actions: dict) -> dict:
    return {"message": "Please choose an option.", "quick_actions": list(actions)}


def test_registry_contains_only_the_two_initial_response_contracts() -> None:
    assert registered_response_schema_ids() == (
        CONVERSATION_SUMMARY_SCHEMA_ID,
        CONFIRMATION_PAYLOAD_SCHEMA_ID,
    )
    assert registered_validator_ids() == (
        SemanticValidatorId.NONE.value,
        SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
    )


def test_summary_registration_is_explicit_text_none_contract() -> None:
    entry = resolve_response_schema(CONVERSATION_SUMMARY_SCHEMA_ID)
    assert entry is not None
    assert entry.kind == ResponseKind.TEXT
    assert entry.semantic_validator_id == SemanticValidatorId.NONE.value
    assert entry.allow_empty is False
    assert entry.schema() is None

    input_entry = resolve_validator_input_schema(SemanticValidatorId.NONE.value)
    assert input_entry is not None
    schema = input_entry.schema()
    Draft7Validator.check_schema(schema)
    validator = Draft7Validator(schema)
    assert validator.is_valid({})
    assert not validator.is_valid({"unexpected": True})


def test_confirmation_schemas_are_valid_draft7_and_closed() -> None:
    response_entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert response_entry is not None
    assert response_entry.kind == ResponseKind.JSON
    assert response_entry.semantic_validator_id == SemanticValidatorId.CONFIRMATION_PAYLOAD.value
    response_schema = response_entry.schema()
    assert response_schema is not None
    Draft7Validator.check_schema(response_schema)

    validator = Draft7Validator(response_schema)
    valid_payload = {
        "message": "Would you prefer work or gaming?",
        "quick_actions": [
            {
                "id": "work",
                "label": "Work",
                "value": "work",
                "preference_patch": {"use_case": "work"},
            },
            {
                "id": "gaming",
                "label": "Gaming",
                "value": "gaming",
                "preference_patch": {"use_case": "gaming"},
            },
        ],
    }
    assert validator.is_valid(valid_payload)
    assert not validator.is_valid({**valid_payload, "unexpected": True})


def test_confirmation_schema_preserves_existing_action_boundaries() -> None:
    entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert entry is not None
    schema = entry.schema()
    assert schema is not None
    validator = Draft7Validator(schema)

    assert validator.is_valid({"message": "Is that correct?"})
    assert not validator.is_valid({"message": "", "quick_actions": []})
    assert not validator.is_valid(
        {
            "message": "Choose one",
            "quick_actions": [
                {
                    "id": "invalid-id",
                    "label": "One",
                    "value": "one",
                    "preference_patch": {"use_case": "one"},
                },
                {
                    "id": "two",
                    "label": "Two",
                    "value": "two",
                    "preference_patch": {"unsupported": "two"},
                },
            ],
        }
    )


@pytest.mark.parametrize(
    ("action_count", "expected_valid"),
    [(1, False), (2, True), (4, True), (5, False)],
)
def test_confirmation_quick_action_count_boundaries(
    action_count: int,
    expected_valid: bool,
) -> None:
    validator = _confirmation_response_validator()
    payload = _confirmation_payload(
        *(_quick_action(index) for index in range(action_count))
    )
    assert validator.is_valid(payload) is expected_valid


@pytest.mark.parametrize(
    ("action_id", "expected_valid"),
    [("a" * 66, True), ("a" * 67, False)],
)
def test_confirmation_action_id_length_boundaries(
    action_id: str,
    expected_valid: bool,
) -> None:
    validator = _confirmation_response_validator()
    payload = _confirmation_payload(
        _quick_action(1, action_id=action_id),
        _quick_action(2),
    )
    assert validator.is_valid(payload) is expected_valid


def test_confirmation_accepts_a_chinese_action_id() -> None:
    validator = _confirmation_response_validator()
    payload = _confirmation_payload(
        _quick_action(1, action_id="工作"),
        _quick_action(2, action_id="游戏"),
    )
    assert validator.is_valid(payload)


@pytest.mark.parametrize(
    ("label", "expected_valid"),
    [("L" * 40, True), ("L" * 41, False)],
)
def test_confirmation_action_label_length_boundaries(
    label: str,
    expected_valid: bool,
) -> None:
    validator = _confirmation_response_validator()
    payload = _confirmation_payload(
        _quick_action(1, label=label),
        _quick_action(2),
    )
    assert validator.is_valid(payload) is expected_valid


@pytest.mark.parametrize(
    ("message", "expected_valid"),
    [("M" * 120, True), ("M" * 121, False)],
)
def test_confirmation_action_message_length_boundaries(
    message: str,
    expected_valid: bool,
) -> None:
    validator = _confirmation_response_validator()
    payload = _confirmation_payload(
        _quick_action(1, message=message),
        _quick_action(2),
    )
    assert validator.is_valid(payload) is expected_valid


def test_invalid_action_id_is_rejected_independently() -> None:
    validator = _confirmation_response_validator()
    payload = _confirmation_payload(
        _quick_action(1, action_id="invalid-id"),
        _quick_action(2),
    )
    assert not validator.is_valid(payload)


def test_unsupported_preference_patch_key_is_rejected_independently() -> None:
    validator = _confirmation_response_validator()
    payload = _confirmation_payload(
        _quick_action(1, patch_key="unsupported"),
        _quick_action(2),
    )
    assert not validator.is_valid(payload)


def test_confirmation_validator_inputs_are_bounded_and_exclude_language() -> None:
    entry = resolve_validator_input_schema(SemanticValidatorId.CONFIRMATION_PAYLOAD.value)
    assert entry is not None
    schema = entry.schema()
    Draft7Validator.check_schema(schema)
    validator = Draft7Validator(schema)

    assert validator.is_valid({"allowed_preference_patch_keys": ["genres", "mood"]})
    assert not validator.is_valid({"allowed_preference_patch_keys": []})
    assert not validator.is_valid({"allowed_preference_patch_keys": ["genres", "genres"]})
    assert not validator.is_valid({"allowed_preference_patch_keys": ["not_registered"]})
    assert not validator.is_valid(
        {"allowed_preference_patch_keys": ["genres"], "language": "en"}
    )


def test_confirmation_validator_inputs_accept_all_23_permitted_keys() -> None:
    entry = resolve_validator_input_schema(SemanticValidatorId.CONFIRMATION_PAYLOAD.value)
    assert entry is not None
    validator = Draft7Validator(entry.schema())
    allowed_keys = [
        "amenities",
        "attraction_types",
        "brand",
        "budget",
        "budget_range",
        "category",
        "dietary_restrictions",
        "dining_purpose",
        "exclude_genres",
        "flavor_profiles",
        "genres",
        "location",
        "min_rating",
        "model",
        "mood",
        "product",
        "restaurant_types",
        "stars",
        "subject",
        "tags",
        "typical_budget",
        "use_case",
        "year",
    ]

    assert len(allowed_keys) == 23
    assert validator.is_valid({"allowed_preference_patch_keys": allowed_keys})
    assert not validator.is_valid(
        {"allowed_preference_patch_keys": allowed_keys[:-1] + ["unknown_key"]}
    )


def test_registry_entries_and_documents_cannot_be_mutated_through_resolvers() -> None:
    entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert entry is not None
    with pytest.raises(FrozenInstanceError):
        entry.schema_id = "changed"  # type: ignore[misc]

    first = entry.schema()
    assert first is not None
    first["required"].append("changed")
    second = entry.schema()
    assert second is not None
    assert second["required"] == ["message"]


def test_unknown_registry_ids_do_not_fall_back() -> None:
    assert resolve_response_schema("unknown.response.v1") is None
    assert resolve_validator_input_schema("unknown.validator.v1") is None
