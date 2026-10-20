"""Fail-closed unit tests for WP5-B response-contract preflight utilities."""

from __future__ import annotations

import pytest

from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import ResponseContractV1, ResponseKind
from llm_gateway.validation import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    CONVERSATION_SUMMARY_SCHEMA_ID,
    PreflightIssueCode,
    preflight_response_contract,
    resolve_response_schema,
    schemas_structurally_equal,
    validate_draft7_schema_definition,
)
from llm_gateway.validation import preflight as preflight_module

pytestmark = pytest.mark.backend_unit


def _summary_contract(**changes: object) -> ResponseContractV1:
    values = {
        "kind": ResponseKind.TEXT,
        "schema_id": CONVERSATION_SUMMARY_SCHEMA_ID,
        "json_schema": None,
        "semantic_validator_id": SemanticValidatorId.NONE.value,
        "validator_inputs": {},
        "allow_empty": False,
    }
    values.update(changes)
    return ResponseContractV1(**values)


def _confirmation_contract(**changes: object) -> ResponseContractV1:
    entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert entry is not None
    values = {
        "kind": ResponseKind.JSON,
        "schema_id": CONFIRMATION_PAYLOAD_SCHEMA_ID,
        "json_schema": entry.schema(),
        "semantic_validator_id": SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
        "validator_inputs": {"allowed_preference_patch_keys": ["use_case"]},
        "allow_empty": False,
    }
    values.update(changes)
    return ResponseContractV1(**values)


def test_approved_summary_and_confirmation_contracts_pass() -> None:
    assert preflight_response_contract(_summary_contract()).passed
    assert preflight_response_contract(_confirmation_contract()).passed


def test_unknown_schema_id_fails_closed() -> None:
    result = preflight_response_contract(_summary_contract(schema_id="unknown.v1"))
    assert not result.passed
    assert result.first_issue is not None
    assert result.first_issue.code == PreflightIssueCode.UNKNOWN_SCHEMA_ID


def test_kind_validator_and_allow_empty_mismatches_fail_closed() -> None:
    kind_result = preflight_response_contract(
        _summary_contract(kind=ResponseKind.TOOL_CALLS)
    )
    assert kind_result.first_issue.code == PreflightIssueCode.RESPONSE_KIND_MISMATCH

    validator_result = preflight_response_contract(
        _summary_contract(semantic_validator_id=SemanticValidatorId.CONFIRMATION_PAYLOAD.value)
    )
    assert validator_result.first_issue.code == PreflightIssueCode.RESPONSE_CONTRACT_MISMATCH

    empty_result = preflight_response_contract(_summary_contract(allow_empty=True))
    assert empty_result.first_issue.code == PreflightIssueCode.RESPONSE_CONTRACT_MISMATCH


def test_inline_schema_mismatch_and_invalid_definition_fail_closed() -> None:
    entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert entry is not None
    mismatched = entry.schema()
    assert mismatched is not None
    mismatched["additionalProperties"] = True
    mismatch_result = preflight_response_contract(
        _confirmation_contract(json_schema=mismatched)
    )
    assert mismatch_result.first_issue.code == PreflightIssueCode.SCHEMA_MISMATCH

    invalid_result = preflight_response_contract(
        _confirmation_contract(json_schema={"type": 7})
    )
    assert invalid_result.first_issue.code == PreflightIssueCode.INVALID_SCHEMA_DEFINITION


def test_text_contract_rejects_an_inline_json_schema() -> None:
    result = preflight_response_contract(
        _summary_contract(json_schema={"type": "string"})
    )
    assert result.first_issue.code == PreflightIssueCode.SCHEMA_MISMATCH


def test_invalid_and_unknown_validator_inputs_fail_closed() -> None:
    invalid_inputs = preflight_response_contract(
        _confirmation_contract(
            validator_inputs={
                "allowed_preference_patch_keys": ["use_case"],
                "language": "en",
            }
        )
    )
    assert invalid_inputs.first_issue.code == PreflightIssueCode.INVALID_VALIDATOR_INPUTS

    none_inputs = preflight_response_contract(
        _summary_contract(validator_inputs={"unexpected": True})
    )
    assert none_inputs.first_issue.code == PreflightIssueCode.INVALID_VALIDATOR_INPUTS

    unknown_validator = preflight_response_contract(
        _summary_contract(semantic_validator_id="unknown.validator.v1")
    )
    assert unknown_validator.first_issue.code == PreflightIssueCode.UNKNOWN_VALIDATOR_ID


def test_missing_draft7_dependency_fails_json_closed_but_not_free_text_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight_module, "_DRAFT7_VALIDATOR_CLASS", None)
    json_result = preflight_response_contract(_confirmation_contract())
    assert json_result.first_issue.code == PreflightIssueCode.VALIDATION_DEPENDENCY_UNAVAILABLE
    assert preflight_response_contract(_summary_contract()).passed


def test_schema_definition_check_is_fail_closed_and_safe() -> None:
    assert validate_draft7_schema_definition({"type": "object"}).passed
    result = validate_draft7_schema_definition({"type": 7})
    assert not result.passed
    assert result.first_issue.code == PreflightIssueCode.INVALID_SCHEMA_DEFINITION
    assert result.first_issue.message == "The JSON Schema definition is invalid for Draft-7."


def test_structural_equality_ignores_object_order_but_preserves_array_order() -> None:
    left = {"type": "object", "required": ["a", "b"], "properties": {"a": {}}}
    reordered = {"properties": {"a": {}}, "required": ["a", "b"], "type": "object"}
    changed_array = {"type": "object", "required": ["b", "a"], "properties": {"a": {}}}
    assert schemas_structurally_equal(left, reordered)
    assert not schemas_structurally_equal(left, changed_array)
    assert not schemas_structurally_equal({"const": True}, {"const": 1})
