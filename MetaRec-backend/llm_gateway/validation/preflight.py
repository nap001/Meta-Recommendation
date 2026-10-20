"""Fail-closed validation preflight utilities.

These functions are reusable before provider invocation, but Increment 1 does
not wire them into the gateway execution path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import ResponseContractV1
from llm_gateway.validation.schemas import (
    ResponseSchemaEntry,
    ValidatorInputSchemaEntry,
    resolve_response_schema,
    resolve_validator_input_schema,
)

try:
    from jsonschema import Draft7Validator as _IMPORTED_DRAFT7_VALIDATOR
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests
    _IMPORTED_DRAFT7_VALIDATOR = None

_DRAFT7_VALIDATOR_CLASS = _IMPORTED_DRAFT7_VALIDATOR


class PreflightIssueCode(str, Enum):
    UNKNOWN_SCHEMA_ID = "unknown_schema_id"
    RESPONSE_KIND_MISMATCH = "response_kind_mismatch"
    RESPONSE_CONTRACT_MISMATCH = "response_contract_mismatch"
    INVALID_SCHEMA_DEFINITION = "invalid_schema_definition"
    SCHEMA_MISMATCH = "schema_mismatch"
    UNKNOWN_VALIDATOR_ID = "unknown_validator_id"
    INVALID_VALIDATOR_INPUTS = "invalid_validator_inputs"
    VALIDATION_DEPENDENCY_UNAVAILABLE = "validation_dependency_unavailable"


@dataclass(frozen=True)
class PreflightIssue:
    code: PreflightIssueCode
    message: str


@dataclass(frozen=True)
class ValidationPreflightResult:
    passed: bool
    issues: Tuple[PreflightIssue, ...] = ()

    @property
    def first_issue(self) -> Optional[PreflightIssue]:
        return self.issues[0] if self.issues else None


def _passed() -> ValidationPreflightResult:
    return ValidationPreflightResult(passed=True)


def _failed(code: PreflightIssueCode, message: str) -> ValidationPreflightResult:
    return ValidationPreflightResult(
        passed=False,
        issues=(PreflightIssue(code=code, message=message),),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def schemas_structurally_equal(left: Any, right: Any) -> bool:
    """Compare JSON structures with object-key order ignored and arrays ordered."""
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def validate_draft7_schema_definition(schema: Any) -> ValidationPreflightResult:
    """Check that ``schema`` is a valid Draft-7 schema definition."""
    if _DRAFT7_VALIDATOR_CLASS is None:
        return _failed(
            PreflightIssueCode.VALIDATION_DEPENDENCY_UNAVAILABLE,
            "Draft-7 JSON Schema validation is unavailable.",
        )
    if not isinstance(schema, Mapping):
        return _failed(
            PreflightIssueCode.INVALID_SCHEMA_DEFINITION,
            "The JSON Schema definition must be an object.",
        )
    try:
        _DRAFT7_VALIDATOR_CLASS.check_schema(dict(schema))
    except Exception:
        return _failed(
            PreflightIssueCode.INVALID_SCHEMA_DEFINITION,
            "The JSON Schema definition is invalid for Draft-7.",
        )
    return _passed()


def _validate_validator_inputs(
    entry: ValidatorInputSchemaEntry,
    validator_inputs: Mapping[str, Any],
) -> ValidationPreflightResult:
    schema = entry.schema()
    definition_result = validate_draft7_schema_definition(schema)
    if not definition_result.passed:
        return definition_result
    try:
        is_valid = _DRAFT7_VALIDATOR_CLASS(schema).is_valid(dict(validator_inputs))
    except Exception:
        return _failed(
            PreflightIssueCode.INVALID_VALIDATOR_INPUTS,
            "Validator inputs could not be checked against their registered schema.",
        )
    if not is_valid:
        return _failed(
            PreflightIssueCode.INVALID_VALIDATOR_INPUTS,
            "Validator inputs do not match their registered schema.",
        )
    return _passed()


def _validate_response_schema(
    entry: ResponseSchemaEntry,
    inline_schema: Any,
) -> ValidationPreflightResult:
    registered_schema = entry.schema()
    if registered_schema is None:
        if inline_schema is not None:
            return _failed(
                PreflightIssueCode.SCHEMA_MISMATCH,
                "This response schema ID does not accept an inline JSON Schema.",
            )
        return _passed()

    registered_result = validate_draft7_schema_definition(registered_schema)
    if not registered_result.passed:
        return registered_result
    inline_result = validate_draft7_schema_definition(inline_schema)
    if not inline_result.passed:
        return inline_result
    if not schemas_structurally_equal(registered_schema, inline_schema):
        return _failed(
            PreflightIssueCode.SCHEMA_MISMATCH,
            "The inline JSON Schema does not match the registered schema ID.",
        )
    return _passed()


def preflight_response_contract(contract: ResponseContractV1) -> ValidationPreflightResult:
    """Validate one approved response contract without invoking a provider."""
    response_entry = resolve_response_schema(contract.schema_id)
    if response_entry is None:
        return _failed(
            PreflightIssueCode.UNKNOWN_SCHEMA_ID,
            "The response schema ID is not registered.",
        )
    if contract.kind != response_entry.kind:
        return _failed(
            PreflightIssueCode.RESPONSE_KIND_MISMATCH,
            "The response kind does not match the registered schema ID.",
        )

    validator_entry = resolve_validator_input_schema(contract.semantic_validator_id)
    if validator_entry is None:
        return _failed(
            PreflightIssueCode.UNKNOWN_VALIDATOR_ID,
            "The semantic validator ID is not registered for preflight.",
        )
    if contract.semantic_validator_id != response_entry.semantic_validator_id:
        return _failed(
            PreflightIssueCode.RESPONSE_CONTRACT_MISMATCH,
            "The semantic validator ID does not match the registered response contract.",
        )
    if contract.allow_empty != response_entry.allow_empty:
        return _failed(
            PreflightIssueCode.RESPONSE_CONTRACT_MISMATCH,
            "The allow_empty setting does not match the registered response contract.",
        )

    response_schema_result = _validate_response_schema(response_entry, contract.json_schema)
    if not response_schema_result.passed:
        return response_schema_result

    # The reviewed free-text validator has one exact input value: the empty
    # object. It needs no JSON Schema runtime to establish that invariant.
    if contract.semantic_validator_id == SemanticValidatorId.NONE.value:
        if contract.validator_inputs != {}:
            return _failed(
                PreflightIssueCode.INVALID_VALIDATOR_INPUTS,
                "The 'none' validator accepts only empty validator inputs.",
            )
        return _passed()

    return _validate_validator_inputs(validator_entry, contract.validator_inputs)
