"""External validation hook for the two initial WP5-B call sites."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

from llm_gateway.adapters.base import RawAdapterResponse
from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import (
    LLMRequestV1,
    OutputPayloadV1,
    ResponseKind,
    ValidationGateResultV1,
    ValidationGateStatus,
    ValidationSummaryV1,
)
from llm_gateway.validation.normalization import (
    ConfirmationNormalizationContext,
    normalize_confirmation_payload,
)
from llm_gateway.validation.preflight import preflight_response_contract
from llm_gateway.validation.schemas import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    CONVERSATION_SUMMARY_SCHEMA_ID,
    resolve_response_schema,
)
from llm_gateway.validation.semantic import (
    is_complete_confirmation_validator_registered,
)
from llm_gateway.validators import ValidatorRegistry, validate_json_schema, validate_json_syntax


ExternalValidationHook = Callable[
    [LLMRequestV1, RawAdapterResponse, Dict[str, str]],
    ValidationGateResultV1,
]


def _contains_non_finite_number(value: object) -> bool:
    """Inspect the complete parsed candidate before normalization can drop fields."""

    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            return True
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return False


def _failed(
    validator_id: str,
    message: str,
    *,
    json_parse: ValidationGateStatus = ValidationGateStatus.NOT_APPLICABLE,
    json_schema: ValidationGateStatus = ValidationGateStatus.NOT_APPLICABLE,
    semantic: ValidationGateStatus = ValidationGateStatus.NOT_APPLICABLE,
) -> ValidationGateResultV1:
    return ValidationGateResultV1(
        passed=False,
        summary=ValidationSummaryV1(
            json_parse=json_parse,
            json_schema=json_schema,
            semantic=semantic,
            validator_id=validator_id,
            errors=[message],
        ),
        payload=None,
    )


def create_wp5_external_validation_hook(
    *,
    registry: ValidatorRegistry,
    confirmation_context: Optional[ConfirmationNormalizationContext] = None,
) -> ExternalValidationHook:
    """Create a request-safe hook while preserving Member A's exact interface."""

    def external_validator_hook(
        request: LLMRequestV1,
        raw_response: RawAdapterResponse,
        sanitized_mapping: Dict[str, str],
    ) -> ValidationGateResultV1:
        del sanitized_mapping  # Both initial call sites are deliberately tool-free.
        contract = request.response_contract
        validator_id = contract.semantic_validator_id

        preflight = preflight_response_contract(contract)
        if not preflight.passed:
            issue_code = preflight.first_issue.code.value if preflight.first_issue else "unknown"
            return _failed(
                validator_id,
                f"Response contract preflight failed: {issue_code}.",
            )

        if contract.schema_id == CONVERSATION_SUMMARY_SCHEMA_ID:
            if registry.get_validator(SemanticValidatorId.NONE.value) is None:
                return _failed(validator_id, "Required semantic validator is not registered.")
            text = (raw_response.text or "").strip()
            if not text:
                return _failed(validator_id, "Summary text is empty.")
            return ValidationGateResultV1(
                passed=True,
                summary=ValidationSummaryV1(validator_id=validator_id),
                payload=OutputPayloadV1(kind=ResponseKind.TEXT, text=text),
            )

        if contract.schema_id != CONFIRMATION_PAYLOAD_SCHEMA_ID:
            return _failed(validator_id, "The response schema is not supported by this hook.")
        if confirmation_context is None:
            return _failed(validator_id, "Confirmation normalization context is unavailable.")

        parse_ok, parsed_json, _ = validate_json_syntax(raw_response.text or "")
        if not parse_ok or _contains_non_finite_number(parsed_json):
            return _failed(
                validator_id,
                "Confirmation response is not valid JSON.",
                json_parse=ValidationGateStatus.FAILED,
            )
        if not isinstance(parsed_json, dict):
            return _failed(
                validator_id,
                "Confirmation response must be a JSON object.",
                json_parse=ValidationGateStatus.PASSED,
                json_schema=ValidationGateStatus.FAILED,
            )

        allowed_keys = contract.validator_inputs.get("allowed_preference_patch_keys", [])
        try:
            normalized = normalize_confirmation_payload(
                parsed_json,
                allowed_preference_patch_keys=allowed_keys,
                context=confirmation_context,
            )
        except Exception:
            return _failed(
                validator_id,
                "Confirmation normalization failed.",
                json_parse=ValidationGateStatus.PASSED,
            )

        response_entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
        registered_schema = response_entry.schema() if response_entry else None
        schema_ok, schema_errors = validate_json_schema(normalized, registered_schema)
        if not schema_ok:
            unavailable = any("validator unavailable" in error.lower() for error in schema_errors)
            message = (
                "JSON Schema validator unavailable."
                if unavailable
                else "Normalized confirmation payload does not match the registered schema."
            )
            return _failed(
                validator_id,
                message,
                json_parse=ValidationGateStatus.PASSED,
                json_schema=ValidationGateStatus.FAILED,
            )

        if not is_complete_confirmation_validator_registered(registry):
            return _failed(
                validator_id,
                "Required complete semantic validator is not registered.",
                json_parse=ValidationGateStatus.PASSED,
                json_schema=ValidationGateStatus.PASSED,
                semantic=ValidationGateStatus.FAILED,
            )
        semantic_validator = registry.get_validator(validator_id)
        if semantic_validator is None:
            return _failed(
                validator_id,
                "Required semantic validator is not registered.",
                json_parse=ValidationGateStatus.PASSED,
                json_schema=ValidationGateStatus.PASSED,
                semantic=ValidationGateStatus.FAILED,
            )
        try:
            semantic_ok, semantic_errors = semantic_validator(
                normalized,
                contract.validator_inputs,
            )
        except Exception:
            return _failed(
                validator_id,
                "Semantic validation failed unexpectedly.",
                json_parse=ValidationGateStatus.PASSED,
                json_schema=ValidationGateStatus.PASSED,
                semantic=ValidationGateStatus.FAILED,
            )
        if not semantic_ok:
            safe_message = semantic_errors[0] if semantic_errors else "Confirmation semantics are invalid."
            return _failed(
                validator_id,
                safe_message,
                json_parse=ValidationGateStatus.PASSED,
                json_schema=ValidationGateStatus.PASSED,
                semantic=ValidationGateStatus.FAILED,
            )

        return ValidationGateResultV1(
            passed=True,
            summary=ValidationSummaryV1(
                json_parse=ValidationGateStatus.PASSED,
                json_schema=ValidationGateStatus.PASSED,
                semantic=ValidationGateStatus.PASSED,
                validator_id=validator_id,
                errors=[],
            ),
            payload=OutputPayloadV1(kind=ResponseKind.JSON, json=normalized),
        )

    return external_validator_hook


__all__ = ["ExternalValidationHook", "create_wp5_external_validation_hook"]
