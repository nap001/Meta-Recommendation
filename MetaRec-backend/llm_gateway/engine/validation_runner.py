"""Output validation gates, schema checking, and semantic validation runner for LLM Gateway."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_gateway.adapters.base import RawAdapterResponse
from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import (
    LLMRequestV1,
    OutputPayloadV1,
    ResponseContractV1,
    ResponseKind,
    ToolCallOutputV1,
    ValidationGateResultV1,
    ValidationGateStatus,
    ValidationSummaryV1,
)
from llm_gateway.validators import (
    get_validator_registry,
    validate_json_schema,
    validate_json_syntax,
)

logger = logging.getLogger(__name__)


def validate_output_gate(
    request: LLMRequestV1,
    resp: RawAdapterResponse,
    sanitized_mapping: Dict[str, str],
    external_validator: Optional[Any] = None,
) -> ValidationGateResultV1:
    """Validate raw response against JSON, JSON Schema, and Semantic Validators."""
    if external_validator is not None:
        return external_validator(request, resp, sanitized_mapping)

    contract = request.response_contract
    val_summary = ValidationSummaryV1(validator_id=contract.semantic_validator_id)
    raw_text = resp.text or ""

    # Handle TEXT response
    if contract.kind == ResponseKind.TEXT:
        if not raw_text and not contract.allow_empty:
            val_summary.errors.append("Empty text response not allowed")
            return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)
        output = OutputPayloadV1(kind=ResponseKind.TEXT, text=raw_text)
        return ValidationGateResultV1(passed=True, summary=val_summary, payload=output)

    # Handle TOOL_CALLS response
    if contract.kind == ResponseKind.TOOL_CALLS:
        out_tool_calls: List[ToolCallOutputV1] = []
        for tc in resp.tool_calls:
            fn = tc.get("function", {})
            wire_name = fn.get("name", "")
            orig_name = sanitized_mapping.get(wire_name, wire_name)

            # Validate tool presence in scoped catalog and ensure not excluded
            if request.tools:
                excluded_tools = set(request.tools.excluded) if request.tools.excluded else set()
                catalog_tool_names = {t.name for t in request.tools.tools} - excluded_tools
                if orig_name not in catalog_tool_names:
                    val_summary.errors.append(f"Tool '{orig_name}' not in scoped catalog or is excluded")
                    val_summary.semantic = ValidationGateStatus.FAILED
                    return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)

            # Parse arguments
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception as e:
                val_summary.errors.append(f"Malformed tool arguments JSON: {e}")
                val_summary.json_parse = ValidationGateStatus.FAILED
                return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)

            out_tool_calls.append(
                ToolCallOutputV1(id=tc.get("id", "call_1"), name=orig_name, arguments=parsed_args)
            )

        val_summary.json_parse = ValidationGateStatus.PASSED
        val_summary.semantic = ValidationGateStatus.PASSED
        output = OutputPayloadV1(kind=ResponseKind.TOOL_CALLS, text=raw_text, tool_calls=out_tool_calls)
        return ValidationGateResultV1(passed=True, summary=val_summary, payload=output)

    # Handle JSON response
    if contract.kind == ResponseKind.JSON:
        ok, parsed_json, err = validate_json_syntax(raw_text)
        if not ok:
            val_summary.json_parse = ValidationGateStatus.FAILED
            val_summary.errors.append(err or "Malformed JSON")
            return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)
        val_summary.json_parse = ValidationGateStatus.PASSED

        # JSON Schema Validation (Draft-7 via jsonschema)
        schema_ok, schema_errs = validate_json_schema(parsed_json, contract.json_schema)
        if not schema_ok:
            val_summary.json_schema = ValidationGateStatus.FAILED
            val_summary.errors.extend(schema_errs)
            return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)
        val_summary.json_schema = ValidationGateStatus.PASSED

        # Semantic Validation (Fail-closed on unknown validator ID)
        if contract.semantic_validator_id and contract.semantic_validator_id != SemanticValidatorId.NONE.value:
            sem_validator = get_validator_registry().get_validator(contract.semantic_validator_id)
            if sem_validator is None:
                val_summary.semantic = ValidationGateStatus.FAILED
                val_summary.errors.append(
                    f"Unknown semantic validator ID: '{contract.semantic_validator_id}' (fail-closed)"
                )
                return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)
            val_inputs = dict(contract.validator_inputs or {})
            if request.tools and contract.semantic_validator_id == SemanticValidatorId.GATHER_ACTION.value:
                excluded_tools = set(request.tools.excluded) if request.tools.excluded else set()
                allowed_from_cat = {t.name for t in request.tools.tools} - excluded_tools
                if "allowed_tools" in val_inputs:
                    val_inputs["allowed_tools"] = [
                        t for t in val_inputs["allowed_tools"] if t in allowed_from_cat
                    ]
                else:
                    val_inputs["allowed_tools"] = list(allowed_from_cat)

            sem_ok, sem_errs = sem_validator(parsed_json, val_inputs)
            if not sem_ok:
                val_summary.semantic = ValidationGateStatus.FAILED
                val_summary.errors.extend(sem_errs)
                return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)
            val_summary.semantic = ValidationGateStatus.PASSED
        else:
            val_summary.semantic = ValidationGateStatus.NOT_APPLICABLE

        output = OutputPayloadV1(kind=ResponseKind.JSON, text=raw_text, json=parsed_json)
        return ValidationGateResultV1(passed=True, summary=val_summary, payload=output)

    return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)


def is_payload_contract_compatible(
    request: LLMRequestV1,
    payload: Optional[OutputPayloadV1],
) -> Tuple[bool, Optional[str]]:
    """Verify that released payload matches the request's response contract."""
    if payload is None:
        return False, "Payload is missing"
    contract = request.response_contract
    if contract.kind == ResponseKind.TEXT:
        if payload.kind != ResponseKind.TEXT:
            return False, f"Expected TEXT payload kind, got {payload.kind.value}"
        if not payload.text and not contract.allow_empty:
            return False, "Empty text response not allowed by contract"
    elif contract.kind == ResponseKind.JSON:
        if payload.kind != ResponseKind.JSON:
            return False, f"Expected JSON payload kind, got {payload.kind.value}"
        if payload.json_payload is None:
            return False, "JSON payload content is null"
    elif contract.kind == ResponseKind.TOOL_CALLS:
        if payload.kind != ResponseKind.TOOL_CALLS:
            return False, f"Expected TOOL_CALLS payload kind, got {payload.kind.value}"
    return True, None


def is_terminal_validation_error(val_summary: ValidationSummaryV1) -> bool:
    """Identify configuration or infrastructure failures that should not trigger route retries."""
    for err in val_summary.errors:
        err_lower = err.lower()
        if "unknown semantic validator" in err_lower:
            return True
        if "json schema validator unavailable" in err_lower:
            return True
        if "json schema error:" in err_lower:
            return True
        if "response contract preflight failed:" in err_lower:
            return True
    return False


def is_registered_contract(request: LLMRequestV1) -> bool:
    """Check whether request contract belongs to Member B's registered contracts."""
    try:
        from llm_gateway.validation.schemas import (
            CONFIRMATION_PAYLOAD_SCHEMA_ID,
            CONVERSATION_SUMMARY_SCHEMA_ID,
            registered_response_schema_ids,
        )
        schema_id = request.response_contract.schema_id
        return schema_id in registered_response_schema_ids() or schema_id in (
            CONFIRMATION_PAYLOAD_SCHEMA_ID,
            CONVERSATION_SUMMARY_SCHEMA_ID,
        )
    except Exception:
        return False


def execute_registered_validation_hook(
    request: LLMRequestV1,
    resp: RawAdapterResponse,
    sanitized_mapping: Dict[str, str],
) -> ValidationGateResultV1:
    """Dispatch to Member B's create_wp5_external_validation_hook for registered contracts."""
    from llm_gateway.validation import (
        CONFIRMATION_PAYLOAD_SCHEMA_ID,
        ConfirmationNormalizationContext,
        create_wp5_external_validation_hook,
        register_wp5_semantic_validators,
    )
    from llm_gateway.validators import ValidatorRegistry

    registry = ValidatorRegistry()
    register_wp5_semantic_validators(registry)

    confirmation_context = None
    if request.response_contract.schema_id == CONFIRMATION_PAYLOAD_SCHEMA_ID:
        lang = str(request.response_contract.validator_inputs.get("language", "en") or "en")
        if lang not in {"en", "zh"}:
            lang = "en"
        fallback = str(request.response_contract.validator_inputs.get("fallback_message", "") or "").strip()
        if not fallback:
            fallback = "Please confirm this request."
        confirmation_context = ConfirmationNormalizationContext(
            language=lang,
            fallback_message=fallback,
        )

    hook = create_wp5_external_validation_hook(
        registry=registry,
        confirmation_context=confirmation_context,
    )
    return hook(request, resp, sanitized_mapping)


def execute_validation_sync(
    request: LLMRequestV1,
    resp: RawAdapterResponse,
    sanitized_mapping: Dict[str, str],
    validator_hook: Optional[Callable[..., Any]] = None,
    engine_validator_hook: Optional[Callable[..., Any]] = None,
) -> ValidationGateResultV1:
    """Execute validation synchronously using external hook if provided, else built-in gates."""
    hook = validator_hook or engine_validator_hook
    if hook is not None:
        try:
            res = hook(request, resp, sanitized_mapping)
            if asyncio.iscoroutine(res):
                raise RuntimeError(
                    "Async validator hook cannot be invoked synchronously in invoke_llm; use invoke_llm_async instead"
                )
            if isinstance(res, ValidationGateResultV1):
                return res
            if isinstance(res, dict):
                return ValidationGateResultV1(**res)
            raise TypeError(f"External validator hook must return ValidationGateResultV1, got {type(res)}")
        except Exception as exc:
            logger.warning("External validator hook error: %s", exc)
            val_summary = ValidationSummaryV1(
                validator_id=request.response_contract.semantic_validator_id,
                errors=[f"External validator error: {exc}"],
            )
            return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)
    elif is_registered_contract(request):
        try:
            return execute_registered_validation_hook(request, resp, sanitized_mapping)
        except Exception as exc:
            logger.warning("execute_validation_hook error: %s", exc)
            val_summary = ValidationSummaryV1(
                validator_id=request.response_contract.semantic_validator_id,
                errors=[f"Validation hook error: {exc}"],
            )
            return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)

    return validate_output_gate(request, resp, sanitized_mapping)


async def execute_validation_async(
    request: LLMRequestV1,
    resp: RawAdapterResponse,
    sanitized_mapping: Dict[str, str],
    validator_hook: Optional[Callable[..., Any]] = None,
    engine_validator_hook: Optional[Callable[..., Any]] = None,
) -> ValidationGateResultV1:
    """Execute validation asynchronously using external hook if provided, else built-in gates."""
    hook = validator_hook or engine_validator_hook
    if hook is not None:
        try:
            if inspect.iscoroutinefunction(hook):
                res = await hook(request, resp, sanitized_mapping)
            else:
                res = hook(request, resp, sanitized_mapping)
                if asyncio.iscoroutine(res):
                    res = await res
            if isinstance(res, ValidationGateResultV1):
                return res
            if isinstance(res, dict):
                return ValidationGateResultV1(**res)
            raise TypeError(f"External validator hook must return ValidationGateResultV1, got {type(res)}")
        except Exception as exc:
            logger.warning("External validator hook error: %s", exc)
            val_summary = ValidationSummaryV1(
                validator_id=request.response_contract.semantic_validator_id,
                errors=[f"External validator error: {exc}"],
            )
            return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)
    elif is_registered_contract(request):
        try:
            return execute_registered_validation_hook(request, resp, sanitized_mapping)
        except Exception as exc:
            logger.warning("execute_validation_hook error: %s", exc)
            val_summary = ValidationSummaryV1(
                validator_id=request.response_contract.semantic_validator_id,
                errors=[f"Validation hook error: {exc}"],
            )
            return ValidationGateResultV1(passed=False, summary=val_summary, payload=None)

    return validate_output_gate(request, resp, sanitized_mapping)
