"""Direct and fake-gateway tests for the WP5-B external validation hook."""

from __future__ import annotations

import asyncio
import json

import pytest

from llm_gateway.adapters.base import RawAdapterResponse
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import (
    BudgetConfigV1,
    GenerationConfigV1,
    LLMRequestV1,
    MessageV1,
    PolicyContextV1,
    PolicyLeg,
    RequestMode,
    RequestRole,
    ResponseContractV1,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    RoutingPolicyV1,
    ValidationGateStatus,
)
from llm_gateway.engine import create_engine
from llm_gateway.validation import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    CONVERSATION_SUMMARY_SCHEMA_ID,
    ConfirmationNormalizationContext,
    create_wp5_external_validation_hook,
    register_wp5_semantic_validators,
    resolve_response_schema,
)
from llm_gateway.validation import hook as hook_module
from llm_gateway.validation import preflight as preflight_module
from llm_gateway import validators as validators_module
from llm_gateway.validators import ValidatorRegistry

pytestmark = pytest.mark.backend_unit


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


def _request(contract: ResponseContractV1 | None = None) -> LLMRequestV1:
    return LLMRequestV1(
        request_id="req-wp5-hook",
        policy_tag="confirmation.payload",
        messages=[MessageV1(role=RequestRole.USER, content="Confirm my request")],
        generation=GenerationConfigV1(max_output_tokens=100),
        response_contract=contract or _confirmation_contract(),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(
                provider_id="fake_provider",
                model_id="fake-strong",
                policy_leg=PolicyLeg.BASELINE,
            ),
            fallback_routes=[],
            allowed_provider_ids=["fake_provider"],
        ),
        budget=BudgetConfigV1(
            deadline_ms=5000,
            max_physical_attempts=1,
            max_output_tokens=100,
            max_cost_usd=1.0,
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
    )


def _summary_request() -> LLMRequestV1:
    return _request(
        ResponseContractV1(
            kind=ResponseKind.TEXT,
            schema_id=CONVERSATION_SUMMARY_SCHEMA_ID,
            json_schema=None,
            semantic_validator_id=SemanticValidatorId.NONE.value,
            validator_inputs={},
            allow_empty=False,
        )
    ).model_copy(update={"policy_tag": "conversation.summary"})


def _registry(*, complete: bool = True) -> ValidatorRegistry:
    registry = ValidatorRegistry()
    if complete:
        register_wp5_semantic_validators(registry)
    return registry


def _hook(*, complete: bool = True):
    return create_wp5_external_validation_hook(
        registry=_registry(complete=complete),
        confirmation_context=ConfirmationNormalizationContext(
            language="en",
            fallback_message="Please confirm this request.",
        ),
    )


def _valid_raw_payload() -> str:
    return json.dumps(
        {
            "message": "Please choose.",
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
    )


@pytest.mark.parametrize("fence", ["```json\n{}\n```", "```JSON\n{}\n```"])
def test_fenced_json_is_parsed_then_normalized(fence: str) -> None:
    result = _hook()(_request(), RawAdapterResponse(text=fence), {})
    assert result.passed
    assert result.summary.json_parse == ValidationGateStatus.PASSED
    assert result.summary.json_schema == ValidationGateStatus.PASSED
    assert result.summary.semantic == ValidationGateStatus.PASSED
    assert result.payload is not None
    assert result.payload.text is None
    assert result.payload.json_payload == {"message": "Please confirm this request."}


@pytest.mark.parametrize("raw", ["plain confirmation text", "{\"message\"", "", "   "])
def test_plain_text_and_malformed_json_fail_closed_without_payload(raw: str) -> None:
    result = _hook()(_request(), RawAdapterResponse(text=raw), {})
    assert not result.passed
    assert result.payload is None
    assert result.summary.json_parse == ValidationGateStatus.FAILED
    assert result.summary.json_schema == ValidationGateStatus.NOT_APPLICABLE
    assert result.summary.semantic == ValidationGateStatus.NOT_APPLICABLE
    assert raw not in result.summary.errors


@pytest.mark.parametrize(
    "raw",
    [
        "NaN",
        "Infinity",
        "-Infinity",
        '{"message":"Confirm?","ignored":NaN}',
        '{"message":"Confirm?","nested":{"value":Infinity}}',
        '```json\n{"message":"Confirm?","ignored":-Infinity}\n```',
    ],
)
def test_non_standard_json_constants_fail_at_parse_stage(raw: str) -> None:
    result = _hook()(_request(), RawAdapterResponse(text=raw), {})

    assert not result.passed
    assert result.payload is None
    assert result.summary.json_parse == ValidationGateStatus.FAILED
    assert result.summary.json_schema == ValidationGateStatus.NOT_APPLICABLE
    assert result.summary.semantic == ValidationGateStatus.NOT_APPLICABLE
    assert result.summary.errors == ["Confirmation response is not valid JSON."]


@pytest.mark.parametrize("raw", ["[]", "null", "123", '"hello"', "true"])
def test_non_object_json_fails_as_contract_shape_violation(raw: str) -> None:
    result = _hook()(_request(), RawAdapterResponse(text=raw), {})

    assert not result.passed
    assert result.payload is None
    assert result.summary.json_parse == ValidationGateStatus.PASSED
    assert result.summary.json_schema == ValidationGateStatus.FAILED
    assert result.summary.semantic == ValidationGateStatus.NOT_APPLICABLE
    assert result.summary.errors == ["Confirmation response must be a JSON object."]


def test_success_releases_only_normalized_json() -> None:
    result = _hook()(_request(), RawAdapterResponse(text=_valid_raw_payload()), {})
    assert result.passed
    assert result.payload is not None
    assert result.payload.kind == ResponseKind.JSON
    assert result.payload.text is None
    assert result.payload.tool_calls == []
    assert result.payload.json_payload["message"].endswith("Would you prefer Work or Gaming?")


def test_weak_placeholder_validator_is_rejected() -> None:
    result = _hook(complete=False)(_request(), RawAdapterResponse(text=_valid_raw_payload()), {})
    assert not result.passed
    assert result.payload is None
    assert result.summary.semantic == ValidationGateStatus.FAILED
    assert result.summary.errors == ["Required complete semantic validator is not registered."]


def test_invalid_configuration_fails_at_defensive_preflight() -> None:
    request = _request(_confirmation_contract(schema_id="unknown.response.v1"))
    result = _hook()(request, RawAdapterResponse(text=_valid_raw_payload()), {})
    assert not result.passed
    assert result.payload is None
    assert result.summary.errors == ["Response contract preflight failed: unknown_schema_id."]


def test_unknown_validator_id_fails_at_defensive_preflight() -> None:
    request = _request(
        _confirmation_contract(semantic_validator_id="unknown.validator.v1")
    )
    result = _hook()(request, RawAdapterResponse(text=_valid_raw_payload()), {})
    assert not result.passed
    assert result.payload is None
    assert result.summary.errors == ["Response contract preflight failed: unknown_validator_id."]


def test_invalid_validator_inputs_fail_at_defensive_preflight() -> None:
    request = _request(
        _confirmation_contract(
            validator_inputs={
                "allowed_preference_patch_keys": ["use_case"],
                "language": "en",
            }
        )
    )
    result = _hook()(request, RawAdapterResponse(text=_valid_raw_payload()), {})
    assert not result.passed
    assert result.summary.errors == ["Response contract preflight failed: invalid_validator_inputs."]


def test_missing_normalization_context_fails_closed() -> None:
    hook = create_wp5_external_validation_hook(registry=_registry())
    result = hook(_request(), RawAdapterResponse(text=_valid_raw_payload()), {})
    assert not result.passed
    assert result.payload is None
    assert result.summary.errors == ["Confirmation normalization context is unavailable."]


def test_missing_schema_dependency_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validators_module, "_JSONSCHEMA_AVAILABLE", False)
    result = _hook()(_request(), RawAdapterResponse(text=_valid_raw_payload()), {})
    assert not result.passed
    assert result.payload is None
    assert result.summary.json_parse == ValidationGateStatus.PASSED
    assert result.summary.json_schema == ValidationGateStatus.FAILED
    assert result.summary.errors == ["JSON Schema validator unavailable."]


def test_missing_preflight_dependency_fails_before_candidate_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight_module, "_DRAFT7_VALIDATOR_CLASS", None)
    result = _hook()(_request(), RawAdapterResponse(text=_valid_raw_payload()), {})
    assert not result.passed
    assert result.summary.json_parse == ValidationGateStatus.NOT_APPLICABLE
    assert result.summary.errors == ["Response contract preflight failed: validation_dependency_unavailable."]


def test_schema_and_semantic_failures_never_release_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hook_module,
        "normalize_confirmation_payload",
        lambda *args, **kwargs: {"message": ""},
    )
    schema_result = _hook()(_request(), RawAdapterResponse(text="{}"), {})
    assert not schema_result.passed
    assert schema_result.payload is None
    assert schema_result.summary.json_schema == ValidationGateStatus.FAILED

    monkeypatch.setattr(
        hook_module,
        "normalize_confirmation_payload",
        lambda *args, **kwargs: {
            "message": "No listed label appears here.",
            "quick_actions": [
                {"id": "work", "label": "Work", "value": "work", "preference_patch": {"use_case": "work"}},
                {"id": "game", "label": "Gaming", "value": "gaming", "preference_patch": {"use_case": "gaming"}},
            ],
        },
    )
    semantic_result = _hook()(_request(), RawAdapterResponse(text="{}"), {})
    assert not semantic_result.passed
    assert semantic_result.payload is None
    assert semantic_result.summary.semantic == ValidationGateStatus.FAILED


def test_summary_contract_accepts_trimmed_text_and_rejects_whitespace() -> None:
    hook = create_wp5_external_validation_hook(registry=_registry())
    passed = hook(_summary_request(), RawAdapterResponse(text="  useful summary  "), {})
    assert passed.passed
    assert passed.payload is not None
    assert passed.payload.text == "useful summary"
    assert passed.summary.json_parse == ValidationGateStatus.NOT_APPLICABLE

    failed = hook(_summary_request(), RawAdapterResponse(text="   "), {})
    assert not failed.passed
    assert failed.payload is None


def test_sync_and_async_fake_gateway_use_hook_without_engine_changes() -> None:
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)
    hook = _hook()

    adapter.enqueue_response(ScriptedResponse(text=_valid_raw_payload()))
    sync_result = engine.invoke_llm(_request(), validator_hook=hook)
    assert sync_result.status == ResultStatus.SUCCEEDED
    assert sync_result.output is not None
    assert sync_result.output.text is None
    assert sync_result.output.json_payload is not None

    adapter.enqueue_response(ScriptedResponse(text="plain text"))
    async_result = asyncio.run(engine.invoke_llm_async(_request(), validator_hook=hook))
    assert async_result.status == ResultStatus.FAILED
    assert async_result.output is None
    assert async_result.attempts[0].validation_summary.json_parse == ValidationGateStatus.FAILED
