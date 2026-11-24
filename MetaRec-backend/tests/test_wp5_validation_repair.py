"""Tests for Gateway Engine dispatch to Member B's validation hook,
bounded validation repair retry, and semantic fallback.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import (
    AttemptKind,
    AttemptOutcome,
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
    resolve_response_schema,
)

pytestmark = pytest.mark.backend_unit


def _valid_confirmation_json() -> str:
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


def _confirmation_contract() -> ResponseContractV1:
    entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert entry is not None
    return ResponseContractV1(
        kind=ResponseKind.JSON,
        schema_id=CONFIRMATION_PAYLOAD_SCHEMA_ID,
        json_schema=entry.schema(),
        semantic_validator_id=SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
        validator_inputs={"allowed_preference_patch_keys": ["use_case"]},
        allow_empty=False,
    )


def _confirmation_request(
    *,
    max_physical_attempts: int = 2,
    routing_policy: RoutingPolicyV1 | None = None,
) -> LLMRequestV1:
    default_routing = RoutingPolicyV1(
        mode=RoutingMode.BASELINE,
        baseline_route=RouteV1(
            provider_id="fake_provider",
            model_id="fake-strong",
            policy_leg=PolicyLeg.BASELINE,
        ),
        fallback_routes=[],
        allowed_provider_ids=["fake_provider"],
    )
    return LLMRequestV1(
        request_id="req-val-repair-test",
        policy_tag="confirmation.payload",
        messages=[MessageV1(role=RequestRole.USER, content="Confirm request")],
        generation=GenerationConfigV1(max_output_tokens=100),
        response_contract=_confirmation_contract(),
        routing_policy=routing_policy or default_routing,
        budget=BudgetConfigV1(
            deadline_ms=5000,
            max_physical_attempts=max_physical_attempts,
            max_output_tokens=100,
            max_cost_usd=1.0,
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
    )


def test_engine_auto_dispatches_registered_contract_without_explicit_hook():
    """Engine automatically dispatches to Member B's execute_validation_hook for registered contracts."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    adapter.enqueue_response(ScriptedResponse(text=_valid_confirmation_json()))

    # Note: validator_hook is NOT passed
    result = engine.invoke_llm(_confirmation_request())

    assert result.status == ResultStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.kind == ResponseKind.JSON
    assert result.output.json_payload is not None
    assert "Would you prefer Work or Gaming?" in result.output.json_payload["message"]
    assert len(result.attempts) == 1
    assert result.attempts[0].validation_summary.semantic == ValidationGateStatus.PASSED


def test_bounded_validation_repair_attempt_on_baseline_route():
    """When baseline route produces invalid output, engine triggers one bounded repair attempt."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    # Attempt 1: Malformed JSON
    adapter.enqueue_response(ScriptedResponse(text="not valid json"))
    # Attempt 2 (Repair): Valid normalized confirmation JSON
    adapter.enqueue_response(ScriptedResponse(text=_valid_confirmation_json()))

    result = engine.invoke_llm(_confirmation_request(max_physical_attempts=2))

    assert result.status == ResultStatus.SUCCEEDED
    assert result.output is not None
    assert len(result.attempts) == 2

    # Attempt 1 failed validation
    assert result.attempts[0].attempt_kind == AttemptKind.INITIAL
    assert result.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR

    # Attempt 2 is a VALIDATION_RETRY
    assert result.attempts[1].attempt_kind == AttemptKind.VALIDATION_RETRY
    assert result.attempts[1].outcome == AttemptOutcome.SUCCESS

    # Verify adapter received the repair guidance in messages kwargs on attempt 2
    assert len(adapter.call_history) == 2
    second_call_messages = adapter.call_history[1]["kwargs"]["messages"]
    assert any("failed contract validation" in m.get("content", "") for m in second_call_messages)


def test_async_bounded_validation_repair_attempt():
    """Async invocation also triggers bounded validation repair retry."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    # Attempt 1: Bad output
    adapter.enqueue_response(ScriptedResponse(text="{bad}"))
    # Attempt 2: Good output
    adapter.enqueue_response(ScriptedResponse(text=_valid_confirmation_json()))

    result = asyncio.run(engine.invoke_llm_async(_confirmation_request(max_physical_attempts=2)))

    assert result.status == ResultStatus.SUCCEEDED
    assert len(result.attempts) == 2
    assert result.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR
    assert result.attempts[1].attempt_kind == AttemptKind.VALIDATION_RETRY
    assert result.attempts[1].outcome == AttemptOutcome.SUCCESS


def test_cheap_route_semantic_violation_triggers_fallback_to_baseline(monkeypatch: pytest.MonkeyPatch):
    """Cheap route failing semantic validation immediately falls back to baseline route without burning repair."""
    from llm_gateway.validation import hook as hook_mod
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    # Mock normalize on attempt 1 so candidate has mismatched message and quick-action labels,
    # causing validate_confirmation_payload_semantics to fail with semantic=FAILED.
    original_normalize = hook_mod.normalize_confirmation_payload
    call_count = [0]

    def mock_normalize(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return {
                "message": "No listed label appears here.",
                "quick_actions": [
                    {"id": "work", "label": "Work", "value": "work", "preference_patch": {"use_case": "work"}},
                    {"id": "game", "label": "Gaming", "value": "gaming", "preference_patch": {"use_case": "gaming"}},
                ],
            }
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(hook_mod, "normalize_confirmation_payload", mock_normalize)

    # Response 1 (Cheap route): will fail semantic validation
    adapter.enqueue_response(ScriptedResponse(text=_valid_confirmation_json()))
    # Response 2 (Baseline route fallback): will pass
    adapter.enqueue_response(ScriptedResponse(text=_valid_confirmation_json()))

    routing = RoutingPolicyV1(
        mode=RoutingMode.CHEAP_FIRST,
        baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE),
        cheap_route=RouteV1(provider_id="fake_provider", model_id="fake-fast", policy_leg=PolicyLeg.CHEAP),
        fallback_routes=[],
        allowed_provider_ids=["fake_provider"],
    )

    req = _confirmation_request(max_physical_attempts=3, routing_policy=routing)
    result = engine.invoke_llm(req)

    assert result.status == ResultStatus.DEGRADED
    assert result.degradation is not None
    assert result.degradation.reason_code == "fallback_route_used"
    assert result.selected_route is not None
    assert result.selected_route.policy_leg == PolicyLeg.BASELINE

    assert len(result.attempts) == 2
    # Leg 1: Cheap route failed semantic validation
    assert result.attempts[0].policy_leg == PolicyLeg.CHEAP
    assert result.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR
    assert result.attempts[0].validation_summary.semantic == ValidationGateStatus.FAILED

    # Leg 2: Fallback to baseline route succeeded
    assert result.attempts[1].policy_leg == PolicyLeg.BASELINE
    assert result.attempts[1].attempt_kind == AttemptKind.ROUTE_FALLBACK
    assert result.attempts[1].outcome == AttemptOutcome.SUCCESS


def test_budget_exhaustion_bounds_repair_attempts():
    """If budget max_physical_attempts=1, no repair attempt is made and result fails closed."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    adapter.enqueue_response(ScriptedResponse(text="invalid json"))

    result = engine.invoke_llm(_confirmation_request(max_physical_attempts=1))

    assert result.status == ResultStatus.FAILED
    assert result.output is None
    assert len(result.attempts) == 1
    assert result.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR


def test_preflight_blocks_invalid_registered_contract_before_provider_sync_and_async():
    """Preflight must reject invalid registered contracts at 0 attempts, $0 cost, ErrorClass.REQUEST."""
    from llm_usage import UsageLedger, push_ledger, reset_ledger

    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    bad_contract = _confirmation_contract().model_copy(
        update={
            "validator_inputs": {
                "allowed_preference_patch_keys": ["use_case"],
                "unexpected_key": True,
            }
        }
    )
    req = _confirmation_request(max_physical_attempts=3).model_copy(
        update={"response_contract": bad_contract}
    )

    # Enqueue response so we prove the adapter is never touched
    adapter.enqueue_response(ScriptedResponse(text=_valid_confirmation_json()))
    ledger = UsageLedger()
    token = push_ledger(ledger)
    try:
        sync_res = engine.invoke_llm(req)
        async_res = asyncio.run(engine.invoke_llm_async(req))
    finally:
        reset_ledger(token)

    for res in (sync_res, async_res):
        assert res.status == ResultStatus.FAILED
        assert res.output is None
        assert res.selected_route is None
        assert res.attempts == []
        assert res.usage.total_tokens == 0
        assert res.usage.cost_usd == 0.0
        assert res.error is not None
        assert res.error.error_class == "request"
        assert res.error.code == "invalid_request"
        assert res.error.retryable is False
        assert res.error.details["preflight_issue_code"] == "invalid_validator_inputs"

    assert adapter.call_history == []
    assert ledger.events == []


def test_preflight_blocks_unknown_schema_id_with_unknown_schema_code_sync_and_async():
    """Unknown schema on registered validator / WP5-B hook emits terminal RequestValidationError(unknown_schema)."""
    from llm_gateway.validation import create_wp5_external_validation_hook, register_wp5_semantic_validators
    from llm_gateway.validators import ValidatorRegistry

    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)
    adapter.enqueue_response(ScriptedResponse(text="should never run"))

    # Case 1: Auto-dispatch path with registered validator ID & unknown schema_id
    bad_conf_contract = _confirmation_contract().model_copy(
        update={"schema_id": "unknown.confirmation.v99"}
    )
    req_conf = _confirmation_request().model_copy(update={"response_contract": bad_conf_contract})
    res_conf = engine.invoke_llm(req_conf)

    assert res_conf.status == ResultStatus.FAILED
    assert res_conf.attempts == []
    assert res_conf.usage.cost_usd == 0.0
    assert res_conf.error is not None
    assert res_conf.error.error_class == "request"
    assert res_conf.error.code == "unknown_schema"
    assert res_conf.error.retryable is False
    assert res_conf.error.details["preflight_issue_code"] == "unknown_schema_id"

    # Case 2: Explicit WP5-B hook with semantic_validator_id="none" & unknown schema_id (async)
    reg = ValidatorRegistry()
    register_wp5_semantic_validators(reg)
    hook = create_wp5_external_validation_hook(registry=reg)
    bad_sum_contract = ResponseContractV1(
        kind=ResponseKind.TEXT,
        schema_id="unknown.summary.v99",
        json_schema=None,
        semantic_validator_id=SemanticValidatorId.NONE.value,
        validator_inputs={},
        allow_empty=False,
    )
    req_sum = _confirmation_request().model_copy(
        update={"policy_tag": "conversation.summary", "response_contract": bad_sum_contract}
    )
    res_sum = asyncio.run(engine.invoke_llm_async(req_sum, validator_hook=hook))

    assert res_sum.status == ResultStatus.FAILED
    assert res_sum.attempts == []
    assert res_sum.usage.cost_usd == 0.0
    assert res_sum.error is not None
    assert res_sum.error.error_class == "request"
    assert res_sum.error.code == "unknown_schema"
    assert res_sum.error.retryable is False
    assert adapter.call_history == []


def test_preflight_blocks_streaming_entry_points_before_provider_call():
    """Streaming entry points fail fast with RequestValidationError before calling adapter."""
    from llm_gateway.errors import RequestValidationError

    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    bad_contract = ResponseContractV1(
        kind=ResponseKind.JSON,
        schema_id=CONVERSATION_SUMMARY_SCHEMA_ID,
        json_schema={"type": "object"},
        semantic_validator_id=SemanticValidatorId.NONE.value,
        allow_empty=False,
    )
    req = _confirmation_request().model_copy(
        update={"policy_tag": "conversation.summary", "response_contract": bad_contract}
    )

    with pytest.raises(RequestValidationError) as exc_sync:
        list(engine.invoke_llm_stream(req))
    assert exc_sync.value.code == "invalid_request"
    assert exc_sync.value.error_class.value == "request"

    async def _drain():
        async for _ in engine.invoke_llm_stream_async(req):
            pass

    with pytest.raises(RequestValidationError) as exc_async:
        asyncio.run(_drain())
    assert exc_async.value.code == "invalid_request"
    assert adapter.call_history == []
