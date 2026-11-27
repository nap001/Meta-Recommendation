"""Exercise WP5 response-contract preflight at the gateway boundary."""

import asyncio

import pytest

from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.contracts import (
    AttemptKind,
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
)
from llm_gateway.engine import create_engine
from llm_gateway.errors import RequestValidationError
from llm_gateway.validation import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    CONVERSATION_SUMMARY_SCHEMA_ID,
    create_wp5_external_validation_hook,
    preflight,
    resolve_response_schema,
)
from llm_gateway.validators import ValidatorRegistry
from llm_usage import UsageLedger, push_ledger, reset_ledger

pytestmark = pytest.mark.backend_unit


def _request(*, summary=False):
    schema_id = CONVERSATION_SUMMARY_SCHEMA_ID if summary else CONFIRMATION_PAYLOAD_SCHEMA_ID
    entry = resolve_response_schema(schema_id)
    assert entry is not None
    return LLMRequestV1(
        request_id="preflight-acceptance",
        policy_tag="conversation.summary" if summary else "confirmation.payload",
        messages=[MessageV1(role=RequestRole.USER, content="Acceptance fixture")],
        generation=GenerationConfigV1(max_output_tokens=50),
        response_contract=ResponseContractV1(
            kind=entry.kind,
            schema_id=schema_id,
            json_schema=entry.schema(),
            semantic_validator_id=entry.semantic_validator_id,
            validator_inputs={} if summary else {"allowed_preference_patch_keys": ["use_case"]},
            allow_empty=entry.allow_empty,
        ),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(
                provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE
            ),
            fallback_routes=[],
            allowed_provider_ids=["fake_provider"],
        ),
        budget=BudgetConfigV1(
            deadline_ms=5000, max_physical_attempts=3, max_output_tokens=100, max_cost_usd=1.0
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN
        ),
    )


def _replace_contract(request, **changes):
    return request.model_copy(
        update={"response_contract": request.response_contract.model_copy(update=changes)}
    )


@pytest.fixture
def gateway():
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)
    adapter.enqueue_response(ScriptedResponse(text="  summary from fake adapter  "))
    return engine, adapter


@pytest.fixture
def ledger():
    active_ledger = UsageLedger()
    token = push_ledger(active_ledger)
    try:
        yield active_ledger
    finally:
        reset_ledger(token)


@pytest.fixture
def reject_route_resolution(gateway, monkeypatch):
    engine, _ = gateway

    def unexpected_route_resolution(*args, **kwargs):
        pytest.fail("Rejected request reached route resolution")

    monkeypatch.setattr(engine, "_determine_initial_route", unexpected_route_resolution)


def _assert_request_error(error, code, issue_code):
    assert error is not None
    assert error.code == code
    assert error.error_class == "request"
    assert error.retryable is False
    assert error.details["preflight_issue_code"] == issue_code
    assert error.details["preflight_issues"][0]["code"] == issue_code


def _assert_rejected(result, adapter, ledger, code, issue_code):
    assert result.status == ResultStatus.FAILED
    assert result.output is None
    assert result.selected_route is None
    assert result.degradation is None
    assert result.attempts == []
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.total_tokens == 0
    assert result.usage.cost_usd == 0.0
    _assert_request_error(result.error, code, issue_code)
    assert adapter.call_history == []
    assert ledger.events == []


@pytest.mark.parametrize("entry_point", ["sync", "async"])
@pytest.mark.parametrize(
    ("changes", "issue_code"),
    [
        ({"json_schema": {"type": "object"}}, "schema_mismatch"),
        ({"json_schema": {"type": 7}}, "invalid_schema_definition"),
        ({"semantic_validator_id": "unregistered.validator.v1"}, "unknown_validator_id"),
        ({"semantic_validator_id": "none"}, "response_contract_mismatch"),
        ({}, "validation_dependency_unavailable"),
    ],
    ids=["inline-mismatch", "invalid-definition", "unknown-validator", "wrong-validator", "no-draft7"],
)
@pytest.mark.usefixtures("reject_route_resolution")
def test_contract_failure_stops_before_routing(
    gateway, ledger, monkeypatch, entry_point, changes, issue_code
):
    if issue_code == "validation_dependency_unavailable":
        monkeypatch.setattr(preflight, "_DRAFT7_VALIDATOR_CLASS", None)
    engine, adapter = gateway
    request = _replace_contract(_request(), **changes)
    result = (
        engine.invoke_llm(request)
        if entry_point == "sync"
        else asyncio.run(engine.invoke_llm_async(request))
    )
    _assert_rejected(result, adapter, ledger, "invalid_request", issue_code)


@pytest.mark.usefixtures("reject_route_resolution")
def test_executor_forwards_explicit_wp5_hook_for_unknown_summary_schema(gateway, ledger):
    engine, adapter = gateway
    request = _replace_contract(_request(summary=True), schema_id="unregistered.summary.v1")
    hook = create_wp5_external_validation_hook(registry=ValidatorRegistry())
    result = asyncio.run(engine.invoke_llm_in_executor(request, validator_hook=hook))
    _assert_rejected(result, adapter, ledger, "unknown_schema", "unknown_schema_id")


@pytest.mark.parametrize("entry_point", ["sync", "async"])
@pytest.mark.usefixtures("reject_route_resolution")
def test_stream_rejection_emits_no_chunks(gateway, ledger, entry_point):
    engine, adapter = gateway
    request = _replace_contract(_request(), json_schema={"type": "object"})
    chunks = []

    async def drain():
        async for chunk in engine.invoke_llm_stream_async(request):
            chunks.append(chunk)

    with pytest.raises(RequestValidationError) as exc:
        if entry_point == "sync":
            chunks.extend(engine.invoke_llm_stream(request))
        else:
            asyncio.run(drain())

    _assert_request_error(exc.value, "invalid_request", "schema_mismatch")
    assert chunks == []
    assert adapter.call_history == []
    assert ledger.events == []


@pytest.mark.parametrize("draft7_available", [True, False], ids=["with-draft7", "without-draft7"])
def test_registered_summary_releases_trimmed_text(gateway, ledger, monkeypatch, draft7_available):
    if not draft7_available:
        monkeypatch.setattr(preflight, "_DRAFT7_VALIDATOR_CLASS", None)
    engine, adapter = gateway
    result = asyncio.run(engine.invoke_llm_async(_request(summary=True)))

    assert result.status == ResultStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.kind == ResponseKind.TEXT
    assert result.output.text == "summary from fake adapter"
    assert result.validation.validator_id == "none"
    assert result.error is None
    assert [attempt.attempt_kind for attempt in result.attempts] == [AttemptKind.INITIAL]
    assert len(adapter.call_history) == len(ledger.events) == 1
