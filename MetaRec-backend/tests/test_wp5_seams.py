"""Unit tests for WP5 Milestone S1 W11–13 Backward-Compatibility Seams.

Tests verify:
1. client.py: Managed clients have max_retries=0 by default to prevent hidden SDK retry amplification.
2. llm_usage.py: UsageEvent tracks attempt_id & metadata; UsageLedger deduplicates by attempt_id.
3. llm_usage.py: mark_response_recorded makes subsequent record_response_usage calls safe no-ops.
4. llm_compat.py: Calls route through GatewayEngine, record physical attempts to the ledger, and project SDK-shaped responses.
5. llm_compat.py: Compatibility retries record every physical attempt in the ledger and prevent double-charging.
"""

from __future__ import annotations

import types
import pytest
from types import SimpleNamespace

import client
import llm_compat
import llm_usage
from llm_usage import (
    UsageEvent,
    UsageLedger,
    current_ledger,
    mark_response_recorded,
    push_ledger,
    record_response_usage,
    recorded_attempt_id,
    reset_ledger,
)

pytestmark = pytest.mark.backend_unit


@pytest.fixture(autouse=True)
def _isolate_ledger():
    token = push_ledger(None)
    try:
        yield
    finally:
        reset_ledger(token)


# ---------------------------------------------------------------------------
# 1. client.py tests
# ---------------------------------------------------------------------------

def test_managed_sync_client_has_zero_sdk_retries(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    c = client.create_sync_client()
    assert c.max_retries == 0


def test_managed_async_client_has_zero_sdk_retries(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    c = client.create_async_client()
    assert c.max_retries == 0


def test_managed_azure_client_has_zero_sdk_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    c = client.create_sync_azure_client()
    assert c.max_retries == 0


def test_managed_client_allows_explicit_retry_override(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    c = client.create_sync_client(max_retries=3)
    assert c.max_retries == 3


# ---------------------------------------------------------------------------
# 2. llm_usage.py attempt_id tracking & idempotency tests
# ---------------------------------------------------------------------------

def test_usage_event_metadata_defaults():
    ev = UsageEvent(
        model="gpt-6",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost_usd=0.001,
        attempt_id="att-123",
        request_id="req-456",
        policy_tag="legacy.unenrolled",
        provider_id="openai",
        policy_leg="baseline",
        attempt_kind="initial",
        pricing_snapshot_id="default-pricing-v1",
        price_table_version="pricing-2026-v1",
    )
    assert ev.attempt_id == "att-123"
    assert ev.policy_tag == "legacy.unenrolled"
    assert ev.is_usage_reported is True


def test_ledger_add_idempotency_by_attempt_id():
    ledger = UsageLedger()
    token = push_ledger(ledger)
    try:
        ev1 = UsageEvent("m", 10, 5, 15, 0.01, attempt_id="att-same")
        ev2 = UsageEvent("m", 10, 5, 15, 0.01, attempt_id="att-same")
        ev3 = UsageEvent("m", 20, 10, 30, 0.02, attempt_id="att-other")

        added1 = ledger.add(ev1)
        added2 = ledger.add(ev2)
        added3 = ledger.add(ev3)

        assert added1 is True
        assert added2 is False  # duplicate attempt_id rejected
        assert added3 is True

        assert len(ledger.events) == 2
        assert ledger.totals["total_tokens"] == 45
        assert ledger.totals["cost_usd"] == 0.03
        assert ledger.has_attempt("att-same")
        assert ledger.has_attempt("att-other")
    finally:
        reset_ledger(token)


def test_record_response_usage_is_noop_when_already_gateway_recorded():
    class DummyResponse:
        def __init__(self):
            self.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            self.model = "gpt-6"

    ledger = UsageLedger()
    token = push_ledger(ledger)
    try:
        resp = DummyResponse()
        mark_response_recorded(resp, "att-already-recorded")
        assert recorded_attempt_id(resp) == "att-already-recorded"

        # Calling record_response_usage on an already recorded response must return None and add nothing
        result = record_response_usage(resp)
        assert result is None
        assert ledger.events == []
    finally:
        reset_ledger(token)


def test_record_response_usage_dedupes_unattributable_dict_response():
    resp_dict = {
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        "model": "gpt-6",
    }
    ledger = UsageLedger()
    token = push_ledger(ledger)
    try:
        mark_response_recorded(resp_dict, "att-dict-1")
        assert recorded_attempt_id(resp_dict) == "att-dict-1"

        res = record_response_usage(resp_dict)
        assert res is None
        assert ledger.events == []
    finally:
        reset_ledger(token)


# ---------------------------------------------------------------------------
# 3. llm_compat.py gateway routing & attempt accounting tests
# ---------------------------------------------------------------------------

def test_sync_chat_completion_records_usage_and_prevents_double_charging():
    fake_response = SimpleNamespace(
        id="chatcmpl-test",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="gateway reply"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        model="test-model",
    )

    def create(**kwargs):
        return fake_response

    client_mock = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    ledger = UsageLedger()
    token = push_ledger(ledger)
    try:
        # 1. Execute chat completion via shim
        completion = llm_compat.create_chat_completion(
            client_mock,
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert completion.choices[0].message.content == "gateway reply"

        # 2. Check that the gateway recorded exactly 1 physical attempt to the ledger
        assert len(ledger.events) == 1
        att_event = ledger.events[0]
        assert att_event.model == "test-model"
        assert att_event.total_tokens == 150
        assert att_event.attempt_id is not None
        assert att_event.policy_tag == "legacy.unenrolled"

        # 3. Simulate legacy call-site invoking record_response_usage(completion)
        legacy_call_result = record_response_usage(completion)
        assert legacy_call_result is None

        # 4. Confirm no double-charging occurred
        assert len(ledger.events) == 1
        assert ledger.totals["total_tokens"] == 150
    finally:
        reset_ledger(token)


@pytest.mark.asyncio
async def test_async_chat_completion_compatibility_retry_records_all_attempts():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "Unsupported value: 'temperature' does not support 0.7 with this model. "
                "Only the default (1) value is supported."
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="retry ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=20, total_tokens=100),
            model="test-compat-model",
        )

    client_mock = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    ledger = UsageLedger()
    token = push_ledger(ledger)
    try:
        result = await llm_compat.create_chat_completion_async(
            client_mock,
            model="test-compat-model",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
        )

        assert result.choices[0].message.content == "retry ok"
        assert len(calls) == 2
        assert calls[0]["temperature"] == 0.7
        assert "temperature" not in calls[1]

        # Both physical attempts were accounted for in the ledger
        # Attempt 1 failed (0 usage reported), Attempt 2 succeeded (100 tokens)
        assert len(ledger.events) == 2
        assert ledger.events[0].attempt_kind == "initial"
        assert ledger.events[1].attempt_kind == "compatibility_retry"
        assert ledger.totals["total_tokens"] == 100

        # Post-call call-site recording is idempotent no-op
        assert record_response_usage(result) is None
        assert len(ledger.events) == 2
    finally:
        reset_ledger(token)


# ---------------------------------------------------------------------------
# 4. Step 1: UsageLedger grouping & Per-tag Enrollment Controls
# ---------------------------------------------------------------------------

def test_usage_ledger_totals_by_tag_provider_and_leg():
    ledger = UsageLedger()
    ledger.add(
        UsageEvent(
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            attempt_id="att-1",
            policy_tag="conversation.summary",
            provider_id="openai",
            policy_leg="cheap",
        )
    )
    ledger.add(
        UsageEvent(
            model="gpt-4o",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            cost_usd=0.005,
            attempt_id="att-2",
            policy_tag="conversation.summary",
            provider_id="openai",
            policy_leg="baseline",
        )
    )
    ledger.add(
        UsageEvent(
            model="claude-sonnet",
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            cost_usd=0.002,
            attempt_id="att-3",
            policy_tag="itinerary.extraction",
            provider_id="anthropic",
            policy_leg="fallback",
        )
    )

    by_tag = ledger.totals_by_tag()
    assert by_tag["conversation.summary"]["total_tokens"] == 450
    assert by_tag["conversation.summary"]["attempt_count"] == 2
    assert by_tag["conversation.summary"]["cost_usd"] == 0.006
    assert by_tag["itinerary.extraction"]["total_tokens"] == 75
    assert by_tag["itinerary.extraction"]["attempt_count"] == 1

    by_provider = ledger.totals_by_provider()
    assert by_provider["openai"]["total_tokens"] == 450
    assert by_provider["openai"]["attempt_count"] == 2
    assert by_provider["anthropic"]["total_tokens"] == 75

    by_leg = ledger.totals_by_leg()
    assert by_leg["cheap"]["total_tokens"] == 150
    assert by_leg["baseline"]["total_tokens"] == 300
    assert by_leg["fallback"]["total_tokens"] == 75


def test_per_tag_enrollment_controls_and_routing_enforcement():
    from llm_gateway.contracts import (
        LLMRequestV1,
        MessageV1,
        PolicyContextV1,
        PolicyLeg,
        RequestRole,
        RouteV1,
        RoutingMode,
        RoutingPolicyV1,
    )
    from llm_gateway.engine.routing import determine_initial_route
    from llm_gateway.kill_switch import (
        disable_tag_enrollment,
        enable_tag_enrollment,
        is_tag_enrolled,
        reset_enrolled_tags,
        set_kill_switch,
    )

    set_kill_switch(False)
    reset_enrolled_tags()
    try:
        assert is_tag_enrolled("conversation.summary") is True
        assert is_tag_enrolled("confirmation.payload") is True
        assert is_tag_enrolled("itinerary.extraction") is False

        now = 1_700_000_000.0
        req = LLMRequestV1(
            request_id="req-enroll-1",
            policy_tag="conversation.summary",
            messages=[MessageV1(role=RequestRole.USER, content="summarize")],
            routing_policy=RoutingPolicyV1(
                mode=RoutingMode.CHEAP_FIRST,
                baseline_route=RouteV1(provider_id="fake", model_id="fake-baseline"),
                cheap_route=RouteV1(provider_id="fake", model_id="fake-cheap"),
            ),
            policy_context=PolicyContextV1(
                context_version="1.0.0",
                safety_class="standard",
                contains_itinerary_state=False,
                request_mode="single_domain",
                produced_at_utc=now - 5.0,
                producer_id="summary_safety_context.v1",
            ),
        )

        route, leg = determine_initial_route(req, now_utc=now)
        assert leg == PolicyLeg.CHEAP
        assert route.model_id == "fake-cheap"

        # Disenroll conversation.summary -> must fall back to BASELINE
        disable_tag_enrollment("conversation.summary")
        assert is_tag_enrolled("conversation.summary") is False
        route2, leg2 = determine_initial_route(req, now_utc=now)
        assert leg2 == PolicyLeg.BASELINE
        assert route2.model_id == "fake-baseline"

        # Re-enroll conversation.summary -> returns to CHEAP
        enable_tag_enrollment("conversation.summary")
        route3, leg3 = determine_initial_route(req, now_utc=now)
        assert leg3 == PolicyLeg.CHEAP
    finally:
        reset_enrolled_tags()
        set_kill_switch(False)


# ---------------------------------------------------------------------------
# 5. Step 2: Upstream Safety Context Producers & Call-site Policy Tagging
# ---------------------------------------------------------------------------

def test_build_summary_safety_context_single_domain_vs_itinerary_and_unknown():
    from llm_gateway import build_summary_safety_context

    # 1. Eligible single-domain messages
    single_domain_msgs = [
        {"role": "user", "content": "Find me a sushi place", "metadata": {"domain": "dining"}},
        {"role": "assistant", "content": "Here are 3 options", "metadata": {"domain": "dining"}},
    ]
    ctx_ok = build_summary_safety_context(single_domain_msgs)
    assert ctx_ok["producer_id"] == "summary_safety_context.v1"
    assert ctx_ok["context_version"] == "1.0.0"
    assert ctx_ok["safety_class"] == "standard"
    assert ctx_ok["contains_itinerary_state"] is False
    assert ctx_ok["request_mode"] == "single_domain"

    # 2. Itinerary-active message -> itinerary / contains_itinerary_state=True
    itinerary_msgs = [
        {"role": "user", "content": "Plan my day", "metadata": {"domain": "itinerary"}},
    ]
    ctx_itin = build_summary_safety_context(itinerary_msgs)
    assert ctx_itin["safety_class"] == "itinerary"
    assert ctx_itin["contains_itinerary_state"] is True
    assert ctx_itin["request_mode"] == "itinerary"

    # 3. Multi-domain messages -> request_mode="multi_domain"
    multi_msgs = [
        {"role": "user", "content": "Find dinner and hotel", "metadata": {"domain": "multi_domain"}},
    ]
    ctx_multi = build_summary_safety_context(multi_msgs)
    assert ctx_multi["safety_class"] == "standard"
    assert ctx_multi["contains_itinerary_state"] is False
    assert ctx_multi["request_mode"] == "multi_domain"

    # 4. Unclassifiable message (missing role) -> fails closed ("unknown")
    bad_msgs = [{"content": "missing role"}]
    ctx_bad = build_summary_safety_context(bad_msgs)
    assert ctx_bad["safety_class"] == "unknown"
    assert ctx_bad["contains_itinerary_state"] is True


def test_build_confirmation_safety_context_modes():
    from llm_gateway import build_confirmation_safety_context

    # 1. Single domain -> eligible
    c1 = build_confirmation_safety_context({"mode": "single_domain", "domain": "dining"})
    assert c1["producer_id"] == "confirmation_context.v1"
    assert c1["safety_class"] == "standard"
    assert c1["contains_itinerary_state"] is False
    assert c1["request_mode"] == "single_domain"

    # 2. Multi domain -> request_mode="multi_domain"
    c2 = build_confirmation_safety_context({"mode": "multi_domain", "domain": "dining"})
    assert c2["safety_class"] == "standard"
    assert c2["contains_itinerary_state"] is False
    assert c2["request_mode"] == "multi_domain"

    # 3. Itinerary -> contains_itinerary_state=True
    c3 = build_confirmation_safety_context({"mode": "itinerary", "domain": "itinerary"})
    assert c3["safety_class"] == "itinerary"
    assert c3["contains_itinerary_state"] is True
    assert c3["request_mode"] == "itinerary"

    # 4. Missing/unknown -> fails closed ("unknown")
    c4 = build_confirmation_safety_context(None)
    assert c4["safety_class"] == "unknown"
    assert c4["contains_itinerary_state"] is True


def test_routing_rejects_stale_or_unregistered_policy_context_and_never_cheap_tags():
    from llm_gateway.contracts import (
        LLMRequestV1,
        MessageV1,
        PolicyContextV1,
        PolicyLeg,
        RequestRole,
        RouteV1,
        RoutingMode,
        RoutingPolicyV1,
    )
    from llm_gateway.engine.routing import determine_initial_route
    from llm_gateway.kill_switch import reset_enrolled_tags, set_kill_switch

    set_kill_switch(False)
    reset_enrolled_tags()
    now = 1_700_000_000.0

    # Stale context (600s old > 300s TTL) -> BASELINE
    stale_req = LLMRequestV1(
        request_id="req-stale-1",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="hi")],
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.CHEAP_FIRST,
            baseline_route=RouteV1(provider_id="fake", model_id="fake-base"),
            cheap_route=RouteV1(provider_id="fake", model_id="fake-cheap"),
        ),
        policy_context=PolicyContextV1(
            context_version="1.0.0",
            safety_class="standard",
            contains_itinerary_state=False,
            request_mode="single_domain",
            produced_at_utc=now - 600.0,
            producer_id="summary_safety_context.v1",
        ),
    )
    _, leg_stale = determine_initial_route(stale_req, now_utc=now)
    assert leg_stale == PolicyLeg.BASELINE

    # Unregistered producer -> BASELINE
    bad_prod_req = stale_req.model_copy(
        update={
            "policy_context": PolicyContextV1(
                context_version="1.0.0",
                safety_class="standard",
                contains_itinerary_state=False,
                request_mode="single_domain",
                produced_at_utc=now - 1.0,
                producer_id="unregistered.producer.v1",
            )
        }
    )
    _, leg_bad_prod = determine_initial_route(bad_prod_req, now_utc=now)
    assert leg_bad_prod == PolicyLeg.BASELINE

    # Never-cheap tag (e.g. itinerary.extraction) -> BASELINE even if fresh context given
    never_cheap_req = bad_prod_req.model_copy(
        update={
            "policy_tag": "itinerary.extraction",
            "policy_context": PolicyContextV1(
                context_version="1.0.0",
                safety_class="standard",
                contains_itinerary_state=False,
                request_mode="single_domain",
                produced_at_utc=now - 1.0,
                producer_id="summary_safety_context.v1",
            ),
        }
    )
    _, leg_never = determine_initial_route(never_cheap_req, now_utc=now)
    assert leg_never == PolicyLeg.BASELINE


def test_legacy_shim_strips_metarec_policy_fields_and_records_policy_tag():
    seen_kwargs = []
    fake_response = SimpleNamespace(
        id="chatcmpl-policy",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="ok"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test-model",
    )

    def create(**kwargs):
        seen_kwargs.append(dict(kwargs))
        return fake_response

    client_mock = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ledger = UsageLedger()
    token = push_ledger(ledger)
    try:
        llm_compat.create_chat_completion(
            client_mock,
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            policy_tag="gather.propose_action",
            policy_context={
                "context_version": "1.0.0",
                "safety_class": "standard",
                "contains_itinerary_state": False,
                "request_mode": "single_domain",
                "produced_at_utc": 1700000000.0,
                "producer_id": "summary_safety_context.v1",
            },
        )
        assert len(seen_kwargs) == 1
        assert "policy_tag" not in seen_kwargs[0]
        assert "policy_context" not in seen_kwargs[0]
        assert len(ledger.events) == 1
        assert ledger.events[0].policy_tag == "gather.propose_action"
    finally:
        reset_ledger(token)


# ---------------------------------------------------------------------------
# 6. Step 3: Injectable FakeClock for Deterministic Backoff & Deadlines
# ---------------------------------------------------------------------------

def test_fake_clock_deterministic_backoff_and_deadline_exhaustion_sync():
    from llm_gateway import (
        BudgetConfigV1,
        FakeClock,
        GatewayErrorCode,
        LLMRequestV1,
        MessageV1,
        PolicyContextV1,
        RequestRole,
        ResultStatus,
        RouteV1,
        RoutingPolicyV1,
        TransportError,
        create_engine,
    )
    from llm_gateway.adapters.base import BaseLLMAdapter

    clock = FakeClock(initial_utc=1_700_000_000.0, initial_monotonic=500.0)

    class FailingTransportAdapter(BaseLLMAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, request, route, kwargs):
            self.calls += 1
            clock.advance(0.05)  # 50ms simulated network call
            raise TransportError("rate_limit", "429 Too Many Requests", http_status=429)

        async def execute_async(self, request, route, kwargs):
            self.calls += 1
            clock.advance(0.05)
            raise TransportError("rate_limit", "429 Too Many Requests", http_status=429)

    # Fixed 400ms backoff per retry; deadline is 600ms
    engine = create_engine(clock=clock, jitter_fn=lambda attempt_num: 0.4)
    adapter = FailingTransportAdapter()
    engine.register_adapter("fake_transport", adapter)

    req = LLMRequestV1(
        request_id="req-fakeclock-sync-1",
        policy_tag="gather.propose_action",
        messages=[MessageV1(role=RequestRole.USER, content="test")],
        routing_policy=RoutingPolicyV1(
            baseline_route=RouteV1(provider_id="fake_transport", model_id="fake-model")
        ),
        policy_context=PolicyContextV1(
            context_version="1.0.0",
            safety_class="standard",
            contains_itinerary_state=False,
            request_mode="single_domain",
            produced_at_utc=clock.time(),
            producer_id="summary_safety_context.v1",
        ),
        budget=BudgetConfigV1(
            deadline_ms=600,
            max_physical_attempts=4,
            max_cost_usd=1.0,
        ),
    )

    result = engine.invoke_llm(req)
    assert result.status == ResultStatus.FAILED
    assert result.error is not None
    assert result.error.code == GatewayErrorCode.DEADLINE_EXHAUSTED.value
    # Attempt 1 takes 50ms + 400ms backoff = 450ms (< 600ms).
    # Attempt 2 takes 50ms (500ms elapsed); next 400ms backoff would reach 900ms >= 600ms -> deadline exhausted!
    assert len(result.attempts) == 2
    assert clock.sleep_history == [0.4, 0.4]
    assert result.timing.total_latency_ms == pytest.approx(900.0)


@pytest.mark.asyncio
async def test_fake_clock_deterministic_backoff_and_circuit_breaker_async():
    from llm_gateway import (
        BudgetConfigV1,
        FakeClock,
        LLMRequestV1,
        MessageV1,
        PolicyContextV1,
        RequestRole,
        ResultStatus,
        RouteV1,
        RoutingPolicyV1,
        TransportError,
        create_engine,
    )
    from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse

    clock = FakeClock(initial_utc=1_700_000_000.0, initial_monotonic=1000.0)

    class FlakyAsyncAdapter(BaseLLMAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, request, route, kwargs):
            raise NotImplementedError

        async def execute_async(self, request, route, kwargs):
            self.calls += 1
            clock.advance(0.02)
            if self.calls == 1:
                raise TransportError("timeout", "Request timed out", retry_after_seconds=0.25)
            return RawAdapterResponse(
                text="recovered after deterministic retry",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                is_usage_reported=True,
            )

    engine = create_engine(clock=clock, jitter_fn=lambda attempt_num: 0.1)
    adapter = FlakyAsyncAdapter()
    engine.register_adapter("flaky_async", adapter)

    req = LLMRequestV1(
        request_id="req-fakeclock-async-1",
        policy_tag="gather.propose_action",
        messages=[MessageV1(role=RequestRole.USER, content="hello")],
        routing_policy=RoutingPolicyV1(
            baseline_route=RouteV1(provider_id="flaky_async", model_id="fake-model")
        ),
        policy_context=PolicyContextV1(
            context_version="1.0.0",
            safety_class="standard",
            contains_itinerary_state=False,
            request_mode="single_domain",
            produced_at_utc=clock.time(),
            producer_id="summary_safety_context.v1",
        ),
        budget=BudgetConfigV1(deadline_ms=2000, max_physical_attempts=3, max_cost_usd=1.0),
    )

    result = await engine.invoke_llm_async(req)
    assert result.status == ResultStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.text == "recovered after deterministic retry"
    # Retry-After (0.25s) > jitter (0.1s), so FakeClock recorded 0.25s sleep in 0ms real time
    assert clock.sleep_history == [0.25]
    assert result.timing.total_latency_ms == pytest.approx(290.0)

