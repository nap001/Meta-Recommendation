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
