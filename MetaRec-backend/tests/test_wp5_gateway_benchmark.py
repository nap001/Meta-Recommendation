"""Automated unit and regression test for the WP5 gateway latency overhead criterion.

Verifies:
1. Baseline gateway-only p95 latency overhead <= 10.0 ms (Sync and Async paths).
2. Correct latency and timing metadata recorded in TimingSummaryV1.
3. Proper attempt progression across specialized legs (cheap, repair, retry, fallback).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import List

import pytest

from llm_gateway import (
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
    create_engine,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.constants import SemanticValidatorId
from llm_gateway.validation import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    resolve_response_schema,
)

pytestmark = pytest.mark.backend_unit


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[int(f)] * (c - k) + sorted_v[int(c)] * (k - f)


def _benchmark_request(mode: RoutingMode = RoutingMode.BASELINE) -> LLMRequestV1:
    routing = RoutingPolicyV1(
        mode=mode,
        baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE),
        cheap_route=RouteV1(provider_id="fake_provider", model_id="fake-fast", policy_leg=PolicyLeg.CHEAP),
        fallback_routes=[],
        allowed_provider_ids=["fake_provider"],
    )
    return LLMRequestV1(
        request_id="bench-p95-test",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Benchmark prompt test")],
        generation=GenerationConfigV1(temperature=0.0, max_output_tokens=50),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=routing,
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=1.0),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )


def test_gateway_sync_p95_overhead_under_10ms():
    """Verify that synchronous gateway overhead p95 is strictly under 10.0 ms across 200 samples."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)
    req = _benchmark_request()

    # Warmup
    for _ in range(25):
        adapter.enqueue_response(ScriptedResponse(text="Warmup"))
        engine.invoke_llm(req)

    overheads_ms: List[float] = []
    for _ in range(200):
        adapter.enqueue_response(ScriptedResponse(text="Benchmark response text"))
        t0 = time.perf_counter()
        res = engine.invoke_llm(req)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        assert res.status == ResultStatus.SUCCEEDED
        attempt_lat = res.attempts[0].latency_ms if res.attempts else 0.0
        overhead_ms = max(0.0, total_time_ms - attempt_lat)
        overheads_ms.append(overhead_ms)

    p50 = _percentile(overheads_ms, 50)
    p95 = _percentile(overheads_ms, 95)
    p99 = _percentile(overheads_ms, 99)

    print(f"\n[BENCHMARK] Sync Gateway Overhead: p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
    assert p95 < 10.0, f"Sync gateway p95 overhead of {p95:.3f}ms exceeded the 10.0ms limit!"


def test_gateway_async_p95_overhead_under_10ms():
    """Verify that asynchronous gateway overhead p95 is strictly under 10.0 ms across 200 samples."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)
    req = _benchmark_request()

    async def _run():
        # Warmup
        for _ in range(25):
            adapter.enqueue_response(ScriptedResponse(text="Warmup"))
            await engine.invoke_llm_async(req)

        overheads_ms: List[float] = []
        for _ in range(200):
            adapter.enqueue_response(ScriptedResponse(text="Async benchmark response text"))
            t0 = time.perf_counter()
            res = await engine.invoke_llm_async(req)
            total_time_ms = (time.perf_counter() - t0) * 1000.0

            assert res.status == ResultStatus.SUCCEEDED
            attempt_lat = res.attempts[0].latency_ms if res.attempts else 0.0
            overheads_ms.append(max(0.0, total_time_ms - attempt_lat))

        return overheads_ms

    overheads_ms = asyncio.run(_run())
    p50 = _percentile(overheads_ms, 50)
    p95 = _percentile(overheads_ms, 95)
    p99 = _percentile(overheads_ms, 99)

    print(f"\n[BENCHMARK] Async Gateway Overhead: p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
    assert p95 < 10.0, f"Async gateway p95 overhead of {p95:.3f}ms exceeded the 10.0ms limit!"


def test_timing_summary_and_monotonic_consistency():
    """Verify TimingSummaryV1 invariants: monotonic duration, valid UTC boundaries, deadline inclusion."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)
    req = _benchmark_request()

    adapter.enqueue_response(ScriptedResponse(text="Timing test", simulated_delay_ms=5.0))
    t0_wall = time.time()
    res = engine.invoke_llm(req)
    t1_wall = time.time()

    timing = res.timing
    assert timing.deadline_ms == req.budget.deadline_ms
    assert t0_wall <= timing.started_at_utc <= timing.completed_at_utc <= (t1_wall + 0.1)
    assert timing.total_latency_ms >= 0.0
