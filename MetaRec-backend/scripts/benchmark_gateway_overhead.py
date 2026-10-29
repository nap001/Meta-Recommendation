"""Automated Gateway Latency Benchmark for WP5 LLM Gateway.

Measures fake-adapter baseline gateway latency to evaluate against the
acceptance criterion: Gateway-only p95 overhead <= 10 ms.

Also measures p50/p95 for accepted-cheap, retry, repair, and fallback escalation legs.

Usage:
    python scripts/benchmark_gateway_overhead.py [--samples 500] [--warmup 50] [--output-json REPORT_PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

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
    TransportError,
    create_engine,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.constants import SemanticValidatorId
from llm_gateway.validation import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    resolve_response_schema,
)


def _percentile(values: List[float], p: float) -> float:
    """Calculate the p-th percentile (0 <= p <= 100) using nearest-rank / linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    d0 = sorted_v[int(f)] * (c - k)
    d1 = sorted_v[int(c)] * (k - f)
    return d0 + d1


def _compute_stats(samples_ms: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max, p50, p90, p95, p99."""
    if not samples_ms:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    n = len(samples_ms)
    mean = sum(samples_ms) / n
    variance = sum((x - mean) ** 2 for x in samples_ms) / n
    std = math.sqrt(variance)

    return {
        "count": float(n),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(min(samples_ms), 3),
        "p50": round(_percentile(samples_ms, 50), 3),
        "p90": round(_percentile(samples_ms, 90), 3),
        "p95": round(_percentile(samples_ms, 95), 3),
        "p99": round(_percentile(samples_ms, 99), 3),
        "max": round(max(samples_ms), 3),
    }


def _make_text_request(mode: RoutingMode = RoutingMode.BASELINE) -> LLMRequestV1:
    routing = RoutingPolicyV1(
        mode=mode,
        baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE),
        cheap_route=RouteV1(provider_id="fake_provider", model_id="fake-fast", policy_leg=PolicyLeg.CHEAP),
        fallback_routes=[],
        allowed_provider_ids=["fake_provider"],
    )
    return LLMRequestV1(
        request_id="bench-req",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Benchmark prompt text")],
        generation=GenerationConfigV1(temperature=0.7, max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=routing,
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=3, max_cost_usd=1.0),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )


def _make_confirmation_request() -> LLMRequestV1:
    entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    assert entry is not None
    routing = RoutingPolicyV1(
        mode=RoutingMode.BASELINE,
        baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE),
        fallback_routes=[RouteV1(provider_id="fake_provider", model_id="fake-fast", policy_leg=PolicyLeg.FALLBACK)],
        allowed_provider_ids=["fake_provider"],
    )
    return LLMRequestV1(
        request_id="bench-conf-req",
        policy_tag="confirmation.payload",
        messages=[MessageV1(role=RequestRole.USER, content="Confirm action")],
        generation=GenerationConfigV1(temperature=0.0, max_output_tokens=100),
        response_contract=ResponseContractV1(
            kind=ResponseKind.JSON,
            schema_id=CONFIRMATION_PAYLOAD_SCHEMA_ID,
            json_schema=entry.schema(),
            semantic_validator_id=SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
            validator_inputs={"allowed_preference_patch_keys": ["use_case"]},
            allow_empty=False,
        ),
        routing_policy=routing,
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=3, max_cost_usd=1.0),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )


def _confirmation_json() -> str:
    return json.dumps({
        "message": "Please choose your preferred category.",
        "quick_actions": [
            {"id": "work", "label": "Work", "value": "work", "preference_patch": {"use_case": "work"}},
            {"id": "gaming", "label": "Gaming", "value": "gaming", "preference_patch": {"use_case": "gaming"}},
        ],
    })


def benchmark_baseline_overhead(samples: int = 500, warmup: int = 50) -> Dict[str, Any]:
    """Run baseline latency benchmarks measuring gateway overhead vs pure adapter execution."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    req = _make_text_request()
    route = req.routing_policy.baseline_route
    kwargs, _ = engine._normalize_request_kwargs(req, route, {})

    # Warmup
    for _ in range(warmup):
        adapter.enqueue_response(ScriptedResponse(text="Warmup response"))
        engine.invoke_llm(req)

    adapter_latencies: List[float] = []
    gateway_sync_latencies: List[float] = []
    gateway_overhead_latencies: List[float] = []

    for _ in range(samples):
        # 1. Pure adapter latency
        t0 = time.perf_counter()
        adapter.execute(req, route, kwargs)
        adapter_ms = (time.perf_counter() - t0) * 1000.0
        adapter_latencies.append(adapter_ms)

        # 2. Complete gateway sync invocation
        adapter.enqueue_response(ScriptedResponse(text="Baseline benchmark response"))
        t0 = time.perf_counter()
        res = engine.invoke_llm(req)
        gw_ms = (time.perf_counter() - t0) * 1000.0
        gateway_sync_latencies.append(gw_ms)

        # Gateway overhead = Gateway end-to-end - Attempt latency recorded by adapter
        attempt_lat = res.attempts[0].latency_ms if res.attempts else 0.0
        overhead_ms = max(0.0, gw_ms - attempt_lat)
        gateway_overhead_latencies.append(overhead_ms)

    return {
        "adapter": _compute_stats(adapter_latencies),
        "gateway_sync": _compute_stats(gateway_sync_latencies),
        "gateway_overhead": _compute_stats(gateway_overhead_latencies),
    }


async def benchmark_async_overhead(samples: int = 500, warmup: int = 50) -> Dict[str, Any]:
    """Benchmark async gateway execution latency."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    req = _make_text_request()

    for _ in range(warmup):
        adapter.enqueue_response(ScriptedResponse(text="Warmup"))
        await engine.invoke_llm_async(req)

    async_latencies: List[float] = []
    async_overhead: List[float] = []

    for _ in range(samples):
        adapter.enqueue_response(ScriptedResponse(text="Async benchmark response"))
        t0 = time.perf_counter()
        res = await engine.invoke_llm_async(req)
        gw_ms = (time.perf_counter() - t0) * 1000.0
        async_latencies.append(gw_ms)
        attempt_lat = res.attempts[0].latency_ms if res.attempts else 0.0
        async_overhead.append(max(0.0, gw_ms - attempt_lat))

    return {
        "gateway_async": _compute_stats(async_latencies),
        "gateway_async_overhead": _compute_stats(async_overhead),
    }


def benchmark_specialized_legs(samples: int = 150) -> Dict[str, Any]:
    """Measure accepted-cheap, retry, validation repair, and escalation legs."""
    engine = create_engine()
    adapter = FakeLLMAdapter()
    engine.register_adapter("fake_provider", adapter)

    cheap_latencies: List[float] = []
    repair_latencies: List[float] = []
    retry_latencies: List[float] = []
    escalation_latencies: List[float] = []

    # 1. Accepted Cheap Route (CHEAP_FIRST mode, single domain, no fallback)
    req_cheap = _make_text_request(mode=RoutingMode.CHEAP_FIRST)
    for _ in range(samples):
        adapter.enqueue_response(ScriptedResponse(text="Cheap model fast response"))
        t0 = time.perf_counter()
        engine.invoke_llm(req_cheap)
        cheap_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 2. Validation Repair Leg (Attempt 1 fails json, Attempt 2 repairs)
    req_repair = _make_confirmation_request()
    for _ in range(samples):
        adapter.enqueue_response(ScriptedResponse(text="bad json"))
        adapter.enqueue_response(ScriptedResponse(text=_confirmation_json()))
        t0 = time.perf_counter()
        engine.invoke_llm(req_repair)
        repair_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 3. Transport Retry Leg (Attempt 1 fails transport, Attempt 2 succeeds)
    for _ in range(samples):
        adapter.enqueue_response(ScriptedResponse(error_to_raise=TransportError("rate_limited", "429 Rate limited")))
        adapter.enqueue_response(ScriptedResponse(text="Success on retry"))
        t0 = time.perf_counter()
        engine.invoke_llm(req_cheap)
        retry_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 4. Route Fallback / Escalation Leg (Attempt 1 fails, falls back to baseline)
    req_fallback = _make_text_request(mode=RoutingMode.CHEAP_FIRST)
    for _ in range(samples):
        # Cheap route fails with non-retryable provider error
        adapter.enqueue_response(ScriptedResponse(error_to_raise=RuntimeError("Cheap model service unavailable")))
        adapter.enqueue_response(ScriptedResponse(text="Baseline model response"))
        t0 = time.perf_counter()
        engine.invoke_llm(req_fallback)
        escalation_latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "accepted_cheap": _compute_stats(cheap_latencies),
        "validation_repair": _compute_stats(repair_latencies),
        "transport_retry": _compute_stats(retry_latencies),
        "route_fallback_escalation": _compute_stats(escalation_latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="WP5 Gateway Latency Benchmark (< 10 ms p95 overhead)")
    parser.add_argument("--samples", type=int, default=500, help="Number of benchmark iterations (default: 500)")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations (default: 50)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    print("=" * 80)
    print("  WP5 LLM GATEWAY LATENCY BENCHMARK")
    print(f"  Samples: {args.samples} | Warmup: {args.warmup}")
    print("  Target Criterion: Gateway-only p95 overhead <= 10.0 ms")
    print("=" * 80)

    print("\n[1/3] Benchmarking Sync Baseline Gateway Overhead...")
    sync_results = benchmark_baseline_overhead(samples=args.samples, warmup=args.warmup)

    print("[2/3] Benchmarking Async Gateway Overhead...")
    async_results = asyncio.run(benchmark_async_overhead(samples=args.samples, warmup=args.warmup))

    print("[3/3] Benchmarking Specialized Legs (Accepted-Cheap, Repair, Retry, Escalation)...")
    leg_results = benchmark_specialized_legs(samples=min(args.samples, 150))

    report = {
        "benchmark_metadata": {
            "samples": args.samples,
            "warmup": args.warmup,
            "timestamp": time.time(),
        },
        **sync_results,
        **async_results,
        **leg_results,
    }

    # Print Formatted Table
    print("\n" + "=" * 80)
    print("  BENCHMARK LATENCY SUMMARY (MILLISECONDS)")
    print("=" * 80)
    print(f"{'Metric / Execution Path':<32s} {'Mean':>7s} {'Std':>7s} {'p50':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'Max':>7s}")
    print("-" * 80)

    rows = [
        ("Pure Fake Adapter", report["adapter"]),
        ("Gateway Sync (Total)", report["gateway_sync"]),
        ("Gateway Overhead (Sync)", report["gateway_overhead"]),
        ("Gateway Async (Total)", report["gateway_async"]),
        ("Gateway Overhead (Async)", report["gateway_async_overhead"]),
        ("Accepted Cheap Leg", report["accepted_cheap"]),
        ("Validation Repair Leg", report["validation_repair"]),
        ("Transport Retry Leg", report["transport_retry"]),
        ("Route Escalation Leg", report["route_fallback_escalation"]),
    ]

    for label, stat in rows:
        print(
            f"{label:<32s} {stat['mean']:>7.2f} {stat['std']:>7.2f} {stat['p50']:>7.2f} "
            f"{stat['p90']:>7.2f} {stat['p95']:>7.2f} {stat['p99']:>7.2f} {stat['max']:>7.2f}"
        )

    print("-" * 80)
    p95_overhead = report["gateway_overhead"]["p95"]
    p95_async_overhead = report["gateway_async_overhead"]["p95"]
    passed = p95_overhead < 10.0 and p95_async_overhead < 10.0

    print(f"Sync Gateway p95 Overhead:  {p95_overhead:.3f} ms (Target: <= 10.0 ms)")
    print(f"Async Gateway p95 Overhead: {p95_async_overhead:.3f} ms (Target: <= 10.0 ms)")
    print(f"Acceptance Criterion:      {'[PASSED]' if passed else '[FAILED]'}")
    print("=" * 80 + "\n")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.output_json}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
