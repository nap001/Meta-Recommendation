"""Unified Multi-Provider CLI Probe for WP5 LLM Gateway.

Tests live or dry-run connectivity against Azure OpenAI, OpenAI, Groq, and Anthropic
under strict budget and token limits.

Usage:
    python scripts/probe_providers.py [--provider {all,openai,azure_openai,groq,anthropic}]
                                      [--dry-run]
                                      [--async]
                                      [--budget-usd 0.01]
                                      [--max-tokens 30]
                                      [--deadline-ms 10000]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

from llm_gateway import (
    BudgetConfigV1,
    GenerationConfigV1,
    LLMRequestV1,
    MessageV1,
    PolicyContextV1,
    PolicyLeg,
    ProviderCapabilityV1,
    RequestMode,
    RequestRole,
    ResponseContractV1,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    RoutingPolicyV1,
    create_engine,
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
    invoke_llm_async,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.constants import ProviderId


SUPPORTED_PROVIDERS: List[str] = [
    ProviderId.OPENAI.value,
    ProviderId.AZURE_OPENAI.value,
    ProviderId.GROQ.value,
    ProviderId.ANTHROPIC.value,
]

DEFAULT_PROBE_MODELS: Dict[str, str] = {
    ProviderId.OPENAI.value: "gpt-4o-mini",
    ProviderId.AZURE_OPENAI.value: "gpt-4o-mini",
    ProviderId.GROQ.value: "llama-3.3-70b-versatile",
    ProviderId.ANTHROPIC.value: "claude-3-5-haiku-latest",
}


def check_provider_credentials(provider_id: str) -> Tuple[bool, str]:
    """Inspect environment variables for the given provider without logging secrets."""
    if provider_id == ProviderId.OPENAI.value:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        ok = bool(key and not key.startswith("Some ") and not key.startswith("sk-placeholder"))
        return ok, "OPENAI_API_KEY"

    if provider_id == ProviderId.AZURE_OPENAI.value:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()
        ok = bool(
            endpoint
            and not endpoint.startswith("Some ")
            and key
            and not key.startswith("Some ")
            and not key.startswith("placeholder")
        )
        return ok, "AZURE_OPENAI_ENDPOINT + API_KEY"

    if provider_id == ProviderId.GROQ.value:
        key = os.getenv("GROQ_API_KEY", "").strip()
        ok = bool(key and not key.startswith("Some ") and not key.startswith("gsk_placeholder"))
        return ok, "GROQ_API_KEY"

    if provider_id == ProviderId.ANTHROPIC.value:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        ok = bool(key and not key.startswith("Some ") and not key.startswith("sk-ant-placeholder"))
        return ok, "ANTHROPIC_API_KEY"

    return False, "UNKNOWN_PROVIDER"


def build_probe_request(
    provider_id: str,
    model_id: str,
    *,
    prompt: str = "Explain in one brief sentence why multi-provider AI gateways improve service reliability.",
    budget_usd: float = 0.01,
    max_output_tokens: int = 30,
    deadline_ms: int = 10000,
) -> LLMRequestV1:
    """Build a strictly budgeted probe request."""
    routing = RoutingPolicyV1(
        mode=RoutingMode.BASELINE,
        baseline_route=RouteV1(provider_id=provider_id, model_id=model_id, policy_leg=PolicyLeg.BASELINE),
        fallback_routes=[],
        allowed_provider_ids=[provider_id],
    )
    return LLMRequestV1(
        request_id=f"probe-{provider_id}-{int(time.time() * 1000) % 1000000}",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content=prompt)],
        generation=GenerationConfigV1(temperature=0.2, max_output_tokens=max_output_tokens),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=routing,
        budget=BudgetConfigV1(
            deadline_ms=deadline_ms,
            max_physical_attempts=1,  # Strict: single attempt, no hidden amplification
            max_output_tokens=max_output_tokens,
            max_cost_usd=budget_usd,
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
    )


def run_probe_sync(
    provider_id: str,
    model_id: str,
    *,
    dry_run: bool = False,
    budget_usd: float = 0.01,
    max_tokens: int = 30,
    deadline_ms: int = 10000,
) -> Dict[str, Any]:
    """Execute synchronous probe for one provider."""
    request = build_probe_request(
        provider_id,
        model_id,
        budget_usd=budget_usd,
        max_output_tokens=max_tokens,
        deadline_ms=deadline_ms,
    )

    if dry_run:
        engine = create_engine()
        fake = FakeLLMAdapter()
        fake.enqueue_response(
            ScriptedResponse(
                text=f"Simulated healthy response from {provider_id}:{model_id}.",
                prompt_tokens=18,
                completion_tokens=12,
            )
        )
        engine.register_adapter(provider_id, fake)
        result = engine.invoke_llm(request)
    else:
        result = invoke_llm(request)

    return _summarize_result(provider_id, model_id, result)


async def run_probe_async(
    provider_id: str,
    model_id: str,
    *,
    dry_run: bool = False,
    budget_usd: float = 0.01,
    max_tokens: int = 30,
    deadline_ms: int = 10000,
) -> Dict[str, Any]:
    """Execute asynchronous probe for one provider."""
    request = build_probe_request(
        provider_id,
        model_id,
        budget_usd=budget_usd,
        max_output_tokens=max_tokens,
        deadline_ms=deadline_ms,
    )

    if dry_run:
        engine = create_engine()
        fake = FakeLLMAdapter()
        fake.enqueue_response(
            ScriptedResponse(
                text=f"Simulated healthy response from {provider_id}:{model_id}.",
                prompt_tokens=18,
                completion_tokens=12,
            )
        )
        engine.register_adapter(provider_id, fake)
        result = await engine.invoke_llm_async(request)
    else:
        result = await invoke_llm_async(request)

    return _summarize_result(provider_id, model_id, result)


def _summarize_result(provider_id: str, model_id: str, result: Any) -> Dict[str, Any]:
    output_text = result.output.text.strip() if (result.output and result.output.text) else ""
    error_code = result.error.code if result.error else ""
    error_msg = result.error.message if result.error else ""

    return {
        "provider": provider_id,
        "model": model_id,
        "status": result.status.value,
        "latency_ms": result.timing.total_latency_ms,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
        "cost_usd": result.usage.cost_usd,
        "output_preview": (output_text[:60] + "...") if len(output_text) > 60 else output_text,
        "error_code": error_code,
        "error_msg": error_msg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Provider LLM Gateway Connectivity Probe")
    parser.add_argument(
        "--provider",
        choices=["all"] + SUPPORTED_PROVIDERS,
        default="all",
        help="Target provider (default: all)",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional model override")
    parser.add_argument("--dry-run", action="store_true", help="Run offline simulation using deterministic fake adapter")
    parser.add_argument("--async", dest="run_async", action="store_true", help="Run via invoke_llm_async")
    parser.add_argument("--budget-usd", type=float, default=0.01, help="Strict cost cap in USD per probe (default: $0.01)")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max completion tokens (default: 30)")
    parser.add_argument("--deadline-ms", type=int, default=10000, help="Per-request deadline in ms (default: 10000)")
    args = parser.parse_args()

    providers_to_test = (
        SUPPORTED_PROVIDERS if args.provider == "all" else [args.provider]
    )

    print("=" * 80)
    print("  WP5 LLM GATEWAY MULTI-PROVIDER PROBE")
    print(f"  Mode: {'DRY-RUN (OFFLINE SIMULATION)' if args.dry_run else 'LIVE CONNECTIVITY'}")
    print(f"  Execution: {'ASYNC' if args.run_async else 'SYNC'}")
    print(f"  Budget Cap: ${args.budget_usd:.4f} per request | Max Tokens: {args.max_tokens} | Deadline: {args.deadline_ms}ms")
    print("=" * 80)

    # 1. Print Credential Status
    print("\n[1/3] Checking Provider Credentials:")
    for p in providers_to_test:
        has_cred, key_name = check_provider_credentials(p)
        status_str = "[CONFIGURED]" if has_cred else "[MISSING / PLACEHOLDER]"
        print(f"  • {p:<15s} : {status_str:<24s} ({key_name})")

    # 2. Execute Probes
    print("\n[2/3] Executing Probes:")
    results: List[Dict[str, Any]] = []

    for p in providers_to_test:
        model = args.model or DEFAULT_PROBE_MODELS.get(p, "default")
        has_cred, _ = check_provider_credentials(p)

        if not has_cred and not args.dry_run:
            print(f"  • {p:<15s} : SKIPPED (missing credentials, pass --dry-run to simulate)")
            results.append({
                "provider": p,
                "model": model,
                "status": "skipped",
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "output_preview": "SKIPPED: Missing live credentials",
                "error_code": "missing_credentials",
                "error_msg": "Missing API key in environment",
            })
            continue

        print(f"  • Probing {p:<14s} (model: {model})... ", end="", flush=True)
        try:
            if args.run_async:
                res = asyncio.run(
                    run_probe_async(
                        p,
                        model,
                        dry_run=args.dry_run,
                        budget_usd=args.budget_usd,
                        max_tokens=args.max_tokens,
                        deadline_ms=args.deadline_ms,
                    )
                )
            else:
                res = run_probe_sync(
                    p,
                    model,
                    dry_run=args.dry_run,
                    budget_usd=args.budget_usd,
                    max_tokens=args.max_tokens,
                    deadline_ms=args.deadline_ms,
                )
            results.append(res)
            print(f"[{res['status'].upper()}] in {res['latency_ms']:.1f} ms (${res['cost_usd']:.6f})")
            if res["output_preview"]:
                print(f"    Output: \"{res['output_preview']}\"")
            if res["error_code"]:
                print(f"    Error:  {res['error_code']}: {res['error_msg']}")
        except Exception as exc:
            print(f"[ERROR] {exc}")
            results.append({
                "provider": p,
                "model": model,
                "status": "failed",
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "output_preview": "",
                "error_code": "probe_exception",
                "error_msg": str(exc),
            })

    # 3. Summary Table
    print("\n" + "=" * 80)
    print("  PROBE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Provider':<14s} {'Model':<25s} {'Status':<10s} {'Latency':<10s} {'Tokens (P/C/T)':<16s} {'Cost USD':<10s}")
    print("-" * 80)

    total_spend = 0.0
    for r in results:
        total_spend += r["cost_usd"]
        tokens_str = f"{r['prompt_tokens']}/{r['completion_tokens']}/{r['total_tokens']}"
        print(f"{r['provider']:<14s} {r['model']:<25s} {r['status']:<10s} {r['latency_ms']:>7.1f} ms {tokens_str:>15s}   ${r['cost_usd']:.6f}")

    print("-" * 80)
    print(f"Total Probes: {len(results)} | Total Cumulative Spend: ${total_spend:.6f} USD")
    budget_ok = total_spend <= (args.budget_usd * len(results))
    print(f"Budget Invariant: {'PASSED (within cap)' if budget_ok else 'EXCEEDED'}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
