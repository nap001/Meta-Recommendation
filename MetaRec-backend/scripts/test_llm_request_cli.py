"""Interactive End-to-End CLI Test Runner & Inspector for WP5 LLMRequestV1.

Provides:
1. Interactive menu-driven preset selection & from-scratch builder for LLMRequestV1.
2. Option-driven configuration for every field, nested model, enum, and tool shape.
3. Pre-flight credential & route-ladder inspection + projected wire kwargs preview.
4. Real end-to-end execution through the WP5 Gateway (sync, async, executor,
   sync streaming, async streaming) as well as offline --dry-run simulation.
5. Non-interactive CLI flags (--list-presets, --preset, --provider, --model,
   --stream, --async, --dry-run, --export, --import-file).

Usage Examples:
    # Launch interactive CLI menu
    python scripts/test_llm_request_cli.py

    # List all discovered LLMRequestV1 presets
    python scripts/test_llm_request_cli.py --list-presets

    # Run a preset live against Groq (default configured provider in .env)
    python scripts/test_llm_request_cli.py --preset minimal_text --provider groq --model llama-3.3-70b-versatile

    # Run streaming live against Groq
    python scripts/test_llm_request_cli.py --preset streaming_token_delta --provider groq --stream

    # Validate & dry-run a preset (or all presets) without network calls
    python scripts/test_llm_request_cli.py --preset wp5b_confirmation_payload --dry-run
    python scripts/test_llm_request_cli.py --preset all --dry-run

    # Export a preset to a reusable JSON fixture
    python scripts/test_llm_request_cli.py --preset full_featured_text --export request_fixture.json --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from llm_usage import UsageLedger, push_ledger, reset_ledger
from llm_gateway import (
    AttemptKind,
    AttemptOutcome,
    BudgetConfigV1,
    GenerationConfigV1,
    LLMRequestV1,
    LLMResultV1,
    MessageV1,
    PolicyContextV1,
    PolicyLeg,
    PrivacyConfigV1,
    ProviderId,
    RequestMode,
    RequestRole,
    ResponseContractV1,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    RoutingPolicyV1,
    ScopedToolCatalogV1,
    SemanticValidatorId,
    ToolSpecV1,
    TraceContextV1,
    build_confirmation_safety_context,
    build_summary_safety_context,
    create_engine,
    get_enrolled_tags,
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
    invoke_llm_async,
    invoke_llm_in_executor,
    invoke_llm_stream,
    invoke_llm_stream_async,
    is_kill_switch_active,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.constants import (
    CHEAP_ROUTE_ELIGIBLE_TAGS,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    NEVER_CHEAP_ROUTE_TAGS,
    POLICY_TAG_AGENT_PLAN,
    POLICY_TAG_AGENT_SUMMARY,
    POLICY_TAG_CONFIRMATION_PAYLOAD,
    POLICY_TAG_CONVERSATION_SUMMARY,
    POLICY_TAG_GATHER_ACTION,
    POLICY_TAG_INTENT_ANALYZE,
    POLICY_TAG_ITINERARY_CONSTRAINTS,
    POLICY_TAG_ITINERARY_CONTAINMENT,
    POLICY_TAG_ITINERARY_DURATIONS,
    POLICY_TAG_ITINERARY_REPAIR,
    POLICY_TAG_ITINERARY_ROLES,
    REGISTERED_CONTEXT_PRODUCERS,
)
from llm_gateway.engine.projection import normalize_request_kwargs
from llm_gateway.engine.routing import build_route_ladder, determine_initial_route
from llm_gateway.legacy_shim import LEGACY_POLICY_TAG, LEGACY_SCHEMA_ID
from llm_gateway.validation.schemas import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    CONVERSATION_SUMMARY_SCHEMA_ID,
    resolve_response_schema,
)


# ==============================================================================
# Constants & Canonical Choices
# ==============================================================================

ALL_POLICY_TAGS: List[Tuple[str, str]] = [
    (POLICY_TAG_CONVERSATION_SUMMARY, "Cheap-eligible when safety_context is single_domain & no itinerary"),
    (POLICY_TAG_CONFIRMATION_PAYLOAD, "Cheap-eligible when safety_context is single_domain & no itinerary"),
    (POLICY_TAG_GATHER_ACTION, "NEVER cheap-routed (domain gather tool proposal)"),
    (POLICY_TAG_INTENT_ANALYZE, "NEVER cheap-routed (intent classification & routing)"),
    (POLICY_TAG_ITINERARY_CONSTRAINTS, "NEVER cheap-routed (itinerary constraint extraction)"),
    (POLICY_TAG_ITINERARY_DURATIONS, "NEVER cheap-routed (itinerary duration enrichment)"),
    (POLICY_TAG_ITINERARY_ROLES, "NEVER cheap-routed (itinerary candidate role classification)"),
    (POLICY_TAG_ITINERARY_CONTAINMENT, "NEVER cheap-routed (itinerary containment check)"),
    (POLICY_TAG_ITINERARY_REPAIR, "NEVER cheap-routed (itinerary repair proposal)"),
    (POLICY_TAG_AGENT_PLAN, "NEVER cheap-routed (legacy restaurant ReAct planner)"),
    (POLICY_TAG_AGENT_SUMMARY, "NEVER cheap-routed (legacy restaurant summarizer)"),
    (LEGACY_POLICY_TAG, "Unenrolled legacy passthrough call site"),
]

EXECUTION_MODES: List[Tuple[str, str]] = [
    ("sync", "invoke_llm(request) — Synchronous unary gateway call"),
    ("async", "await invoke_llm_async(request) — Asynchronous unary gateway call"),
    ("executor", "await invoke_llm_in_executor(request) — Sync engine in ThreadPoolExecutor"),
    ("stream_sync", "invoke_llm_stream(request) — Synchronous real-time token stream"),
    ("stream_async", "invoke_llm_stream_async(request) — Asynchronous real-time token stream"),
]


def _default_live_provider_and_model() -> Tuple[str, str]:
    """Detect the best default live provider and model from .env."""
    groq_key = (os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    azure_ep = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if groq_key and not groq_key.startswith("Some "):
        model = (os.getenv("LLM_MODEL") or "openai/gpt-oss-120b").strip()
        if "Your LLM" in model:
            model = "openai/gpt-oss-120b"
        return ProviderId.GROQ.value, model
    if openai_key and not openai_key.startswith("Some ") and not azure_ep:
        return ProviderId.OPENAI.value, "gpt-6-mini"
    if azure_ep and not azure_ep.startswith("Some ") and (os.getenv("AZURE_OPENAI_API_KEY") or openai_key):
        return ProviderId.AZURE_OPENAI.value, "gpt-6-mini"
    if anthropic_key and not anthropic_key.startswith("Some "):
        return ProviderId.ANTHROPIC.value, "claude-3-5-haiku-20241022"
    return ProviderId.GROQ.value, "openai/gpt-oss-120b"


# ==============================================================================
# Credential & Endpoint Pre-Flight Inspector
# ==============================================================================

def inspect_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Inspect .env credentials and target endpoint for a given provider_id."""
    if provider_id == ProviderId.FAKE.value:
        return {
            "provider_id": provider_id,
            "configured": True,
            "env_vars": "None (in-memory FakeLLMAdapter)",
            "endpoint": "memory://fake_provider",
            "instructions": "Ready (deterministic offline adapter).",
        }

    if provider_id == ProviderId.GROQ.value:
        key = (os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
        base_url = (os.getenv("GROQ_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_GROQ_BASE_URL).strip()
        ok = bool(key and not key.startswith("Some ") and not key.startswith("gsk_placeholder"))
        return {
            "provider_id": provider_id,
            "configured": ok,
            "env_vars": "GROQ_API_KEY (or LLM_API_KEY)",
            "endpoint": f"{base_url.rstrip('/')}/chat/completions",
            "instructions": "Set GROQ_API_KEY=\"gsk_...\" in MetaRec-backend/.env" if not ok else "Ready",
        }

    if provider_id in (ProviderId.OPENAI.value, ProviderId.OPENAI_BASE.value):
        key = (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
        base_url = (os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).strip()
        ok = bool(key and not key.startswith("Some ") and not key.startswith("sk-placeholder"))
        return {
            "provider_id": provider_id,
            "configured": ok,
            "env_vars": "OPENAI_API_KEY",
            "endpoint": f"{base_url.rstrip('/')}/chat/completions",
            "instructions": "Set OPENAI_API_KEY=\"sk-...\" in MetaRec-backend/.env" if not ok else "Ready",
        }

    if provider_id == ProviderId.AZURE_OPENAI.value:
        key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_ver = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION).strip()
        ok = bool(
            endpoint
            and not endpoint.startswith("Some ")
            and key
            and not key.startswith("Some ")
        )
        return {
            "provider_id": provider_id,
            "configured": ok,
            "env_vars": "AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY",
            "endpoint": f"{endpoint or '<unset>'} (api-version={api_ver})",
            "instructions": (
                "Set AZURE_OPENAI_ENDPOINT=\"https://<resource>.openai.azure.com/\" "
                "and AZURE_OPENAI_API_KEY=\"...\" in MetaRec-backend/.env"
                if not ok
                else "Ready"
            ),
        }

    if provider_id == ProviderId.ANTHROPIC.value:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        base_url = (os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL).strip()
        ok = bool(key and not key.startswith("Some ") and not key.startswith("sk-ant-placeholder"))
        return {
            "provider_id": provider_id,
            "configured": ok,
            "env_vars": "ANTHROPIC_API_KEY",
            "endpoint": f"{base_url.rstrip('/')}/v1/messages",
            "instructions": "Set ANTHROPIC_API_KEY=\"sk-ant-...\" in MetaRec-backend/.env" if not ok else "Ready",
        }

    return {
        "provider_id": provider_id,
        "configured": bool(os.getenv("LLM_API_KEY")),
        "env_vars": "LLM_API_KEY + LLM_BASE_URL",
        "endpoint": os.getenv("LLM_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        "instructions": "Set LLM_API_KEY and LLM_BASE_URL in .env",
    }


def print_environment_and_registry_status() -> None:
    """Display all registered providers, models, pricing, and .env credential status."""
    registry = get_provider_registry()
    pricing = registry.get_pricing_snapshot("default-pricing-v1")

    print("\n" + "=" * 88)
    print("  WP5 LLM GATEWAY — PROVIDER REGISTRY, PRICING & CREDENTIAL STATUS")
    print("=" * 88)
    print(f"  Global Kill Switch Active : {is_kill_switch_active()}")
    print(f"  Enrolled Cheap-First Tags : {', '.join(sorted(get_enrolled_tags())) or '(none)'}")
    print("-" * 88)

    for cap in registry.list_providers():
        cred = inspect_provider_credentials(cap.provider_id)
        status_badge = "[READY]" if cred["configured"] else "[MISSING KEY]"
        print(f"\n  Provider: {cap.provider_id:<15s} | Adapter: {cap.adapter_kind:<18s} | Status: {status_badge}")
        print(f"    Endpoint   : {cred['endpoint']}")
        print(f"    Env Vars   : {cred['env_vars']}")
        print(
            f"    Capabilities: stream={cap.features.supports_streaming}, "
            f"json_obj={cap.features.supports_json_object}, "
            f"json_schema={cap.features.supports_json_schema}, "
            f"tools={cap.features.supports_tools}, "
            f"seed={cap.features.supports_seed}"
        )
        print(f"    Data Policy: {cap.data_policy_ref} | Price Table: {cap.price_table_version}")
        print("    Registered Models:")
        for m in cap.models:
            rate = pricing.get_price(cap.provider_id, m.model_id)
            reas_flag = " [reasoning]" if m.is_reasoning_model else ""
            print(
                f"      - {m.model_id:<30s} ctx={m.context_window:<7d} max_out={m.max_output_tokens:<5d} "
                f"in=${rate.input_price_per_1m:<6.2f}/1M out=${rate.output_price_per_1m:<6.2f}/1M{reas_flag}"
            )
    print("=" * 88 + "\n")


# ==============================================================================
# Exhaustive Test Case Matrix (18 Presets)
# ==============================================================================

class PresetDefinition:
    def __init__(
        self,
        preset_id: str,
        category: str,
        title: str,
        description: str,
        builder: Callable[[str, str], LLMRequestV1],
        default_exec_mode: str = "sync",
        dry_run_script: Optional[Callable[[LLMRequestV1], ScriptedResponse]] = None,
    ) -> None:
        self.preset_id = preset_id
        self.category = category
        self.title = title
        self.description = description
        self.builder = builder
        self.default_exec_mode = default_exec_mode
        self.dry_run_script = dry_run_script


def _default_cheap_model(prov: str, fallback_mod: str) -> str:
    env_cheap = (os.getenv("LLM_CHEAP_MODEL") or "").strip()
    if env_cheap:
        return env_cheap
    return "openai/gpt-oss-20b" if prov == "groq" else fallback_mod


def _build_presets() -> List[PresetDefinition]:
    confirmation_schema_entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
    confirmation_schema = confirmation_schema_entry.schema() if confirmation_schema_entry else {
        "type": "object",
        "required": ["message"],
        "properties": {"message": {"type": "string"}},
    }

    presets: List[PresetDefinition] = [
        # ── Category A: Baseline & Core Contracts ──────────────────────────────
        PresetDefinition(
            preset_id="minimal_text",
            category="A. Baseline & Core Contracts",
            title="Minimal Valid LLMRequestV1 (Unary Text)",
            description="Smallest valid LLMRequestV1 with defaults for GenerationConfigV1, ResponseContractV1, BudgetConfigV1, and PrivacyConfigV1.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-min-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(
                        role=RequestRole.USER,
                        content="Explain in one brief sentence why a unified LLM gateway prevents retry amplification.",
                    )
                ],
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),
        PresetDefinition(
            preset_id="full_featured_text",
            category="A. Baseline & Core Contracts",
            title="Full-Featured Multi-Turn LLMRequestV1",
            description="Populates system + developer + multi-turn user/assistant messages, temperature, seed, stop sequences, trace_context, privacy, and fallback route.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-full-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_AGENT_SUMMARY,
                messages=[
                    MessageV1(role=RequestRole.SYSTEM, content="You are a concise Singapore dining concierge."),
                    MessageV1(role=RequestRole.DEVELOPER, content="Always mention the neighborhood and price tier."),
                    MessageV1(role=RequestRole.USER, content="I want casual Peranakan food near Bugis."),
                    MessageV1(role=RequestRole.ASSISTANT, content="I can help with Peranakan options in the Bugis area."),
                    MessageV1(role=RequestRole.USER, content="Give me a 2-sentence recommendation summary."),
                ],
                generation=GenerationConfigV1(
                    temperature=0.3,
                    max_output_tokens=150,
                    seed=42,
                    stop=["\n\n---"],
                ),
                response_contract=ResponseContractV1(
                    kind=ResponseKind.TEXT,
                    schema_id="default.text.v1",
                    allow_empty=False,
                ),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE, alias="primary-strong"),
                    fallback_routes=[
                        RouteV1(provider_id=prov, model_id=_default_cheap_model(prov, mod), policy_leg=PolicyLeg.FALLBACK, alias="secondary-backup")
                    ],
                    allowed_provider_ids=[prov],
                ),
                budget=BudgetConfigV1(
                    deadline_ms=20000,
                    max_physical_attempts=3,
                    max_input_tokens=4000,
                    max_output_tokens=300,
                    max_total_tokens=4300,
                    max_cost_usd=0.05,
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                    safety_class="standard",
                    producer_id="default_producer",
                    context_version="1.0.0",
                ),
                trace_context=TraceContextV1(
                    trace_id=f"trace-{uuid.uuid4().hex[:12]}",
                    parent_span_id="span-root-01",
                    correlation_id="corr-demo-001",
                ),
                privacy=PrivacyConfigV1(
                    data_class="standard",
                    allow_prompt_logging=False,
                    allow_response_cache=False,
                ),
            ),
        ),
        PresetDefinition(
            preset_id="structured_json_schema",
            category="A. Baseline & Core Contracts",
            title="Structured JSON Output + Draft-7 JSON Schema Gate",
            description="Uses ResponseKind.JSON with a strict Draft-7 json_schema (type=object, required fields, additionalProperties=False).",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-json-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_INTENT_ANALYZE,
                messages=[
                    MessageV1(
                        role=RequestRole.SYSTEM,
                        content=(
                            "Return ONLY a JSON object with keys: "
                            '{"domain": "restaurant"|"hotel"|"movie", "confidence": float between 0 and 1, "keywords": [string]}.'
                        ),
                    ),
                    MessageV1(role=RequestRole.USER, content="Find me a romantic Italian dinner in Marina Bay."),
                ],
                generation=GenerationConfigV1(temperature=0.1, max_output_tokens=150),
                response_contract=ResponseContractV1(
                    kind=ResponseKind.JSON,
                    schema_id="intent.classification.v1",
                    json_schema={
                        "type": "object",
                        "required": ["domain", "confidence", "keywords"],
                        "additionalProperties": False,
                        "properties": {
                            "domain": {"type": "string", "enum": ["restaurant", "hotel", "movie"]},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        },
                    },
                ),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
            dry_run_script=lambda req: ScriptedResponse(
                text=json.dumps({"domain": "restaurant", "confidence": 0.98, "keywords": ["romantic", "Italian", "Marina Bay"]}),
                prompt_tokens=45,
                completion_tokens=22,
            ),
        ),
        PresetDefinition(
            preset_id="wp5b_confirmation_payload",
            category="A. Baseline & Core Contracts",
            title="WP5-B Registered Confirmation Payload Contract + Semantic Validator",
            description="Triggers Member B's external validation hook (confirmation.payload.response.v1 + confirmation_payload.v1 + normalization).",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-wp5b-conf-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONFIRMATION_PAYLOAD,
                messages=[
                    MessageV1(
                        role=RequestRole.SYSTEM,
                        content=(
                            "Return ONLY valid JSON matching confirmation.payload.response.v1: "
                            '{"message": "Would you prefer Work or Gaming?", "quick_actions": ['
                            '{"id": "work", "label": "Work", "value": "work", "preference_patch": {"use_case": "work"}}, '
                            '{"id": "gaming", "label": "Gaming", "value": "gaming", "preference_patch": {"use_case": "gaming"}}]}'
                        ),
                    ),
                    MessageV1(role=RequestRole.USER, content="Ask the user whether this laptop is for Work or Gaming."),
                ],
                generation=GenerationConfigV1(temperature=0.1, max_output_tokens=250),
                response_contract=ResponseContractV1(
                    kind=ResponseKind.JSON,
                    schema_id=CONFIRMATION_PAYLOAD_SCHEMA_ID,
                    json_schema=confirmation_schema,
                    semantic_validator_id=SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
                    validator_inputs={"allowed_preference_patch_keys": ["use_case"]},
                    allow_empty=False,
                ),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                budget=BudgetConfigV1(deadline_ms=20000, max_physical_attempts=2, max_output_tokens=250, max_cost_usd=0.05),
                policy_context=PolicyContextV1(**build_confirmation_safety_context({"mode": "single_domain", "domain": "product"})),
            ),
            dry_run_script=lambda req: ScriptedResponse(
                text=json.dumps({
                    "message": "Would you prefer Work or Gaming?",
                    "quick_actions": [
                        {"id": "work", "label": "Work", "value": "work", "preference_patch": {"use_case": "work"}},
                        {"id": "gaming", "label": "Gaming", "value": "gaming", "preference_patch": {"use_case": "gaming"}},
                    ],
                }),
                prompt_tokens=60,
                completion_tokens=45,
            ),
        ),
        PresetDefinition(
            preset_id="wp5b_conversation_summary",
            category="A. Baseline & Core Contracts",
            title="WP5-B Registered Conversation Summary Contract",
            description="Dispatches to Member B's external validation hook for conversation.summary.text.v1.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-wp5b-sum-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(role=RequestRole.SYSTEM, content="Summarize the user's dining preferences in 1 concise paragraph."),
                    MessageV1(role=RequestRole.USER, content="User asked for spicy Sichuan noodles under $25 near Chinatown."),
                ],
                generation=GenerationConfigV1(temperature=0.2, max_output_tokens=120),
                response_contract=ResponseContractV1(
                    kind=ResponseKind.TEXT,
                    schema_id=CONVERSATION_SUMMARY_SCHEMA_ID,
                    semantic_validator_id=SemanticValidatorId.NONE.value,
                    validator_inputs={},
                    allow_empty=False,
                ),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    **build_summary_safety_context(
                        [{"role": "user", "content": "spicy Sichuan", "metadata": {"domain": "restaurant"}}]
                    )
                ),
            ),
        ),

        # ── Category B: Tool Calling & WP7 Scoping (Two Tool Shapes) ───────────
        PresetDefinition(
            preset_id="native_tool_calling",
            category="B. Tool Calling & WP7 Scoping",
            title="Native Function Tool Calling + Dotted Tool Name Sanitization",
            description="Uses ResponseKind.TOOL_CALLS + ScopedToolCatalogV1 with dotted names (restaurant.google_maps.search) and excluded tools.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-tools-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_AGENT_PLAN,
                messages=[
                    MessageV1(role=RequestRole.SYSTEM, content="Call the restaurant search tool to find ramen in Orchard."),
                    MessageV1(role=RequestRole.USER, content="Find top-rated ramen restaurants in Orchard Road."),
                ],
                generation=GenerationConfigV1(temperature=0.0, max_output_tokens=200),
                response_contract=ResponseContractV1(kind=ResponseKind.TOOL_CALLS),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                tools=ScopedToolCatalogV1(
                    catalog_id="wp7-restaurant-catalog-v1",
                    health_snapshot_id="health-snap-2026",
                    tools=[
                        ToolSpecV1(
                            name="restaurant.google_maps.search",
                            description="Search restaurants on Google Maps by query and location.",
                            parameters={
                                "type": "object",
                                "required": ["query", "location"],
                                "properties": {
                                    "query": {"type": "string"},
                                    "location": {"type": "string"},
                                },
                            },
                        ),
                        ToolSpecV1(
                            name="restaurant.xiaohongshu.search",
                            description="Search Xiaohongshu notes (currently unhealthy/excluded).",
                            parameters={"type": "object", "properties": {"keyword": {"type": "string"}}},
                        ),
                    ],
                    excluded=["restaurant.xiaohongshu.search"],
                ),
                tool_choice="restaurant.google_maps.search",
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
            dry_run_script=lambda req: ScriptedResponse(
                text="",
                tool_calls=[
                    {
                        "id": "call_ramen_01",
                        "type": "function",
                        "function": {
                            "name": "restaurant_google_maps_search",
                            "arguments": json.dumps({"query": "ramen", "location": "Orchard Road"}),
                        },
                    }
                ],
                prompt_tokens=68,
                completion_tokens=24,
            ),
        ),
        PresetDefinition(
            preset_id="prompt_embedded_tools_json",
            category="B. Tool Calling & WP7 Scoping",
            title="Prompt-Embedded 'available_tools' JSON + gather_action.v1 Validator",
            description="Tests Shape #2: Gateway automatically strips excluded WP7 tools from JSON prompt content and enforces gather_action.v1.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-gather-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_GATHER_ACTION,
                messages=[
                    MessageV1(
                        role=RequestRole.SYSTEM,
                        content='Select one allowed tool and return JSON: {"action": "call_tool", "tool": "<tool_name>", "args": {"query": "jazz"}}',
                    ),
                    MessageV1(
                        role=RequestRole.USER,
                        content=json.dumps({
                            "task": "Find jazz tracks",
                            "available_tools": {
                                "musicbrainz.recording.search": {"description": "Search recordings on MusicBrainz"},
                                "lastfm.track.discover": {"description": "Discover tracks on Last.fm (excluded by WP7)"},
                            },
                        }),
                    ),
                ],
                generation=GenerationConfigV1(temperature=0.0, max_output_tokens=150),
                response_contract=ResponseContractV1(
                    kind=ResponseKind.JSON,
                    schema_id="gather.action.v1",
                    json_schema={
                        "type": "object",
                        "required": ["action", "tool"],
                        "properties": {
                            "action": {"type": "string"},
                            "tool": {"type": "string"},
                            "args": {"type": "object"},
                        },
                    },
                    semantic_validator_id=SemanticValidatorId.GATHER_ACTION.value,
                ),
                tools=ScopedToolCatalogV1(
                    catalog_id="wp7-music-catalog",
                    tools=[
                        ToolSpecV1(name="musicbrainz.recording.search", description="Search MusicBrainz"),
                        ToolSpecV1(name="lastfm.track.discover", description="Discover on Last.fm"),
                    ],
                    excluded=["lastfm.track.discover"],
                ),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
            dry_run_script=lambda req: ScriptedResponse(
                text=json.dumps({"action": "call_tool", "tool": "musicbrainz.recording.search", "args": {"query": "jazz"}}),
                prompt_tokens=55,
                completion_tokens=20,
            ),
        ),
        PresetDefinition(
            preset_id="prompt_embedded_tools_react",
            category="B. Tool Calling & WP7 Scoping",
            title="Prompt-Embedded ReAct [可用工具] Block Filtering",
            description="Demonstrates WP7 tool filtering inside Chinese ReAct prompts ([可用工具] numbered list) before sending to the provider.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-react-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_AGENT_PLAN,
                messages=[
                    MessageV1(
                        role=RequestRole.USER,
                        content=(
                            "请根据用户需求选择工具：\n"
                            "[可用工具]\n"
                            "1) search_google_maps: 搜索谷歌地图餐厅\n"
                            "2) search_xiaohongshu: 搜索小红书笔记\n"
                            "[用户问题]\n"
                            "推荐乌节路的日本料理，只需回答你会调用哪个工具名称。"
                        ),
                    )
                ],
                generation=GenerationConfigV1(temperature=0.1, max_output_tokens=100),
                response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
                tools=ScopedToolCatalogV1(
                    catalog_id="wp7-react-catalog",
                    tools=[
                        ToolSpecV1(name="search_google_maps", description="Google Maps"),
                        ToolSpecV1(name="search_xiaohongshu", description="Xiaohongshu"),
                    ],
                    excluded=["search_xiaohongshu"],
                ),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),

        # ── Category C: Routing, CHEAP_FIRST & Safety Context Rules ────────────
        PresetDefinition(
            preset_id="cheap_first_safe_summary",
            category="C. Routing & Safety Context Rules",
            title="CHEAP_FIRST Routing with Clean Safety Context (Routes to Cheap Leg)",
            description="Enrolled tag 'conversation.summary' + clean single_domain safety context starts on cheap_route (e.g. openai/gpt-oss-20b) with baseline fallback.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-cheap-ok-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(role=RequestRole.USER, content="Summarize in one sentence: User likes sci-fi movies from the 1990s.")
                ],
                generation=GenerationConfigV1(temperature=0.2, max_output_tokens=80),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.CHEAP_FIRST,
                    cheap_route=RouteV1(
                        provider_id=prov,
                        model_id=_default_cheap_model(prov, mod),
                        policy_leg=PolicyLeg.CHEAP,
                    ),
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    **build_summary_safety_context(
                        [{"role": "user", "content": "I like 90s sci-fi movies", "metadata": {"domain": "movie"}}]
                    )
                ),
            ),
        ),
        PresetDefinition(
            preset_id="cheap_first_unsafe_itinerary",
            category="C. Routing & Safety Context Rules",
            title="CHEAP_FIRST Blocked by Itinerary State (Never-Cheapen Rule -> Baseline)",
            description="Even though mode=CHEAP_FIRST and cheap_route is set, contains_itinerary_state=True forces initial route to baseline_route.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-cheap-blocked-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(role=RequestRole.USER, content="Summarize day 1 of the Singapore itinerary: Marina Bay Sands -> Gardens by the Bay.")
                ],
                generation=GenerationConfigV1(temperature=0.2, max_output_tokens=80),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.CHEAP_FIRST,
                    cheap_route=RouteV1(
                        provider_id=prov,
                        model_id=_default_cheap_model(prov, mod),
                        policy_leg=PolicyLeg.CHEAP,
                    ),
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    **build_summary_safety_context(
                        [{"role": "user", "content": "Plan itinerary", "metadata": {"domain": "itinerary", "itinerary": True}}]
                    )
                ),
            ),
        ),
        PresetDefinition(
            preset_id="never_cheap_tag_guard",
            category="C. Routing & Safety Context Rules",
            title="CHEAP_FIRST Blocked by NEVER_CHEAP_ROUTE_TAGS (itinerary.extract_constraints)",
            description="High-risk policy tag refuses cheap_route even when PolicyContextV1 claims single_domain & contains_itinerary_state=False.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-never-cheap-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_ITINERARY_CONSTRAINTS,
                messages=[
                    MessageV1(role=RequestRole.USER, content="Extract time window constraints from: 'Start at 9am and finish by 6pm'.")
                ],
                generation=GenerationConfigV1(temperature=0.1, max_output_tokens=80),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.CHEAP_FIRST,
                    cheap_route=RouteV1(
                        provider_id=prov,
                        model_id=_default_cheap_model(prov, mod),
                        policy_leg=PolicyLeg.CHEAP,
                    ),
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),
        PresetDefinition(
            preset_id="multi_provider_fallback_ladder",
            category="C. Routing & Safety Context Rules",
            title="Multi-Provider 3-Step Route Ladder (Cheap -> Baseline -> Fallback)",
            description="Constructs a full 3-step route ladder across models/providers to inspect fallback ordering.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-ladder-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(role=RequestRole.USER, content="In one sentence, how does circuit breaking complement fallback routing?")
                ],
                generation=GenerationConfigV1(temperature=0.2, max_output_tokens=80),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.CHEAP_FIRST,
                    cheap_route=RouteV1(
                        provider_id=prov,
                        model_id=_default_cheap_model(prov, mod),
                        policy_leg=PolicyLeg.CHEAP,
                    ),
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                    fallback_routes=[
                        RouteV1(provider_id="openai", model_id="gpt-6-mini", policy_leg=PolicyLeg.FALLBACK),
                        RouteV1(provider_id="anthropic", model_id="claude-3-5-haiku-20241022", policy_leg=PolicyLeg.FALLBACK),
                    ],
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),

        # ── Category D: Provider-Specific & Streaming Execution ────────────────
        PresetDefinition(
            preset_id="streaming_token_delta",
            category="D. Provider-Specific & Streaming",
            title="Real-Time Token Streaming (invoke_llm_stream / invoke_llm_stream_async)",
            description="Streams delta tokens in real time across the route ladder and measures Time-To-First-Token (TTFT).",
            default_exec_mode="stream_sync",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-stream-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(
                        role=RequestRole.USER,
                        content="List 3 short bullet points on why streaming improves perceived latency in AI chat interfaces.",
                    )
                ],
                generation=GenerationConfigV1(temperature=0.3, max_output_tokens=120),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),
        PresetDefinition(
            preset_id="reasoning_model_adaptation",
            category="D. Provider-Specific & Streaming",
            title="Reasoning Model Wire Projection (o3 / o3-mini / fake-reasoning)",
            description="Targets a reasoning model (is_reasoning_model=True): gateway automatically strips temperature and maps max_output_tokens to extra_body.max_completion_tokens.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-reasoning-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_ITINERARY_REPAIR,
                messages=[
                    MessageV1(role=RequestRole.USER, content="Solve: If an itinerary has a 30-minute overlap between lunch and a museum tour, how should it be repaired?")
                ],
                generation=GenerationConfigV1(temperature=0.7, max_output_tokens=256),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(
                        provider_id="openai" if prov not in ("openai", "azure_openai", "fake_provider") else prov,
                        model_id="o3-mini" if prov != "fake_provider" else "fake-reasoning",
                        policy_leg=PolicyLeg.BASELINE,
                    ),
                    fallback_routes=[RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.FALLBACK)],
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=True,
                    request_mode=RequestMode.ITINERARY,
                ),
            ),
        ),
        PresetDefinition(
            preset_id="anthropic_messages_translation",
            category="D. Provider-Specific & Streaming",
            title="Anthropic Messages API Wire Translation (Claude 3.5 Haiku)",
            description="Routes to AnthropicAdapter (extracts system/developer messages to top-level system field, consolidates consecutive user turns).",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-anthropic-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(role=RequestRole.SYSTEM, content="You are a helpful travel assistant."),
                    MessageV1(role=RequestRole.DEVELOPER, content="Keep answers under 2 sentences."),
                    MessageV1(role=RequestRole.USER, content="Context: User is visiting Kyoto in autumn."),
                    MessageV1(role=RequestRole.USER, content="What is one must-see temple for autumn foliage?"),
                ],
                generation=GenerationConfigV1(temperature=0.3, max_output_tokens=100),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(
                        provider_id=ProviderId.ANTHROPIC.value,
                        model_id="claude-3-5-haiku-20241022",
                        policy_leg=PolicyLeg.BASELINE,
                    ),
                    fallback_routes=[RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.FALLBACK)],
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),

        # ── Category E: Boundary, Policy Gates & Error Handling ────────────────
        PresetDefinition(
            preset_id="output_token_clamping",
            category="E. Boundary, Policy Gates & Error Handling",
            title="Pre-Flight Output Token Clamping (max_output_tokens=32000 -> Clamped)",
            description="Requests 32,000 output tokens on a model with a 4,096/8,192 cap; Gateway clamps max_tokens automatically to prevent HTTP 400.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-clamp-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[
                    MessageV1(role=RequestRole.USER, content="Reply with a 1-sentence greeting.")
                ],
                generation=GenerationConfigV1(temperature=0.2, max_output_tokens=32000),
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),
        PresetDefinition(
            preset_id="policy_blocked_provider",
            category="E. Boundary, Policy Gates & Error Handling",
            title="Policy Gate: Disallowed Provider (provider_not_allowed)",
            description="Sets allowed_provider_ids=['openai'] while routing to 'groq' -> Gateway blocks pre-dispatch with PolicyViolationError.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-policy-block-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[MessageV1(role=RequestRole.USER, content="Test provider allowlist enforcement.")],
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                    allowed_provider_ids=["only_permitted_vendor"],
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),
        PresetDefinition(
            preset_id="privacy_restricted_mismatch",
            category="E. Boundary, Policy Gates & Error Handling",
            title="Privacy Gate: data_class='restricted' vs Standard Provider Policy",
            description="Sets PrivacyConfigV1(data_class='restricted') on a provider with standard retention -> fails closed with privacy_policy_mismatch.",
            builder=lambda prov, mod: LLMRequestV1(
                request_id=f"req-privacy-block-{uuid.uuid4().hex[:8]}",
                policy_tag=POLICY_TAG_CONVERSATION_SUMMARY,
                messages=[MessageV1(role=RequestRole.USER, content="Sensitive payload requiring zero data retention.")],
                routing_policy=RoutingPolicyV1(
                    mode=RoutingMode.BASELINE,
                    baseline_route=RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE),
                ),
                privacy=PrivacyConfigV1(
                    data_class="restricted",
                    allow_prompt_logging=False,
                    allow_response_cache=False,
                ),
                policy_context=PolicyContextV1(
                    contains_itinerary_state=False,
                    request_mode=RequestMode.SINGLE_DOMAIN,
                ),
            ),
        ),
    ]
    return presets


PRESETS_LIST: List[PresetDefinition] = _build_presets()
PRESETS_BY_ID: Dict[str, PresetDefinition] = {p.preset_id: p for p in PRESETS_LIST}


# ==============================================================================
# Inspection & Execution Helpers
# ==============================================================================

def inspect_request_and_routing(request: LLMRequestV1) -> None:
    """Display full LLMRequestV1 payload, route ladder resolution, pre-flight credentials, and wire projection."""
    registry = get_provider_registry()
    initial_route, initial_leg = determine_initial_route(request)
    ladder = build_route_ladder(request, initial_route, initial_leg)

    print("\n" + "=" * 88)
    print(f"  REQUEST & ROUTING INSPECTION (request_id={request.request_id})")
    print("=" * 88)

    # 1. Route Ladder & Credential Pre-Flight
    print("\n[1/3] Resolved Route Ladder & Pre-Flight Credential Check:")
    print(f"  • Policy Tag           : {request.policy_tag}")
    print(f"  • Routing Mode         : {request.routing_policy.mode.value}")
    print(f"  • Initial Route        : {initial_route.provider_id}:{initial_route.model_id} (leg={initial_leg.value})")
    if request.routing_policy.mode == RoutingMode.CHEAP_FIRST and initial_leg != PolicyLeg.CHEAP:
        print("  • Cheap-First Decision : BYPASSED -> Failed closed to BASELINE (safety context or tag rule)")
    elif initial_leg == PolicyLeg.CHEAP:
        print("  • Cheap-First Decision : ELIGIBLE -> Starting on CHEAP route")

    print("\n  Ordered Route Ladder:")
    for idx, (route, leg) in enumerate(ladder, 1):
        cred = inspect_provider_credentials(route.provider_id)
        badge = "CONFIGURED" if cred["configured"] else "MISSING CREDENTIALS"
        print(f"    {idx}. [{leg.value.upper():<8s}] {route.provider_id}:{route.model_id:<28s} -> [{badge}] ({cred['endpoint']})")
        if not cred["configured"]:
            print(f"       [!] Action Required: {cred['instructions']}")

    # 2. Projected Wire Payload for Initial Route
    print("\n[2/3] Projected Wire Kwargs (normalize_request_kwargs for Step 1):")
    adaptations: Dict[str, Any] = {"removed_params": []}
    wire_kwargs, sanitized_map, tool_proj = normalize_request_kwargs(request, initial_route, adaptations, registry)
    print(json.dumps(wire_kwargs, indent=2, ensure_ascii=False))
    if sanitized_map:
        print(f"  • Tool Name Wire Sanitization Mapping: {sanitized_map}")
    if tool_proj:
        print(f"  • WP7 Tool Projection ID: {tool_proj.projection_id} (catalog={tool_proj.catalog_id}, count={len(tool_proj.definitions)})")

    # 3. Full Serialized LLMRequestV1 Contract
    print("\n[3/3] Canonical Serialized LLMRequestV1 (Pydantic model_dump):")
    print(request.model_dump_json(indent=2))
    print("=" * 88 + "\n")


def execute_request_end_to_end(
    request: LLMRequestV1,
    *,
    exec_mode: str = "sync",
    dry_run: bool = False,
    preset: Optional[PresetDefinition] = None,
) -> Optional[LLMResultV1]:
    """Execute LLMRequestV1 through the gateway and print detailed telemetry."""
    inspect_request_and_routing(request)

    initial_route, initial_leg = determine_initial_route(request)
    ladder = build_route_ladder(request, initial_route, initial_leg)

    # Check if any route in ladder has live credentials when not in dry_run
    if not dry_run:
        missing_routes = [
            (r, inspect_provider_credentials(r.provider_id))
            for r, _ in ladder
            if not inspect_provider_credentials(r.provider_id)["configured"]
        ]
        if missing_routes and len(missing_routes) == len(ladder):
            print("!" * 88)
            print("  [PRE-FLIGHT WARNING] None of the routes in the ladder have live API credentials configured!")
            for r, info in missing_routes:
                print(f"    • {r.provider_id}: {info['instructions']}")
            print("  Proceeding will test the Gateway's live error handling (or pass --dry-run to simulate).")
            print("!" * 88 + "\n")

    ledger = UsageLedger()
    ledger_token = push_ledger(ledger)

    try:
        if dry_run:
            engine = create_engine()
            for route, _ in ladder:
                fake = FakeLLMAdapter()
                if preset and preset.dry_run_script is not None:
                    fake.enqueue_response(preset.dry_run_script(request))
                else:
                    fake.enqueue_response(
                        ScriptedResponse(
                            text=f"[DRY-RUN SIMULATED RESPONSE] Served by {route.provider_id}:{route.model_id} for policy_tag='{request.policy_tag}'.",
                            prompt_tokens=42,
                            completion_tokens=18,
                        )
                    )
                engine.register_adapter(route.provider_id, fake)
        else:
            engine = get_gateway_engine()

        print("=" * 88)
        print(f"  EXECUTING GATEWAY CALL | Mode: {exec_mode.upper()} | Target: {'DRY-RUN (FakeLLMAdapter)' if dry_run else 'LIVE UPSTREAM PROVIDER'}")
        print("=" * 88)

        # Streaming execution modes
        if exec_mode in ("stream_sync", "stream_async"):
            t0 = time.perf_counter()
            ttft_ms: Optional[float] = None
            chunks: List[str] = []
            print("\n  [Live Stream Output]:\n  ", end="", flush=True)
            try:
                if exec_mode == "stream_sync":
                    stream_iter = engine.invoke_llm_stream(request) if dry_run else invoke_llm_stream(request)
                    for token_chunk in stream_iter:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        chunks.append(token_chunk)
                        print(token_chunk, end="", flush=True)
                else:
                    async def _consume_async_stream():
                        nonlocal ttft_ms
                        aiter = engine.invoke_llm_stream_async(request) if dry_run else invoke_llm_stream_async(request)
                        async for token_chunk in aiter:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000.0
                            chunks.append(token_chunk)
                            print(token_chunk, end="", flush=True)

                    asyncio.run(_consume_async_stream())
                total_ms = (time.perf_counter() - t0) * 1000.0
                print("\n\n  [Stream Telemetry]:")
                print(f"    • Status           : SUCCEEDED")
                print(f"    • TTFT             : {ttft_ms:.2f} ms" if ttft_ms is not None else "    • TTFT             : N/A")
                print(f"    • Total Stream Time: {total_ms:.2f} ms")
                print(f"    • Chunks Received  : {len(chunks)}")
                print(f"    • Ledger Events    : {len(ledger.events)} physical attempt(s) recorded")
                if ledger.events:
                    ev = ledger.events[-1]
                    print(f"    • Recorded Route   : {ev.provider_id}:{ev.model} (leg={ev.policy_leg}, cost=${ev.cost_usd:.6f})")
                print("=" * 88 + "\n")
            except Exception as exc:
                total_ms = (time.perf_counter() - t0) * 1000.0
                print(f"\n\n  [Stream Error after {total_ms:.2f} ms]: {type(exc).__name__}: {exc}")
                print("=" * 88 + "\n")
            return None

        # Unary execution modes (sync, async, executor)
        t0 = time.perf_counter()
        if exec_mode == "async":
            result = asyncio.run(engine.invoke_llm_async(request) if dry_run else invoke_llm_async(request))
        elif exec_mode == "executor":
            result = asyncio.run(engine.invoke_llm_in_executor(request) if dry_run else invoke_llm_in_executor(request))
        else:
            result = engine.invoke_llm(request) if dry_run else invoke_llm(request)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        _print_result_telemetry(result, ledger, wall_ms)
        return result
    finally:
        reset_ledger(ledger_token)


def _print_result_telemetry(result: LLMResultV1, ledger: UsageLedger, wall_ms: float) -> None:
    """Render comprehensive LLMResultV1 telemetry, attempts, validation, ledger, and output."""
    print(f"\n  [Execution Result Summary]")
    print(f"    • Result Status    : {result.status.value.upper()}")
    if result.selected_route:
        print(
            f"    • Selected Route   : {result.selected_route.provider_id}:{result.selected_route.model_id} "
            f"(leg={result.selected_route.policy_leg.value})"
        )
    else:
        print("    • Selected Route   : None (No route succeeded)")
    print(f"    • Gateway Latency  : {result.timing.total_latency_ms:.2f} ms (Wall clock: {wall_ms:.2f} ms / Deadline: {result.timing.deadline_ms} ms)")
    print(
        f"    • Token Usage      : prompt={result.usage.prompt_tokens}, "
        f"completion={result.usage.completion_tokens}, total={result.usage.total_tokens}"
    )
    print(f"    • Total Cost (USD) : ${result.usage.cost_usd:.6f} (snapshot={result.usage.pricing_snapshot_id})")

    if result.degradation:
        from_r = f"{result.degradation.from_route.provider_id}:{result.degradation.from_route.model_id}" if result.degradation.from_route else "N/A"
        to_r = f"{result.degradation.to_route.provider_id}:{result.degradation.to_route.model_id}" if result.degradation.to_route else "N/A"
        print(f"    • Degradation      : {result.degradation.reason_code} ({from_r} -> {to_r})")

    print(f"\n  [Validation Gate Summary]")
    print(
        f"    • json_parse={result.validation.json_parse.value} | "
        f"json_schema={result.validation.json_schema.value} | "
        f"semantic={result.validation.semantic.value} (validator='{result.validation.validator_id}')"
    )
    if result.validation.errors:
        print(f"    • Validation Errors: {result.validation.errors}")

    print(f"\n  [Physical Attempt Trace ({len(result.attempts)} attempt(s))]")
    for att in result.attempts:
        print(
            f"    #{att.attempt_number} [{att.attempt_id}] {att.provider_id}:{att.model_id} "
            f"| leg={att.policy_leg.value} | kind={att.attempt_kind.value} "
            f"| outcome={att.outcome.value} (contract={att.contract_outcome.value}) "
            f"| http={att.http_status} | {att.latency_ms:.1f}ms | ${att.cost_usd:.6f} "
            f"| table={att.price_table_version}"
        )
        if att.applied_compatibility_changes:
            print(f"       Applied Adaptations: {att.applied_compatibility_changes}")
        if att.error_code:
            print(f"       Error [{att.error_code}]: {att.error_message}")

    print(f"\n  [Usage Ledger Verification (llm_usage.py)]")
    print(
        f"    • Recorded Events  : {len(ledger.events)} event(s) | totals={ledger.totals}\n"
        f"    • By Policy Tag    : {ledger.totals_by_tag()}\n"
        f"    • By Provider      : {ledger.totals_by_provider()}\n"
        f"    • By Policy Leg    : {ledger.totals_by_leg()}"
    )

    if result.output is not None:
        print(f"\n  [Released Output Payload (kind={result.output.kind.value})]")
        if result.output.kind == ResponseKind.JSON and result.output.json_payload is not None:
            print(json.dumps(result.output.json_payload, indent=2, ensure_ascii=False))
        elif result.output.kind == ResponseKind.TOOL_CALLS and result.output.tool_calls:
            for tc in result.output.tool_calls:
                print(f"    • Tool Call [{tc.id}]: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})")
        else:
            print(f"    {result.output.text}")

    if result.error is not None:
        print(f"\n  [Terminal Gateway Error Payload]")
        print(f"    • Code       : {result.error.code}")
        print(f"    • Class      : {result.error.error_class}")
        print(f"    • Retryable  : {result.error.retryable}")
        print(f"    • Message    : {result.error.message}")

    print("=" * 88 + "\n")


# ==============================================================================
# Interactive Option-Driven Parameter Configuration
# ==============================================================================

def _prompt_choice(title: str, options: List[Tuple[str, str]], default_idx: int = 1) -> str:
    """Present a numbered single-select menu and return the chosen key."""
    print(f"\n  --- {title} ---")
    for idx, (val, desc) in enumerate(options, 1):
        marker = "*" if idx == default_idx else " "
        print(f"   {marker}[{idx}] {val:<34s} {desc}")
    raw = input(f"  Select option [1-{len(options)}] (default={default_idx}): ").strip()
    if not raw:
        return options[default_idx - 1][0]
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
    except ValueError:
        for val, _ in options:
            if raw.lower() == val.lower():
                return val
    print("  [!] Invalid selection, keeping default.")
    return options[default_idx - 1][0]


def _prompt_bool(label: str, current: bool) -> bool:
    raw = input(f"  {label} [{'Y/n' if current else 'y/N'}]: ").strip().lower()
    if not raw:
        return current
    return raw in ("y", "yes", "1", "true", "on")


def _select_provider_and_model(role_label: str, default_provider: str, default_model: str) -> Tuple[str, str]:
    """Interactive provider & model selector backed by ProviderRegistry."""
    registry = get_provider_registry()
    providers = registry.list_providers()
    prov_options: List[Tuple[str, str]] = []
    default_p_idx = 1
    for idx, p in enumerate(providers, 1):
        cred = inspect_provider_credentials(p.provider_id)
        status = "READY" if cred["configured"] else "NO KEY"
        prov_options.append((p.provider_id, f"[{status}] adapter={p.adapter_kind} ({len(p.models)} models)"))
        if p.provider_id == default_provider:
            default_p_idx = idx

    chosen_prov = _prompt_choice(f"Select Provider for {role_label}", prov_options, default_idx=default_p_idx)
    cap = registry.get_provider(chosen_prov)
    model_options: List[Tuple[str, str]] = []
    default_m_idx = 1
    if cap and cap.models:
        for idx, m in enumerate(cap.models, 1):
            reas = " [reasoning]" if m.is_reasoning_model else ""
            model_options.append((m.model_id, f"ctx={m.context_window}, max_out={m.max_output_tokens}{reas}"))
            if m.model_id == default_model:
                default_m_idx = idx
    model_options.append(("__custom__", "Enter a custom model ID string"))

    chosen_model = _prompt_choice(f"Select Model for {chosen_prov} ({role_label})", model_options, default_idx=default_m_idx)
    if chosen_model == "__custom__":
        custom_m = input(f"  Enter custom model ID (default={default_model}): ").strip()
        chosen_model = custom_m or default_model
    return chosen_prov, chosen_model


def _edit_policy_tag(req: LLMRequestV1) -> LLMRequestV1:
    options = list(ALL_POLICY_TAGS) + [("__custom__", "Enter a custom policy_tag string")]
    default_idx = next((i for i, (k, _) in enumerate(options, 1) if k == req.policy_tag), 1)
    chosen = _prompt_choice("Select policy_tag", options, default_idx=default_idx)
    if chosen == "__custom__":
        chosen = input(f"  Enter custom policy_tag [{req.policy_tag}]: ").strip() or req.policy_tag
    new_id = input(f"  Enter request_id [{req.request_id}]: ").strip() or req.request_id
    return req.model_copy(update={"policy_tag": chosen, "request_id": new_id})


def _edit_messages(req: LLMRequestV1) -> LLMRequestV1:
    messages = list(req.messages)
    while True:
        print("\n  --- Current Messages ---")
        for i, m in enumerate(messages, 1):
            name_str = f" (name={m.name})" if m.name else ""
            preview = m.content if len(m.content) <= 75 else m.content[:72] + "..."
            print(f"    [{i}] {m.role.value.upper()}{name_str}: {preview}")
        print("\n    [A] Add message  | [E] Edit message | [D] Delete message | [R] Replace with single user prompt | [B] Done")
        act = input("  Select action [B]: ").strip().upper() or "B"
        if act == "B":
            break
        elif act == "R":
            text = input("  Enter user prompt: ").strip()
            if text:
                messages = [MessageV1(role=RequestRole.USER, content=text)]
        elif act == "A":
            role_opts = [(r.value, f"{r.name} role") for r in RequestRole]
            role_val = _prompt_choice("Select Message Role", role_opts, default_idx=3)
            content = input("  Enter message content: ").strip()
            name = input("  Optional tool/author name (press Enter for None): ").strip() or None
            messages.append(MessageV1(role=RequestRole(role_val), content=content, name=name))
        elif act == "E" and messages:
            idx_raw = input(f"  Message number to edit [1-{len(messages)}]: ").strip()
            if idx_raw.isdigit() and 1 <= int(idx_raw) <= len(messages):
                idx = int(idx_raw) - 1
                old = messages[idx]
                new_c = input(f"  New content [{old.content}]: ").strip() or old.content
                messages[idx] = old.model_copy(update={"content": new_c})
        elif act == "D" and len(messages) > 1:
            idx_raw = input(f"  Message number to delete [1-{len(messages)}]: ").strip()
            if idx_raw.isdigit() and 1 <= int(idx_raw) <= len(messages):
                messages.pop(int(idx_raw) - 1)
    return req.model_copy(update={"messages": messages})


def _edit_routing_policy(req: LLMRequestV1) -> LLMRequestV1:
    rp = req.routing_policy
    mode_opts = [
        (RoutingMode.BASELINE.value, "Start directly on baseline_route"),
        (RoutingMode.CHEAP_FIRST.value, "Start on cheap_route if tag & safety_context allow, else baseline_route"),
    ]
    default_m = 2 if rp.mode == RoutingMode.CHEAP_FIRST else 1
    chosen_mode = RoutingMode(_prompt_choice("Select RoutingMode", mode_opts, default_idx=default_m))

    b_prov, b_mod = _select_provider_and_model("Baseline Route", rp.baseline_route.provider_id, rp.baseline_route.model_id)
    baseline_route = RouteV1(provider_id=b_prov, model_id=b_mod, policy_leg=PolicyLeg.BASELINE)

    cheap_route = rp.cheap_route
    if chosen_mode == RoutingMode.CHEAP_FIRST or _prompt_bool("Configure a cheap_route?", cheap_route is not None):
        c_def_p = cheap_route.provider_id if cheap_route else b_prov
        c_def_m = cheap_route.model_id if cheap_route else ("llama-3.1-8b-instant" if b_prov == "groq" else b_mod)
        c_prov, c_mod = _select_provider_and_model("Cheap Route", c_def_p, c_def_m)
        cheap_route = RouteV1(provider_id=c_prov, model_id=c_mod, policy_leg=PolicyLeg.CHEAP)
    else:
        cheap_route = None

    fallback_routes = list(rp.fallback_routes)
    if _prompt_bool(f"Modify fallback_routes? (currently {len(fallback_routes)} configured)", False):
        fallback_routes = []
        while _prompt_bool("Add a fallback route to the ladder?", len(fallback_routes) == 0):
            f_prov, f_mod = _select_provider_and_model(f"Fallback Route #{len(fallback_routes) + 1}", "openai", "gpt-4o-mini")
            fallback_routes.append(RouteV1(provider_id=f_prov, model_id=f_mod, policy_leg=PolicyLeg.FALLBACK))

    allowed_ids = list(rp.allowed_provider_ids)
    raw_allowed = input(
        f"  Allowed provider IDs comma-separated (empty = allow all) [{','.join(allowed_ids)}]: "
    ).strip()
    if raw_allowed == "-":
        allowed_ids = []
    elif raw_allowed:
        allowed_ids = [x.strip() for x in raw_allowed.split(",") if x.strip()]

    new_rp = RoutingPolicyV1(
        mode=chosen_mode,
        baseline_route=baseline_route,
        cheap_route=cheap_route,
        fallback_routes=fallback_routes,
        allowed_provider_ids=allowed_ids,
    )
    return req.model_copy(update={"routing_policy": new_rp})


def _edit_generation(req: LLMRequestV1) -> LLMRequestV1:
    gen = req.generation
    t_raw = input(f"  Temperature (0.0 - 2.0, or 'none') [{gen.temperature}]: ").strip()
    if t_raw.lower() == "none":
        temp = None
    elif t_raw:
        temp = float(t_raw)
    else:
        temp = gen.temperature

    max_out_raw = input(f"  max_output_tokens [{gen.max_output_tokens}]: ").strip()
    max_out = int(max_out_raw) if max_out_raw.isdigit() else gen.max_output_tokens

    seed_raw = input(f"  seed (integer or 'none') [{gen.seed}]: ").strip()
    if seed_raw.lower() == "none":
        seed = None
    elif seed_raw.lstrip("-").isdigit():
        seed = int(seed_raw)
    else:
        seed = gen.seed

    stop_raw = input(f"  stop sequences comma-separated (or 'none') [{','.join(gen.stop or [])}]: ").strip()
    if stop_raw.lower() == "none":
        stop = None
    elif stop_raw:
        stop = [s for s in stop_raw.split(",") if s]
    else:
        stop = gen.stop

    return req.model_copy(
        update={
            "generation": GenerationConfigV1(
                temperature=temp,
                max_output_tokens=max_out,
                seed=seed,
                stop=stop,
            )
        }
    )


def _edit_response_contract(req: LLMRequestV1) -> LLMRequestV1:
    rc = req.response_contract
    kind_opts = [
        (ResponseKind.TEXT.value, "Free-form text response"),
        (ResponseKind.JSON.value, "Structured JSON validated against Draft-7 json_schema"),
        (ResponseKind.TOOL_CALLS.value, "Function/tool call invocation output"),
    ]
    default_k = next((i for i, (k, _) in enumerate(kind_opts, 1) if k == rc.kind.value), 1)
    chosen_kind = ResponseKind(_prompt_choice("Select ResponseKind", kind_opts, default_idx=default_k))

    schema_opts = [
        ("default.text.v1", "Default text contract"),
        (CONVERSATION_SUMMARY_SCHEMA_ID, "WP5-B Registered Conversation Summary Schema"),
        (CONFIRMATION_PAYLOAD_SCHEMA_ID, "WP5-B Registered Confirmation Payload Schema"),
        (LEGACY_SCHEMA_ID, "Legacy passthrough schema"),
        ("__custom__", "Enter custom schema_id"),
    ]
    chosen_schema_id = _prompt_choice("Select schema_id", schema_opts, default_idx=1)
    if chosen_schema_id == "__custom__":
        chosen_schema_id = input(f"  Enter custom schema_id [{rc.schema_id}]: ").strip() or rc.schema_id

    json_schema = rc.json_schema
    if chosen_kind == ResponseKind.JSON:
        if chosen_schema_id == CONFIRMATION_PAYLOAD_SCHEMA_ID:
            entry = resolve_response_schema(CONFIRMATION_PAYLOAD_SCHEMA_ID)
            json_schema = entry.schema() if entry else json_schema
        elif not json_schema:
            json_schema = {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            }
        raw_js = input(f"  Override json_schema JSON (press Enter to keep current): ").strip()
        if raw_js:
            json_schema = json.loads(raw_js)
    else:
        json_schema = None

    val_opts = [
        (SemanticValidatorId.NONE.value, "No semantic validator"),
        (SemanticValidatorId.GATHER_ACTION.value, "Verify proposed tool is in allowed_tools"),
        (SemanticValidatorId.CONFIRMATION_PAYLOAD.value, "Verify confirmation payload domain & quick_actions"),
    ]
    chosen_val = _prompt_choice("Select semantic_validator_id", val_opts, default_idx=1)
    allow_empty = _prompt_bool("allow_empty response?", rc.allow_empty)

    val_inputs = dict(rc.validator_inputs)
    if chosen_val == SemanticValidatorId.CONFIRMATION_PAYLOAD.value and "allowed_preference_patch_keys" not in val_inputs:
        val_inputs = {"allowed_preference_patch_keys": ["use_case"]}

    return req.model_copy(
        update={
            "response_contract": ResponseContractV1(
                kind=chosen_kind,
                schema_id=chosen_schema_id,
                json_schema=json_schema,
                semantic_validator_id=chosen_val,
                validator_inputs=val_inputs,
                allow_empty=allow_empty,
            )
        }
    )


def _edit_policy_context(req: LLMRequestV1) -> LLMRequestV1:
    pc = req.policy_context
    has_itin = _prompt_bool("contains_itinerary_state?", pc.contains_itinerary_state)
    mode_opts = [(m.value, f"RequestMode.{m.name}") for m in RequestMode]
    default_m = next((i for i, (k, _) in enumerate(mode_opts, 1) if k == pc.request_mode.value), 1)
    chosen_mode = RequestMode(_prompt_choice("Select request_mode", mode_opts, default_idx=default_m))

    safety_opts = [
        ("standard", "Standard clean context"),
        ("itinerary", "Itinerary context (blocks cheap route)"),
        ("unknown", "Unknown/unclassified context (fails closed to baseline)"),
    ]
    chosen_safety = _prompt_choice("Select safety_class", safety_opts, default_idx=1)

    prod_opts = [(p, "Registered context producer") for p in sorted(REGISTERED_CONTEXT_PRODUCERS)] + [
        ("unregistered_producer", "Unregistered producer (tests fail-closed freshness check)")
    ]
    chosen_prod = _prompt_choice("Select producer_id", prod_opts, default_idx=1)

    stale = _prompt_bool("Simulate stale timestamp (>300s old to test TTL rejection)?", False)
    ts = (time.time() - 600.0) if stale else time.time()

    return req.model_copy(
        update={
            "policy_context": PolicyContextV1(
                contains_itinerary_state=has_itin,
                request_mode=chosen_mode,
                safety_class=chosen_safety,
                producer_id=chosen_prod,
                context_version=pc.context_version,
                produced_at_utc=ts,
            )
        }
    )


def _edit_budget(req: LLMRequestV1) -> LLMRequestV1:
    b = req.budget
    dl = input(f"  deadline_ms [{b.deadline_ms}]: ").strip()
    att = input(f"  max_physical_attempts (1-10) [{b.max_physical_attempts}]: ").strip()
    cost = input(f"  max_cost_usd [{b.max_cost_usd}]: ").strip()
    max_in = input(f"  max_input_tokens (or 'none') [{b.max_input_tokens}]: ").strip()
    max_tot = input(f"  max_total_tokens (or 'none') [{b.max_total_tokens}]: ").strip()

    return req.model_copy(
        update={
            "budget": BudgetConfigV1(
                deadline_ms=int(dl) if dl.isdigit() else b.deadline_ms,
                max_physical_attempts=int(att) if att.isdigit() else b.max_physical_attempts,
                max_output_tokens=b.max_output_tokens,
                max_cost_usd=float(cost) if cost else b.max_cost_usd,
                max_input_tokens=None if max_in.lower() == "none" else (int(max_in) if max_in.isdigit() else b.max_input_tokens),
                max_total_tokens=None if max_tot.lower() == "none" else (int(max_tot) if max_tot.isdigit() else b.max_total_tokens),
            )
        }
    )


def _edit_privacy_and_trace(req: LLMRequestV1) -> LLMRequestV1:
    priv = req.privacy
    dc_opts = [
        ("standard", "Standard data retention permitted"),
        ("restricted", "Requires zero_data_retention or restricted_approved provider policy"),
    ]
    dc = _prompt_choice("Select PrivacyConfigV1.data_class", dc_opts, default_idx=1)
    log_ok = _prompt_bool("allow_prompt_logging?", priv.allow_prompt_logging)
    cache_ok = _prompt_bool("allow_response_cache?", priv.allow_response_cache)

    trace_id = input(f"  trace_id [{req.trace_context.trace_id if req.trace_context else ''}]: ").strip()
    trace = TraceContextV1(trace_id=trace_id) if trace_id else req.trace_context

    return req.model_copy(
        update={
            "privacy": PrivacyConfigV1(
                data_class=dc,
                allow_prompt_logging=log_ok,
                allow_response_cache=cache_ok,
            ),
            "trace_context": trace,
        }
    )


def interactive_edit_before_send_loop(
    request: LLMRequestV1,
    *,
    exec_mode: str = "sync",
    dry_run: bool = False,
    preset: Optional[PresetDefinition] = None,
) -> None:
    """Interactive dashboard allowing inspection, field tweaking, JSON export, and live execution."""
    curr_req = request
    curr_mode = exec_mode
    curr_dry = dry_run

    while True:
        initial_route, initial_leg = determine_initial_route(curr_req)
        ladder = build_route_ladder(curr_req, initial_route, initial_leg)
        ladder_str = " -> ".join(f"{r.provider_id}:{r.model_id}[{leg.value}]" for r, leg in ladder)

        print("\n" + "=" * 88)
        print("  CONFIGURE & SEND LLMRequestV1 (Edit-Before-Send Dashboard)")
        print("=" * 88)
        print(f"  [1] policy_tag & request_id : tag='{curr_req.policy_tag}' | id='{curr_req.request_id}'")
        print(f"  [2] messages                : {len(curr_req.messages)} message(s) (first='{curr_req.messages[0].content[:48]}...')")
        print(f"  [3] routing_policy          : mode={curr_req.routing_policy.mode.value} | ladder: {ladder_str}")
        print(
            f"  [4] generation              : temp={curr_req.generation.temperature}, "
            f"max_out={curr_req.generation.max_output_tokens}, seed={curr_req.generation.seed}, stop={curr_req.generation.stop}"
        )
        print(
            f"  [5] response_contract       : kind={curr_req.response_contract.kind.value}, "
            f"schema_id='{curr_req.response_contract.schema_id}', validator='{curr_req.response_contract.semantic_validator_id}'"
        )
        print(
            f"  [6] policy_context          : itinerary={curr_req.policy_context.contains_itinerary_state}, "
            f"mode={curr_req.policy_context.request_mode.value}, safety={curr_req.policy_context.safety_class}, "
            f"producer={curr_req.policy_context.producer_id}"
        )
        print(
            f"  [7] budget                  : deadline={curr_req.budget.deadline_ms}ms, "
            f"max_attempts={curr_req.budget.max_physical_attempts}, max_cost=${curr_req.budget.max_cost_usd}"
        )
        print(
            f"  [8] privacy & trace         : data_class='{curr_req.privacy.data_class}', "
            f"prompt_log={curr_req.privacy.allow_prompt_logging}"
        )
        print(
            f"  [9] Execution Mode & Target : exec_mode='{curr_mode}' | "
            f"target={'DRY-RUN (FakeLLMAdapter)' if curr_dry else 'LIVE UPSTREAM PROVIDER'}"
        )
        print("-" * 88)
        print("  [V] View Full JSON & Wire Projection  | [E] Export JSON Fixture  | [R] RUN / SEND NOW  | [B] Back")
        print("=" * 88)

        choice = input("  Choose field to edit or action [R]: ").strip().upper() or "R"
        try:
            if choice == "B":
                return
            elif choice == "R":
                execute_request_end_to_end(curr_req, exec_mode=curr_mode, dry_run=curr_dry, preset=preset)
                again = _prompt_bool("Stay in Edit-Before-Send dashboard for this request?", True)
                if not again:
                    return
            elif choice == "V":
                inspect_request_and_routing(curr_req)
            elif choice == "E":
                out_path = input(f"  Export file path [{curr_req.request_id}.json]: ").strip() or f"{curr_req.request_id}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(curr_req.model_dump_json(indent=2))
                print(f"  [+] Saved LLMRequestV1 JSON fixture to: {os.path.abspath(out_path)}")
            elif choice == "1":
                curr_req = _edit_policy_tag(curr_req)
            elif choice == "2":
                curr_req = _edit_messages(curr_req)
            elif choice == "3":
                curr_req = _edit_routing_policy(curr_req)
            elif choice == "4":
                curr_req = _edit_generation(curr_req)
            elif choice == "5":
                curr_req = _edit_response_contract(curr_req)
            elif choice == "6":
                curr_req = _edit_policy_context(curr_req)
            elif choice == "7":
                curr_req = _edit_budget(curr_req)
            elif choice == "8":
                curr_req = _edit_privacy_and_trace(curr_req)
            elif choice == "9":
                curr_mode = _prompt_choice("Select Execution Mode", EXECUTION_MODES, default_idx=1)
                curr_dry = _prompt_bool("Use offline --dry-run FakeLLMAdapter instead of live upstream?", curr_dry)
        except Exception as exc:
            print(f"\n  [!] Validation error while updating LLMRequestV1: {exc}")


def run_interactive_main_menu(default_prov: str, default_mod: str, initial_dry_run: bool = False) -> int:
    """Primary interactive menu loop."""
    while True:
        print("\n" + "=" * 88)
        print("  WP5 LLM GATEWAY — INTERACTIVE END-TO-END CLI TEST RUNNER")
        print(f"  Default Live Route: {default_prov}:{default_mod}")
        print("=" * 88)

        current_cat = ""
        for idx, p in enumerate(PRESETS_LIST, 1):
            if p.category != current_cat:
                current_cat = p.category
                print(f"\n  [{current_cat}]")
            print(f"    {idx:2d}. {p.preset_id:<32s} — {p.title}")

        print("\n  [Other Actions]")
        print("     C. Build Custom / Blank LLMRequestV1 From Scratch")
        print("     L. Load LLMRequestV1 From JSON Fixture File")
        print("     P. Inspect Provider Registry, Pricing Snapshot & Live .env Credentials")
        print("     Q. Quit")
        print("=" * 88)

        sel = input("  Select a preset [1-18] or action [C/L/P/Q]: ").strip()
        if not sel or sel.upper() == "Q":
            return 0
        if sel.upper() == "P":
            print_environment_and_registry_status()
            continue
        if sel.upper() == "L":
            fpath = input("  Enter JSON file path to load: ").strip()
            if os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    loaded_req = LLMRequestV1.model_validate_json(f.read())
                interactive_edit_before_send_loop(loaded_req, dry_run=initial_dry_run)
            else:
                print(f"  [!] File not found: {fpath}")
            continue
        if sel.upper() == "C":
            base_req = PRESETS_BY_ID["minimal_text"].builder(default_prov, default_mod)
            base_req = _edit_policy_tag(base_req)
            base_req = _edit_routing_policy(base_req)
            base_req = _edit_messages(base_req)
            interactive_edit_before_send_loop(base_req, dry_run=initial_dry_run)
            continue

        preset_chosen: Optional[PresetDefinition] = None
        if sel.isdigit() and 1 <= int(sel) <= len(PRESETS_LIST):
            preset_chosen = PRESETS_LIST[int(sel) - 1]
        elif sel in PRESETS_BY_ID:
            preset_chosen = PRESETS_BY_ID[sel]

        if preset_chosen is None:
            print(f"  [!] Unrecognized selection: {sel}")
            continue

        print(f"\n  Loaded Preset: {preset_chosen.preset_id} — {preset_chosen.title}")
        print(f"  Description  : {preset_chosen.description}")
        req = preset_chosen.builder(default_prov, default_mod)
        interactive_edit_before_send_loop(
            req,
            exec_mode=preset_chosen.default_exec_mode,
            dry_run=initial_dry_run,
            preset=preset_chosen,
        )


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def print_presets_table() -> None:
    print("\n" + "=" * 96)
    print("  DISCOVERED LLMRequestV1 TEST CASE PRESETS (18 Presets)")
    print("=" * 96)
    curr_cat = ""
    for idx, p in enumerate(PRESETS_LIST, 1):
        if p.category != curr_cat:
            curr_cat = p.category
            print(f"\n  {curr_cat}")
            print("  " + "-" * 90)
        print(f"  {idx:2d}. {p.preset_id:<32s} | mode={p.default_exec_mode:<11s} | {p.title}")
        print(f"      {p.description}")
    print("=" * 96 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive & CLI End-to-End Test Runner for WP5 LLMRequestV1."
    )
    parser.add_argument("--list-presets", action="store_true", help="List all discovered LLMRequestV1 test presets and exit")
    parser.add_argument("--preset", type=str, default=None, help="Preset ID to run (or 'all' to run all presets)")
    parser.add_argument("--provider", type=str, default=None, help="Override baseline provider_id (e.g. groq, openai, azure_openai, anthropic)")
    parser.add_argument("--model", type=str, default=None, help="Override baseline model_id (e.g. llama-3.3-70b-versatile)")
    parser.add_argument(
        "--exec-mode",
        choices=[m for m, _ in EXECUTION_MODES],
        default=None,
        help="Execution mode: sync, async, executor, stream_sync, stream_async",
    )
    parser.add_argument("--stream", action="store_true", help="Shortcut to run in streaming mode")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Shortcut to run in async mode")
    parser.add_argument("--dry-run", action="store_true", help="Validate and simulate execution using FakeLLMAdapter without live network calls")
    parser.add_argument("--export", type=str, default=None, help="Export the configured LLMRequestV1 to a JSON file")
    parser.add_argument("--import-file", type=str, default=None, help="Load an LLMRequestV1 from a JSON file and execute/inspect it")
    parser.add_argument("--interactive", action="store_true", help="Open the Edit-Before-Send dashboard even when --preset or --import-file is given")
    parser.add_argument("--status", action="store_true", help="Print provider registry, pricing, and .env credential status and exit")
    args = parser.parse_args()

    if args.list_presets:
        print_presets_table()
        return 0

    if args.status:
        print_environment_and_registry_status()
        return 0

    default_prov, default_mod = _default_live_provider_and_model()
    prov = args.provider or default_prov
    mod = args.model or default_mod

    # Determine execution mode override
    exec_mode_override = args.exec_mode
    if exec_mode_override is None:
        if args.stream and args.use_async:
            exec_mode_override = "stream_async"
        elif args.stream:
            exec_mode_override = "stream_sync"
        elif args.use_async:
            exec_mode_override = "async"

    # Handle --import-file
    if args.import_file:
        with open(args.import_file, "r", encoding="utf-8") as f:
            req = LLMRequestV1.model_validate_json(f.read())
        mode = exec_mode_override or "sync"
        if args.export:
            with open(args.export, "w", encoding="utf-8") as out_f:
                out_f.write(req.model_dump_json(indent=2))
            print(f"[+] Exported LLMRequestV1 to {os.path.abspath(args.export)}")
        if args.interactive:
            interactive_edit_before_send_loop(req, exec_mode=mode, dry_run=args.dry_run)
        else:
            execute_request_end_to_end(req, exec_mode=mode, dry_run=args.dry_run)
        return 0

    # Handle --preset
    if args.preset:
        if args.preset.lower() == "all":
            print(f"\nRunning all {len(PRESETS_LIST)} presets (dry_run={args.dry_run}, provider={prov}, model={mod})...\n")
            for p in PRESETS_LIST:
                req = p.builder(prov, mod)
                mode = exec_mode_override or p.default_exec_mode
                execute_request_end_to_end(req, exec_mode=mode, dry_run=args.dry_run, preset=p)
            return 0

        if args.preset not in PRESETS_BY_ID:
            print(f"[ERROR] Unknown preset '{args.preset}'. Use --list-presets to see valid IDs.")
            return 1

        preset = PRESETS_BY_ID[args.preset]
        req = preset.builder(prov, mod)
        if args.provider or args.model:
            new_base = RouteV1(provider_id=prov, model_id=mod, policy_leg=PolicyLeg.BASELINE)
            req = req.model_copy(
                update={"routing_policy": req.routing_policy.model_copy(update={"baseline_route": new_base})}
            )
        mode = exec_mode_override or preset.default_exec_mode

        if args.export:
            with open(args.export, "w", encoding="utf-8") as out_f:
                out_f.write(req.model_dump_json(indent=2))
            print(f"[+] Exported LLMRequestV1 to {os.path.abspath(args.export)}")

        if args.interactive:
            interactive_edit_before_send_loop(req, exec_mode=mode, dry_run=args.dry_run, preset=preset)
        else:
            execute_request_end_to_end(req, exec_mode=mode, dry_run=args.dry_run, preset=preset)
        return 0

    # Default: Interactive Menu
    return run_interactive_main_menu(prov, mod, initial_dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
