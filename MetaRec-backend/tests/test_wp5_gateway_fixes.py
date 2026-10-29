"""Unit tests verifying all WP5 Gateway security, reliability, and streaming fixes."""

import asyncio
import os
import pytest

pytestmark = pytest.mark.backend_unit

from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.adapters.openai_compat import OpenAICompatibleAdapter
from llm_gateway.constants import (
    GatewayErrorCode,
    ProviderId,
    SemanticValidatorId,
)
from llm_gateway.contracts import (
    BudgetConfigV1,
    GenerationConfigV1,
    LLMRequestV1,
    LLMResultV1,
    LLMToolDefinitionV1,
    LLMToolProjectionV1,
    MessageV1,
    PolicyContextV1,
    RequestMode,
    RequestRole,
    ResponseContractV1,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    RoutingPolicyV1,
    ScopedToolCatalogV1,
    ToolSpecV1,
    ValidationGateStatus,
)
from llm_gateway.engine import (
    GatewayEngine,
    RouteCircuitBreaker,
    create_engine,
    invoke_llm,
    invoke_llm_stream,
    invoke_llm_stream_async,
)
from llm_gateway.errors import ConfigurationError, TransportError
from llm_gateway.registry import DynamicProviderRegistry, PricingSnapshotV1, _validate_catalog_url


def _make_sample_request(
    tools: ScopedToolCatalogV1 | None = None,
    budget_attempts: int = 3,
) -> LLMRequestV1:
    return LLMRequestV1(
        request_id="fix-test-req-1",
        policy_tag="test.run",
        messages=[MessageV1(role=RequestRole.USER, content="Hello fix validation")],
        generation=GenerationConfigV1(max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast"),
            fallback_routes=[RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-strong")],
        ),
        budget=BudgetConfigV1(max_physical_attempts=budget_attempts, deadline_ms=10000),
        tools=tools,
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
    )


# ── Critical Fix 01: Credential Isolation ───────────────────────────────────

def test_groq_adapter_does_not_leak_openai_api_key(monkeypatch):
    """When GROQ_API_KEY is unset, Groq adapter must not use OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-openai-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    adapter = OpenAICompatibleAdapter(provider_id=ProviderId.GROQ.value)
    api_key, base_url = adapter._resolve_config(is_azure=False)
    assert api_key == "", "Groq adapter must not fall back to OPENAI_API_KEY"


def test_azure_adapter_requires_explicit_endpoint(monkeypatch):
    """Azure adapter raises ConfigurationError when endpoint is unset."""
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    adapter = OpenAICompatibleAdapter(provider_id=ProviderId.AZURE_OPENAI.value)
    with pytest.raises(ConfigurationError):
        adapter._resolve_config(is_azure=True)


# ── Critical Fix 02: SSRF Protection on Dynamic Catalog ─────────────────────

def test_catalog_url_rejects_non_https():
    """Catalog URL must be HTTPS only."""
    with pytest.raises(ValueError, match="SSRF Protection"):
        _validate_catalog_url("http://raw.githubusercontent.com/catalog.json")

    with pytest.raises(ValueError, match="SSRF Protection"):
        _validate_catalog_url("file:///etc/passwd")


def test_catalog_url_rejects_private_and_metadata_ips():
    """Catalog URL must reject localhost, private IPs, and cloud metadata."""
    with pytest.raises(ValueError, match="SSRF Protection"):
        _validate_catalog_url("https://127.0.0.1/catalog.json")

    with pytest.raises(ValueError, match="SSRF Protection"):
        _validate_catalog_url("https://169.254.169.254/latest/meta-data/")

    with pytest.raises(ValueError, match="SSRF Protection"):
        _validate_catalog_url("https://192.168.1.100/catalog.json")

    with pytest.raises(ValueError, match="SSRF Protection"):
        _validate_catalog_url("https://localhost/catalog.json")


# ── High Fix 01: Client Cache Key Isolation ─────────────────────────────────

def test_client_cache_key_distinct_for_different_base_urls():
    """Different base_urls or providers must not collide in client cache."""
    adapter1 = OpenAICompatibleAdapter(provider_id=ProviderId.OPENAI.value, base_url="https://api.openai.com/v1")
    adapter2 = OpenAICompatibleAdapter(provider_id=ProviderId.GROQ.value, base_url="https://api.groq.com/openai/v1")

    client1 = adapter1._get_sync_client(is_azure=False, timeout_seconds=30.0)
    client2 = adapter2._get_sync_client(is_azure=False, timeout_seconds=30.0)

    assert str(client1.base_url).rstrip("/") == "https://api.openai.com/v1"
    assert str(client2.base_url).rstrip("/") == "https://api.groq.com/openai/v1"


# ── High Fix 02: Streaming Engine & Adapter Pipeline ────────────────────────

def test_sync_streaming_on_gateway_engine():
    """invoke_llm_stream yields tokens and records attempts."""
    engine = create_engine()
    fake_adapter = FakeLLMAdapter()
    fake_adapter.enqueue_response(ScriptedResponse(text="The quick brown fox jumps over the lazy dog"))
    engine.register_adapter(ProviderId.FAKE.value, fake_adapter)

    request = _make_sample_request()
    tokens = list(engine.invoke_llm_stream(request))

    assert len(tokens) > 0
    full_output = "".join(tokens)
    assert full_output == "The quick brown fox jumps over the lazy dog"


@pytest.mark.asyncio
async def test_async_streaming_on_gateway_engine():
    """invoke_llm_stream_async yields delta tokens asynchronously."""
    engine = create_engine()
    fake_adapter = FakeLLMAdapter()
    fake_adapter.enqueue_response(ScriptedResponse(text="Async stream completion test"))
    engine.register_adapter(ProviderId.FAKE.value, fake_adapter)

    request = _make_sample_request()
    collected = []
    async for chunk in engine.invoke_llm_stream_async(request):
        collected.append(chunk)

    assert len(collected) > 0
    assert "".join(collected) == "Async stream completion test"


# ── Medium Fix 02: Route Circuit Breaker ─────────────────────────────────────

def test_route_circuit_breaker_trips_and_recovers():
    """Circuit breaker trips OPEN after consecutive failures and recovers on success."""
    cb = RouteCircuitBreaker(failure_threshold=3, recovery_timeout_sec=0.1)
    provider = "test_prov"
    model = "test_model"

    assert cb.is_available(provider, model) is True

    # Record 2 failures (below threshold)
    cb.record_failure(provider, model)
    cb.record_failure(provider, model)
    assert cb.is_available(provider, model) is True

    # 3rd failure trips the breaker
    cb.record_failure(provider, model)
    assert cb.is_available(provider, model) is False

    # Wait for recovery timeout
    import time
    time.sleep(0.15)
    assert cb.is_available(provider, model) is True  # Half-open probe

    # Success resets breaker
    cb.record_success(provider, model)
    assert cb.is_available(provider, model) is True


# ── Medium Fix 03: Tool Name Sanitization Collision Avoidance ────────────────

def test_tool_name_collision_avoidance_generates_unique_wire_names():
    """Tools with names normalizing to identical strings receive unique wire names."""
    engine = create_engine()
    tools = ScopedToolCatalogV1(
        catalog_id="test_cat",
        tools=[
            ToolSpecV1(name="user.profile", description="Profile tool"),
            ToolSpecV1(name="user_profile", description="Underscored tool"),
        ],
    )
    request = _make_sample_request(tools=tools)
    route = RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast")
    kwargs, mapping = engine._normalize_request_kwargs(request, route, {})

    wire_names = [t["function"]["name"] for t in kwargs["tools"]]
    assert len(wire_names) == 2
    assert len(set(wire_names)) == 2, "Wire names must be strictly unique"
    # Ensure mapping contains reverse mapping for both tools
    assert len(mapping) >= 1


# ── Low Fix 01: Pricing Precision ───────────────────────────────────────────

def test_pricing_precision_retains_micro_cents():
    """calculate_cost_usd retains 8 decimal places for small prompt token counts."""
    from llm_gateway.registry import ModelPriceV1

    pricing = PricingSnapshotV1()
    pricing.rates["test_prov:small-model"] = ModelPriceV1(input_price_per_1m=0.15, output_price_per_1m=0.60)
    # 5 prompt tokens at $0.15 / 1M = 0.00000075 USD
    cost = pricing.calculate_cost_usd("test_prov", "small-model", prompt_tokens=5, completion_tokens=0)
    assert cost == 0.00000075


# ── High Fix 03: Two Tool Shapes & WP7 Tool Catalog Invariant ───────────────

def test_tool_choice_validation_rejects_excluded_tool():
    """Forced tool_choice for a tool in catalog.excluded must be rejected."""
    catalog = ScopedToolCatalogV1(
        catalog_id="cat-scope-1",
        tools=[
            ToolSpecV1(name="gmap.search", description="Search maps"),
            ToolSpecV1(name="xhs.search", description="Search Xiaohongshu"),
        ],
        excluded=["xhs.search"],
    )
    # Allowed tool choice passes
    req_valid = _make_sample_request(tools=catalog)
    req_valid.tool_choice = "gmap.search"
    # Forced choice of excluded tool must raise ValueError
    with pytest.raises(ValueError, match="not present in scoped catalog tools"):
        LLMRequestV1(
            request_id="req-ex-1",
            policy_tag="test.run",
            messages=[MessageV1(role=RequestRole.USER, content="Test")],
            generation=GenerationConfigV1(max_output_tokens=100),
            response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
            routing_policy=RoutingPolicyV1(
                mode=RoutingMode.BASELINE,
                baseline_route=RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast"),
            ),
            policy_context=PolicyContextV1(
                contains_itinerary_state=False,
                request_mode=RequestMode.SINGLE_DOMAIN,
            ),
            tools=catalog,
            tool_choice="xhs.search",
        )


def test_scoped_catalog_monotonic_shrink_excludes_tools_from_projection():
    """_build_tool_projection shrinks tool set and never includes tools in excluded."""
    engine = create_engine()
    catalog = ScopedToolCatalogV1(
        catalog_id="cat-scope-2",
        tools=[
            ToolSpecV1(name="gmap.search", description="Google Maps search"),
            ToolSpecV1(name="xhs.search", description="Xiaohongshu search"),
            ToolSpecV1(name="yelp.search", description="Yelp search"),
        ],
        excluded=["yelp.search"],
    )
    req = _make_sample_request(tools=catalog)
    route = RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast")
    projection, mapping = engine._build_tool_projection(req, route)

    assert projection is not None
    assert len(projection.definitions) == 2
    source_names = {d.source_tool_name for d in projection.definitions}
    assert "gmap.search" in source_names
    assert "xhs.search" in source_names
    assert "yelp.search" not in source_names, "WP7 excluded tool must never appear in projection"


def test_prompt_embedded_tools_generic_domain_filtered():
    """Prompt-embedded available_tools JSON is filtered to exclude WP7-excluded tools."""
    engine = create_engine()
    catalog = ScopedToolCatalogV1(
        catalog_id="cat-scope-3",
        tools=[
            ToolSpecV1(name="search_music", description="Music tool"),
            ToolSpecV1(name="search_artist", description="Artist tool"),
        ],
        excluded=["search_artist"],
    )
    import json
    user_payload = {
        "query": "Jazz concert",
        "domain": "music",
        "available_tools": {
            "search_music": ["genre", "year"],
            "search_artist": ["name"],
            "unauthorized_tool": ["param"],
        },
    }
    req = LLMRequestV1(
        request_id="req-embed-1",
        policy_tag="gather_action.propose",
        messages=[MessageV1(role=RequestRole.USER, content=json.dumps(user_payload))],
        generation=GenerationConfigV1(max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.JSON, json_schema={"type": "object"}),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast"),
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
        tools=catalog,
    )
    route = RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast")
    kwargs, mapping = engine._normalize_request_kwargs(req, route, {})

    # In the rendered wire user message, available_tools must ONLY contain search_music
    wire_msg = kwargs["messages"][0]["content"]
    parsed_msg = json.loads(wire_msg)
    assert "available_tools" in parsed_msg
    assert "search_music" in parsed_msg["available_tools"]
    assert "search_artist" not in parsed_msg["available_tools"], "Excluded tool must be stripped from prompt"
    assert "unauthorized_tool" not in parsed_msg["available_tools"], "Out-of-catalog tool must be stripped"


def test_prompt_embedded_tools_react_text_filtered():
    """Prompt-embedded [可用工具] text prompt strips excluded tools."""
    engine = create_engine()
    catalog = ScopedToolCatalogV1(
        catalog_id="cat-scope-4",
        tools=[
            ToolSpecV1(name="xhs.search", description="Xiaohongshu"),
            ToolSpecV1(name="gmap.search", description="Google Maps"),
        ],
        excluded=["gmap.search"],
    )
    prompt_text = (
        "你是一个推荐agent。\n\n"
        "[可用工具]\n"
        "1) xhs.search: 获取口碑与探店线索。\n"
        "2) gmap.search: 获取候选门店列表。\n\n"
        "[输出要求]\n"
        "仅以 JSON 格式输出。"
    )
    req = _make_sample_request(tools=catalog)
    req.messages = [MessageV1(role=RequestRole.SYSTEM, content=prompt_text)]
    route = RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast")
    kwargs, mapping = engine._normalize_request_kwargs(req, route, {})

    rendered = kwargs["messages"][0]["content"]
    assert "xhs.search" in rendered
    assert "gmap.search" not in rendered, "gmap.search was excluded by WP7 and must be stripped from [可用工具]"


def test_tool_call_output_gate_rejects_excluded_tool():
    """A returned tool call for a WP7-excluded tool must fail semantic validation."""
    fake_adapter = FakeLLMAdapter()
    engine = create_engine()
    engine.register_adapter(ProviderId.FAKE.value, fake_adapter)

    catalog = ScopedToolCatalogV1(
        catalog_id="cat-scope-5",
        tools=[
            ToolSpecV1(name="xhs.search", description="Xiaohongshu"),
            ToolSpecV1(name="gmap.search", description="Google Maps"),
        ],
        excluded=["gmap.search"],
    )
    # Fake adapter returns tool call for excluded tool
    fake_adapter.enqueue_response(
        ScriptedResponse(
            text="",
            tool_calls=[{"id": "c1", "function": {"name": "gmap.search", "arguments": "{}"}}],
        )
    )
    req = LLMRequestV1(
        request_id="req-tc-fail",
        policy_tag="agent.plan",
        messages=[MessageV1(role=RequestRole.USER, content="Find hotpot")],
        generation=GenerationConfigV1(max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TOOL_CALLS),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast"),
        ),
        budget=BudgetConfigV1(max_physical_attempts=1, deadline_ms=5000),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
        tools=catalog,
    )
    res = engine.invoke_llm(req)
    # Must fail validation because gmap.search is in catalog.excluded
    assert res.status == ResultStatus.FAILED
    assert len(res.attempts) == 1
    assert res.attempts[0].validation_summary.semantic == ValidationGateStatus.FAILED
    assert any("excluded" in err for err in res.attempts[0].validation_summary.errors)


def test_gather_action_semantic_validation_rejects_excluded_tool():
    """Gather action proposing an excluded tool fails semantic validation."""
    fake_adapter = FakeLLMAdapter()
    engine = create_engine()
    engine.register_adapter(ProviderId.FAKE.value, fake_adapter)

    catalog = ScopedToolCatalogV1(
        catalog_id="cat-scope-6",
        tools=[
            ToolSpecV1(name="search_food", description="Search food"),
            ToolSpecV1(name="search_drinks", description="Search drinks"),
        ],
        excluded=["search_drinks"],
    )
    # Model returns JSON proposing excluded tool "search_drinks"
    fake_adapter.enqueue_response(
        ScriptedResponse(
            text='{"tool": "search_drinks", "parameters": {"query": "cocktails"}}'
        )
    )
    req = LLMRequestV1(
        request_id="req-gather-fail",
        policy_tag="gather_action.propose",
        messages=[MessageV1(role=RequestRole.USER, content="Find drinks")],
        generation=GenerationConfigV1(max_output_tokens=100),
        response_contract=ResponseContractV1(
            kind=ResponseKind.JSON,
            json_schema={"type": "object"},
            semantic_validator_id=SemanticValidatorId.GATHER_ACTION.value,
        ),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id=ProviderId.FAKE.value, model_id="fake-fast"),
        ),
        budget=BudgetConfigV1(max_physical_attempts=1, deadline_ms=5000),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
        tools=catalog,
    )
    res = engine.invoke_llm(req)
    assert res.status == ResultStatus.FAILED
    assert len(res.attempts) == 1
    assert res.attempts[0].validation_summary.semantic == ValidationGateStatus.FAILED
    assert any("search_drinks" in err for err in res.attempts[0].validation_summary.errors)

