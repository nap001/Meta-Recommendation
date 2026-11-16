"""Unit tests for Anthropic Claude integration in LLM Gateway (WP5)."""

from __future__ import annotations

import json
import pytest
import httpx

from llm_gateway import (
    AnthropicAdapter,
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
    ScopedToolCatalogV1,
    ToolSpecV1,
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
    invoke_llm_async,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.errors import CompatibilityError, ProviderError, TransportError

pytestmark = pytest.mark.backend_unit


def test_anthropic_provider_registration():
    registry = get_provider_registry()
    anthropic_prov = registry.get_provider("anthropic")

    assert anthropic_prov is not None
    assert anthropic_prov.enabled is True
    assert anthropic_prov.adapter_kind == "anthropic"

    models = {m.model_id for m in anthropic_prov.models}
    assert "claude-3-5-sonnet-20241022" in models
    assert "claude-3-5-haiku-20241022" in models
    assert "claude-3-opus-20240229" not in models
    assert "claude-3-haiku-20240307" not in models

    pricing = registry.get_pricing_snapshot("default-pricing-v1")
    cost_sonnet = pricing.calculate_cost_usd("anthropic", "claude-3-5-sonnet-20241022", 1000, 500)
    cost_haiku = pricing.calculate_cost_usd("anthropic", "claude-3-5-haiku-20241022", 1000, 500)

    assert cost_sonnet > 0.0
    assert cost_haiku > 0.0
    assert cost_sonnet > cost_haiku


def test_anthropic_payload_formatting():
    adapter = AnthropicAdapter(api_key="test-key")
    route = RouteV1(provider_id="anthropic", model_id="claude-3-5-sonnet-20241022", policy_leg=PolicyLeg.BASELINE)

    req = LLMRequestV1(
        request_id="req-format-test",
        policy_tag="conversation.summary",
        messages=[
            MessageV1(role=RequestRole.SYSTEM, content="You are a system summarizer."),
            MessageV1(role=RequestRole.USER, content="Hello there!"),
            MessageV1(role=RequestRole.USER, content="Can you summarize this?"),
        ],
        generation=GenerationConfigV1(temperature=0.4, max_output_tokens=256, stop=["\nUser:"]),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=route,
        ),
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
        tools=ScopedToolCatalogV1(
            catalog_id="cat-v1",
            health_snapshot_id="health-v1",
            tools=[
                ToolSpecV1(
                    name="search_attractions",
                    description="Search tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            ],
        ),
        tool_choice="search_attractions",
    )

    payload = adapter._format_anthropic_payload(req, route, {"max_tokens": 256, "temperature": 0.4})

    assert payload["model"] == "claude-3-5-sonnet-20241022"
    assert payload["max_tokens"] == 256
    assert payload["system"] == "You are a system summarizer."
    assert payload["temperature"] == 0.4
    assert payload["stop_sequences"] == ["\nUser:"]

    # Consecutive user messages should be merged
    assert len(payload["messages"]) == 1
    assert "Hello there!" in payload["messages"][0]["content"]
    assert "Can you summarize this?" in payload["messages"][0]["content"]

    # Verify tool conversion
    assert "tools" in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "search_attractions"
    assert "input_schema" in payload["tools"][0]
    assert payload["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"

    # Verify forced tool choice
    assert payload["tool_choice"] == {"type": "tool", "name": "search_attractions"}


def test_anthropic_response_parsing():
    adapter = AnthropicAdapter()
    sample_anthropic_json = {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {"type": "text", "text": "I can help with attractions. "},
            {
                "type": "tool_use",
                "id": "toolu_01A09q90qw90lq9178350184",
                "name": "search_attractions",
                "input": {"city": "Tokyo"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 125,
            "output_tokens": 42,
        },
    }

    parsed = adapter._parse_anthropic_response(sample_anthropic_json, "claude-3-5-sonnet-20241022")

    assert parsed.text == "I can help with attractions. "
    assert parsed.prompt_tokens == 125
    assert parsed.completion_tokens == 42
    assert parsed.total_tokens == 167
    assert parsed.provider_request_id == "msg_01XFDUDYJgAACzvnptvVoYEL"
    assert len(parsed.tool_calls) == 1

    tc = parsed.tool_calls[0]
    assert tc["id"] == "toolu_01A09q90qw90lq9178350184"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search_attractions"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}


def test_anthropic_error_mapping():
    adapter = AnthropicAdapter()

    # 1. 429 Rate Limit
    err_429 = adapter._handle_http_error(429, {"error": {"type": "rate_limit_error", "message": "Tokens exceeded"}})
    assert isinstance(err_429, TransportError)
    assert err_429.code == "rate_limited"
    assert err_429.retryable is True

    # 2. 529 Overloaded
    err_529 = adapter._handle_http_error(529, {"error": {"type": "overloaded_error", "message": "Claude is overloaded"}})
    assert isinstance(err_529, TransportError)
    assert err_529.code == "provider_5xx"
    assert err_529.retryable is True

    # 3. 401 Unauthorized
    err_401 = adapter._handle_http_error(401, {"error": {"type": "authentication_error", "message": "Invalid API Key"}})
    assert isinstance(err_401, ProviderError)
    assert err_401.code == "auth_failed"
    assert err_401.retryable is False

    # 4. 400 Unsupported Parameter
    err_400 = adapter._handle_http_error(400, {"error": {"type": "invalid_request_error", "message": "Unsupported parameter: seed"}})
    assert isinstance(err_400, CompatibilityError)
    assert err_400.code == "unsupported_parameter"
    assert err_400.retryable is True


def test_anthropic_mock_execute_sync():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-claude-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        req_body = json.loads(request.content)
        assert req_body["model"] == "claude-3-5-sonnet-20241022"

        return httpx.Response(
            200,
            json={
                "id": "msg_mock_001",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "Claude simulated answer."}],
                "usage": {"input_tokens": 50, "output_tokens": 15},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    adapter = AnthropicAdapter(api_key="test-claude-key", http_client=mock_client)

    route = RouteV1(provider_id="anthropic", model_id="claude-3-5-sonnet-20241022", policy_leg=PolicyLeg.BASELINE)
    req = LLMRequestV1(
        request_id="req-sync-mock",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Hello Claude")],
        generation=GenerationConfigV1(temperature=0.5, max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(mode=RoutingMode.BASELINE, baseline_route=route),
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    resp = adapter.execute(req, route, {"max_tokens": 100})
    assert resp.text == "Claude simulated answer."
    assert resp.prompt_tokens == 50
    assert resp.completion_tokens == 15
    assert resp.provider_request_id == "msg_mock_001"


@pytest.mark.asyncio
async def test_anthropic_mock_execute_async():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_async_002",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-haiku-20241022",
                "content": [{"type": "text", "text": "Async Claude response."}],
                "usage": {"input_tokens": 30, "output_tokens": 10},
            },
        )

    mock_async_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    adapter = AnthropicAdapter(api_key="test-claude-key", async_http_client=mock_async_client)

    route = RouteV1(provider_id="anthropic", model_id="claude-3-5-haiku-20241022", policy_leg=PolicyLeg.BASELINE)
    req = LLMRequestV1(
        request_id="req-async-mock",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Async test")],
        generation=GenerationConfigV1(temperature=0.5, max_output_tokens=50),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(mode=RoutingMode.BASELINE, baseline_route=route),
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    resp = await adapter.execute_async(req, route, {"max_tokens": 50})
    assert resp.text == "Async Claude response."
    assert resp.prompt_tokens == 30
    assert resp.completion_tokens == 10


def test_anthropic_gateway_fallback_to_openai():
    engine = get_gateway_engine()

    fake_anthropic = FakeLLMAdapter()
    fake_openai = FakeLLMAdapter()

    # Simulate Anthropic provider outage
    fake_anthropic.enqueue_response(
        ScriptedResponse(error_to_raise=ProviderError("outage", "Anthropic 529 outage"))
    )
    # Simulate OpenAI fallback success
    fake_openai.enqueue_response(
        ScriptedResponse(text="OpenAI fallback recovered answer", prompt_tokens=40, completion_tokens=15)
    )

    engine.register_adapter("anthropic", fake_anthropic)
    engine.register_adapter("openai", fake_openai)

    req = LLMRequestV1(
        request_id="req-anthropic-fallback",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Summarize itinerary options.")],
        generation=GenerationConfigV1(temperature=0.2, max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(
                provider_id="anthropic",
                model_id="claude-3-5-sonnet-20241022",
                policy_leg=PolicyLeg.BASELINE,
            ),
            fallback_routes=[
                RouteV1(provider_id="openai", model_id="gpt-6-mini", policy_leg=PolicyLeg.FALLBACK)
            ],
        ),
        budget=BudgetConfigV1(deadline_ms=6000, max_physical_attempts=3, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    result = invoke_llm(req)

    assert result.status in (ResultStatus.SUCCEEDED, ResultStatus.DEGRADED)
    assert result.selected_route.provider_id == "openai"
    assert result.selected_route.policy_leg == PolicyLeg.FALLBACK
    assert len(result.attempts) == 2
    assert result.attempts[0].provider_id == "anthropic"
    assert result.attempts[0].outcome.value == "provider_error"
    assert result.attempts[1].provider_id == "openai"
    assert result.attempts[1].outcome.value == "success"
    assert result.output.text == "OpenAI fallback recovered answer"
    assert result.usage.cost_usd > 0.0
