"""Unit tests for OpenAIBaseAdapter in LLM Gateway (WP5)."""

from __future__ import annotations

import json
import pytest
import httpx

from llm_gateway import (
    BudgetConfigV1,
    GenerationConfigV1,
    LLMRequestV1,
    MessageV1,
    OpenAIBaseAdapter,
    OpenAIAdapter,
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


def test_openai_base_adapter_defaults_and_headers():
    # 1. Default base URL resolution
    adapter = OpenAIBaseAdapter(api_key="sk-testkey123", organization="org-meta-test")
    assert adapter._get_base_url() == "https://api.openai.com/v1"
    assert adapter._get_completions_url() == "https://api.openai.com/v1/chat/completions"

    headers = adapter._get_headers()
    assert headers["Authorization"] == "Bearer sk-testkey123"
    assert headers["Content-Type"] == "application/json"
    assert headers["OpenAI-Organization"] == "org-meta-test"

    # 2. Custom base URL (e.g. local vLLM or custom gateway)
    custom_adapter = OpenAIBaseAdapter(base_url="http://localhost:8000/v1/", api_key="custom-key")
    assert custom_adapter._get_base_url() == "http://localhost:8000/v1"
    assert custom_adapter._get_completions_url() == "http://localhost:8000/v1/chat/completions"

    # 3. Alias check
    assert OpenAIAdapter is OpenAIBaseAdapter


def test_openai_base_adapter_formatting():
    adapter = OpenAIBaseAdapter(api_key="sk-test")
    route = RouteV1(provider_id="openai_base", model_id="o3-mini", policy_leg=PolicyLeg.BASELINE)

    req = LLMRequestV1(
        request_id="req-openai-base-format",
        policy_tag="conversation.summary",
        messages=[
            MessageV1(role=RequestRole.USER, content="Summarize the plan"),
        ],
        generation=GenerationConfigV1(temperature=0.7, max_output_tokens=500),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(mode=RoutingMode.BASELINE, baseline_route=route),
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    normalized_kwargs = {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "Summarize the plan"}],
        "extra_body": {"max_completion_tokens": 500},
    }

    payload = adapter._format_payload(req, route, normalized_kwargs)
    assert payload["model"] == "o3-mini"
    assert payload["max_completion_tokens"] == 500
    assert "extra_body" not in payload
    assert len(payload["messages"]) == 1


def test_openai_base_adapter_response_parsing():
    adapter = OpenAIBaseAdapter()
    sample_json = {
        "id": "chatcmpl-test999",
        "object": "chat.completion",
        "created": 1720000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here is your requested itinerary summary.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_restaurant_details",
                                "arguments": "{\"id\": 42}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 30,
            "total_tokens": 110,
        },
    }

    parsed = adapter._parse_response(sample_json, "gpt-4o-mini")
    assert parsed.text == "Here is your requested itinerary summary."
    assert parsed.prompt_tokens == 80
    assert parsed.completion_tokens == 30
    assert parsed.total_tokens == 110
    assert parsed.provider_request_id == "chatcmpl-test999"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["id"] == "call_abc123"
    assert parsed.tool_calls[0]["function"]["name"] == "get_restaurant_details"
    assert parsed.tool_calls[0]["function"]["arguments"] == "{\"id\": 42}"


def test_openai_base_adapter_error_mapping():
    adapter = OpenAIBaseAdapter()

    # 429 Rate limit
    e_429 = adapter._handle_http_error(429, {"error": {"message": "Rate limit exceeded", "type": "requests"}})
    assert isinstance(e_429, TransportError)
    assert e_429.code == "rate_limited"
    assert e_429.retryable is True

    # 503 Server error
    e_503 = adapter._handle_http_error(503, {"error": {"message": "Service unavailable", "type": "server_error"}})
    assert isinstance(e_503, TransportError)
    assert e_503.code == "provider_5xx"
    assert e_503.retryable is True

    # 401 Auth error
    e_401 = adapter._handle_http_error(401, {"error": {"message": "Incorrect API key", "type": "invalid_request_error", "code": "invalid_api_key"}})
    assert isinstance(e_401, ProviderError)
    assert e_401.code == "auth_failed"
    assert e_401.retryable is False

    # 400 Unsupported param
    e_400 = adapter._handle_http_error(400, {"error": {"message": "Unsupported parameter: seed", "type": "invalid_request_error"}})
    assert isinstance(e_400, CompatibilityError)
    assert e_400.code == "unsupported_parameter"
    assert e_400.retryable is True


def test_openai_base_mock_execute_sync():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.openai.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-sk-key"
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"][0]["content"] == "Hello OpenAI Base"

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-sync-001",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Sync response from OpenAI Base adapter.",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    adapter = OpenAIBaseAdapter(api_key="test-sk-key", http_client=mock_client)

    route = RouteV1(provider_id="openai_base", model_id="gpt-4o-mini", policy_leg=PolicyLeg.BASELINE)
    req = LLMRequestV1(
        request_id="req-sync-base-test",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Hello OpenAI Base")],
        generation=GenerationConfigV1(temperature=0.3, max_output_tokens=50),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(mode=RoutingMode.BASELINE, baseline_route=route),
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    resp = adapter.execute(req, route, {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello OpenAI Base"}]})
    assert resp.text == "Sync response from OpenAI Base adapter."
    assert resp.prompt_tokens == 15
    assert resp.completion_tokens == 8
    assert resp.provider_request_id == "chatcmpl-sync-001"


@pytest.mark.asyncio
async def test_openai_base_mock_execute_async():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.openai.com/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-async-002",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Async response from OpenAI Base adapter.",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

    mock_async_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    adapter = OpenAIBaseAdapter(api_key="test-sk-key", async_http_client=mock_async_client)

    route = RouteV1(provider_id="openai_base", model_id="gpt-4o-mini", policy_leg=PolicyLeg.BASELINE)
    req = LLMRequestV1(
        request_id="req-async-base-test",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Async test")],
        generation=GenerationConfigV1(temperature=0.3, max_output_tokens=50),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(mode=RoutingMode.BASELINE, baseline_route=route),
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    resp = await adapter.execute_async(req, route, {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Async test"}]})
    assert resp.text == "Async response from OpenAI Base adapter."
    assert resp.prompt_tokens == 20
    assert resp.completion_tokens == 10


def test_openai_base_gateway_fallback_ladder():
    engine = get_gateway_engine()

    fake_openai_base = FakeLLMAdapter()
    fake_azure = FakeLLMAdapter()

    # Simulate OpenAI Base 500 outage
    fake_openai_base.enqueue_response(
        ScriptedResponse(error_to_raise=ProviderError("outage", "OpenAI Base 500 outage"))
    )
    # Simulate Azure fallback success
    fake_azure.enqueue_response(
        ScriptedResponse(text="Azure fallback recovered response", prompt_tokens=30, completion_tokens=12)
    )

    engine.register_adapter("openai_base", fake_openai_base)
    engine.register_adapter("azure_openai", fake_azure)

    req = LLMRequestV1(
        request_id="req-base-fallback-test",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Summarize context.")],
        generation=GenerationConfigV1(temperature=0.5, max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(
                provider_id="openai_base",
                model_id="gpt-6-mini",
                policy_leg=PolicyLeg.BASELINE,
            ),
            fallback_routes=[
                RouteV1(provider_id="azure_openai", model_id="gpt-6-mini", policy_leg=PolicyLeg.FALLBACK)
            ],
        ),
        budget=BudgetConfigV1(deadline_ms=5000, max_physical_attempts=3, max_cost_usd=0.50),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    result = invoke_llm(req)

    assert result.status in (ResultStatus.SUCCEEDED, ResultStatus.DEGRADED)
    assert result.selected_route.provider_id == "azure_openai"
    assert result.selected_route.policy_leg == PolicyLeg.FALLBACK
    assert len(result.attempts) == 2
    assert result.attempts[0].provider_id == "openai_base"
    assert result.attempts[0].outcome.value == "provider_error"
    assert result.attempts[1].provider_id == "azure_openai"
    assert result.attempts[1].outcome.value == "success"
    assert result.output.text == "Azure fallback recovered response"
    assert result.usage.cost_usd > 0.0
