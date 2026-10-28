"""Unit tests for OpenAI and Azure OpenAI integration in LLM Gateway (WP5)."""

from __future__ import annotations

import pytest

from llm_gateway import (
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
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_gateway.adapters.openai_compat import OpenAICompatibleAdapter
from llm_gateway.errors import ProviderError

pytestmark = pytest.mark.backend_unit


def test_openai_and_azure_provider_registrations():
    registry = get_provider_registry()

    openai_prov = registry.get_provider("openai")
    assert openai_prov is not None
    assert openai_prov.enabled is True
    openai_models = {m.model_id for m in openai_prov.models}
    assert "gpt-6" in openai_models
    assert "gpt-6-mini" in openai_models

    azure_prov = registry.get_provider("azure_openai")
    assert azure_prov is not None
    assert azure_prov.enabled is True
    azure_models = {m.model_id for m in azure_prov.models}
    assert "gpt-6" in azure_models
    assert "gpt-6-mini" in azure_models

    pricing = registry.get_pricing_snapshot("default-pricing-v1")
    # Verify pricing calculation for OpenAI and Azure OpenAI
    cost_openai = pricing.calculate_cost_usd("openai", "gpt-6-mini", 1000, 500)
    cost_azure = pricing.calculate_cost_usd("azure_openai", "gpt-6-mini", 1000, 500)
    assert cost_openai > 0.0
    assert cost_azure > 0.0


def test_openai_to_azure_fallback_ladder():
    engine = get_gateway_engine()

    fake_openai = FakeLLMAdapter()
    fake_azure = FakeLLMAdapter()

    # Simulate OpenAI 500 failure
    fake_openai.enqueue_response(
        ScriptedResponse(error_to_raise=ProviderError("outage", "OpenAI 500 outage"))
    )
    # Simulate Azure OpenAI success
    fake_azure.enqueue_response(
        ScriptedResponse(text="Azure fallback response", prompt_tokens=25, completion_tokens=10)
    )

    engine.register_adapter("openai", fake_openai)
    engine.register_adapter("azure_openai", fake_azure)

    req = LLMRequestV1(
        request_id="req-fallback-test",
        policy_tag="conversation.summary",
        messages=[MessageV1(role=RequestRole.USER, content="Hello")],
        generation=GenerationConfigV1(temperature=0.7, max_output_tokens=50),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id="openai", model_id="gpt-6-mini", policy_leg=PolicyLeg.BASELINE),
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
    assert result.attempts[0].provider_id == "openai"
    assert result.attempts[1].provider_id == "azure_openai"
    assert result.output.text == "Azure fallback response"
