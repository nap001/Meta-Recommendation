"""Unit tests for DynamicProviderRegistry and dynamic catalog synchronization (WP5)."""

from __future__ import annotations

import json
import pytest
import httpx

from llm_gateway import (
    DynamicProviderRegistry,
    ModelLimitV1,
    ModelPriceV1,
    PricingSnapshotV1,
    ProviderCapabilityV1,
    ProviderRegistry,
    get_dynamic_provider_registry,
    get_provider_registry,
)

pytestmark = pytest.mark.backend_unit


def test_dynamic_registry_inheritance_and_defaults():
    # 1. Verify inheritance
    assert issubclass(DynamicProviderRegistry, ProviderRegistry)

    reg = DynamicProviderRegistry()
    assert isinstance(reg, ProviderRegistry)
    assert isinstance(reg, DynamicProviderRegistry)

    # 2. Verify singleton factory
    global_reg = get_provider_registry()
    assert isinstance(global_reg, DynamicProviderRegistry)
    assert get_dynamic_provider_registry() is global_reg

    # 3. Verify defaults are pre-registered
    prov_openai = reg.get_provider("openai")
    prov_azure = reg.get_provider("azure_openai")
    prov_anthropic = reg.get_provider("anthropic")
    prov_fake = reg.get_provider("fake_provider")

    assert prov_openai is not None
    assert prov_azure is not None
    assert prov_anthropic is not None
    assert prov_fake is not None

    snapshot = reg.get_pricing_snapshot("default-pricing-v1")
    assert snapshot is not None
    assert "openai:gpt-6" in snapshot.rates
    assert snapshot.rates["openai:gpt-6"].input_price_per_1m == 2.50


@pytest.mark.asyncio
async def test_dynamic_catalog_sync_success():
    sample_litellm_feed = {
        "gpt-4o": {
            "max_tokens": 4096,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "litellm_provider": "openai",
            "mode": "chat",
        },
        "azure/gpt-4o": {
            "max_tokens": 4096,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "litellm_provider": "azure",
            "mode": "chat",
        },
        "claude-3-7-sonnet-20250219": {
            "max_output_tokens": 16384,
            "max_input_tokens": 200000,
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
            "litellm_provider": "anthropic",
            "is_reasoning_model": True,
        },
        "groq/llama-3.3-70b-versatile": {
            "max_tokens": 8192,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0.00000059,
            "output_cost_per_token": 0.00000079,
            "litellm_provider": "groq",
        },
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=sample_litellm_feed)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    registry = DynamicProviderRegistry()
    await registry.fetch_and_sync_catalog("https://fake-feed.com/catalog.json", http_client=mock_client)

    # Verify Anthropic received the new model
    prov_anthropic = registry.get_provider("anthropic")
    model_ids = {m.model_id for m in prov_anthropic.models}
    assert "claude-3-7-sonnet-20250219" in model_ids

    claude_37 = next(m for m in prov_anthropic.models if m.model_id == "claude-3-7-sonnet-20250219")
    assert isinstance(claude_37, ModelLimitV1)
    assert claude_37.context_window == 200000
    assert claude_37.max_output_tokens == 16384
    assert claude_37.is_reasoning_model is True

    # Verify PricingSnapshot rates are populated and strictly validated as ModelPriceV1
    snapshot = registry.get_pricing_snapshot("default-pricing-v1")
    assert "anthropic:claude-3-7-sonnet-20250219" in snapshot.rates

    price = snapshot.rates["anthropic:claude-3-7-sonnet-20250219"]
    assert isinstance(price, ModelPriceV1)
    assert price.input_price_per_1m == 3.0  # 0.000003 * 1,000,000
    assert price.output_price_per_1m == 15.0  # 0.000015 * 1,000,000

    # Verify Azure pricing rate
    assert "azure_openai:gpt-4o" in snapshot.rates
    assert snapshot.rates["azure_openai:gpt-4o"].input_price_per_1m == 2.50


@pytest.mark.asyncio
async def test_dynamic_catalog_sync_skips_invalid_entries():
    feed_with_malformed_entries = {
        # Valid entry
        "gpt-4o-mini": {
            "max_tokens": 4096,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.00000060,
            "litellm_provider": "openai",
        },
        # Invalid entry 1: non-numeric pricing string
        "corrupt-model-1": {
            "max_tokens": 4096,
            "input_cost_per_token": "not-a-number",
            "litellm_provider": "openai",
        },
        # Invalid entry 2: not a dict
        "corrupt-model-2": "invalid-string-payload",
        # Invalid entry 3: empty model ID
        "": {
            "max_tokens": 2048,
            "input_cost_per_token": 0.00001,
        },
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=feed_with_malformed_entries)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    registry = DynamicProviderRegistry()
    await registry.fetch_and_sync_catalog("https://fake-feed.com/catalog.json", http_client=mock_client)

    # Valid entry must be successfully processed
    snapshot = registry.get_pricing_snapshot("default-pricing-v1")
    assert "openai:gpt-4o-mini" in snapshot.rates
    assert snapshot.rates["openai:gpt-4o-mini"].input_price_per_1m == 0.15

    # Corrupt model must not be registered
    assert "corrupt-model-1" not in snapshot.rates
    assert "corrupt-model-2" not in snapshot.rates


@pytest.mark.asyncio
async def test_dynamic_catalog_sync_graceful_fallback_on_network_error():
    # Simulate network 500 error
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    registry = DynamicProviderRegistry()

    # Capture initial state of defaults
    initial_prov_count = len(registry.list_providers())
    initial_gpt4o_price = registry.get_pricing_snapshot("default-pricing-v1").get_price("openai", "gpt-4o")

    # Should not raise exception; catches and logs warning
    await registry.fetch_and_sync_catalog("https://fake-feed.com/fail.json", http_client=mock_client)

    # Defaults must be completely intact
    assert len(registry.list_providers()) == initial_prov_count
    assert registry.get_provider("openai") is not None
    assert registry.get_pricing_snapshot("default-pricing-v1").get_price("openai", "gpt-4o") == initial_gpt4o_price


@pytest.mark.asyncio
async def test_dynamic_catalog_sync_graceful_fallback_on_timeout():
    # Simulate timeout
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connection timed out")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    registry = DynamicProviderRegistry()
    # Must not raise
    await registry.fetch_and_sync_catalog("https://timeout-feed.com/catalog.json", http_client=mock_client)

    # Verify registry remains healthy
    assert registry.get_provider("anthropic") is not None
    assert registry.get_provider("fake_provider") is not None


@pytest.mark.asyncio
async def test_dynamic_catalog_sync_graceful_fallback_on_non_dict_json():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "list", "format"])

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    registry = DynamicProviderRegistry()
    await registry.fetch_and_sync_catalog("https://invalid-json-feed.com/catalog.json", http_client=mock_client)

    # Defaults preserved
    assert registry.get_provider("azure_openai") is not None
