"""In-memory ProviderRegistry store and default registrations (v1)."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from llm_gateway.constants import (
    AdapterKind,
    CredentialStatus,
    DEFAULT_CURRENCY,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    DEFAULT_PRICE_TABLE_VERSION,
    DEFAULT_PRICING_SNAPSHOT_ID,
    ProviderId,
)
from llm_gateway.registry.models import (
    ModelLimitV1,
    ParameterRulesV1,
    ProviderCapabilityV1,
    ProviderFeaturesV1,
)
from llm_gateway.registry.pricing import ModelPriceV1, PricingSnapshotV1

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """In-memory store of ProviderCapabilityV1 and PricingSnapshotV1 configurations."""

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderCapabilityV1] = {}
        self._pricing_snapshots: Dict[str, PricingSnapshotV1] = {}
        self._register_defaults()

    def register_provider(self, capability: ProviderCapabilityV1) -> None:
        self._providers[capability.provider_id] = capability

    def get_provider(self, provider_id: str) -> Optional[ProviderCapabilityV1]:
        return self._providers.get(provider_id)

    def list_providers(self) -> List[ProviderCapabilityV1]:
        return list(self._providers.values())

    def register_pricing_snapshot(self, snapshot: PricingSnapshotV1) -> None:
        self._pricing_snapshots[snapshot.pricing_snapshot_id] = snapshot

    def get_pricing_snapshot(self, snapshot_id: str) -> PricingSnapshotV1:
        if snapshot_id in self._pricing_snapshots:
            return self._pricing_snapshots[snapshot_id]
        # Return default zero-price snapshot if not found
        return PricingSnapshotV1(pricing_snapshot_id=snapshot_id)

    def _register_defaults(self) -> None:
        # 1. Deterministic Fake Provider
        fake_cap = ProviderCapabilityV1(
            provider_id=ProviderId.FAKE.value,
            adapter_kind=AdapterKind.FAKE.value,
            enabled=True,
            credential_status=CredentialStatus.READY.value,
            models=[
                ModelLimitV1(model_id="fake-fast", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="fake-strong", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192),
                ModelLimitV1(model_id="fake-reasoning", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192, is_reasoning_model=True),
            ],
            features=ProviderFeaturesV1(
                supports_sync=True,
                supports_async=True,
                supports_streaming=False,
                supports_json_object=True,
                supports_json_schema=True,
                supports_tools=True,
                supports_forced_tool_choice=True,
                supports_seed=True,
                supports_provider_usage=True,
            ),
        )
        self.register_provider(fake_cap)

        # 2. Groq Provider (OpenAI-compatible)
        groq_cap = ProviderCapabilityV1(
            provider_id=ProviderId.GROQ.value,
            adapter_kind=AdapterKind.OPENAI_COMPATIBLE.value,
            enabled=True,
            credential_status=CredentialStatus.READY.value if os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") else CredentialStatus.MISSING.value,
            models=[
                ModelLimitV1(model_id="qwen/qwen3.8-27b", context_window=32768, max_output_tokens=8192),
                ModelLimitV1(model_id="allam-2-7b", context_window=8192, max_output_tokens=4096),
                ModelLimitV1(model_id="llama-3.3-70b-versatile", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192),
                ModelLimitV1(model_id="llama-3.1-8b-instant", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
            ],
            features=ProviderFeaturesV1(
                supports_sync=True,
                supports_async=True,
                supports_streaming=False,  # Fix #5: no adapter implements streaming yet
                supports_json_object=True,
                supports_json_schema=False,
                supports_tools=True,
                supports_forced_tool_choice=True,
                supports_seed=True,
                supports_provider_usage=True,
            ),
        )
        self.register_provider(groq_cap)

        # 3. Azure OpenAI Provider
        azure_cap = ProviderCapabilityV1(
            provider_id=ProviderId.AZURE_OPENAI.value,
            adapter_kind=AdapterKind.AZURE_OPENAI.value,
            enabled=True,
            credential_status=CredentialStatus.READY.value if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") else CredentialStatus.MISSING.value,
            models=[
                ModelLimitV1(model_id="gpt-6", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="gpt-6-mini", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="o3", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192, is_reasoning_model=True),
            ],
            parameter_rules=ParameterRulesV1(
                forbidden_parameters=["seed"]
            ),
        )
        self.register_provider(azure_cap)

        # 4. OpenAI Direct Provider
        openai_cap = ProviderCapabilityV1(
            provider_id=ProviderId.OPENAI.value,
            adapter_kind=AdapterKind.OPENAI_COMPATIBLE.value,
            enabled=True,
            credential_status=CredentialStatus.READY.value if os.getenv("OPENAI_API_KEY") else CredentialStatus.MISSING.value,
            models=[
                ModelLimitV1(model_id="gpt-6", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="gpt-6-mini", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="o3", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192, is_reasoning_model=True),
                ModelLimitV1(model_id="o3-mini", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192, is_reasoning_model=True),
            ],
            features=ProviderFeaturesV1(
                supports_sync=True,
                supports_async=True,
                supports_streaming=False,  # Fix #5: no adapter implements streaming yet
                supports_json_object=True,
                supports_json_schema=True,
                supports_tools=True,
                supports_forced_tool_choice=True,
                supports_seed=True,
                supports_provider_usage=True,
            ),
        )
        self.register_provider(openai_cap)

        # 5. OpenAI Base HTTP Provider
        openai_base_cap = ProviderCapabilityV1(
            provider_id=ProviderId.OPENAI_BASE.value,
            adapter_kind=AdapterKind.OPENAI_BASE.value,
            enabled=True,
            credential_status=CredentialStatus.READY.value if os.getenv("OPENAI_API_KEY") else CredentialStatus.MISSING.value,
            models=[
                ModelLimitV1(model_id="gpt-6", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="gpt-6-mini", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="o3", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192, is_reasoning_model=True),
                ModelLimitV1(model_id="o3-mini", context_window=DEFAULT_MODEL_CONTEXT_WINDOW, max_output_tokens=8192, is_reasoning_model=True),
            ],
            features=ProviderFeaturesV1(
                supports_sync=True,
                supports_async=True,
                supports_streaming=False,  # Fix #5: no adapter implements streaming yet
                supports_json_object=True,
                supports_json_schema=True,
                supports_tools=True,
                supports_forced_tool_choice=True,
                supports_seed=True,
                supports_provider_usage=True,
            ),
            price_table_version=DEFAULT_PRICE_TABLE_VERSION,
            data_policy_ref="openai_standard_terms",
        )
        self.register_provider(openai_base_cap)

        # 6. Anthropic Claude Provider
        anthropic_cap = ProviderCapabilityV1(
            provider_id=ProviderId.ANTHROPIC.value,
            adapter_kind=AdapterKind.ANTHROPIC.value,
            enabled=True,
            credential_status=CredentialStatus.READY.value if os.getenv("ANTHROPIC_API_KEY") else CredentialStatus.MISSING.value,
            models=[
                ModelLimitV1(model_id="claude-3-5-sonnet-20241022", context_window=200000, max_output_tokens=8192),
                ModelLimitV1(model_id="claude-3-5-haiku-20241022", context_window=200000, max_output_tokens=8192),
                ModelLimitV1(model_id="claude-3-opus-20240229", context_window=200000, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
                ModelLimitV1(model_id="claude-3-haiku-20240307", context_window=200000, max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS),
            ],
            features=ProviderFeaturesV1(
                supports_sync=True,
                supports_async=True,
                supports_streaming=False,  # Fix #5: no adapter implements streaming yet
                supports_json_object=True,
                supports_json_schema=False,
                supports_tools=True,
                supports_forced_tool_choice=True,
                supports_seed=False,
                supports_provider_usage=True,
            ),
            parameter_rules=ParameterRulesV1(
                forbidden_parameters=["seed"]
            ),
            price_table_version=DEFAULT_PRICE_TABLE_VERSION,
            data_policy_ref="anthropic_standard_terms",
        )
        self.register_provider(anthropic_cap)

        # 7. Standard Pricing Snapshot
        default_pricing = PricingSnapshotV1(
            pricing_snapshot_id=DEFAULT_PRICING_SNAPSHOT_ID,
            currency=DEFAULT_CURRENCY,
            effective_time_utc=0.0,
            rates={
                "fake_provider:fake-fast": ModelPriceV1(input_price_per_1m=0.10, output_price_per_1m=0.20),
                "fake_provider:fake-strong": ModelPriceV1(input_price_per_1m=2.50, output_price_per_1m=10.00),
                "fake_provider:fake-reasoning": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=12.00),
                "groq:qwen/qwen3.8-27b": ModelPriceV1(input_price_per_1m=0.20, output_price_per_1m=0.20),
                "groq:allam-2-7b": ModelPriceV1(input_price_per_1m=0.10, output_price_per_1m=0.10),
                "groq:llama-3.3-70b-versatile": ModelPriceV1(input_price_per_1m=0.59, output_price_per_1m=0.79),
                "groq:llama-3.1-8b-instant": ModelPriceV1(input_price_per_1m=0.05, output_price_per_1m=0.08),
                "azure_openai:gpt-6": ModelPriceV1(input_price_per_1m=2.50, output_price_per_1m=10.00),
                "azure_openai:gpt-6-mini": ModelPriceV1(input_price_per_1m=0.15, output_price_per_1m=0.60),
                "azure_openai:o3": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=12.00),
                "openai:gpt-6": ModelPriceV1(input_price_per_1m=2.50, output_price_per_1m=10.00),
                "openai:gpt-6-mini": ModelPriceV1(input_price_per_1m=0.15, output_price_per_1m=0.60),
                "openai:o3": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=12.00),
                "openai:o3-mini": ModelPriceV1(input_price_per_1m=1.10, output_price_per_1m=4.40),
                "openai_base:gpt-6": ModelPriceV1(input_price_per_1m=2.50, output_price_per_1m=10.00),
                "openai_base:gpt-6-mini": ModelPriceV1(input_price_per_1m=0.15, output_price_per_1m=0.60),
                "openai_base:o3": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=12.00),
                "openai_base:o3-mini": ModelPriceV1(input_price_per_1m=1.10, output_price_per_1m=4.40),
                # Legacy rates retained for audit log replay & backwards compatibility
                "azure_openai:gpt-4o": ModelPriceV1(input_price_per_1m=2.50, output_price_per_1m=10.00),
                "azure_openai:gpt-4o-mini": ModelPriceV1(input_price_per_1m=0.15, output_price_per_1m=0.60),
                "azure_openai:o4-mini": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=12.00),
                "openai:gpt-4o": ModelPriceV1(input_price_per_1m=2.50, output_price_per_1m=10.00),
                "openai:gpt-4o-mini": ModelPriceV1(input_price_per_1m=0.15, output_price_per_1m=0.60),
                "openai:o1-mini": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=12.00),
                "openai_base:gpt-4o": ModelPriceV1(input_price_per_1m=2.50, output_price_per_1m=10.00),
                "openai_base:gpt-4o-mini": ModelPriceV1(input_price_per_1m=0.15, output_price_per_1m=0.60),
                "openai_base:o1-mini": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=12.00),
                "anthropic:claude-3-5-sonnet-20241022": ModelPriceV1(input_price_per_1m=3.00, output_price_per_1m=15.00),
                "anthropic:claude-3-5-haiku-20241022": ModelPriceV1(input_price_per_1m=0.80, output_price_per_1m=4.00),
                "anthropic:claude-3-opus-20240229": ModelPriceV1(input_price_per_1m=15.00, output_price_per_1m=75.00),
                "anthropic:claude-3-haiku-20240307": ModelPriceV1(input_price_per_1m=0.25, output_price_per_1m=1.25),
            },
        )
        self.register_pricing_snapshot(default_pricing)


_GLOBAL_REGISTRY: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Return the global ProviderRegistry singleton, initializing defaults if needed."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ProviderRegistry()
    return _GLOBAL_REGISTRY
