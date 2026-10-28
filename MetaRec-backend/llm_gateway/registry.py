"""Provider Capability Registry and Pricing Snapshot Management (v1)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from llm_gateway.constants import (
    AdapterKind,
    CredentialStatus,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_CURRENCY,
    DEFAULT_LITELLM_CATALOG_URL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    DEFAULT_PRICE_TABLE_VERSION,
    DEFAULT_PRICING_SNAPSHOT_ID,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DEFAULT_SDK_RETRIES,
    DEFAULT_TOOL_NAME_PATTERN,
    ProviderId,
    SCHEMA_VERSION_PROVIDER_CAPABILITY,
    TOKENS_PER_MILLION,
)

logger = logging.getLogger(__name__)


class ModelLimitV1(BaseModel):
    model_id: str
    context_window: int = DEFAULT_MODEL_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MODEL_MAX_OUTPUT_TOKENS
    is_reasoning_model: bool = False


class ProviderFeaturesV1(BaseModel):
    supports_sync: bool = True
    supports_async: bool = True
    supports_streaming: bool = False
    supports_json_object: bool = True
    supports_json_schema: bool = True
    supports_tools: bool = True
    supports_forced_tool_choice: bool = True
    supports_seed: bool = True
    supports_provider_usage: bool = True


class ParameterRulesV1(BaseModel):
    forbidden_parameters: List[str] = Field(default_factory=list)
    renamed_parameters: Dict[str, str] = Field(default_factory=dict)
    tool_name_pattern: str = DEFAULT_TOOL_NAME_PATTERN


class TransportDefaultsV1(BaseModel):
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS
    sdk_internal_retries: int = DEFAULT_SDK_RETRIES  # In WP5, always 0 on managed path
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY


class ProviderCapabilityV1(BaseModel):
    schema_version: str = SCHEMA_VERSION_PROVIDER_CAPABILITY
    provider_id: str
    adapter_kind: str  # e.g. AdapterKind.OPENAI_COMPATIBLE.value, AdapterKind.AZURE_OPENAI.value
    enabled: bool = True
    credential_status: str = CredentialStatus.READY.value
    models: List[ModelLimitV1]
    features: ProviderFeaturesV1 = Field(default_factory=ProviderFeaturesV1)
    parameter_rules: ParameterRulesV1 = Field(default_factory=ParameterRulesV1)
    transport_defaults: TransportDefaultsV1 = Field(default_factory=TransportDefaultsV1)
    price_table_version: str = DEFAULT_PRICE_TABLE_VERSION
    data_policy_ref: str = "standard_retention"
    health_ref: str = "healthy"


class ModelPriceV1(BaseModel):
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0


class PricingSnapshotV1(BaseModel):
    pricing_snapshot_id: str = DEFAULT_PRICING_SNAPSHOT_ID
    currency: str = DEFAULT_CURRENCY
    effective_time_utc: float = 0.0
    # Key format: "{provider_id}:{model_id}" -> ModelPriceV1
    rates: Dict[str, ModelPriceV1] = Field(default_factory=dict)

    def get_price(self, provider_id: str, model_id: str) -> ModelPriceV1:
        key = f"{provider_id}:{model_id}"
        if key in self.rates:
            return self.rates[key]
        # Fallback to model_id only or default 0.0
        if model_id in self.rates:
            return self.rates[model_id]
        return ModelPriceV1(input_price_per_1m=0.0, output_price_per_1m=0.0)

    def calculate_cost_usd(self, provider_id: str, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        rate = self.get_price(provider_id, model_id)
        cost = (prompt_tokens / TOKENS_PER_MILLION) * rate.input_price_per_1m + (
            completion_tokens / TOKENS_PER_MILLION
        ) * rate.output_price_per_1m
        return round(cost, 6)



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


class DynamicProviderRegistry(ProviderRegistry):
    """Provider registry with dynamic external catalog synchronization capabilities."""

    async def fetch_and_sync_catalog(
        self,
        url: str = DEFAULT_LITELLM_CATALOG_URL,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        """Fetch model context windows, max output limits, and pricing from an external JSON feed.

        Guarantees:
        - Strict Pydantic validation against ModelLimitV1 and ModelPriceV1.
        - Non-conforming/malformed entries are skipped individually.
        - Entire operation is wrapped in try/except; on network or JSON failure, it logs a warning
          and preserves all existing registered defaults without overwriting.
        """
        try:
            raw_data: Any = None
            if http_client is not None:
                resp = await http_client.get(url, timeout=timeout_seconds)
                resp.raise_for_status()
                raw_data = resp.json()
            else:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    raw_data = resp.json()

            if not isinstance(raw_data, dict):
                logger.warning(
                    "Dynamic catalog sync: fetched data from %s is not a JSON object (type=%s)",
                    url,
                    type(raw_data).__name__,
                )
                return

            self._apply_catalog_data(raw_data)
            logger.info("Dynamic catalog sync completed successfully from %s", url)

        except Exception as exc:
            logger.warning(
                "Dynamic catalog sync from %s failed gracefully; preserving existing defaults: %s",
                url,
                exc,
            )
            return

    def _apply_catalog_data(self, catalog_data: Dict[str, Any]) -> None:
        """Process, validate, and apply catalog entries to memory."""
        for raw_key, entry in catalog_data.items():
            if not isinstance(entry, dict) or not raw_key:
                continue

            # Parse prefix and model ID
            if "/" in raw_key:
                prefix, model_id = raw_key.split("/", 1)
            else:
                prefix = ""
                model_id = raw_key

            model_id = model_id.strip()
            if not model_id:
                continue

            prefix = prefix.lower().strip()
            litellm_provider = str(entry.get("litellm_provider", "")).lower().strip()

            # Map to target registered provider(s)
            target_providers: List[str] = []
            if prefix == "azure" or litellm_provider == "azure":
                target_providers = [ProviderId.AZURE_OPENAI.value]
            elif prefix == "anthropic" or litellm_provider == "anthropic":
                target_providers = [ProviderId.ANTHROPIC.value]
            elif prefix == "groq" or litellm_provider == "groq":
                target_providers = [ProviderId.GROQ.value]
            elif prefix in ("openai", "openai_base") or litellm_provider == "openai":
                target_providers = [p for p in (ProviderId.OPENAI.value, ProviderId.OPENAI_BASE.value) if p in self._providers]
            else:
                # Check if model already belongs to any registered provider
                for p_id, p_cap in self._providers.items():
                    if any(m.model_id.lower() == model_id.lower() for m in p_cap.models):
                        target_providers.append(p_id)

            # 1. Parse & validate ModelLimitV1
            self._sync_model_limit(entry, model_id, target_providers)

            # 2. Parse & validate ModelPriceV1
            self._sync_model_price(entry, model_id, target_providers)

    def _sync_model_limit(self, entry: Dict[str, Any], model_id: str, target_providers: List[str]) -> None:
        """Strictly validate and update model context window & max output tokens.

        Fix #9: Replaced bare ``except: pass`` with a debug-level log so that
        programming errors and genuinely malformed catalog entries are visible
        without crashing the sync operation.
        """
        try:
            cw_raw = entry.get("max_input_tokens") or entry.get("context_window")
            context_window = int(cw_raw) if cw_raw is not None else DEFAULT_MODEL_CONTEXT_WINDOW

            mot_raw = entry.get("max_output_tokens") or entry.get("max_tokens")
            max_output_tokens = int(mot_raw) if mot_raw is not None else DEFAULT_MODEL_MAX_OUTPUT_TOKENS

            is_reasoning = bool(
                entry.get("is_reasoning_model", False)
                or model_id.lower().startswith(("o1", "o3", "o4"))
            )

            # Strict Pydantic instantiation
            limit = ModelLimitV1(
                model_id=model_id,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                is_reasoning_model=is_reasoning,
            )

            # Safely update or append to matching registered providers
            for p_id in target_providers:
                p_cap = self._providers.get(p_id)
                if not p_cap:
                    continue
                existing = next((m for m in p_cap.models if m.model_id.lower() == model_id.lower()), None)
                if existing:
                    existing.context_window = limit.context_window
                    existing.max_output_tokens = limit.max_output_tokens
                    existing.is_reasoning_model = limit.is_reasoning_model
                else:
                    p_cap.models.append(limit)

        except Exception as exc:  # Fix #9 – was bare ``except: pass``
            logger.debug(
                "Dynamic catalog sync: skipping malformed model-limit entry for '%s': %s",
                model_id,
                exc,
            )

    def _sync_model_price(
        self,
        entry: Dict[str, Any],
        model_id: str,
        target_providers: List[str],
    ) -> None:
        """Strictly validate and update model pricing rates.

        Fix #9: Replaced bare ``except: pass`` with a debug-level log.
        """
        try:
            in_price_1m = entry.get("input_price_per_1m")
            out_price_1m = entry.get("output_price_per_1m")
            in_cost_token = entry.get("input_cost_per_token")
            out_cost_token = entry.get("output_cost_per_token")

            if in_price_1m is not None:
                final_in = float(in_price_1m)
            elif in_cost_token is not None:
                final_in = round(float(in_cost_token) * TOKENS_PER_MILLION, 6)
            else:
                final_in = None

            if out_price_1m is not None:
                final_out = float(out_price_1m)
            elif out_cost_token is not None:
                final_out = round(float(out_cost_token) * TOKENS_PER_MILLION, 6)
            else:
                final_out = None

            if final_in is None and final_out is None:
                return

            # Strict Pydantic instantiation
            price = ModelPriceV1(
                input_price_per_1m=float(final_in or 0.0),
                output_price_per_1m=float(final_out or 0.0),
            )

            # Safely update existing pricing snapshots
            target_snapshots = list(self._pricing_snapshots.values())
            if not target_snapshots:
                default_snap = self.get_pricing_snapshot(DEFAULT_PRICING_SNAPSHOT_ID)
                target_snapshots = [default_snap]

            for snap in target_snapshots:
                if target_providers:
                    for p_id in target_providers:
                        snap.rates[f"{p_id}:{model_id}"] = price
                # Store model_id directly for fallback rate lookups
                snap.rates[model_id] = price

        except Exception as exc:  # Fix #9 – was bare ``except: pass``
            logger.debug(
                "Dynamic catalog sync: skipping malformed pricing entry for '%s': %s",
                model_id,
                exc,
            )



# Global registry singleton (DynamicProviderRegistry subclass of ProviderRegistry)
_GLOBAL_REGISTRY = DynamicProviderRegistry()


def get_provider_registry() -> DynamicProviderRegistry:
    return _GLOBAL_REGISTRY


def get_dynamic_provider_registry() -> DynamicProviderRegistry:
    return _GLOBAL_REGISTRY
