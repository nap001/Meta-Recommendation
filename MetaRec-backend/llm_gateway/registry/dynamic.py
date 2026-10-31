"""Dynamic external catalog synchronization and SSRF validation for LLM Gateway (v1)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

from llm_gateway.constants import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_LITELLM_CATALOG_URL,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    DEFAULT_PRICING_SNAPSHOT_ID,
    ProviderId,
    TOKENS_PER_MILLION,
)
from llm_gateway.registry.models import ModelLimitV1
from llm_gateway.registry.pricing import ModelPriceV1
from llm_gateway.registry.store import ProviderRegistry

logger = logging.getLogger(__name__)


def _validate_catalog_url(url: str) -> None:
    """Validate external catalog URL to prevent SSRF against internal/metadata services."""
    import ipaddress
    import urllib.parse

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"SSRF Protection: Only HTTPS catalog URLs are permitted, got '{parsed.scheme}'.")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("SSRF Protection: Catalog URL must specify a valid hostname.")

    # Check for direct IP address
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"SSRF Protection: Access to private/internal IP {ip} is forbidden.")
        # Specific check for cloud instance metadata service (169.254.169.254)
        if str(ip) == "169.254.169.254":
            raise ValueError("SSRF Protection: Access to cloud instance metadata service is forbidden.")
    except ValueError as val_err:
        if "SSRF Protection" in str(val_err):
            raise
        # Hostname string check
        if hostname in ("localhost", "127.0.0.1", "::1", "metadata.google.internal") or hostname.endswith(".local") or hostname.endswith(".internal"):
            raise ValueError(f"SSRF Protection: Access to internal host '{hostname}' is forbidden.")


class DynamicProviderRegistry(ProviderRegistry):
    """Provider registry with dynamic external catalog synchronization capabilities."""

    async def fetch_and_sync_catalog(
        self,
        url: str = DEFAULT_LITELLM_CATALOG_URL,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        """Fetch model context windows, max output limits, and pricing from an external JSON feed."""
        try:
            _validate_catalog_url(url)
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
        """Strictly validate and update model context window & max output tokens."""
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

        except Exception as exc:
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
        """Strictly validate and update model pricing rates."""
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

        except Exception as exc:
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
