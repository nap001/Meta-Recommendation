"""Pricing models and cost calculation for LLM Gateway Registry (v1)."""

from __future__ import annotations

from typing import Dict
from pydantic import BaseModel, Field

from llm_gateway.constants import (
    DEFAULT_CURRENCY,
    DEFAULT_PRICING_SNAPSHOT_ID,
    TOKENS_PER_MILLION,
)


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

    def calculate_cost_usd(
        self, provider_id: str, model_id: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        rate = self.get_price(provider_id, model_id)
        cost = (prompt_tokens / TOKENS_PER_MILLION) * rate.input_price_per_1m + (
            completion_tokens / TOKENS_PER_MILLION
        ) * rate.output_price_per_1m
        return round(cost, 8)
