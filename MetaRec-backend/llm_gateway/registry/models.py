"""Provider capability and limits data models for LLM Gateway Registry (v1)."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field

from llm_gateway.constants import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    DEFAULT_PRICE_TABLE_VERSION,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DEFAULT_SDK_RETRIES,
    DEFAULT_TOOL_NAME_PATTERN,
    CredentialStatus,
    SCHEMA_VERSION_PROVIDER_CAPABILITY,
)


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
