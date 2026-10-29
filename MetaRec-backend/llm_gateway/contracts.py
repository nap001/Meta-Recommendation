"""Pydantic schemas and typed data contracts for LLM Gateway (v1)."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_gateway.constants import (
    DEFAULT_BUDGET_DEADLINE_MS,
    DEFAULT_BUDGET_MAX_COST_USD,
    DEFAULT_BUDGET_MAX_OUTPUT_TOKENS,
    DEFAULT_BUDGET_MAX_PHYSICAL_ATTEMPTS,
    DEFAULT_CONTEXT_VERSION,
    DEFAULT_DATA_CLASS,
    DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
    DEFAULT_HEALTH_SNAPSHOT_ID,
    DEFAULT_PRICE_TABLE_VERSION,
    DEFAULT_PRICING_SNAPSHOT_ID,
    DEFAULT_PRODUCER_ID,
    DEFAULT_SAFETY_CLASS,
    DEFAULT_TEXT_SCHEMA_ID,
    SCHEMA_VERSION_LLM_REQUEST,
    SCHEMA_VERSION_LLM_RESULT,
    SemanticValidatorId,
)


class RequestRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageV1(BaseModel):
    role: RequestRole
    content: str
    name: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class GenerationConfigV1(BaseModel):
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=DEFAULT_GENERATION_MAX_OUTPUT_TOKENS, gt=0)
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    model_config = ConfigDict(extra="forbid")


class ResponseKind(str, Enum):
    TEXT = "text"
    JSON = "json"
    TOOL_CALLS = "tool_calls"


class ResponseContractV1(BaseModel):
    kind: ResponseKind = ResponseKind.TEXT
    schema_id: str = DEFAULT_TEXT_SCHEMA_ID
    json_schema: Optional[Dict[str, Any]] = None
    semantic_validator_id: str = SemanticValidatorId.NONE.value
    validator_inputs: Dict[str, Any] = Field(default_factory=dict)
    allow_empty: bool = False
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_schema_if_json(self) -> "ResponseContractV1":
        """Enforce that json_schema is provided and non-empty when kind is JSON."""
        if self.kind == ResponseKind.JSON:
            if not self.json_schema or not isinstance(self.json_schema, dict):
                raise ValueError("json_schema is strictly required and must be a non-empty dictionary when kind is 'json'")
        return self


class PolicyLeg(str, Enum):
    BASELINE = "baseline"
    CHEAP = "cheap"
    ESCALATION = "escalation"
    FALLBACK = "fallback"


class RouteV1(BaseModel):
    provider_id: str
    model_id: str
    policy_leg: PolicyLeg = PolicyLeg.BASELINE
    alias: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class RoutingMode(str, Enum):
    BASELINE = "baseline"
    CHEAP_FIRST = "cheap_first"


class RoutingPolicyV1(BaseModel):
    mode: RoutingMode = RoutingMode.BASELINE
    baseline_route: RouteV1
    cheap_route: Optional[RouteV1] = None
    fallback_routes: List[RouteV1] = Field(default_factory=list)
    allowed_provider_ids: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class BudgetConfigV1(BaseModel):
    deadline_ms: int = Field(default=DEFAULT_BUDGET_DEADLINE_MS, gt=0)  # End-to-end deadline
    max_physical_attempts: int = Field(default=DEFAULT_BUDGET_MAX_PHYSICAL_ATTEMPTS, ge=1, le=10)
    max_input_tokens: Optional[int] = Field(default=None, gt=0)
    max_output_tokens: int = Field(default=DEFAULT_BUDGET_MAX_OUTPUT_TOKENS, gt=0)
    max_total_tokens: Optional[int] = Field(default=None, gt=0)
    max_cost_usd: float = Field(default=DEFAULT_BUDGET_MAX_COST_USD, gt=0.0)
    model_config = ConfigDict(extra="forbid")


class RequestMode(str, Enum):
    SINGLE_DOMAIN = "single_domain"
    MULTI_DOMAIN = "multi_domain"
    ITINERARY = "itinerary"
    DOMAIN_ERROR = "domain_error"


class PolicyContextV1(BaseModel):
    """Call-site safety facts. contains_itinerary_state and request_mode MUST be explicitly provided."""
    contains_itinerary_state: bool  # Required, no default
    request_mode: RequestMode        # Required, no default
    safety_class: str = DEFAULT_SAFETY_CLASS
    producer_id: str = DEFAULT_PRODUCER_ID
    context_version: str = DEFAULT_CONTEXT_VERSION
    produced_at_utc: float = Field(default_factory=time.time)
    model_config = ConfigDict(extra="forbid")


class PrivacyConfigV1(BaseModel):
    data_class: str = DEFAULT_DATA_CLASS
    allow_prompt_logging: bool = False
    allow_response_cache: bool = False
    model_config = ConfigDict(extra="forbid")


class TraceContextV1(BaseModel):
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class ToolSpecV1(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ScopedToolCatalogV1(BaseModel):
    catalog_id: str
    health_snapshot_id: str = DEFAULT_HEALTH_SNAPSHOT_ID
    tools: List[ToolSpecV1] = Field(default_factory=list)
    excluded: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class LLMToolDefinitionV1(BaseModel):
    """Outbound tool definition projected from WP7 tool catalog (WP5 spec L226-234)."""
    source_tool_name: str
    llm_tool_name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class LLMToolProjectionV1(BaseModel):
    """Internal tool projection for selected LLM source (WP5 spec L226-234)."""
    projection_id: str
    catalog_id: str
    health_snapshot_id: str
    provider_id: str
    definitions: List[LLMToolDefinitionV1] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class LLMRequestV1(BaseModel):
    schema_version: str = SCHEMA_VERSION_LLM_REQUEST
    request_id: str
    policy_tag: str
    messages: List[MessageV1]
    generation: GenerationConfigV1 = Field(default_factory=GenerationConfigV1)
    response_contract: ResponseContractV1 = Field(default_factory=ResponseContractV1)
    routing_policy: RoutingPolicyV1
    budget: BudgetConfigV1 = Field(default_factory=BudgetConfigV1)
    pricing_snapshot_id: str = DEFAULT_PRICING_SNAPSHOT_ID
    tools: Optional[ScopedToolCatalogV1] = None
    tool_choice: Optional[str] = None
    policy_context: PolicyContextV1
    trace_context: Optional[TraceContextV1] = None
    privacy: PrivacyConfigV1 = Field(default_factory=PrivacyConfigV1)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_tool_choice(self) -> "LLMRequestV1":
        """Ensure tool_choice is 'none', 'auto', or belongs to scoped catalog and not excluded."""
        if self.tool_choice is not None:
            if self.tool_choice in ("none", "auto"):
                return self
            if not self.tools:
                raise ValueError(f"Forced tool_choice '{self.tool_choice}' specified but no tools catalog provided")
            excluded = set(self.tools.excluded) if self.tools.excluded else set()
            allowed_names = {t.name for t in self.tools.tools} - excluded
            if self.tool_choice not in allowed_names:
                raise ValueError(
                    f"Forced tool_choice '{self.tool_choice}' is not present in scoped catalog tools: {allowed_names}"
                )
        return self


# ── Attempt & Validation Models ──────────────────────────────────────────────

class AttemptKind(str, Enum):
    INITIAL = "initial"
    TRANSPORT_RETRY = "transport_retry"
    COMPATIBILITY_RETRY = "compatibility_retry"
    VALIDATION_RETRY = "validation_retry"
    ROUTE_FALLBACK = "route_fallback"


class AttemptOutcome(str, Enum):
    # Two-value contract outcomes (for evaluation and cost accounting alignment)
    RESPONSE_RETURNED = "response_returned"
    NO_RESPONSE_RETURNED = "no_response_returned"

    # Detailed attempt outcomes
    SUCCESS = "success"
    TRANSPORT_ERROR = "transport_error"
    COMPATIBILITY_ERROR = "compatibility_error"
    VALIDATION_ERROR = "validation_error"
    PROVIDER_ERROR = "provider_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ValidationGateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class ValidationSummaryV1(BaseModel):
    json_parse: ValidationGateStatus = ValidationGateStatus.NOT_APPLICABLE
    json_schema: ValidationGateStatus = ValidationGateStatus.NOT_APPLICABLE
    semantic: ValidationGateStatus = ValidationGateStatus.NOT_APPLICABLE
    validator_id: str = SemanticValidatorId.NONE.value
    errors: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class PhysicalAttemptV1(BaseModel):
    attempt_id: str
    request_id: str
    attempt_number: int
    provider_id: str
    model_id: str
    policy_leg: PolicyLeg
    attempt_kind: AttemptKind
    started_at_utc: float
    ended_at_utc: Optional[float] = None
    latency_ms: float
    outcome: AttemptOutcome
    detailed_outcome: Optional[AttemptOutcome] = None
    http_status: Optional[int] = None
    provider_request_id: Optional[str] = None
    # Usage Provenance: None indicates unreported/missing usage; 0 means explicitly reported 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    is_usage_reported: bool = False
    cost_usd: float = 0.0
    price_table_version: str = DEFAULT_PRICE_TABLE_VERSION
    applied_compatibility_changes: List[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    validation_summary: Optional[ValidationSummaryV1] = None
    model_config = ConfigDict(extra="forbid")

    @property
    def contract_outcome(self) -> AttemptOutcome:
        """Map internal attempt outcome to two-value contract: response_returned / no_response_returned."""
        if self.outcome in (
            AttemptOutcome.RESPONSE_RETURNED,
            AttemptOutcome.SUCCESS,
            AttemptOutcome.VALIDATION_ERROR,
        ):
            return AttemptOutcome.RESPONSE_RETURNED
        return AttemptOutcome.NO_RESPONSE_RETURNED

    @property
    def provider_response_received(self) -> bool:
        """True if an actual completion HTTP response was returned from the provider."""
        return self.contract_outcome == AttemptOutcome.RESPONSE_RETURNED


class ToolCallOutputV1(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class OutputPayloadV1(BaseModel):
    kind: ResponseKind
    text: Optional[str] = None
    json_payload: Optional[Union[Dict[str, Any], List[Any]]] = Field(default=None, alias="json")
    tool_calls: List[ToolCallOutputV1] = Field(default_factory=list)
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    FAILED = "failed"


class UsageSummaryV1(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    pricing_snapshot_id: str = DEFAULT_PRICING_SNAPSHOT_ID
    is_estimated: bool = False
    model_config = ConfigDict(extra="forbid")


class TimingSummaryV1(BaseModel):
    started_at_utc: float
    completed_at_utc: float
    total_latency_ms: float
    deadline_ms: int
    model_config = ConfigDict(extra="forbid")


class DegradationSummaryV1(BaseModel):
    reason_code: str
    from_route: Optional[RouteV1] = None
    to_route: Optional[RouteV1] = None
    released_fallback: bool = True
    model_config = ConfigDict(extra="forbid")


class LLMErrorPayloadV1(BaseModel):
    """Pydantic representation of LLMErrorV1 for serialized result payloads."""
    code: str
    error_class: str
    message: str
    retryable: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)
    last_attempt_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, item: str) -> Any:
        """Provide dictionary-like subscripting for backward compatibility."""
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


LLMErrorV1 = LLMErrorPayloadV1  # Schema alias for contract consistency


class ValidationGateResultV1(BaseModel):
    """Structured dependency injection response for external validation modules."""
    passed: bool
    summary: ValidationSummaryV1
    payload: Optional[OutputPayloadV1] = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_payload_consistency(self) -> "ValidationGateResultV1":
        """Enforce: passed=True -> payload present; passed=False -> payload absent."""
        if self.passed and self.payload is None:
            raise ValueError("ValidationGateResultV1 with passed=True must have a non-null payload")
        if not self.passed and self.payload is not None:
            raise ValueError("ValidationGateResultV1 with passed=False must have payload=None")
        return self



class LLMResultV1(BaseModel):
    schema_version: str = SCHEMA_VERSION_LLM_RESULT
    request_id: str
    policy_tag: str
    status: ResultStatus
    output: Optional[OutputPayloadV1] = None
    selected_route: Optional[RouteV1] = None
    validation: ValidationSummaryV1 = Field(default_factory=ValidationSummaryV1)
    attempts: List[PhysicalAttemptV1] = Field(default_factory=list)
    usage: UsageSummaryV1 = Field(default_factory=UsageSummaryV1)
    timing: TimingSummaryV1
    degradation: Optional[DegradationSummaryV1] = None
    error: Optional[LLMErrorPayloadV1] = None
    trace: Optional[TraceContextV1] = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "LLMResultV1":
        """Enforce strict result invariants based on status."""
        if self.status == ResultStatus.FAILED:
            if self.output is not None:
                raise ValueError("Failed result must have output=None")
            if self.selected_route is not None:
                raise ValueError("Failed result must have selected_route=None")
            if self.error is None:
                raise ValueError("Failed result must have a populated error payload")
        elif self.status in (ResultStatus.SUCCEEDED, ResultStatus.DEGRADED):
            if self.output is None:
                raise ValueError("Succeeded or degraded result must have a released output payload")
            if self.selected_route is None:
                raise ValueError("Succeeded or degraded result must have a selected_route")
            if self.error is not None:
                raise ValueError("Succeeded or degraded result must have error=None")
        return self
