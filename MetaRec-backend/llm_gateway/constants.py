"""Centralized constants, enums, and default configurations for the LLM Gateway.

Replaces scattered magic numbers and strings across adapters, engine, contracts,
and registries with descriptive, strongly-typed constants and enums.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Provider & Adapter Identifiers
# ---------------------------------------------------------------------------

class ProviderId(str, Enum):
    """Canonical provider IDs supported by the gateway."""
    FAKE = "fake_provider"
    GROQ = "groq"
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    OPENAI_BASE = "openai_base"
    ANTHROPIC = "anthropic"


class AdapterKind(str, Enum):
    """Underlying adapter implementation family."""
    FAKE = "fake"
    OPENAI_COMPATIBLE = "openai_compatible"
    AZURE_OPENAI = "azure_openai"
    OPENAI_BASE = "openai_base"
    OPENAI_HTTP = "openai_http"
    ANTHROPIC = "anthropic"


class CredentialStatus(str, Enum):
    """Lifecycle / readiness state of provider credentials."""
    READY = "ready"
    MISSING = "missing"
    INVALID = "invalid"
    UNCHECKED = "unchecked"


# ---------------------------------------------------------------------------
# Error Codes & Reasons
# ---------------------------------------------------------------------------

class GatewayErrorCode(str, Enum):
    """Standardized error taxonomy code strings."""
    # Request class
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_SCHEMA = "unknown_schema"
    INVALID_TOOL_SCOPE = "invalid_tool_scope"
    # Policy class
    UNSAFE_SITE = "unsafe_site"
    PROVIDER_NOT_ALLOWED = "provider_not_allowed"
    CAPABILITY_MISMATCH = "capability_mismatch"
    PRIVACY_POLICY_MISMATCH = "privacy_policy_mismatch"
    # Budget class
    DEADLINE_EXHAUSTED = "deadline_exhausted"
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    # Transport class
    CONNECT_ERROR = "connect_error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    PROVIDER_5XX = "provider_5xx"
    # Compatibility class
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    UNSUPPORTED_RESPONSE_FORMAT = "unsupported_response_format"
    UNSUPPORTED_TOOLS = "unsupported_tools"
    # Validation class
    VALIDATION_FAILED = "validation_failed"
    MALFORMED_JSON = "malformed_json"
    JSON_SCHEMA_VIOLATION = "json_schema_violation"
    SEMANTIC_VIOLATION = "semantic_violation"
    INVALID_TOOL_CALL = "invalid_tool_call"
    # Provider class
    AUTH_FAILED = "auth_failed"
    PROVIDER_4XX = "provider_4xx"
    PROVIDER_ERROR = "provider_error"
    CONTENT_BLOCKED = "content_blocked"
    EMPTY_RESPONSE = "empty_response"
    USAGE_MISSING = "usage_missing"
    CONFIGURATION_ERROR = "configuration_error"
    # Internal / Control class
    ADAPTER_ERROR = "adapter_error"
    VALIDATOR_ERROR = "validator_error"
    CANCELLED = "cancelled"
    UNKNOWN_FAILURE = "unknown_failure"
    KILL_SWITCH_ACTIVE = "kill_switch_active"


class DegradationReason(str, Enum):
    """Reasons why execution degraded to fallback policy legs."""
    FALLBACK_ROUTE_USED = "fallback_route_used"


# ---------------------------------------------------------------------------
# Semantic Validator Identifiers
# ---------------------------------------------------------------------------

class SemanticValidatorId(str, Enum):
    """Built-in registered semantic validator IDs."""
    NONE = "none"
    GATHER_ACTION = "gather_action.v1"
    CONFIRMATION_PAYLOAD = "confirmation_payload.v1"


# ---------------------------------------------------------------------------
# HTTP Status Codes & Status Classes
# ---------------------------------------------------------------------------

HTTP_STATUS_OK: int = 200
HTTP_STATUS_BAD_REQUEST: int = 400
HTTP_STATUS_UNAUTHORIZED: int = 401
HTTP_STATUS_FORBIDDEN: int = 403
HTTP_STATUS_TOO_MANY_REQUESTS: int = 429
HTTP_STATUS_INTERNAL_SERVER_ERROR: int = 500
HTTP_STATUS_BAD_GATEWAY: int = 502
HTTP_STATUS_SERVICE_UNAVAILABLE: int = 503
HTTP_STATUS_GATEWAY_TIMEOUT: int = 504
HTTP_STATUS_ANTHROPIC_OVERLOADED: int = 529

RETRYABLE_HTTP_STATUSES: tuple[int, ...] = (
    HTTP_STATUS_TOO_MANY_REQUESTS,
    HTTP_STATUS_INTERNAL_SERVER_ERROR,
    HTTP_STATUS_BAD_GATEWAY,
    HTTP_STATUS_SERVICE_UNAVAILABLE,
    HTTP_STATUS_GATEWAY_TIMEOUT,
    HTTP_STATUS_ANTHROPIC_OVERLOADED,
)

AUTH_HTTP_STATUSES: tuple[int, ...] = (
    HTTP_STATUS_UNAUTHORIZED,
    HTTP_STATUS_FORBIDDEN,
)


# ---------------------------------------------------------------------------
# Conversion & Mathematical Constants
# ---------------------------------------------------------------------------

TOKENS_PER_MILLION: float = 1_000_000.0
MS_PER_SECOND: float = 1000.0


# ---------------------------------------------------------------------------
# Budget & Generation Default Numbers
# ---------------------------------------------------------------------------

DEFAULT_GENERATION_MAX_OUTPUT_TOKENS: int = 1024
DEFAULT_BUDGET_DEADLINE_MS: int = 30000
DEFAULT_BUDGET_MAX_PHYSICAL_ATTEMPTS: int = 3
DEFAULT_BUDGET_MAX_OUTPUT_TOKENS: int = 2048
DEFAULT_BUDGET_MAX_COST_USD: float = 0.50

DEFAULT_MODEL_CONTEXT_WINDOW: int = 128000
DEFAULT_MODEL_MAX_OUTPUT_TOKENS: int = 4096

DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 10.0
DEFAULT_READ_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_CONCURRENCY: int = 20
DEFAULT_SDK_RETRIES: int = 0


# ---------------------------------------------------------------------------
# Back-off and Retry Constants
# ---------------------------------------------------------------------------

DEFAULT_BACKOFF_BASE_MS: float = 200.0
DEFAULT_BACKOFF_CAP_MS: float = 5000.0
DEFAULT_MAX_ATTEMPTS_PER_ROUTE: int = 2


# ---------------------------------------------------------------------------
# Schema Versions, Snapshots & Metadata
# ---------------------------------------------------------------------------

SCHEMA_VERSION_LLM_REQUEST: str = "llm-request.v1"
SCHEMA_VERSION_LLM_RESULT: str = "llm-result.v1"
SCHEMA_VERSION_PROVIDER_CAPABILITY: str = "llm-provider-capability.v1"

DEFAULT_PRICING_SNAPSHOT_ID: str = "default-pricing-v1"
DEFAULT_PRICE_TABLE_VERSION: str = "pricing-2026-v1"
DEFAULT_TEXT_SCHEMA_ID: str = "default.text.v1"
DEFAULT_HEALTH_SNAPSHOT_ID: str = "health-v1"
DEFAULT_CURRENCY: str = "USD"

DEFAULT_DATA_CLASS: str = "standard"
DEFAULT_SAFETY_CLASS: str = "standard"
DEFAULT_PRODUCER_ID: str = "default_producer"
DEFAULT_CONTEXT_VERSION: str = "1.0.0"
DEFAULT_TOOL_NAME_PATTERN: str = r"^[a-zA-Z0-9_-]+$"
DEFAULT_KILL_SWITCH_REASON: str = "Manual override"



# ---------------------------------------------------------------------------
# Endpoint & Remote Catalog Defaults
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_BASE_URL: str = "https://api.openai.com/v1"
DEFAULT_GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
DEFAULT_AZURE_API_VERSION: str = "2024-12-01-preview"
DEFAULT_ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION: str = "2023-06-01"
DEFAULT_LITELLM_CATALOG_URL: str = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)


# ---------------------------------------------------------------------------
# Policy Tags & Safety Context Producers
# ---------------------------------------------------------------------------

POLICY_TAG_CONVERSATION_SUMMARY: str = "conversation.summary"
POLICY_TAG_CONFIRMATION_PAYLOAD: str = "confirmation.payload"
POLICY_TAG_GATHER_ACTION: str = "gather.propose_action"
POLICY_TAG_INTENT_ANALYZE: str = "intent.analyze_message"
POLICY_TAG_ITINERARY_CONSTRAINTS: str = "itinerary.extract_constraints"
POLICY_TAG_ITINERARY_DURATIONS: str = "itinerary.enrich_durations"
POLICY_TAG_ITINERARY_ROLES: str = "itinerary.classify_roles"
POLICY_TAG_ITINERARY_CONTAINMENT: str = "itinerary.classify_containment"
POLICY_TAG_ITINERARY_REPAIR: str = "itinerary.propose_repair"
POLICY_TAG_AGENT_PLAN: str = "agent.restaurant_plan"
POLICY_TAG_AGENT_SUMMARY: str = "agent.restaurant_summary"

CHEAP_ROUTE_ELIGIBLE_TAGS: frozenset[str] = frozenset({
    POLICY_TAG_CONVERSATION_SUMMARY,
    POLICY_TAG_CONFIRMATION_PAYLOAD,
})

NEVER_CHEAP_ROUTE_TAGS: frozenset[str] = frozenset({
    POLICY_TAG_GATHER_ACTION,
    POLICY_TAG_INTENT_ANALYZE,
    POLICY_TAG_ITINERARY_CONSTRAINTS,
    POLICY_TAG_ITINERARY_DURATIONS,
    POLICY_TAG_ITINERARY_ROLES,
    POLICY_TAG_ITINERARY_CONTAINMENT,
    POLICY_TAG_ITINERARY_REPAIR,
    POLICY_TAG_AGENT_PLAN,
    POLICY_TAG_AGENT_SUMMARY,
})

PRODUCER_ID_SUMMARY_SAFETY: str = "summary_safety_context.v1"
PRODUCER_ID_CONFIRMATION_CONTEXT: str = "confirmation_context.v1"

REGISTERED_CONTEXT_PRODUCERS: frozenset[str] = frozenset({
    DEFAULT_PRODUCER_ID,
    PRODUCER_ID_SUMMARY_SAFETY,
    PRODUCER_ID_CONFIRMATION_CONTEXT,
    "test_producer",
    "test",
})

REGISTERED_CONTEXT_VERSIONS: frozenset[str] = frozenset({
    DEFAULT_CONTEXT_VERSION,
    "v1",
    PRODUCER_ID_SUMMARY_SAFETY,
    PRODUCER_ID_CONFIRMATION_CONTEXT,
})

DEFAULT_CONTEXT_MAX_AGE_SECONDS: float = 300.0
DEFAULT_CONTEXT_MAX_FUTURE_SKEW_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Fake Adapter Deterministic Defaults
# ---------------------------------------------------------------------------

DEFAULT_FAKE_PROMPT_TOKENS: int = 40
DEFAULT_FAKE_COMPLETION_TOKENS: int = 20
DEFAULT_SCRIPTED_PROMPT_TOKENS: int = 50
DEFAULT_SCRIPTED_COMPLETION_TOKENS: int = 25
DEFAULT_SCRIPTED_REQUEST_ID: str = "fake-req-001"
