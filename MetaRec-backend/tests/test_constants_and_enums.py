"""Unit tests for LLM Gateway constants, enums, and centralized configuration."""

from __future__ import annotations

import pytest

from llm_gateway import (
    AdapterKind,
    AttemptOutcome,
    BudgetConfigV1,
    ConfigurationError,
    CredentialStatus,
    DegradationReason,
    GatewayErrorCode,
    GenerationConfigV1,
    LLMRequestV1,
    MessageV1,
    OutputPayloadV1,
    PhysicalAttemptV1,
    PolicyContextV1,
    PolicyLeg,
    PricingSnapshotV1,
    ProviderCapabilityV1,
    ProviderId,
    RequestMode,
    RequestRole,
    ResponseContractV1,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    RoutingPolicyV1,
    SemanticValidatorId,
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
)
from llm_gateway.constants import (
    AUTH_HTTP_STATUSES,
    DEFAULT_BACKOFF_BASE_MS,
    DEFAULT_BACKOFF_CAP_MS,
    DEFAULT_BUDGET_DEADLINE_MS,
    DEFAULT_BUDGET_MAX_COST_USD,
    DEFAULT_BUDGET_MAX_OUTPUT_TOKENS,
    DEFAULT_BUDGET_MAX_PHYSICAL_ATTEMPTS,
    DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
    DEFAULT_KILL_SWITCH_REASON,
    DEFAULT_MAX_ATTEMPTS_PER_ROUTE,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    DEFAULT_PRICING_SNAPSHOT_ID,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_INTERNAL_SERVER_ERROR,
    HTTP_STATUS_OK,
    HTTP_STATUS_TOO_MANY_REQUESTS,
    HTTP_STATUS_UNAUTHORIZED,
    MS_PER_SECOND,
    RETRYABLE_HTTP_STATUSES,
    TOKENS_PER_MILLION,
)
from llm_gateway.kill_switch import (
    KillSwitchManager,
    get_kill_switch_reason,
    is_kill_switch_active,
    set_kill_switch,
)

pytestmark = pytest.mark.backend_unit


def test_provider_id_enum_and_string_compatibility():
    """ProviderId enums must evaluate equal to raw strings for backwards compatibility."""
    assert ProviderId.FAKE == "fake_provider"
    assert ProviderId.GROQ == "groq"
    assert ProviderId.AZURE_OPENAI == "azure_openai"
    assert ProviderId.OPENAI == "openai"
    assert ProviderId.OPENAI_BASE == "openai_base"
    assert ProviderId.ANTHROPIC == "anthropic"

    assert isinstance(ProviderId.FAKE, str)
    assert ProviderId("fake_provider") == ProviderId.FAKE


def test_adapter_kind_enum():
    """AdapterKind enums must match expected adapter identifiers."""
    assert AdapterKind.FAKE == "fake"
    assert AdapterKind.OPENAI_COMPATIBLE == "openai_compatible"
    assert AdapterKind.AZURE_OPENAI == "azure_openai"
    assert AdapterKind.OPENAI_BASE == "openai_base"
    assert AdapterKind.OPENAI_HTTP == "openai_http"
    assert AdapterKind.ANTHROPIC == "anthropic"


def test_credential_status_enum():
    """CredentialStatus enums must match provider readiness states."""
    assert CredentialStatus.READY == "ready"
    assert CredentialStatus.MISSING == "missing"
    assert CredentialStatus.INVALID == "invalid"
    assert CredentialStatus.UNCHECKED == "unchecked"


def test_gateway_error_codes():
    """GatewayErrorCode enums must match error taxonomy and be string-compatible."""
    assert GatewayErrorCode.RATE_LIMITED == "rate_limited"
    assert GatewayErrorCode.TIMEOUT == "timeout"
    assert GatewayErrorCode.CONNECT_ERROR == "connect_error"
    assert GatewayErrorCode.PROVIDER_5XX == "provider_5xx"
    assert GatewayErrorCode.UNSUPPORTED_PARAMETER == "unsupported_parameter"
    assert GatewayErrorCode.AUTH_FAILED == "auth_failed"
    assert GatewayErrorCode.INVALID_REQUEST == "invalid_request"
    assert GatewayErrorCode.PROVIDER_ERROR == "provider_error"
    assert GatewayErrorCode.VALIDATION_FAILED == "validation_failed"
    assert GatewayErrorCode.DEADLINE_EXHAUSTED == "deadline_exhausted"
    assert GatewayErrorCode.ATTEMPT_BUDGET_EXHAUSTED == "attempt_budget_exhausted"
    assert GatewayErrorCode.COST_BUDGET_EXHAUSTED == "cost_budget_exhausted"
    assert GatewayErrorCode.CONFIGURATION_ERROR == "configuration_error"
    assert GatewayErrorCode.UNKNOWN_FAILURE == "unknown_failure"


def test_degradation_and_validator_enums():
    """Degradation reasons and semantic validator IDs must match system defaults."""
    assert DegradationReason.FALLBACK_ROUTE_USED == "fallback_route_used"
    assert SemanticValidatorId.NONE == "none"
    assert SemanticValidatorId.GATHER_ACTION == "gather_action.v1"
    assert SemanticValidatorId.CONFIRMATION_PAYLOAD == "confirmation_payload.v1"


def test_configuration_error_exception():
    """ConfigurationError should be a ProviderError with CONFIGURATION_ERROR code."""
    err = ConfigurationError("Azure endpoint missing")
    assert err.code == GatewayErrorCode.CONFIGURATION_ERROR.value
    assert err.retryable is False
    assert "Azure endpoint missing" in str(err)
    err_dict = err.to_dict()
    assert err_dict["code"] == "configuration_error"
    assert err_dict["error_class"] == "provider"


def test_http_status_constants():
    """HTTP status codes and category groupings must be consistent."""
    assert HTTP_STATUS_OK == 200
    assert HTTP_STATUS_BAD_REQUEST == 400
    assert HTTP_STATUS_UNAUTHORIZED == 401
    assert HTTP_STATUS_FORBIDDEN == 403
    assert HTTP_STATUS_TOO_MANY_REQUESTS == 429
    assert HTTP_STATUS_INTERNAL_SERVER_ERROR == 500

    assert HTTP_STATUS_TOO_MANY_REQUESTS in RETRYABLE_HTTP_STATUSES
    assert HTTP_STATUS_INTERNAL_SERVER_ERROR in RETRYABLE_HTTP_STATUSES
    assert 529 in RETRYABLE_HTTP_STATUSES

    assert HTTP_STATUS_UNAUTHORIZED in AUTH_HTTP_STATUSES
    assert HTTP_STATUS_FORBIDDEN in AUTH_HTTP_STATUSES


def test_contract_defaults_use_constants():
    """Default values in contract models must match centralized constants."""
    budget = BudgetConfigV1()
    assert budget.deadline_ms == DEFAULT_BUDGET_DEADLINE_MS
    assert budget.max_physical_attempts == DEFAULT_BUDGET_MAX_PHYSICAL_ATTEMPTS
    assert budget.max_output_tokens == DEFAULT_BUDGET_MAX_OUTPUT_TOKENS
    assert budget.max_cost_usd == DEFAULT_BUDGET_MAX_COST_USD

    gen = GenerationConfigV1()
    assert gen.max_output_tokens == DEFAULT_GENERATION_MAX_OUTPUT_TOKENS

    resp = ResponseContractV1()
    assert resp.semantic_validator_id == SemanticValidatorId.NONE.value


def test_token_math_and_pricing_calculation():
    """Pricing calculation must accurately use TOKENS_PER_MILLION divisor."""
    assert TOKENS_PER_MILLION == 1_000_000.0
    assert MS_PER_SECOND == 1000.0

    registry = get_provider_registry()
    snap = registry.get_pricing_snapshot(DEFAULT_PRICING_SNAPSHOT_ID)
    # fake-fast is $0.10 input / 1M, $0.20 output / 1M
    cost = snap.calculate_cost_usd("fake_provider", "fake-fast", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(0.30, rel=1e-5)


def test_kill_switch_default_reason():
    """KillSwitch should use DEFAULT_KILL_SWITCH_REASON when reason is omitted."""
    set_kill_switch(True)
    try:
        assert is_kill_switch_active() is True
        assert get_kill_switch_reason() == DEFAULT_KILL_SWITCH_REASON
    finally:
        set_kill_switch(False)


def test_attempt_outcome_contract_and_detailed_mapping():
    """AttemptOutcome contract enum values and PhysicalAttemptV1 mapping."""
    assert AttemptOutcome.RESPONSE_RETURNED == "response_returned"
    assert AttemptOutcome.NO_RESPONSE_RETURNED == "no_response_returned"
    assert AttemptOutcome.SUCCESS == "success"
    assert AttemptOutcome.VALIDATION_ERROR == "validation_error"
    assert AttemptOutcome.TRANSPORT_ERROR == "transport_error"

    # Verify PhysicalAttemptV1 mapping behavior
    att_success = PhysicalAttemptV1(
        attempt_id="att-1",
        request_id="req-1",
        attempt_number=1,
        provider_id="fake_provider",
        model_id="fake-strong",
        policy_leg=PolicyLeg.BASELINE,
        attempt_kind="initial",
        started_at_utc=100.0,
        latency_ms=10.0,
        outcome=AttemptOutcome.SUCCESS,
        detailed_outcome=AttemptOutcome.SUCCESS,
    )
    assert att_success.contract_outcome == AttemptOutcome.RESPONSE_RETURNED
    assert att_success.provider_response_received is True
    assert att_success.detailed_outcome == AttemptOutcome.SUCCESS

    att_val_err = PhysicalAttemptV1(
        attempt_id="att-2",
        request_id="req-1",
        attempt_number=2,
        provider_id="fake_provider",
        model_id="fake-strong",
        policy_leg=PolicyLeg.BASELINE,
        attempt_kind="validation_retry",
        started_at_utc=100.0,
        latency_ms=10.0,
        outcome=AttemptOutcome.VALIDATION_ERROR,
    )
    assert att_val_err.contract_outcome == AttemptOutcome.RESPONSE_RETURNED
    assert att_val_err.provider_response_received is True

    att_transport_err = PhysicalAttemptV1(
        attempt_id="att-3",
        request_id="req-1",
        attempt_number=3,
        provider_id="fake_provider",
        model_id="fake-strong",
        policy_leg=PolicyLeg.BASELINE,
        attempt_kind="transport_retry",
        started_at_utc=100.0,
        latency_ms=10.0,
        outcome=AttemptOutcome.TRANSPORT_ERROR,
    )
    assert att_transport_err.contract_outcome == AttemptOutcome.NO_RESPONSE_RETURNED
    assert att_transport_err.provider_response_received is False


def test_patched_error_taxonomy_and_policy_budgets():
    """Verify full error taxonomy codes, streaming capability flags, price_table_version, and policy filters."""
    from llm_gateway.constants import DEFAULT_PRICE_TABLE_VERSION
    from llm_gateway.contracts import PrivacyConfigV1
    from llm_gateway.engine import create_engine
    from llm_gateway.registry import ProviderRegistry

    assert GatewayErrorCode.TOKEN_BUDGET_EXHAUSTED == "token_budget_exhausted"
    assert GatewayErrorCode.PROVIDER_NOT_ALLOWED == "provider_not_allowed"
    assert GatewayErrorCode.CAPABILITY_MISMATCH == "capability_mismatch"
    assert GatewayErrorCode.PRIVACY_POLICY_MISMATCH == "privacy_policy_mismatch"

    reg = ProviderRegistry()
    for prov in reg.list_providers():
        assert prov.features.supports_streaming is True

    engine = create_engine(registry=reg)
    req_ok = LLMRequestV1(
        request_id="req-ptv-1",
        policy_tag="intent.analyze_message",
        messages=[MessageV1(role=RequestRole.USER, content="hi")],
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-fast"),
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
    )
    res_ok = engine.invoke_llm(req_ok)
    assert res_ok.status == ResultStatus.SUCCEEDED
    assert res_ok.attempts[0].price_table_version == DEFAULT_PRICE_TABLE_VERSION

    # Disallowed provider check
    req_disallowed = LLMRequestV1(
        request_id="req-disallow-1",
        policy_tag="intent.analyze_message",
        messages=[MessageV1(role=RequestRole.USER, content="hi")],
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-fast"),
            allowed_provider_ids=["openai"],
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
    )
    res_disallowed = engine.invoke_llm(req_disallowed)
    assert res_disallowed.status == ResultStatus.FAILED
    assert res_disallowed.error.code == GatewayErrorCode.PROVIDER_NOT_ALLOWED.value
    assert res_disallowed.error.error_class == "policy"

    # Privacy policy mismatch check
    req_restricted = LLMRequestV1(
        request_id="req-restricted-1",
        policy_tag="intent.analyze_message",
        messages=[MessageV1(role=RequestRole.USER, content="hi")],
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-fast"),
        ),
        privacy=PrivacyConfigV1(data_class="restricted"),
        policy_context=PolicyContextV1(
            contains_itinerary_state=False,
            request_mode=RequestMode.SINGLE_DOMAIN,
        ),
    )
    res_restricted = engine.invoke_llm(req_restricted)
    assert res_restricted.status == ResultStatus.FAILED
    assert res_restricted.error.code == GatewayErrorCode.PRIVACY_POLICY_MISMATCH.value
    assert res_restricted.error.error_class == "policy"


