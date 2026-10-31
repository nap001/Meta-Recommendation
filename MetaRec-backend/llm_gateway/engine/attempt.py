"""Physical attempt recording, privacy logging, and attempt execution loops."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.constants import (
    DegradationReason,
    GatewayErrorCode,
    HTTP_STATUS_INTERNAL_SERVER_ERROR,
    HTTP_STATUS_OK,
    MS_PER_SECOND,
)
from llm_gateway.contracts import (
    AttemptKind,
    AttemptOutcome,
    DegradationSummaryV1,
    LLMRequestV1,
    OutputPayloadV1,
    PhysicalAttemptV1,
    PolicyLeg,
    RouteV1,
    ValidationGateStatus,
    ValidationSummaryV1,
)
from llm_gateway.errors import (
    BudgetExhaustedError,
    CompatibilityError,
    LLMErrorV1,
    OutputValidationError,
    ProviderError,
    TransportError,
)
from llm_gateway.registry import ProviderRegistry
from llm_gateway.engine.projection import normalize_request_kwargs
from llm_gateway.engine.resilience import RouteCircuitBreaker, _jitter_sleep_seconds
from llm_gateway.engine.validation_runner import (
    execute_validation_async,
    execute_validation_sync,
    is_payload_contract_compatible,
    is_registered_contract,
    is_terminal_validation_error,
)

logger = logging.getLogger(__name__)


def record_attempt(
    registry: ProviderRegistry,
    request: LLMRequestV1,
    route: RouteV1,
    leg: PolicyLeg,
    attempt_number: int,
    attempt_kind: AttemptKind,
    started_at: float,
    latency_ms: float,
    outcome: AttemptOutcome,
    resp: Optional[RawAdapterResponse] = None,
    error: Optional[Exception] = None,
    applied_changes: Optional[List[str]] = None,
    validation_summary: Optional[ValidationSummaryV1] = None,
) -> PhysicalAttemptV1:
    """Create a PhysicalAttemptV1 record and record in usage ledger."""
    attempt_id = f"att-{uuid.uuid4().hex[:12]}"
    pricing = registry.get_pricing_snapshot(request.pricing_snapshot_id)
    ended_at = started_at + (latency_ms / MS_PER_SECOND)
    started_at_utc = round(started_at, 6)
    ended_at_utc = max(started_at_utc, round(ended_at, 6))

    # Distinguish missing usage (None) vs explicitly reported 0
    if resp is not None:
        p_tokens = resp.prompt_tokens
        c_tokens = resp.completion_tokens
        t_tokens = resp.total_tokens
        is_usage_reported = resp.is_usage_reported if (p_tokens is not None) else False
    else:
        p_tokens = getattr(error, "prompt_tokens", None)
        c_tokens = getattr(error, "completion_tokens", None)
        t_tokens = getattr(error, "total_tokens", None)
        is_usage_reported = p_tokens is not None

    if is_usage_reported and p_tokens is not None and c_tokens is not None:
        if registry.get_provider(route.provider_id) is None:
            # Unregistered provider: keep existing LLM_PRICE_* env price table
            from llm_usage import estimate_cost
            cost_usd = estimate_cost(route.model_id, p_tokens, c_tokens)
        else:
            cost_usd = pricing.calculate_cost_usd(route.provider_id, route.model_id, p_tokens, c_tokens)
    else:
        cost_usd = 0.0

    error_code = None
    error_msg = None
    if error:
        error_code = getattr(error, "code", type(error).__name__)
        error_msg = str(error)

    attempt = PhysicalAttemptV1(
        attempt_id=attempt_id,
        request_id=request.request_id,
        attempt_number=attempt_number,
        provider_id=route.provider_id,
        model_id=route.model_id,
        policy_leg=leg,
        attempt_kind=attempt_kind,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        latency_ms=round(latency_ms, 2),
        outcome=outcome,
        detailed_outcome=outcome,
        http_status=resp.http_status if resp else (HTTP_STATUS_INTERNAL_SERVER_ERROR if error else HTTP_STATUS_OK),
        provider_request_id=resp.provider_request_id if resp else None,
        prompt_tokens=p_tokens if is_usage_reported else None,
        completion_tokens=c_tokens if is_usage_reported else None,
        total_tokens=t_tokens if is_usage_reported else None,
        is_usage_reported=is_usage_reported,
        cost_usd=cost_usd,
        price_table_version=pricing.pricing_snapshot_id,
        applied_compatibility_changes=applied_changes or [],
        error_code=error_code,
        error_message=error_msg,
        validation_summary=validation_summary,
    )

    # Update llm_usage ledger: every counted physical attempt emits exactly one event
    try:
        from llm_usage import UsageEvent, current_ledger
        ledger = current_ledger()
        if ledger is not None:
            event_prompt = p_tokens if (is_usage_reported and p_tokens is not None) else 0
            event_comp = c_tokens if (is_usage_reported and c_tokens is not None) else 0
            event_total = t_tokens if (is_usage_reported and t_tokens is not None) else (event_prompt + event_comp)
            event_cost = cost_usd if is_usage_reported else 0.0
            event = UsageEvent(
                model=route.model_id,
                prompt_tokens=event_prompt,
                completion_tokens=event_comp,
                total_tokens=event_total,
                cost_usd=event_cost,
                attempt_id=attempt_id,
                request_id=request.request_id,
                policy_tag=request.policy_tag,
                provider_id=route.provider_id,
                policy_leg=leg.value,
                attempt_kind=attempt_kind.value,
                pricing_snapshot_id=request.pricing_snapshot_id,
                price_table_version=pricing.pricing_snapshot_id,
                is_usage_reported=is_usage_reported,
            )
            ledger.add(event)  # idempotent on attempt_id
    except Exception as exc:
        logger.debug("Could not record to llm_usage ledger: %s", exc)

    return attempt


def log_privacy_safe(request: LLMRequestV1, label: str, content: str) -> None:
    """Log content only when prompt logging is permitted by the privacy policy."""
    if request.privacy.allow_prompt_logging:
        logger.debug("%s [req=%s]: %s", label, request.request_id, content)
    else:
        logger.debug("%s [req=%s]: [REDACTED – prompt logging disabled]", label, request.request_id)


def run_attempt_loop_sync(
    request: LLMRequestV1,
    route_ladder: List[Tuple[RouteV1, PolicyLeg]],
    initial_route: RouteV1,
    initial_leg: PolicyLeg,
    start_time_monotonic: float,
    get_adapter: Callable[[str], BaseLLMAdapter],
    circuit_breaker: RouteCircuitBreaker,
    registry: ProviderRegistry,
    route_attempt_cap: int,
    validator_hook: Optional[Callable[..., Any]] = None,
    engine_validator_hook: Optional[Callable[..., Any]] = None,
) -> Tuple[
    List[PhysicalAttemptV1],
    Optional[OutputPayloadV1],
    Optional[RouteV1],
    ValidationSummaryV1,
    Optional[DegradationSummaryV1],
    Optional[LLMErrorV1],
    Optional[Any],
]:
    """Synchronous attempt-loop body shared by invoke_llm."""
    deadline_ms = request.budget.deadline_ms
    attempts: List[PhysicalAttemptV1] = []
    adaptations: Dict[str, Any] = {"removed_params": []}

    attempt_count = 0
    last_error: Optional[LLMErrorV1] = None
    selected_route: Optional[RouteV1] = None
    released_output: Optional[OutputPayloadV1] = None
    final_val_summary = ValidationSummaryV1(validator_id=request.response_contract.semantic_validator_id)
    degradation: Optional[DegradationSummaryV1] = None
    terminal_failure = False
    last_projection = None

    for route, leg in route_ladder:
        if released_output is not None or terminal_failure:
            break

        if not circuit_breaker.is_available(route.provider_id, route.model_id):
            logger.info(
                "Circuit breaker OPEN for %s/%s; fast-failing to next ladder route.",
                route.provider_id,
                route.model_id,
            )
            continue

        route_attempts = 0
        validation_retries = 0
        retry_kind = AttemptKind.TRANSPORT_RETRY
        while route_attempts < route_attempt_cap:
            elapsed_ms = (time.monotonic() - start_time_monotonic) * MS_PER_SECOND
            if elapsed_ms >= deadline_ms:
                last_error = BudgetExhaustedError(GatewayErrorCode.DEADLINE_EXHAUSTED.value, f"Request deadline of {deadline_ms}ms exceeded")
                break

            if attempt_count >= request.budget.max_physical_attempts:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.ATTEMPT_BUDGET_EXHAUSTED.value,
                    f"Max physical attempts ({request.budget.max_physical_attempts}) reached",
                )
                break

            total_cost_so_far = sum(a.cost_usd for a in attempts)
            if total_cost_so_far >= request.budget.max_cost_usd:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.COST_BUDGET_EXHAUSTED.value, f"Max cost budget ${request.budget.max_cost_usd} exceeded"
                )
                break

            total_tokens_so_far = sum(a.total_tokens for a in attempts if a.total_tokens is not None)
            if request.budget.max_total_tokens and total_tokens_so_far >= request.budget.max_total_tokens:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.TOKEN_BUDGET_EXHAUSTED.value,
                    f"Max total tokens budget {request.budget.max_total_tokens} exceeded (consumed {total_tokens_so_far})",
                )
                break

            attempt_count += 1
            route_attempts += 1
            att_kind = AttemptKind.INITIAL if attempt_count == 1 else (
                AttemptKind.ROUTE_FALLBACK if route_attempts == 1 else retry_kind
            )

            att_start_utc = time.time()
            att_start_mono = time.monotonic()
            adapter = get_adapter(route.provider_id)
            kwargs, san_mapping, proj = normalize_request_kwargs(request, route, adaptations, registry)
            last_projection = proj

            # Fix #8: log prompt content only when privacy policy allows
            log_privacy_safe(request, "PROMPT", str([m["content"] for m in kwargs.get("messages", [])]))

            try:
                raw_resp = adapter.execute(request, route, kwargs)
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND

                # Fix #8: log response only when allowed
                log_privacy_safe(request, "RESPONSE", raw_resp.text or "")

                gate_res = execute_validation_sync(
                    request, raw_resp, san_mapping, validator_hook=validator_hook, engine_validator_hook=engine_validator_hook
                )
                val_summary = gate_res.summary
                final_val_summary = val_summary

                is_compat, compat_err = is_payload_contract_compatible(request, gate_res.payload)
                if not is_compat and gate_res.passed:
                    val_summary.errors.append(compat_err or "Payload incompatible with response contract")
                    is_valid = False
                else:
                    is_valid = gate_res.passed and is_compat

                if is_valid and gate_res.payload is not None:
                    circuit_breaker.record_success(route.provider_id, route.model_id)
                    attempt = record_attempt(
                        registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.SUCCESS, resp=raw_resp, applied_changes=adaptations.get("removed_params"),
                        validation_summary=val_summary,
                    )
                    attempts.append(attempt)
                    selected_route = route
                    released_output = gate_res.payload
                    if leg != initial_leg:
                        degradation = DegradationSummaryV1(
                            reason_code=DegradationReason.FALLBACK_ROUTE_USED.value, from_route=initial_route, to_route=route
                        )
                    break
                else:
                    val_err = OutputValidationError(GatewayErrorCode.VALIDATION_FAILED.value, f"Validation errors: {val_summary.errors}")
                    attempt = record_attempt(
                        registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.VALIDATION_ERROR, resp=raw_resp, error=val_err,
                        validation_summary=val_summary,
                    )
                    attempts.append(attempt)
                    last_error = val_err
                    if is_terminal_validation_error(val_summary):
                        terminal_failure = True
                        break

                    # Cheap-first route triggers immediate fallback to baseline on semantic violation
                    if val_summary.semantic == ValidationGateStatus.FAILED and leg == PolicyLeg.CHEAP:
                        break

                    # Trigger bounded validation repair attempt if within budget and eligible contract
                    can_repair = (
                        validation_retries < 1
                        and route_attempts < route_attempt_cap
                        and attempt_count < request.budget.max_physical_attempts
                        and (is_registered_contract(request) or request.policy_tag == "confirmation.payload")
                    )
                    if can_repair:
                        retry_kind = AttemptKind.VALIDATION_RETRY
                        validation_retries += 1
                        adaptations["validation_repair"] = True
                        adaptations["validation_errors"] = list(val_summary.errors)
                        continue

                    break  # Fall back to next route in ladder unless terminal failure

            except CompatibilityError as exc:
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                attempt = record_attempt(
                    registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                    AttemptOutcome.COMPATIBILITY_ERROR, error=exc
                )
                attempts.append(attempt)
                last_error = exc
                retry_kind = AttemptKind.COMPATIBILITY_RETRY
                if "temperature" in str(exc).lower():
                    adaptations["removed_params"].append("temperature")
                elif "max_tokens" in str(exc).lower() or "extra_body" in str(exc).lower():
                    adaptations["use_max_completion_tokens"] = True
                else:
                    break

            except TransportError as exc:
                circuit_breaker.record_failure(route.provider_id, route.model_id)
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                attempt = record_attempt(
                    registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                    AttemptOutcome.TRANSPORT_ERROR, error=exc
                )
                attempts.append(attempt)
                last_error = exc
                retry_kind = AttemptKind.TRANSPORT_RETRY
                # Fix #1: exponential back-off with full jitter
                sleep_sec = _jitter_sleep_seconds(route_attempts)
                logger.debug(
                    "TransportError on attempt %d/%d; sleeping %.0f ms before retry (Fix #1 jitter).",
                    attempt_count,
                    request.budget.max_physical_attempts,
                    sleep_sec * MS_PER_SECOND,
                )
                time.sleep(sleep_sec)

            except Exception as exc:
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                pe = ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, str(exc))
                attempt = record_attempt(
                    registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                    AttemptOutcome.PROVIDER_ERROR, error=pe
                )
                attempts.append(attempt)
                last_error = pe
                break

    return attempts, released_output, selected_route, final_val_summary, degradation, last_error, last_projection


async def run_attempt_loop_async(
    request: LLMRequestV1,
    route_ladder: List[Tuple[RouteV1, PolicyLeg]],
    initial_route: RouteV1,
    initial_leg: PolicyLeg,
    start_time_monotonic: float,
    get_adapter: Callable[[str], BaseLLMAdapter],
    circuit_breaker: RouteCircuitBreaker,
    registry: ProviderRegistry,
    route_attempt_cap: int,
    validator_hook: Optional[Callable[..., Any]] = None,
    engine_validator_hook: Optional[Callable[..., Any]] = None,
) -> Tuple[
    List[PhysicalAttemptV1],
    Optional[OutputPayloadV1],
    Optional[RouteV1],
    ValidationSummaryV1,
    Optional[DegradationSummaryV1],
    Optional[LLMErrorV1],
    Optional[Any],
]:
    """Asynchronous attempt-loop body shared by invoke_llm_async."""
    deadline_ms = request.budget.deadline_ms
    attempts: List[PhysicalAttemptV1] = []
    adaptations: Dict[str, Any] = {"removed_params": []}

    attempt_count = 0
    last_error: Optional[LLMErrorV1] = None
    selected_route: Optional[RouteV1] = None
    released_output: Optional[OutputPayloadV1] = None
    final_val_summary = ValidationSummaryV1(validator_id=request.response_contract.semantic_validator_id)
    degradation: Optional[DegradationSummaryV1] = None
    terminal_failure = False
    last_projection = None

    for route, leg in route_ladder:
        if released_output is not None or terminal_failure:
            break

        if not circuit_breaker.is_available(route.provider_id, route.model_id):
            logger.info(
                "Circuit breaker OPEN for %s/%s; fast-failing to next ladder route.",
                route.provider_id,
                route.model_id,
            )
            continue

        route_attempts = 0
        validation_retries = 0
        retry_kind = AttemptKind.TRANSPORT_RETRY
        while route_attempts < route_attempt_cap:
            elapsed_ms = (time.monotonic() - start_time_monotonic) * MS_PER_SECOND
            if elapsed_ms >= deadline_ms:
                last_error = BudgetExhaustedError(GatewayErrorCode.DEADLINE_EXHAUSTED.value, f"Request deadline of {deadline_ms}ms exceeded")
                break

            if attempt_count >= request.budget.max_physical_attempts:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.ATTEMPT_BUDGET_EXHAUSTED.value,
                    f"Max physical attempts ({request.budget.max_physical_attempts}) reached",
                )
                break

            total_cost_so_far = sum(a.cost_usd for a in attempts)
            if total_cost_so_far >= request.budget.max_cost_usd:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.COST_BUDGET_EXHAUSTED.value, f"Max cost budget ${request.budget.max_cost_usd} exceeded"
                )
                break

            total_tokens_so_far = sum(a.total_tokens for a in attempts if a.total_tokens is not None)
            if request.budget.max_total_tokens and total_tokens_so_far >= request.budget.max_total_tokens:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.TOKEN_BUDGET_EXHAUSTED.value,
                    f"Max total tokens budget {request.budget.max_total_tokens} exceeded (consumed {total_tokens_so_far})",
                )
                break

            attempt_count += 1
            route_attempts += 1
            att_kind = AttemptKind.INITIAL if attempt_count == 1 else (
                AttemptKind.ROUTE_FALLBACK if route_attempts == 1 else retry_kind
            )

            att_start_utc = time.time()
            att_start_mono = time.monotonic()
            adapter = get_adapter(route.provider_id)
            kwargs, san_mapping, proj = normalize_request_kwargs(request, route, adaptations, registry)
            last_projection = proj

            log_privacy_safe(request, "PROMPT", str([m["content"] for m in kwargs.get("messages", [])]))

            try:
                raw_resp = await adapter.execute_async(request, route, kwargs)
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND

                log_privacy_safe(request, "RESPONSE", raw_resp.text or "")

                gate_res = await execute_validation_async(
                    request, raw_resp, san_mapping, validator_hook=validator_hook, engine_validator_hook=engine_validator_hook
                )
                val_summary = gate_res.summary
                final_val_summary = val_summary

                is_compat, compat_err = is_payload_contract_compatible(request, gate_res.payload)
                if not is_compat and gate_res.passed:
                    val_summary.errors.append(compat_err or "Payload incompatible with response contract")
                    is_valid = False
                else:
                    is_valid = gate_res.passed and is_compat

                if is_valid and gate_res.payload is not None:
                    circuit_breaker.record_success(route.provider_id, route.model_id)
                    attempt = record_attempt(
                        registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.SUCCESS, resp=raw_resp, applied_changes=adaptations.get("removed_params"),
                        validation_summary=val_summary,
                    )
                    attempts.append(attempt)
                    selected_route = route
                    released_output = gate_res.payload
                    if leg != initial_leg:
                        degradation = DegradationSummaryV1(
                            reason_code=DegradationReason.FALLBACK_ROUTE_USED.value, from_route=initial_route, to_route=route
                        )
                    break
                else:
                    val_err = OutputValidationError(GatewayErrorCode.VALIDATION_FAILED.value, f"Validation errors: {val_summary.errors}")
                    attempt = record_attempt(
                        registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.VALIDATION_ERROR, resp=raw_resp, error=val_err,
                        validation_summary=val_summary,
                    )
                    attempts.append(attempt)
                    last_error = val_err
                    if is_terminal_validation_error(val_summary):
                        terminal_failure = True
                        break

                    # Cheap-first route triggers immediate fallback to baseline on semantic violation
                    if val_summary.semantic == ValidationGateStatus.FAILED and leg == PolicyLeg.CHEAP:
                        break

                    # Trigger bounded validation repair attempt if within budget and eligible contract
                    can_repair = (
                        validation_retries < 1
                        and route_attempts < route_attempt_cap
                        and attempt_count < request.budget.max_physical_attempts
                        and (is_registered_contract(request) or request.policy_tag == "confirmation.payload")
                    )
                    if can_repair:
                        retry_kind = AttemptKind.VALIDATION_RETRY
                        validation_retries += 1
                        adaptations["validation_repair"] = True
                        adaptations["validation_errors"] = list(val_summary.errors)
                        continue

                    break

            except CompatibilityError as exc:
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                attempt = record_attempt(
                    registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                    AttemptOutcome.COMPATIBILITY_ERROR, error=exc
                )
                attempts.append(attempt)
                last_error = exc
                retry_kind = AttemptKind.COMPATIBILITY_RETRY
                if "temperature" in str(exc).lower():
                    adaptations["removed_params"].append("temperature")
                elif "max_tokens" in str(exc).lower() or "extra_body" in str(exc).lower():
                    adaptations["use_max_completion_tokens"] = True
                else:
                    break

            except TransportError as exc:
                circuit_breaker.record_failure(route.provider_id, route.model_id)
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                attempt = record_attempt(
                    registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                    AttemptOutcome.TRANSPORT_ERROR, error=exc
                )
                attempts.append(attempt)
                last_error = exc
                retry_kind = AttemptKind.TRANSPORT_RETRY
                # Fix #1: exponential back-off with full jitter (async-safe)
                sleep_sec = _jitter_sleep_seconds(route_attempts)
                logger.debug(
                    "TransportError on attempt %d/%d; sleeping %.0f ms before retry (Fix #1 jitter).",
                    attempt_count,
                    request.budget.max_physical_attempts,
                    sleep_sec * MS_PER_SECOND,
                )
                await asyncio.sleep(sleep_sec)

            except Exception as exc:
                lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                pe = ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, str(exc))
                attempt = record_attempt(
                    registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                    AttemptOutcome.PROVIDER_ERROR, error=pe
                )
                attempts.append(attempt)
                last_error = pe
                break

    return attempts, released_output, selected_route, final_val_summary, degradation, last_error, last_projection
