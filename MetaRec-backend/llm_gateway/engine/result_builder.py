"""Result assembly and accounting for LLM Gateway Engine."""

from __future__ import annotations

import time
from typing import List, Optional

from llm_gateway.constants import (
    DegradationReason,
    GatewayErrorCode,
    MS_PER_SECOND,
)
from llm_gateway.contracts import (
    DegradationSummaryV1,
    LLMErrorPayloadV1,
    LLMRequestV1,
    LLMResultV1,
    OutputPayloadV1,
    PhysicalAttemptV1,
    ResultStatus,
    RouteV1,
    TimingSummaryV1,
    UsageSummaryV1,
    ValidationGateStatus,
    ValidationSummaryV1,
)
from llm_gateway.errors import BudgetExhaustedError, ErrorClass, LLMErrorV1


def build_result(
    request: LLMRequestV1,
    attempts: List[PhysicalAttemptV1],
    released_output: Optional[OutputPayloadV1],
    selected_route: Optional[RouteV1],
    final_val_summary: ValidationSummaryV1,
    degradation: Optional[DegradationSummaryV1],
    last_error: Optional[LLMErrorV1],
    start_time_utc: float,
    start_time_monotonic: float,
    deadline_ms: int,
) -> LLMResultV1:
    """Assemble LLMResultV1 from attempt data (shared by sync and async paths)."""
    total_lat_ms = (time.monotonic() - start_time_monotonic) * MS_PER_SECOND
    tot_prompt = sum(a.prompt_tokens for a in attempts if a.prompt_tokens is not None)
    tot_comp = sum(a.completion_tokens for a in attempts if a.completion_tokens is not None)
    tot_tokens = sum(a.total_tokens for a in attempts if a.total_tokens is not None)
    tot_cost = sum(a.cost_usd for a in attempts)

    usage = UsageSummaryV1(
        prompt_tokens=tot_prompt,
        completion_tokens=tot_comp,
        total_tokens=tot_tokens,
        cost_usd=round(tot_cost, 6),
        pricing_snapshot_id=request.pricing_snapshot_id,
        is_estimated=False,
    )
    timing = TimingSummaryV1(
        started_at_utc=start_time_utc,
        completed_at_utc=time.time(),
        total_latency_ms=round(total_lat_ms, 2),
        deadline_ms=deadline_ms,
    )

    if released_output is not None:
        status = ResultStatus.DEGRADED if degradation else ResultStatus.SUCCEEDED
        err_payload = None
        res_val_summary = final_val_summary
    else:
        status = ResultStatus.FAILED
        selected_route = None  # Invariant: failed results must have selected_route=None
        # Result-level validation represents the released output. Since output is None,
        # top-level validation uses NOT_APPLICABLE with errors=[] while keeping the validator_id.
        # Attempt-level validation retains candidate validation details in each PhysicalAttemptV1.
        res_val_summary = ValidationSummaryV1(
            json_parse=ValidationGateStatus.NOT_APPLICABLE,
            json_schema=ValidationGateStatus.NOT_APPLICABLE,
            semantic=ValidationGateStatus.NOT_APPLICABLE,
            validator_id=request.response_contract.semantic_validator_id,
            errors=[],
        )
        attempt_count = len(attempts)
        if attempt_count >= request.budget.max_physical_attempts:
            last_error = BudgetExhaustedError(
                GatewayErrorCode.ATTEMPT_BUDGET_EXHAUSTED.value,
                f"Max physical attempts ({request.budget.max_physical_attempts}) reached",
            )
        elif tot_cost >= request.budget.max_cost_usd:
            last_error = BudgetExhaustedError(
                GatewayErrorCode.COST_BUDGET_EXHAUSTED.value, f"Max cost budget ${request.budget.max_cost_usd} exceeded"
            )
        elif total_lat_ms >= deadline_ms:
            last_error = BudgetExhaustedError(GatewayErrorCode.DEADLINE_EXHAUSTED.value, f"Request deadline of {deadline_ms}ms exceeded")

        err_dict = last_error.to_dict() if last_error else {
            "code": GatewayErrorCode.UNKNOWN_FAILURE.value,
            "error_class": ErrorClass.INTERNAL.value,
            "message": "Execution failed",
        }
        err_payload = LLMErrorPayloadV1(**err_dict)

    return LLMResultV1(
        request_id=request.request_id,
        policy_tag=request.policy_tag,
        status=status,
        output=released_output,
        selected_route=selected_route,
        validation=res_val_summary,
        attempts=attempts,
        usage=usage,
        timing=timing,
        degradation=degradation,
        error=err_payload,
        trace=request.trace_context,
    )
