"""Backward-compatibility seam: route legacy ``llm_compat`` calls through the gateway.

``llm_compat.create_chat_completion`` / ``create_chat_completion_async`` keep
their public signatures and return values. Internally each call now:

1. builds a logical ``LLMRequestV1`` describing the call (policy tag
   ``legacy.unenrolled``, baseline route on the caller's model);
2. binds a per-call ``LegacyClientAdapter`` wrapping the caller's own client
   and the already-prepared wire request;
3. runs the gateway engine's ``invoke_llm`` / ``invoke_llm_async``, which owns
   budgets, transport retries, compatibility retries and attempt accounting
   (exactly one ``llm_usage`` event per physical SDK call); and
4. projects the ``LLMResultV1`` back to what legacy callers expect: the raw
   SDK response on success, or the caller's original exception on failure.

The response is marked as already recorded so the existing
``record_response_usage(response)`` calls at every call site become no-ops
instead of double charging.

Unenrolled legacy calls use a pass-through validation gate: the call sites
still parse and validate their own output, exactly as before. Enrolling a call
site with a real ``policy_tag`` / response contract is separate, per-site work.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import llm_usage
from llm_gateway.adapters.base import RawAdapterResponse
from llm_gateway.adapters.legacy_client import (
    LEGACY_PROVIDER_ID,
    AdaptFn,
    LegacyClientAdapter,
    LegacyClientDispatcher,
    bind_legacy_adapter,
)
from llm_gateway.constants import DEFAULT_GENERATION_MAX_OUTPUT_TOKENS, SemanticValidatorId
from llm_gateway.contracts import (
    BudgetConfigV1,
    GenerationConfigV1,
    LLMRequestV1,
    LLMResultV1,
    MessageV1,
    OutputPayloadV1,
    PolicyContextV1,
    RequestMode,
    RequestRole,
    ResponseContractV1,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    RoutingPolicyV1,
    ValidationGateResultV1,
    ValidationSummaryV1,
)
from llm_gateway.engine import GatewayEngine, create_engine

logger = logging.getLogger(__name__)

LEGACY_POLICY_TAG = "legacy.unenrolled"
LEGACY_SCHEMA_ID = "legacy.passthrough.v1"
LEGACY_PRODUCER_ID = "llm_compat.legacy_shim"
COMPAT_RETRY_LIMIT_MESSAGE = "Chat Completions compatibility retry limit exceeded"

# The pre-gateway compat loop allowed 3 tries (initial + 2 adaptations), and
# the SDK allowed 1 + LLM_SDK_MAX_RETRIES transport attempts. The gateway's
# physical-attempt budget preserves the larger of the two.
_LEGACY_COMPAT_TRIES = 3
_DEFAULT_TRANSPORT_RETRIES = 2
_MAX_BUDGET_ATTEMPTS = 10  # BudgetConfigV1.max_physical_attempts upper bound
_DEFAULT_DEADLINE_MS = 300_000
_DEFAULT_MAX_COST_USD = 5.0

_VALID_ROLES = {role.value for role in RequestRole}


def _env_int(name: str, default: int, *, min_value: int) -> int:
    try:
        return max(min_value, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, min_value: float) -> float:
    try:
        return max(min_value, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def transport_retry_count() -> int:
    """Gateway-owned transport retries for legacy calls.

    ``LLM_SDK_MAX_RETRIES`` used to configure the SDK's hidden retries. With
    the shim active, managed SDK clients run with ``max_retries=0`` and the same
    setting is honoured here instead, where every attempt is counted.
    """
    return _env_int("LLM_SDK_MAX_RETRIES", _DEFAULT_TRANSPORT_RETRIES, min_value=0)


def _max_physical_attempts() -> int:
    return min(_MAX_BUDGET_ATTEMPTS, max(_LEGACY_COMPAT_TRIES, 1 + transport_retry_count()))


# ── Engine ──────────────────────────────────────────────────────────────────

_ENGINE: Optional[GatewayEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_legacy_shim_engine() -> GatewayEngine:
    """Return the isolated engine that serves legacy calls (created lazily).

    A dedicated engine keeps the legacy dispatcher out of the global engine's
    adapter map and lets the single legacy route use the whole attempt budget
    (``budget.max_physical_attempts`` stays the binding limit). The kill
    switch, provider registry and pricing snapshots are shared globals.
    """
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                engine = create_engine()
                engine.register_adapter(LEGACY_PROVIDER_ID, LegacyClientDispatcher())
                engine.max_attempts_per_route = _MAX_BUDGET_ATTEMPTS
                _ENGINE = engine
    return _ENGINE


# ── LLMRequestV1 construction ───────────────────────────────────────────────

def _to_message(raw: Any) -> MessageV1:
    role = str(raw.get("role") or "user") if isinstance(raw, dict) else "user"
    if role == "function":  # pre-tools OpenAI role
        role = RequestRole.TOOL.value
    if role not in _VALID_ROLES:
        role = RequestRole.USER.value

    content = raw.get("content") if isinstance(raw, dict) else raw
    if content is None:
        content_text = ""
    elif isinstance(content, str):
        content_text = content
    else:  # multimodal parts, structured content, ...
        content_text = json.dumps(content, ensure_ascii=False, default=str)

    name = raw.get("name") if isinstance(raw, dict) else None
    return MessageV1(role=RequestRole(role), content=content_text, name=name if isinstance(name, str) else None)


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _generation_from_wire(wire: Dict[str, Any]) -> GenerationConfigV1:
    temperature = wire.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
        temperature = None

    extra_body = wire.get("extra_body") if isinstance(wire.get("extra_body"), dict) else {}
    max_output = (
        _positive_int(wire.get("max_tokens"))
        or _positive_int(wire.get("max_completion_tokens"))
        or _positive_int(extra_body.get("max_completion_tokens"))
        or DEFAULT_GENERATION_MAX_OUTPUT_TOKENS
    )

    seed = wire.get("seed")
    seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None

    stop = wire.get("stop")
    if isinstance(stop, str):
        stop_list: Optional[List[str]] = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        stop_list = stop or None
    else:
        stop_list = None

    return GenerationConfigV1(temperature=temperature, max_output_tokens=max_output, seed=seed, stop=stop_list)


def build_legacy_request(wire: Dict[str, Any]) -> LLMRequestV1:
    """Describe a legacy wire request as a logical ``LLMRequestV1``.

    The policy context is deliberately conservative (itinerary state assumed,
    non-single-domain mode) so an unenrolled call can never become
    cheap-route eligible; routing is ``baseline`` on the caller's own model.
    """
    model = str(wire.get("model") or "unknown-model")
    try:
        messages = [_to_message(m) for m in (wire.get("messages") or [])]
        generation = _generation_from_wire(wire)
    except Exception as exc:  # never let request description break a legacy call
        logger.debug("Legacy shim could not describe request faithfully: %s", exc)
        messages, generation = [], GenerationConfigV1()

    return LLMRequestV1(
        request_id=f"legacy-{uuid.uuid4().hex}",
        policy_tag=LEGACY_POLICY_TAG,
        messages=messages,
        generation=generation,
        response_contract=ResponseContractV1(
            kind=ResponseKind.TEXT,
            schema_id=LEGACY_SCHEMA_ID,
            semantic_validator_id=SemanticValidatorId.NONE.value,
            allow_empty=True,  # tool-call-only responses carry no text
        ),
        routing_policy=RoutingPolicyV1(
            mode=RoutingMode.BASELINE,
            baseline_route=RouteV1(provider_id=LEGACY_PROVIDER_ID, model_id=model),
            allowed_provider_ids=[LEGACY_PROVIDER_ID],
        ),
        budget=BudgetConfigV1(
            deadline_ms=_env_int("LLM_GATEWAY_LEGACY_DEADLINE_MS", _DEFAULT_DEADLINE_MS, min_value=1),
            max_physical_attempts=_max_physical_attempts(),
            max_cost_usd=_env_float("LLM_GATEWAY_LEGACY_MAX_COST_USD", _DEFAULT_MAX_COST_USD, min_value=0.000001),
        ),
        policy_context=PolicyContextV1(
            contains_itinerary_state=True,
            request_mode=RequestMode.MULTI_DOMAIN,
            producer_id=LEGACY_PRODUCER_ID,
        ),
    )


# ── Validation + result projection ──────────────────────────────────────────

def _passthrough_validator(
    request: LLMRequestV1, resp: RawAdapterResponse, sanitized_mapping: Dict[str, str]
) -> ValidationGateResultV1:
    """Release every provider response unchanged; legacy call sites validate their own output."""
    return ValidationGateResultV1(
        passed=True,
        summary=ValidationSummaryV1(validator_id=SemanticValidatorId.NONE.value),
        payload=OutputPayloadV1(kind=ResponseKind.TEXT, text=resp.text or ""),
    )


def _project_result(result: LLMResultV1, adapter: LegacyClientAdapter) -> Any:
    if result.status != ResultStatus.FAILED and adapter.has_response:
        response = adapter.raw_response
        if result.attempts:
            llm_usage.mark_response_recorded(response, result.attempts[-1].attempt_id)
        return response

    if adapter.last_error_was_adaptation:
        raise RuntimeError(COMPAT_RETRY_LIMIT_MESSAGE) from adapter.last_exception
    if adapter.last_exception is not None:
        raise adapter.last_exception

    error = result.error
    code = error.code if error else "unknown_failure"
    message = error.message if error else "Execution failed"
    raise RuntimeError(f"LLM gateway call failed [{code}]: {message}")


def invoke_legacy_chat_completion(
    client: Any, wire_request: Dict[str, Any], adapt_fn: Optional[AdaptFn] = None
) -> Any:
    """Execute a prepared legacy wire request through ``GatewayEngine.invoke_llm``."""
    engine = get_legacy_shim_engine()
    request = build_legacy_request(wire_request)
    adapter = LegacyClientAdapter(client, wire_request, adapt_fn)
    with bind_legacy_adapter(adapter):
        result = engine.invoke_llm(request, validator_hook=_passthrough_validator)
    return _project_result(result, adapter)


async def invoke_legacy_chat_completion_async(
    client: Any, wire_request: Dict[str, Any], adapt_fn: Optional[AdaptFn] = None
) -> Any:
    """Execute a prepared legacy wire request through ``GatewayEngine.invoke_llm_async``."""
    engine = get_legacy_shim_engine()
    request = build_legacy_request(wire_request)
    adapter = LegacyClientAdapter(client, wire_request, adapt_fn)
    with bind_legacy_adapter(adapter):
        result = await engine.invoke_llm_async(request, validator_hook=_passthrough_validator)
    return _project_result(result, adapter)
