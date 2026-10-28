"""Per-request LLM token-usage capture.

The dashboard's "Token Consumption" card sums real usage recorded here. Each LLM
call site records ``response.usage`` into the *current* ledger — a request- or
task-scoped accumulator held in a ``contextvars.ContextVar`` so no plumbing has
to thread through every function signature. When no ledger is active (e.g. a
debug ping, or a code path outside a scoped request) recording is a silent no-op.

Cost is intentionally env-configurable and defaults to **$0** so token counts
light up immediately without baking in prices that could be wrong for the
deployed model. Configure real prices via:

- ``LLM_PRICE_INPUT_PER_1M`` / ``LLM_PRICE_OUTPUT_PER_1M`` — global USD price per
  1,000,000 prompt / completion tokens.
- ``LLM_PRICE_TABLE_JSON`` — optional per-model override, e.g.
  ``{"gpt-4o": {"input": 2.5, "output": 10}}`` (falls back to the globals).

This module is deliberately dependency-free (stdlib only) so it can be imported
from the low-level LLM client layer without pulling in the DB/service layers.
The service layer owns starting a scope and flushing the ledger to storage.

Exactly-once accounting with the LLM gateway (WP5): the gateway records one
event per *physical* provider attempt, keyed by ``attempt_id`` — failed,
retried and compatibility attempts included. ``UsageLedger.add`` ignores a
repeated ``attempt_id``, and responses returned through the ``llm_compat``
shim are marked as already recorded, so the ``record_response_usage(...)``
calls that legacy call sites still make become no-ops instead of double
charging.
"""
from __future__ import annotations

import contextvars
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UsageEvent:
    """One LLM call's token usage (and its estimated cost).

    The gateway metadata fields are optional so legacy constructors keep
    working; ``attempt_id`` is the idempotency key for exactly-once recording.
    """

    model: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    attempt_id: Optional[str] = None
    request_id: Optional[str] = None
    policy_tag: Optional[str] = None
    provider_id: Optional[str] = None
    policy_leg: Optional[str] = None
    attempt_kind: Optional[str] = None
    pricing_snapshot_id: Optional[str] = None
    price_table_version: Optional[str] = None
    is_usage_reported: bool = True


@dataclass
class UsageLedger:
    """Accumulates the usage of every LLM call within a single scope."""

    events: list[UsageEvent] = field(default_factory=list)
    _attempt_ids: set[str] = field(default_factory=set, repr=False, compare=False)
    # id(response) -> (response, attempt_id). The strong reference pins the
    # object so its id() cannot be reused while this (request-scoped) ledger
    # lives; used for responses that cannot carry the marker attribute.
    _recorded_responses: dict[int, tuple[Any, str]] = field(default_factory=dict, repr=False, compare=False)

    def add(self, event: UsageEvent) -> bool:
        """Append ``event``; return ``False`` if its ``attempt_id`` was already recorded."""
        if event.attempt_id is not None:
            if event.attempt_id in self._attempt_ids:
                return False
            self._attempt_ids.add(event.attempt_id)
        self.events.append(event)
        return True

    def has_attempt(self, attempt_id: str) -> bool:
        return attempt_id in self._attempt_ids

    def mark_response(self, response: Any, attempt_id: str) -> None:
        self._recorded_responses[id(response)] = (response, attempt_id)

    def recorded_attempt_for(self, response: Any) -> Optional[str]:
        entry = self._recorded_responses.get(id(response))
        if entry is not None and entry[0] is response:
            return entry[1]
        return None

    @property
    def totals(self) -> dict[str, Any]:
        return {
            "prompt_tokens": sum(e.prompt_tokens for e in self.events),
            "completion_tokens": sum(e.completion_tokens for e in self.events),
            "total_tokens": sum(e.total_tokens for e in self.events),
            "cost_usd": round(sum(e.cost_usd for e in self.events), 6),
        }


# The active ledger for the current async context (request or background task).
_current_ledger: contextvars.ContextVar[Optional[UsageLedger]] = contextvars.ContextVar(
    "metarec_usage_ledger", default=None
)


def current_ledger() -> Optional[UsageLedger]:
    return _current_ledger.get()


def push_ledger(ledger: UsageLedger) -> contextvars.Token:
    """Make ``ledger`` current; returns a token to restore the previous one."""
    return _current_ledger.set(ledger)


def reset_ledger(token: contextvars.Token) -> None:
    _current_ledger.reset(token)


def _env_float(name: str) -> float:
    raw = os.getenv(name)
    if not raw:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _price_for_model(model: Optional[str]) -> tuple[float, float]:
    """(input_price, output_price) in USD per 1,000,000 tokens for ``model``.

    A per-model entry in ``LLM_PRICE_TABLE_JSON`` wins; otherwise the global
    ``LLM_PRICE_INPUT_PER_1M`` / ``LLM_PRICE_OUTPUT_PER_1M`` (default 0) apply.
    """
    default = (_env_float("LLM_PRICE_INPUT_PER_1M"), _env_float("LLM_PRICE_OUTPUT_PER_1M"))
    raw_table = os.getenv("LLM_PRICE_TABLE_JSON")
    if not (raw_table and model):
        return default
    try:
        table = json.loads(raw_table)
        entry = table.get(model)
        if not isinstance(entry, dict):
            return default
        return (
            float(entry.get("input", default[0])),
            float(entry.get("output", default[1])),
        )
    except (ValueError, TypeError):
        return default


def estimate_cost(model: Optional[str], prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = _price_for_model(model)
    cost = (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price
    return round(cost, 6)


def _usage_field(usage: Any, name: str) -> int:
    # Providers return usage either as an object with attributes or a dict.
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# Marker attribute set on responses whose attempts the gateway already charged.
GATEWAY_RECORDED_ATTR = "_llm_gateway_attempt_id"


def mark_response_recorded(response: Any, attempt_id: str) -> None:
    """Flag ``response`` as already charged by the gateway under ``attempt_id``.

    The marker is set directly on the object when possible (SDK models,
    namespaces); ``object.__setattr__`` bypasses pydantic field validation.
    It is also registered on the active ledger, which covers objects that
    cannot carry attributes (str, dict, ``__slots__`` classes).
    """
    if response is None:
        return
    try:
        object.__setattr__(response, GATEWAY_RECORDED_ATTR, attempt_id)
    except (AttributeError, TypeError):
        pass
    ledger = current_ledger()
    if ledger is not None:
        ledger.mark_response(response, attempt_id)


def recorded_attempt_id(response: Any) -> Optional[str]:
    """Return the gateway attempt that already charged ``response``, if any."""
    if response is None or isinstance(response, dict):
        attempt_id = None
    else:
        try:
            attempt_id = object.__getattribute__(response, GATEWAY_RECORDED_ATTR)
        except AttributeError:
            attempt_id = None
    if attempt_id is None:
        ledger = current_ledger()
        if ledger is not None:
            attempt_id = ledger.recorded_attempt_for(response)
    return attempt_id if isinstance(attempt_id, str) else None


def record_response_usage(response: Any, model_hint: Optional[str] = None) -> Optional[UsageEvent]:
    """Record an LLM response's token usage into the current ledger.

    No-op (returns ``None``) when there is no active ledger, the response
    carries no usage, or the gateway already recorded the physical attempt(s)
    behind this response — never raises, so it is safe to sprinkle at call sites.
    """
    ledger = current_ledger()
    if ledger is None:
        return None
    if recorded_attempt_id(response) is not None:
        return None
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None
    prompt = _usage_field(usage, "prompt_tokens")
    completion = _usage_field(usage, "completion_tokens")
    total = _usage_field(usage, "total_tokens") or (prompt + completion)
    model = getattr(response, "model", None)
    if model is None and isinstance(response, dict):
        model = response.get("model")
    model = model or model_hint
    event = UsageEvent(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cost_usd=estimate_cost(model, prompt, completion),
    )
    ledger.add(event)
    return event
