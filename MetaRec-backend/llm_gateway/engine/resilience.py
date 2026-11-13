"""Resilience, exponential jitter backoff, and circuit breaker for LLM Gateway Engine."""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Callable, Dict, List, Tuple

from llm_gateway.constants import (
    DEFAULT_BACKOFF_BASE_MS,
    DEFAULT_BACKOFF_CAP_MS,
    DEFAULT_MAX_ATTEMPTS_PER_ROUTE,
    MS_PER_SECOND,
)

logger = logging.getLogger(__name__)

# ── Fix #1: Retry back-off constants ────────────────────────────────────────
_BACKOFF_BASE_MS: float = DEFAULT_BACKOFF_BASE_MS
_BACKOFF_CAP_MS: float = DEFAULT_BACKOFF_CAP_MS
_MAX_ATTEMPTS_PER_ROUTE: int = DEFAULT_MAX_ATTEMPTS_PER_ROUTE


def _jitter_sleep_seconds(attempt_number: int) -> float:
    """Compute full-jitter exponential back-off sleep duration in seconds.

    Fix #1: Replaces flat sleep with full jitter: sleep = random(0, min(cap, base * 2^attempt)).
    ``attempt_number`` is 1-indexed (first retry = 1).
    """
    cap = _BACKOFF_CAP_MS
    base = _BACKOFF_BASE_MS
    window_ms = min(cap, base * math.pow(2.0, attempt_number - 1))
    sleep_ms = random.uniform(0.0, window_ms)
    return sleep_ms / MS_PER_SECOND


class FakeClock:
    """Deterministic virtual clock for testing retry back-off, deadlines, and circuit breakers in 0ms real time."""

    def __init__(
        self,
        initial_utc: float = 1_700_000_000.0,
        initial_monotonic: float = 1_000.0,
    ) -> None:
        self._utc = float(initial_utc)
        self._monotonic = float(initial_monotonic)
        self.sleep_history: List[float] = []

    def time(self) -> float:
        """Return current virtual wall-clock UTC timestamp in seconds."""
        return self._utc

    def monotonic(self) -> float:
        """Return current virtual monotonic timestamp in seconds."""
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Advance both monotonic and UTC clocks by ``seconds``."""
        delta = max(0.0, float(seconds))
        self._monotonic += delta
        self._utc += delta

    def sleep(self, seconds: float) -> None:
        """Record a synchronous sleep and advance virtual time immediately."""
        duration = max(0.0, float(seconds))
        self.sleep_history.append(duration)
        self.advance(duration)

    async def async_sleep(self, seconds: float) -> None:
        """Record an asynchronous sleep and advance virtual time immediately."""
        duration = max(0.0, float(seconds))
        self.sleep_history.append(duration)
        self.advance(duration)


class RouteCircuitBreaker:
    """Automated circuit breaker per (provider_id, model_id) to prevent cascading outage storms."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self._monotonic_fn = monotonic_fn
        self._failures: Dict[Tuple[str, str], int] = {}
        self._opened_at: Dict[Tuple[str, str], float] = {}

    def is_available(self, provider_id: str, model_id: str) -> bool:
        key = (provider_id, model_id)
        opened = self._opened_at.get(key)
        if opened is None:
            return True
        if self._monotonic_fn() - opened > self.recovery_timeout_sec:
            return True  # Half-open probe attempt
        return False

    def record_failure(self, provider_id: str, model_id: str) -> None:
        key = (provider_id, model_id)
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= self.failure_threshold:
            self._opened_at[key] = self._monotonic_fn()
            logger.warning(
                "Circuit breaker tripped OPEN for %s/%s after %d consecutive failures.",
                provider_id,
                model_id,
                self._failures[key],
            )

    def record_success(self, provider_id: str, model_id: str) -> None:
        key = (provider_id, model_id)
        self._failures.pop(key, None)
        self._opened_at.pop(key, None)
