"""Resilience, exponential jitter backoff, and circuit breaker for LLM Gateway Engine."""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Dict, Tuple

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


class RouteCircuitBreaker:
    """Automated circuit breaker per (provider_id, model_id) to prevent cascading outage storms."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_sec: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self._failures: Dict[Tuple[str, str], int] = {}
        self._opened_at: Dict[Tuple[str, str], float] = {}

    def is_available(self, provider_id: str, model_id: str) -> bool:
        key = (provider_id, model_id)
        opened = self._opened_at.get(key)
        if opened is None:
            return True
        if time.monotonic() - opened > self.recovery_timeout_sec:
            return True  # Half-open probe attempt
        return False

    def record_failure(self, provider_id: str, model_id: str) -> None:
        key = (provider_id, model_id)
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= self.failure_threshold:
            self._opened_at[key] = time.monotonic()
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
