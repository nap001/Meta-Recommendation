"""Core Gateway Engine implementing sync and async multi-provider execution (v1)."""

from __future__ import annotations

from llm_gateway.engine.core import GatewayEngine
from llm_gateway.engine.factories import (
    create_engine,
    create_engine_for_registry,
    get_gateway_engine,
    invoke_llm,
    invoke_llm_async,
    invoke_llm_in_executor,
    invoke_llm_stream,
    invoke_llm_stream_async,
)
from llm_gateway.engine.resilience import (
    RouteCircuitBreaker,
    _BACKOFF_BASE_MS,
    _BACKOFF_CAP_MS,
    _MAX_ATTEMPTS_PER_ROUTE,
    _jitter_sleep_seconds,
)

__all__ = [
    "GatewayEngine",
    "create_engine",
    "create_engine_for_registry",
    "get_gateway_engine",
    "invoke_llm",
    "invoke_llm_async",
    "invoke_llm_in_executor",
    "invoke_llm_stream",
    "invoke_llm_stream_async",
    "RouteCircuitBreaker",
    "_jitter_sleep_seconds",
    "_BACKOFF_BASE_MS",
    "_BACKOFF_CAP_MS",
    "_MAX_ATTEMPTS_PER_ROUTE",
]
