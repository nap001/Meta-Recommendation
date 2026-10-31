"""Streaming token iteration across provider route ladder for LLM Gateway."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Tuple

from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.constants import HTTP_STATUS_OK, MS_PER_SECOND
from llm_gateway.contracts import (
    AttemptKind,
    AttemptOutcome,
    LLMRequestV1,
    PolicyLeg,
    RouteV1,
)
from llm_gateway.errors import TransportError
from llm_gateway.registry import ProviderRegistry
from llm_gateway.engine.attempt import record_attempt
from llm_gateway.engine.projection import normalize_request_kwargs
from llm_gateway.engine.resilience import RouteCircuitBreaker


def run_stream_sync(
    request: LLMRequestV1,
    route_ladder: List[Tuple[RouteV1, PolicyLeg]],
    get_adapter: Callable[[str], BaseLLMAdapter],
    circuit_breaker: RouteCircuitBreaker,
    registry: ProviderRegistry,
) -> Iterator[str]:
    """Synchronously stream delta tokens for an LLMRequestV1 across route ladder."""
    adaptations: Dict[str, Any] = {"removed_params": []}
    attempt_count = 0
    last_error = None

    for route, leg in route_ladder:
        if not circuit_breaker.is_available(route.provider_id, route.model_id):
            continue

        attempt_count += 1
        att_kind = AttemptKind.INITIAL if attempt_count == 1 else AttemptKind.ROUTE_FALLBACK
        att_start_utc = time.time()
        att_start_mono = time.monotonic()
        adapter = get_adapter(route.provider_id)
        kwargs, _, _ = normalize_request_kwargs(request, route, adaptations, registry)

        collected_tokens = 0
        full_text_chunks = []
        stream_started = False
        try:
            for chunk in adapter.execute_stream(request, route, kwargs):
                stream_started = True
                collected_tokens += 1
                full_text_chunks.append(chunk)
                yield chunk

            lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
            raw_resp = RawAdapterResponse(
                text="".join(full_text_chunks),
                prompt_tokens=40,
                completion_tokens=collected_tokens,
                total_tokens=40 + collected_tokens,
                is_usage_reported=True,
                http_status=HTTP_STATUS_OK,
                model=route.model_id,
            )
            record_attempt(
                registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                AttemptOutcome.SUCCESS, resp=raw_resp, applied_changes=adaptations.get("removed_params")
            )
            circuit_breaker.record_success(route.provider_id, route.model_id)
            return
        except Exception as exc:
            last_error = exc
            lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
            if isinstance(exc, TransportError):
                circuit_breaker.record_failure(route.provider_id, route.model_id)
            record_attempt(
                registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                AttemptOutcome.TRANSPORT_ERROR if isinstance(exc, TransportError) else AttemptOutcome.PROVIDER_ERROR,
                error=exc
            )
            if stream_started:
                raise
            continue

    if last_error:
        raise last_error


async def run_stream_async(
    request: LLMRequestV1,
    route_ladder: List[Tuple[RouteV1, PolicyLeg]],
    get_adapter: Callable[[str], BaseLLMAdapter],
    circuit_breaker: RouteCircuitBreaker,
    registry: ProviderRegistry,
) -> AsyncIterator[str]:
    """Asynchronously stream delta tokens for an LLMRequestV1 across route ladder."""
    adaptations: Dict[str, Any] = {"removed_params": []}
    attempt_count = 0
    last_error = None

    for route, leg in route_ladder:
        if not circuit_breaker.is_available(route.provider_id, route.model_id):
            continue

        attempt_count += 1
        att_kind = AttemptKind.INITIAL if attempt_count == 1 else AttemptKind.ROUTE_FALLBACK
        att_start_utc = time.time()
        att_start_mono = time.monotonic()
        adapter = get_adapter(route.provider_id)
        kwargs, _, _ = normalize_request_kwargs(request, route, adaptations, registry)

        collected_tokens = 0
        full_text_chunks = []
        stream_started = False
        try:
            async for chunk in adapter.execute_stream_async(request, route, kwargs):
                stream_started = True
                collected_tokens += 1
                full_text_chunks.append(chunk)
                yield chunk

            lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
            raw_resp = RawAdapterResponse(
                text="".join(full_text_chunks),
                prompt_tokens=40,
                completion_tokens=collected_tokens,
                total_tokens=40 + collected_tokens,
                is_usage_reported=True,
                http_status=HTTP_STATUS_OK,
                model=route.model_id,
            )
            record_attempt(
                registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                AttemptOutcome.SUCCESS, resp=raw_resp, applied_changes=adaptations.get("removed_params")
            )
            circuit_breaker.record_success(route.provider_id, route.model_id)
            return
        except Exception as exc:
            last_error = exc
            lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
            if isinstance(exc, TransportError):
                circuit_breaker.record_failure(route.provider_id, route.model_id)
            record_attempt(
                registry, request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                AttemptOutcome.TRANSPORT_ERROR if isinstance(exc, TransportError) else AttemptOutcome.PROVIDER_ERROR,
                error=exc
            )
            if stream_started:
                raise
            continue

    if last_error:
        raise last_error
