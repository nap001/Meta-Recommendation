"""Core GatewayEngine class implementing lifecycle orchestration for LLM Gateway."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_gateway.adapters.anthropic import AnthropicAdapter
from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.adapters.fake import FakeLLMAdapter
from llm_gateway.adapters.openai_base import OpenAIBaseAdapter
from llm_gateway.adapters.openai_compat import OpenAICompatibleAdapter
from llm_gateway.constants import (
    AdapterKind,
    DEFAULT_MAX_ATTEMPTS_PER_ROUTE,
    ProviderId,
)
from llm_gateway.contracts import (
    AttemptKind,
    AttemptOutcome,
    DegradationSummaryV1,
    LLMRequestV1,
    LLMResultV1,
    LLMToolProjectionV1,
    OutputPayloadV1,
    PhysicalAttemptV1,
    PolicyLeg,
    RouteV1,
    ValidationGateResultV1,
    ValidationSummaryV1,
)
from llm_gateway.errors import LLMErrorV1
from llm_gateway.registry import ProviderRegistry, get_provider_registry
from llm_gateway.engine.attempt import (
    log_privacy_safe,
    record_attempt,
    run_attempt_loop_async,
    run_attempt_loop_sync,
)
from llm_gateway.engine.projection import (
    build_tool_projection,
    clamp_output_tokens,
    normalize_request_kwargs,
    project_prompt_embedded_tools,
)
from llm_gateway.engine.resilience import RouteCircuitBreaker, _MAX_ATTEMPTS_PER_ROUTE
from llm_gateway.engine.result_builder import build_result
from llm_gateway.engine.routing import build_route_ladder, determine_initial_route
from llm_gateway.engine.stream import run_stream_async, run_stream_sync
from llm_gateway.engine.validation_runner import (
    execute_registered_validation_hook,
    execute_validation_async,
    execute_validation_sync,
    is_payload_contract_compatible,
    is_registered_contract,
    is_terminal_validation_error,
    validate_output_gate,
)

logger = logging.getLogger(__name__)


class GatewayEngine:
    """Execution engine for LLMRequestV1 -> LLMResultV1 lifecycle.

    Fix #3: Constructor accepts an optional ``registry`` so isolated engine
    instances can be created for testing without mutating the global singleton.
    Use ``create_engine()`` or ``create_engine_for_registry()`` factories.
    """

    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.registry = registry or get_provider_registry()
        self.validator_hook = validator_hook
        self.circuit_breaker = RouteCircuitBreaker()
        # Optional per-engine override of the module-level per-route attempt cap.
        self.max_attempts_per_route: Optional[int] = None
        self._adapters: Dict[str, BaseLLMAdapter] = {
            ProviderId.FAKE.value: FakeLLMAdapter(),
            ProviderId.GROQ.value: OpenAICompatibleAdapter(provider_id=ProviderId.GROQ.value),
            ProviderId.AZURE_OPENAI.value: OpenAICompatibleAdapter(provider_id=ProviderId.AZURE_OPENAI.value),
            ProviderId.OPENAI.value: OpenAICompatibleAdapter(provider_id=ProviderId.OPENAI.value),
            ProviderId.OPENAI_BASE.value: OpenAIBaseAdapter(provider_id=ProviderId.OPENAI_BASE.value),
            ProviderId.ANTHROPIC.value: AnthropicAdapter(provider_id=ProviderId.ANTHROPIC.value),
        }
        self.last_tool_projection: Optional[LLMToolProjectionV1] = None

    def set_validator_hook(self, hook: Optional[Callable[..., Any]]) -> None:
        """Register or clear an external validator hook."""
        self.validator_hook = hook

    def register_adapter(self, provider_id: str, adapter: BaseLLMAdapter) -> None:
        self._adapters[provider_id] = adapter

    def _route_attempt_cap(self) -> int:
        return self.max_attempts_per_route or _MAX_ATTEMPTS_PER_ROUTE

    def get_adapter(self, provider_id: str) -> BaseLLMAdapter:
        if provider_id in self._adapters:
            return self._adapters[provider_id]

        prov_cap = self.registry.get_provider(provider_id)
        if prov_cap:
            if prov_cap.adapter_kind == AdapterKind.ANTHROPIC.value:
                adapter = AnthropicAdapter(provider_id=provider_id)
                self._adapters[provider_id] = adapter
                return adapter
            if prov_cap.adapter_kind in (AdapterKind.OPENAI_BASE.value, AdapterKind.OPENAI_HTTP.value):
                adapter = OpenAIBaseAdapter(provider_id=provider_id)
                self._adapters[provider_id] = adapter
                return adapter

        # Default to OpenAI compatible adapter
        adapter = OpenAICompatibleAdapter(provider_id=provider_id)
        self._adapters[provider_id] = adapter
        return adapter

    def _determine_initial_route(self, request: LLMRequestV1) -> Tuple[RouteV1, PolicyLeg]:
        return determine_initial_route(request)

    def _build_route_ladder(
        self, request: LLMRequestV1, initial_route: RouteV1, initial_leg: PolicyLeg
    ) -> List[Tuple[RouteV1, PolicyLeg]]:
        return build_route_ladder(request, initial_route, initial_leg)

    def _clamp_output_tokens(self, request: LLMRequestV1, route: RouteV1) -> int:
        return clamp_output_tokens(request, route, self.registry)

    def _build_tool_projection(
        self, request: LLMRequestV1, route: RouteV1
    ) -> Tuple[Optional[LLMToolProjectionV1], Dict[str, str]]:
        return build_tool_projection(request, route)

    def _project_prompt_embedded_tools(
        self,
        request: LLMRequestV1,
        messages: List[Dict[str, Any]],
        projection: Optional[LLMToolProjectionV1],
    ) -> List[Dict[str, Any]]:
        return project_prompt_embedded_tools(request, messages, projection)

    def _normalize_request_kwargs(
        self, request: LLMRequestV1, route: RouteV1, adaptations: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        kwargs, mapping, proj = normalize_request_kwargs(request, route, adaptations, self.registry)
        self.last_tool_projection = proj
        return kwargs, mapping

    def _record_attempt(
        self,
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
        return record_attempt(
            self.registry, request, route, leg, attempt_number, attempt_kind,
            started_at, latency_ms, outcome, resp=resp, error=error,
            applied_changes=applied_changes, validation_summary=validation_summary,
        )

    def _log_privacy_safe(self, request: LLMRequestV1, label: str, content: str) -> None:
        log_privacy_safe(request, label, content)

    def _validate_output_gate(
        self,
        request: LLMRequestV1,
        resp: RawAdapterResponse,
        sanitized_mapping: Dict[str, str],
        external_validator: Optional[Any] = None,
    ) -> ValidationGateResultV1:
        return validate_output_gate(request, resp, sanitized_mapping, external_validator=external_validator)

    def _is_payload_contract_compatible(
        self, request: LLMRequestV1, payload: Optional[OutputPayloadV1]
    ) -> Tuple[bool, Optional[str]]:
        return is_payload_contract_compatible(request, payload)

    @staticmethod
    def _is_terminal_validation_error(val_summary: ValidationSummaryV1) -> bool:
        return is_terminal_validation_error(val_summary)

    def _is_registered_contract(self, request: LLMRequestV1) -> bool:
        return is_registered_contract(request)

    def _execute_registered_validation_hook(
        self, request: LLMRequestV1, resp: RawAdapterResponse, sanitized_mapping: Dict[str, str]
    ) -> ValidationGateResultV1:
        return execute_registered_validation_hook(request, resp, sanitized_mapping)

    def _execute_validation_sync(
        self,
        request: LLMRequestV1,
        resp: RawAdapterResponse,
        sanitized_mapping: Dict[str, str],
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> ValidationGateResultV1:
        return execute_validation_sync(
            request, resp, sanitized_mapping, validator_hook=validator_hook,
            engine_validator_hook=self.validator_hook
        )

    async def _execute_validation_async(
        self,
        request: LLMRequestV1,
        resp: RawAdapterResponse,
        sanitized_mapping: Dict[str, str],
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> ValidationGateResultV1:
        return await execute_validation_async(
            request, resp, sanitized_mapping, validator_hook=validator_hook,
            engine_validator_hook=self.validator_hook
        )

    def _validate_output(
        self,
        request: LLMRequestV1,
        resp: RawAdapterResponse,
        sanitized_mapping: Dict[str, str],
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> Tuple[ValidationSummaryV1, Optional[OutputPayloadV1]]:
        gate_res = self._execute_validation_sync(request, resp, sanitized_mapping, validator_hook)
        return gate_res.summary, gate_res.payload

    def _run_attempt_loop_sync(
        self,
        request: LLMRequestV1,
        route_ladder: List[Tuple[RouteV1, PolicyLeg]],
        initial_route: RouteV1,
        initial_leg: PolicyLeg,
        start_time_monotonic: float,
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> Tuple[
        List[PhysicalAttemptV1],
        Optional[OutputPayloadV1],
        Optional[RouteV1],
        ValidationSummaryV1,
        Optional[DegradationSummaryV1],
        Optional[LLMErrorV1],
    ]:
        attempts, released_output, selected_route, final_val_summary, degradation, last_error, last_proj = (
            run_attempt_loop_sync(
                request, route_ladder, initial_route, initial_leg, start_time_monotonic,
                self.get_adapter, self.circuit_breaker, self.registry, self._route_attempt_cap(),
                validator_hook=validator_hook, engine_validator_hook=self.validator_hook,
            )
        )
        if last_proj is not None:
            self.last_tool_projection = last_proj
        return attempts, released_output, selected_route, final_val_summary, degradation, last_error

    async def _run_attempt_loop_async(
        self,
        request: LLMRequestV1,
        route_ladder: List[Tuple[RouteV1, PolicyLeg]],
        initial_route: RouteV1,
        initial_leg: PolicyLeg,
        start_time_monotonic: float,
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> Tuple[
        List[PhysicalAttemptV1],
        Optional[OutputPayloadV1],
        Optional[RouteV1],
        ValidationSummaryV1,
        Optional[DegradationSummaryV1],
        Optional[LLMErrorV1],
    ]:
        attempts, released_output, selected_route, final_val_summary, degradation, last_error, last_proj = (
            await run_attempt_loop_async(
                request, route_ladder, initial_route, initial_leg, start_time_monotonic,
                self.get_adapter, self.circuit_breaker, self.registry, self._route_attempt_cap(),
                validator_hook=validator_hook, engine_validator_hook=self.validator_hook,
            )
        )
        if last_proj is not None:
            self.last_tool_projection = last_proj
        return attempts, released_output, selected_route, final_val_summary, degradation, last_error

    def invoke_llm(
        self,
        request: LLMRequestV1,
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> LLMResultV1:
        """Synchronous execution of LLMRequestV1 across provider route ladder."""
        start_time_monotonic = time.monotonic()
        start_time_utc = time.time()
        deadline_ms = request.budget.deadline_ms

        initial_route, initial_leg = self._determine_initial_route(request)
        route_ladder = self._build_route_ladder(request, initial_route, initial_leg)

        attempts, released_output, selected_route, final_val_summary, degradation, last_error = (
            self._run_attempt_loop_sync(
                request, route_ladder, initial_route, initial_leg, start_time_monotonic,
                validator_hook=validator_hook,
            )
        )

        return self._build_result(
            request, attempts, released_output, selected_route, final_val_summary,
            degradation, last_error, start_time_utc, start_time_monotonic, deadline_ms,
        )

    async def invoke_llm_async(
        self,
        request: LLMRequestV1,
        validator_hook: Optional[Callable[..., Any]] = None,
    ) -> LLMResultV1:
        """Asynchronous execution of LLMRequestV1 across provider route ladder."""
        start_time_monotonic = time.monotonic()
        start_time_utc = time.time()
        deadline_ms = request.budget.deadline_ms

        initial_route, initial_leg = self._determine_initial_route(request)
        route_ladder = self._build_route_ladder(request, initial_route, initial_leg)

        attempts, released_output, selected_route, final_val_summary, degradation, last_error = (
            await self._run_attempt_loop_async(
                request, route_ladder, initial_route, initial_leg, start_time_monotonic,
                validator_hook=validator_hook,
            )
        )

        return self._build_result(
            request, attempts, released_output, selected_route, final_val_summary,
            degradation, last_error, start_time_utc, start_time_monotonic, deadline_ms,
        )

    async def invoke_llm_in_executor(
        self,
        request: LLMRequestV1,
        validator_hook: Optional[Callable[..., Any]] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> LLMResultV1:
        """Run synchronous invoke_llm safely in a thread-pool executor."""
        loop = asyncio.get_running_loop()
        from functools import partial
        call_fn = partial(self.invoke_llm, request, validator_hook=validator_hook)
        return await loop.run_in_executor(executor, call_fn)

    def invoke_llm_stream(self, request: LLMRequestV1):
        """Synchronously stream delta tokens for an LLMRequestV1 across route ladder."""
        initial_route, initial_leg = self._determine_initial_route(request)
        route_ladder = self._build_route_ladder(request, initial_route, initial_leg)
        return run_stream_sync(request, route_ladder, self.get_adapter, self.circuit_breaker, self.registry)

    async def invoke_llm_stream_async(self, request: LLMRequestV1):
        """Asynchronously stream delta tokens for an LLMRequestV1 across route ladder."""
        initial_route, initial_leg = self._determine_initial_route(request)
        route_ladder = self._build_route_ladder(request, initial_route, initial_leg)
        async for chunk in run_stream_async(request, route_ladder, self.get_adapter, self.circuit_breaker, self.registry):
            yield chunk

    def _build_result(
        self,
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
        return build_result(
            request, attempts, released_output, selected_route, final_val_summary,
            degradation, last_error, start_time_utc, start_time_monotonic, deadline_ms,
        )
