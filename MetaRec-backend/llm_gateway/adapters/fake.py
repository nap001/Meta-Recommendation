"""Deterministic Fake LLM Adapter for offline testing and fault injection."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Union

from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.constants import (
    DEFAULT_FAKE_COMPLETION_TOKENS,
    DEFAULT_FAKE_PROMPT_TOKENS,
    DEFAULT_SCRIPTED_COMPLETION_TOKENS,
    DEFAULT_SCRIPTED_PROMPT_TOKENS,
    DEFAULT_SCRIPTED_REQUEST_ID,
    HTTP_STATUS_OK,
    MS_PER_SECOND,
)
from llm_gateway.contracts import LLMRequestV1, ResponseKind, RouteV1
from llm_gateway.errors import (
    CompatibilityError,
    ProviderError,
    TransportError,
)


class ScriptedResponse:
    """Pre-configured response or error trigger for FakeLLMAdapter."""

    def __init__(
        self,
        *,
        text: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        prompt_tokens: int = DEFAULT_SCRIPTED_PROMPT_TOKENS,
        completion_tokens: int = DEFAULT_SCRIPTED_COMPLETION_TOKENS,
        http_status: int = HTTP_STATUS_OK,
        provider_request_id: str = DEFAULT_SCRIPTED_REQUEST_ID,
        simulated_delay_ms: float = 0.0,
        error_to_raise: Optional[Exception] = None,
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls or []
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.http_status = http_status
        self.provider_request_id = provider_request_id
        self.simulated_delay_ms = simulated_delay_ms
        self.error_to_raise = error_to_raise


class FakeLLMAdapter(BaseLLMAdapter):
    """Deterministic, zero-network test double for LLM Gateway testing."""

    def __init__(self) -> None:
        self.response_queue: List[Union[ScriptedResponse, Callable[..., ScriptedResponse]]] = []
        self.call_history: List[Dict[str, Any]] = []
        self.default_prompt_tokens: int = DEFAULT_FAKE_PROMPT_TOKENS
        self.default_completion_tokens: int = DEFAULT_FAKE_COMPLETION_TOKENS

    def enqueue_response(self, response: Union[ScriptedResponse, Callable[..., ScriptedResponse]]) -> None:
        self.response_queue.append(response)

    def clear(self) -> None:
        self.response_queue.clear()
        self.call_history.clear()

    def _generate_default_response(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        # If response contract is JSON, provide a deterministic valid JSON string
        if request.response_contract.kind == ResponseKind.JSON:
            schema_id = request.response_contract.schema_id
            if "confirmation" in schema_id:
                text = '{"domain": "restaurant", "intent": "book", "slots": {"cuisine": "Italian"}}'
            elif "gather" in schema_id:
                text = '{"thought": "search candidates", "tool": "search_restaurants", "params": {"query": "Italian"}}'
            else:
                text = '{"status": "ok", "message": "Default fake json response"}'
        else:
            text = f"Default fake response from {route.provider_id}:{route.model_id}"

        return RawAdapterResponse(
            text=text,
            tool_calls=[],
            prompt_tokens=self.default_prompt_tokens,
            completion_tokens=self.default_completion_tokens,
            total_tokens=self.default_prompt_tokens + self.default_completion_tokens,
            is_usage_reported=True,
            http_status=HTTP_STATUS_OK,
            provider_request_id=f"fake-req-{len(self.call_history)}",
            model=route.model_id,
        )

    def _append_history(self, entry: Dict[str, Any]) -> None:
        if len(self.call_history) >= 1000:
            self.call_history.pop(0)
        self.call_history.append(entry)

    def execute(self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]) -> RawAdapterResponse:
        self._append_history(
            {"request": request, "route": route, "kwargs": normalized_kwargs, "timestamp": time.time()}
        )

        if not self.response_queue:
            return self._generate_default_response(request, route, normalized_kwargs)

        item = self.response_queue.pop(0)
        if callable(item):
            scripted = item(request, route, normalized_kwargs)
        else:
            scripted = item

        if scripted.simulated_delay_ms > 0:
            time.sleep(scripted.simulated_delay_ms / MS_PER_SECOND)

        if scripted.error_to_raise:
            setattr(scripted.error_to_raise, "prompt_tokens", scripted.prompt_tokens)
            setattr(scripted.error_to_raise, "completion_tokens", scripted.completion_tokens)
            raise scripted.error_to_raise

        p_tok = scripted.prompt_tokens
        c_tok = scripted.completion_tokens
        t_tok = (p_tok + c_tok) if (p_tok is not None and c_tok is not None) else None
        is_rep = p_tok is not None

        return RawAdapterResponse(
            text=scripted.text,
            tool_calls=scripted.tool_calls,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            is_usage_reported=is_rep,
            http_status=scripted.http_status,
            provider_request_id=scripted.provider_request_id,
            model=route.model_id,
        )

    async def execute_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        self._append_history(
            {"request": request, "route": route, "kwargs": normalized_kwargs, "timestamp": time.time()}
        )

        if not self.response_queue:
            return self._generate_default_response(request, route, normalized_kwargs)

        item = self.response_queue.pop(0)
        if callable(item):
            scripted = item(request, route, normalized_kwargs)
        else:
            scripted = item

        if scripted.simulated_delay_ms > 0:
            await asyncio.sleep(scripted.simulated_delay_ms / MS_PER_SECOND)

        if scripted.error_to_raise:
            setattr(scripted.error_to_raise, "prompt_tokens", scripted.prompt_tokens)
            setattr(scripted.error_to_raise, "completion_tokens", scripted.completion_tokens)
            raise scripted.error_to_raise

        p_tok = scripted.prompt_tokens
        c_tok = scripted.completion_tokens
        t_tok = (p_tok + c_tok) if (p_tok is not None and c_tok is not None) else None
        is_rep = p_tok is not None

        return RawAdapterResponse(
            text=scripted.text,
            tool_calls=scripted.tool_calls,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            is_usage_reported=is_rep,
            http_status=scripted.http_status,
            provider_request_id=scripted.provider_request_id,
            model=route.model_id,
        )

    def execute_stream(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Yield deterministic streaming delta tokens."""
        raw_resp = self.execute(request, route, normalized_kwargs)
        text = raw_resp.text or ""
        # Chunk text by words or default token slices
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    async def execute_stream_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Asynchronously yield deterministic streaming delta tokens."""
        raw_resp = await self.execute_async(request, route, normalized_kwargs)
        text = raw_resp.text or ""
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.001)

