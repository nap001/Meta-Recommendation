"""Anthropic Claude adapter for LLM Gateway (WP5) with zero internal retries."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional
import httpx

from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.constants import (
    AUTH_HTTP_STATUSES,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_VERSION,
    GatewayErrorCode,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_OK,
    HTTP_STATUS_TOO_MANY_REQUESTS,
    MS_PER_SECOND,
    ProviderId,
    RETRYABLE_HTTP_STATUSES,
)
from llm_gateway.contracts import LLMRequestV1, RouteV1
from llm_gateway.errors import CompatibilityError, ProviderError, TransportError

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude messages API with strict zero SDK retries."""

    def __init__(
        self,
        provider_id: str = ProviderId.ANTHROPIC.value,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.provider_id = provider_id
        self.base_url = base_url
        self.api_key = api_key
        self._http_client = http_client
        self._async_http_client = async_http_client
        self._shared_http_client: Optional[httpx.Client] = None
        self._shared_async_client: Optional[httpx.AsyncClient] = None

    def _get_sync_client(self, timeout_sec: float) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        if self._shared_http_client is None or self._shared_http_client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self._shared_http_client = httpx.Client(limits=limits, timeout=timeout_sec)
        return self._shared_http_client

    def _get_async_client(self, timeout_sec: float) -> httpx.AsyncClient:
        if self._async_http_client is not None:
            return self._async_http_client
        if self._shared_async_client is None or self._shared_async_client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self._shared_async_client = httpx.AsyncClient(limits=limits, timeout=timeout_sec)
        return self._shared_async_client

    def _get_base_url(self) -> str:
        url = self.base_url or os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL
        return url.rstrip("/")

    def _get_api_key(self) -> str:
        return (self.api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()


    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._get_api_key(),
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION),
            "content-type": "application/json",
        }

    def _format_anthropic_payload(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert standard LLMRequestV1 / normalized kwargs into Anthropic Messages API wire format."""
        system_parts: List[str] = []
        messages_wire: List[Dict[str, Any]] = []

        # 1. Extract system / developer messages; format user and assistant turns
        for msg in request.messages:
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if role_val in ("system", "developer"):
                if msg.content:
                    system_parts.append(msg.content)
            elif role_val == "assistant":
                messages_wire.append({"role": "assistant", "content": msg.content})
            elif role_val == "user":
                messages_wire.append({"role": "user", "content": msg.content})
            elif role_val == "tool":
                messages_wire.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.name or "tool_call_0",
                                "content": msg.content,
                            }
                        ],
                    }
                )

        # 2. Merge consecutive turns with the same role (Anthropic requires strict alternating roles)
        consolidated: List[Dict[str, Any]] = []
        for m in messages_wire:
            if consolidated and consolidated[-1]["role"] == m["role"]:
                prev = consolidated[-1]["content"]
                curr = m["content"]
                if isinstance(prev, str) and isinstance(curr, str):
                    consolidated[-1]["content"] = f"{prev}\n\n{curr}"
                elif isinstance(prev, list) and isinstance(curr, list):
                    consolidated[-1]["content"].extend(curr)
                elif isinstance(prev, list) and isinstance(curr, str):
                    consolidated[-1]["content"].append({"type": "text", "text": curr})
                elif isinstance(prev, str) and isinstance(curr, list):
                    consolidated[-1]["content"] = [{"type": "text", "text": prev}] + curr
            else:
                consolidated.append(m)

        if not consolidated:
            # Fix (minor): Previously injected a silent {"role":"user","content":"Hello"}
            # which is a data corruption — the caller sent no messages, so we raise.
            from llm_gateway.errors import RequestValidationError
            raise RequestValidationError(
                "empty_messages",
                "Anthropic adapter requires at least one non-system message. "
                "The consolidated message list is empty after processing.",
            )

        # 3. max_tokens is strictly required by Anthropic
        max_tokens = (
            normalized_kwargs.get("max_tokens")
            or request.generation.max_output_tokens
            or 4096
        )

        payload: Dict[str, Any] = {
            "model": route.model_id,
            "messages": consolidated,
            "max_tokens": max_tokens,
        }

        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        if "temperature" in normalized_kwargs and normalized_kwargs["temperature"] is not None:
            payload["temperature"] = float(normalized_kwargs["temperature"])

        stop = normalized_kwargs.get("stop") or request.generation.stop
        if stop:
            payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # 4. Map tools to Anthropic format: (name, description, input_schema)
        if request.tools and request.tools.tools:
            anthropic_tools = []
            for t in request.tools.tools:
                anthropic_tools.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.parameters or {"type": "object", "properties": {}},
                    }
                )
            payload["tools"] = anthropic_tools

            # Tool choice mapping
            if request.tool_choice:
                if request.tool_choice == "auto":
                    payload["tool_choice"] = {"type": "auto"}
                elif request.tool_choice == "any":
                    payload["tool_choice"] = {"type": "any"}
                elif isinstance(request.tool_choice, str) and request.tool_choice not in ("none", "auto"):
                    payload["tool_choice"] = {"type": "tool", "name": request.tool_choice}
                elif isinstance(request.tool_choice, dict):
                    fn_name = (
                        request.tool_choice.get("function", {}).get("name")
                        or request.tool_choice.get("name")
                    )
                    if fn_name:
                        payload["tool_choice"] = {"type": "tool", "name": fn_name}

        return payload

    def _parse_anthropic_response(self, data: Dict[str, Any], model: str) -> RawAdapterResponse:
        """Parse Anthropic JSON response into standard RawAdapterResponse."""
        content_blocks = data.get("content", []) or []
        text_pieces: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in content_blocks:
            b_type = block.get("type")
            if b_type == "text":
                text_pieces.append(block.get("text", ""))
            elif b_type == "tool_use":
                inp = block.get("input", {})
                tool_calls.append(
                    {
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(inp) if isinstance(inp, dict) else str(inp or "{}"),
                        },
                    }
                )

        text = "".join(text_pieces) if text_pieces else None
        usage = data.get("usage")
        if usage is not None:
            p_tokens = usage.get("input_tokens")
            c_tokens = usage.get("output_tokens")
            t_tokens = (p_tokens + c_tokens) if (p_tokens is not None and c_tokens is not None) else None
            is_reported = p_tokens is not None
        else:
            p_tokens = None
            c_tokens = None
            t_tokens = None
            is_reported = False

        return RawAdapterResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            is_usage_reported=is_reported,
            http_status=HTTP_STATUS_OK,
            provider_request_id=data.get("id"),
            model=data.get("model", model),
            raw_response=data,
        )

    def _handle_http_error(self, status_code: int, error_data: Any) -> Exception:
        msg = ""
        err_type = ""
        if isinstance(error_data, dict):
            err_obj = error_data.get("error", {})
            if isinstance(err_obj, dict):
                msg = err_obj.get("message", "")
                err_type = err_obj.get("type", "")
            else:
                msg = str(error_data)
        else:
            msg = str(error_data)

        msg_lower = msg.lower()
        if status_code == HTTP_STATUS_TOO_MANY_REQUESTS or "rate_limit" in err_type or "rate_limit" in msg_lower:
            return TransportError(GatewayErrorCode.RATE_LIMITED.value, f"Anthropic rate limit exceeded: {msg}", retryable=True)
        if status_code in RETRYABLE_HTTP_STATUSES or "overloaded" in msg_lower or "overloaded_error" in err_type:
            return TransportError(GatewayErrorCode.PROVIDER_5XX.value, f"Anthropic server/overload error ({status_code}): {msg}", retryable=True)
        if status_code in AUTH_HTTP_STATUSES or "authentication" in err_type or "permission" in err_type:
            return ProviderError(GatewayErrorCode.AUTH_FAILED.value, f"Anthropic authentication failed ({status_code}): {msg}", retryable=False)
        if status_code == HTTP_STATUS_BAD_REQUEST:
            if "unsupported" in msg_lower or "unknown parameter" in msg_lower or "invalid_request_error" in err_type:
                return CompatibilityError(GatewayErrorCode.UNSUPPORTED_PARAMETER.value, f"Anthropic parameter error: {msg}", retryable=True)
            return ProviderError(GatewayErrorCode.INVALID_REQUEST.value, f"Anthropic bad request (400): {msg}", retryable=False)

        return ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, f"Anthropic invocation error ({status_code}): {msg}", retryable=False)

    def _handle_exception(self, exc: Exception) -> Exception:
        if isinstance(exc, (TransportError, CompatibilityError, ProviderError)):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return TransportError(GatewayErrorCode.TIMEOUT.value, f"Anthropic request timed out: {exc}", retryable=True)
        if isinstance(exc, httpx.ConnectError):
            return TransportError(GatewayErrorCode.CONNECT_ERROR.value, f"Anthropic connection error: {exc}", retryable=True)
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                err_body = exc.response.json()
            except Exception:
                err_body = exc.response.text
            return self._handle_http_error(exc.response.status_code, err_body)
        return ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, f"Anthropic unexpected error: {exc}", retryable=False)

    def execute(self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]) -> RawAdapterResponse:
        """Synchronously execute Anthropic Messages API call with 0 internal retries."""
        payload = self._format_anthropic_payload(request, route, normalized_kwargs)
        url = f"{self._get_base_url()}/v1/messages"
        headers = self._get_headers()
        timeout_sec = request.budget.deadline_ms / MS_PER_SECOND

        try:
            client = self._get_sync_client(timeout_sec)
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != HTTP_STATUS_OK:
                try:
                    err_json = resp.json()
                except Exception:
                    err_json = resp.text
                raise self._handle_http_error(resp.status_code, err_json)
            return self._parse_anthropic_response(resp.json(), route.model_id)
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def execute_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        """Asynchronously execute Anthropic Messages API call with 0 internal retries."""
        payload = self._format_anthropic_payload(request, route, normalized_kwargs)
        url = f"{self._get_base_url()}/v1/messages"
        headers = self._get_headers()
        timeout_sec = request.budget.deadline_ms / MS_PER_SECOND

        try:
            client = self._get_async_client(timeout_sec)
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != HTTP_STATUS_OK:
                try:
                    err_json = resp.json()
                except Exception:
                    err_json = resp.text
                raise self._handle_http_error(resp.status_code, err_json)
            return self._parse_anthropic_response(resp.json(), route.model_id)
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    def execute_stream(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute synchronous streaming Messages request yielding delta tokens."""
        payload = self._format_anthropic_payload(request, route, normalized_kwargs)
        payload["stream"] = True
        url = f"{self._get_base_url()}/v1/messages"
        headers = self._get_headers()
        timeout_sec = request.budget.deadline_ms / MS_PER_SECOND

        try:
            client = self._get_sync_client(timeout_sec)
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != HTTP_STATUS_OK:
                    resp.read()
                    raise self._handle_http_error(resp.status_code, resp.text)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        raw_data = line[len("data: ") :].strip()
                        try:
                            event = json.loads(raw_data)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {}) or {}
                                text = delta.get("text")
                                if text:
                                    yield text
                        except Exception:
                            continue
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def execute_stream_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute asynchronous streaming Messages request yielding delta tokens."""
        payload = self._format_anthropic_payload(request, route, normalized_kwargs)
        payload["stream"] = True
        url = f"{self._get_base_url()}/v1/messages"
        headers = self._get_headers()
        timeout_sec = request.budget.deadline_ms / MS_PER_SECOND

        try:
            client = self._get_async_client(timeout_sec)
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != HTTP_STATUS_OK:
                    await resp.aread()
                    raise self._handle_http_error(resp.status_code, resp.text)
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        raw_data = line[len("data: ") :].strip()
                        try:
                            event = json.loads(raw_data)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {}) or {}
                                text = delta.get("text")
                                if text:
                                    yield text
                        except Exception:
                            continue
        except Exception as exc:
            raise self._handle_exception(exc) from exc

