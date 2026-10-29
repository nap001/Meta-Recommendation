"""OpenAI Base and OpenAI-compatible HTTP adapter for LLM Gateway (WP5) with zero internal retries."""

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
    DEFAULT_OPENAI_BASE_URL,
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


class OpenAIBaseAdapter(BaseLLMAdapter):
    """Direct HTTP adapter for OpenAI and any OpenAI-compatible API base (Groq, Together, DeepSeek, vLLM, Ollama).

    Features:
    - Direct httpx calls guaranteeing strict zero internal SDK retries (WP5 Rule 2).
    - Custom base_url, api_key, and organization headers.
    - Client injection support (http_client / async_http_client) for deterministic testing and connection pooling.
    - Full OpenAI Chat Completions payload formatting and response parsing.
    """

    def __init__(
        self,
        provider_id: str = ProviderId.OPENAI.value,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.provider_id = provider_id
        self.base_url = base_url
        self.api_key = api_key
        self.organization = organization
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
        """Resolve base URL ensuring clean endpoint formatting."""
        if self.base_url:
            raw = self.base_url.strip().rstrip("/")
        else:
            raw = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")

        # If base url ends in /chat/completions, strip it back to base
        if raw.endswith("/chat/completions"):
            raw = raw[: -len("/chat/completions")]
        return raw

    def _get_api_key(self) -> str:
        """Resolve API key strictly scoped to provider (no cross-vendor key leakage)."""
        if self.api_key:
            return self.api_key.strip()
        if self.provider_id == ProviderId.OPENAI.value:
            key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        elif self.provider_id == ProviderId.GROQ.value:
            key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""
        else:
            key = os.getenv("LLM_API_KEY") or ""
        return key.strip()

    def _get_headers(self) -> Dict[str, str]:
        """Construct standard HTTP headers for OpenAI API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_api_key()}",
        }
        org = self.organization or os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
        if org:
            headers["OpenAI-Organization"] = org.strip()
        return headers

    def _get_completions_url(self) -> str:
        base = self._get_base_url()
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _format_payload(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare wire payload for /chat/completions endpoint."""
        payload: Dict[str, Any] = dict(normalized_kwargs)
        payload["model"] = route.model_id

        # Merge extra_body if provided by engine (e.g. max_completion_tokens)
        extra = payload.pop("extra_body", None)
        if isinstance(extra, dict):
            payload.update(extra)

        return payload

    def _parse_response(self, data: Dict[str, Any], model: str) -> RawAdapterResponse:
        """Parse raw OpenAI Chat Completion response JSON into RawAdapterResponse."""
        choices = data.get("choices", []) or []
        text: Optional[str] = None
        tool_calls: List[Dict[str, Any]] = []

        if choices:
            first_choice = choices[0]
            message = first_choice.get("message", {}) or {}
            text = message.get("content")
            raw_tc = message.get("tool_calls", []) or []
            for tc in raw_tc:
                func = tc.get("function", {}) or {}
                args = func.get("arguments", "{}")
                tool_calls.append(
                    {
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args if isinstance(args, str) else json.dumps(args),
                        },
                    }
                )

        usage = data.get("usage")
        if usage is not None:
            p_tokens = usage.get("prompt_tokens")
            c_tokens = usage.get("completion_tokens")
            t_tokens = usage.get("total_tokens") or (
                (p_tokens + c_tokens) if p_tokens is not None and c_tokens is not None else None
            )
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
        """Map HTTP error status and response payload into WP5 exception taxonomy."""
        msg = ""
        err_type = ""
        err_code = ""

        if isinstance(error_data, dict):
            err_obj = error_data.get("error", {})
            if isinstance(err_obj, dict):
                msg = err_obj.get("message", "")
                err_type = err_obj.get("type", "")
                err_code = str(err_obj.get("code", ""))
            else:
                msg = str(error_data)
        else:
            msg = str(error_data)

        msg_lower = msg.lower()

        # Rate limited or quota exceeded
        if status_code == HTTP_STATUS_TOO_MANY_REQUESTS or "rate_limit" in err_type or "rate_limit" in msg_lower:
            return TransportError(GatewayErrorCode.RATE_LIMITED.value, f"OpenAI rate limit exceeded: {msg}", retryable=True)

        # Server errors / gateway timeouts
        if status_code in RETRYABLE_HTTP_STATUSES or "server error" in msg_lower:
            return TransportError(GatewayErrorCode.PROVIDER_5XX.value, f"OpenAI server error ({status_code}): {msg}", retryable=True)

        # Authentication or permissions
        if status_code in AUTH_HTTP_STATUSES or "invalid_api_key" in err_code or "auth" in err_type:
            return ProviderError(GatewayErrorCode.AUTH_FAILED.value, f"OpenAI authentication failed ({status_code}): {msg}", retryable=False)

        # Incompatible parameters or bad requests
        if status_code == HTTP_STATUS_BAD_REQUEST:
            if any(term in msg_lower for term in ("unsupported", "parameter", "extra_body", "unknown parameter")):
                return CompatibilityError(GatewayErrorCode.UNSUPPORTED_PARAMETER.value, f"OpenAI parameter incompatible: {msg}", retryable=True)
            return ProviderError(GatewayErrorCode.INVALID_REQUEST.value, f"OpenAI bad request (400): {msg}", retryable=False)

        return ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, f"OpenAI error ({status_code}): {msg}", retryable=False)

    def _handle_exception(self, exc: Exception) -> Exception:
        """Classify unexpected transport or system exceptions."""
        if isinstance(exc, (TransportError, CompatibilityError, ProviderError)):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return TransportError(GatewayErrorCode.TIMEOUT.value, f"OpenAI request timed out: {exc}", retryable=True)
        if isinstance(exc, httpx.ConnectError):
            return TransportError(GatewayErrorCode.CONNECT_ERROR.value, f"OpenAI connection error: {exc}", retryable=True)
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                err_body = exc.response.json()
            except Exception:
                err_body = exc.response.text
            return self._handle_http_error(exc.response.status_code, err_body)

        return ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, f"OpenAI unexpected error: {exc}", retryable=False)

    def execute(self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]) -> RawAdapterResponse:
        """Synchronously execute Chat Completions request via direct HTTP."""
        payload = self._format_payload(request, route, normalized_kwargs)
        url = self._get_completions_url()
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
            return self._parse_response(resp.json(), route.model_id)
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def execute_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        """Asynchronously execute Chat Completions request via direct HTTP."""
        payload = self._format_payload(request, route, normalized_kwargs)
        url = self._get_completions_url()
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
            return self._parse_response(resp.json(), route.model_id)
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    def execute_stream(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute synchronous HTTP streaming yielding delta tokens."""
        payload = self._format_payload(request, route, normalized_kwargs)
        payload["stream"] = True
        url = self._get_completions_url()
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
                        if raw_data == "[DONE]":
                            break
                        try:
                            data = json.loads(raw_data)
                            choices = data.get("choices", []) or []
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                content = delta.get("content")
                                if content:
                                    yield content
                        except Exception:
                            continue
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def execute_stream_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute asynchronous HTTP streaming yielding delta tokens."""
        payload = self._format_payload(request, route, normalized_kwargs)
        payload["stream"] = True
        url = self._get_completions_url()
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
                        if raw_data == "[DONE]":
                            break
                        try:
                            data = json.loads(raw_data)
                            choices = data.get("choices", []) or []
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                content = delta.get("content")
                                if content:
                                    yield content
                        except Exception:
                            continue
        except Exception as exc:
            raise self._handle_exception(exc) from exc



# Alias for convenience
OpenAIAdapter = OpenAIBaseAdapter
