"""OpenAI-compatible and Azure OpenAI live adapter with zero internal retries.

Fix #7: Removed the hard-coded ``https://agenthiack.openai.azure.com/`` default
fallback for ``AZURE_OPENAI_ENDPOINT``.  A missing env var now raises
``ConfigurationError`` immediately so misconfigured deployments fail fast and
visibly instead of silently routing to the wrong endpoint.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

if "proxies" not in httpx.Client.__init__.__code__.co_varnames:
    _orig_client_init = httpx.Client.__init__
    _orig_async_init = httpx.AsyncClient.__init__

    def _patched_client_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            p = kwargs.pop("proxies")
            if p is not None and "proxy" not in kwargs:
                kwargs["proxy"] = p
        return _orig_client_init(self, *args, **kwargs)

    def _patched_async_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            p = kwargs.pop("proxies")
            if p is not None and "proxy" not in kwargs:
                kwargs["proxy"] = p
        return _orig_async_init(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_client_init
    httpx.AsyncClient.__init__ = _patched_async_init

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.constants import (
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_READ_TIMEOUT_SECONDS,
    GatewayErrorCode,
    HTTP_STATUS_OK,
    MS_PER_SECOND,
    ProviderId,
)
from llm_gateway.contracts import LLMRequestV1, RouteV1
from llm_gateway.errors import (
    CompatibilityError,
    ConfigurationError,
    ProviderError,
    TransportError,
)


class OpenAICompatibleAdapter(BaseLLMAdapter):
    """Adapter for live OpenAI-compatible and Azure endpoints.

    Fix #6: SDK clients are now cached per (is_azure, timeout_seconds) key
    instead of being recreated on every call.  This prevents per-call socket
    exhaustion under concurrent load while still respecting different deadline
    budgets for different request types.
    """

    def __init__(self, provider_id: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.provider_id = provider_id
        self.base_url = base_url
        self.api_key = api_key
        # Client caches: keyed by (is_azure: bool, timeout_seconds: float)
        self._sync_client_cache: Dict[tuple, Any] = {}
        self._async_client_cache: Dict[tuple, Any] = {}

    def _resolve_config(self, is_azure: bool) -> tuple[str, str]:
        """Resolve API key and base URL strictly scoped to this provider (no cross-vendor leakage)."""
        if is_azure:
            azure_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY") or ""
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            if not azure_endpoint:
                raise ConfigurationError(
                    "AZURE_OPENAI_ENDPOINT environment variable is not set. "
                    "Set it to your Azure OpenAI resource endpoint, e.g. "
                    "https://<resource>.openai.azure.com/"
                )
            return azure_key, azure_endpoint
        elif self.provider_id == ProviderId.OPENAI.value:
            api_key = self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
            base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
            return api_key, base_url
        elif self.provider_id == ProviderId.GROQ.value:
            api_key = self.api_key or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""
            base_url = self.base_url or os.getenv("GROQ_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_GROQ_BASE_URL
            return api_key, base_url
        else:
            api_key = self.api_key or os.getenv("LLM_API_KEY") or ""
            base_url = self.base_url or os.getenv("LLM_BASE_URL") or DEFAULT_OPENAI_BASE_URL
            return api_key, base_url

    def _get_sync_client(self, is_azure: bool = False, timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS) -> Any:
        """Return a cached sync SDK client, creating one if needed."""
        api_key, base_url = self._resolve_config(is_azure)
        key_id = api_key[:8] if api_key else "none"
        cache_key = (self.provider_id, is_azure, base_url, key_id, round(timeout_seconds, 2))
        if cache_key in self._sync_client_cache:
            return self._sync_client_cache[cache_key]

        if is_azure:
            client = AzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
                max_retries=0,  # CRITICAL: zero SDK retries
                timeout=timeout_seconds,
            )
        else:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=0,
                timeout=timeout_seconds,
            )

        self._sync_client_cache[cache_key] = client
        return client

    def _get_async_client(self, is_azure: bool = False, timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS) -> Any:
        """Return a cached async SDK client, creating one if needed."""
        api_key, base_url = self._resolve_config(is_azure)
        key_id = api_key[:8] if api_key else "none"
        cache_key = (self.provider_id, is_azure, base_url, key_id, round(timeout_seconds, 2))
        if cache_key in self._async_client_cache:
            return self._async_client_cache[cache_key]

        if is_azure:
            client = AsyncAzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
                max_retries=0,  # CRITICAL: zero SDK retries
                timeout=timeout_seconds,
            )
        else:
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=0,
                timeout=timeout_seconds,
            )

        self._async_client_cache[cache_key] = client
        return client

    def _parse_sdk_response(self, response: Any, model: str) -> RawAdapterResponse:
        choices = getattr(response, "choices", []) or []
        text: Optional[str] = None
        tool_calls: List[Dict[str, Any]] = []

        if choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message:
                text = getattr(message, "content", None)
                raw_tc = getattr(message, "tool_calls", None) or []
                for tc in raw_tc:
                    func = getattr(tc, "function", None)
                    if func:
                        tool_calls.append(
                            {
                                "id": getattr(tc, "id", "call_1"),
                                "type": "function",
                                "function": {
                                    "name": getattr(func, "name", ""),
                                    "arguments": getattr(func, "arguments", "{}"),
                                },
                            }
                        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            p_tokens = getattr(usage, "prompt_tokens", None)
            c_tokens = getattr(usage, "completion_tokens", None)
            t_tokens = getattr(usage, "total_tokens", None) or (
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
            provider_request_id=getattr(response, "id", None),
            model=model,
            raw_response=response,
        )

    def _handle_exception(self, exc: Exception) -> Exception:
        exc_str = str(exc).lower()
        if "rate limit" in exc_str or "429" in exc_str:
            return TransportError(GatewayErrorCode.RATE_LIMITED.value, f"Provider rate limit exceeded: {exc}", retryable=True)
        if "timeout" in exc_str or "timed out" in exc_str:
            return TransportError(GatewayErrorCode.TIMEOUT.value, f"Provider request timed out: {exc}", retryable=True)
        if "connection error" in exc_str or "connect" in exc_str:
            return TransportError(GatewayErrorCode.CONNECT_ERROR.value, f"Provider connection error: {exc}", retryable=True)
        if "500" in exc_str or "502" in exc_str or "503" in exc_str or "504" in exc_str:
            return TransportError(GatewayErrorCode.PROVIDER_5XX.value, f"Provider 5xx server error: {exc}", retryable=True)
        if "unsupported" in exc_str or "does not support" in exc_str or "extra_body" in exc_str:
            return CompatibilityError(GatewayErrorCode.UNSUPPORTED_PARAMETER.value, f"Parameter incompatible with provider: {exc}", retryable=True)
        if "authentication" in exc_str or "401" in exc_str or "403" in exc_str:
            return ProviderError(GatewayErrorCode.AUTH_FAILED.value, f"Provider authentication failed: {exc}", retryable=False)

        return ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, f"Provider invocation error: {exc}", retryable=False)

    def execute(self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]) -> RawAdapterResponse:
        is_azure = route.provider_id == ProviderId.AZURE_OPENAI.value
        client = self._get_sync_client(is_azure=is_azure, timeout_seconds=request.budget.deadline_ms / MS_PER_SECOND)
        try:
            resp = client.chat.completions.create(**normalized_kwargs)
            return self._parse_sdk_response(resp, route.model_id)
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def execute_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        is_azure = route.provider_id == ProviderId.AZURE_OPENAI.value
        client = self._get_async_client(is_azure=is_azure, timeout_seconds=request.budget.deadline_ms / MS_PER_SECOND)
        try:
            resp = await client.chat.completions.create(**normalized_kwargs)
            return self._parse_sdk_response(resp, route.model_id)
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    def execute_stream(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute synchronous streaming request yielding delta tokens."""
        is_azure = route.provider_id == ProviderId.AZURE_OPENAI.value
        client = self._get_sync_client(is_azure=is_azure, timeout_seconds=request.budget.deadline_ms / MS_PER_SECOND)
        stream_kwargs = dict(normalized_kwargs)
        stream_kwargs["stream"] = True
        try:
            stream_resp = client.chat.completions.create(**stream_kwargs)
            for chunk in stream_resp:
                choices = getattr(chunk, "choices", []) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def execute_stream_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute asynchronous streaming request yielding delta tokens."""
        is_azure = route.provider_id == ProviderId.AZURE_OPENAI.value
        client = self._get_async_client(is_azure=is_azure, timeout_seconds=request.budget.deadline_ms / MS_PER_SECOND)
        stream_kwargs = dict(normalized_kwargs)
        stream_kwargs["stream"] = True
        try:
            stream_resp = await client.chat.completions.create(**stream_kwargs)
            async for chunk in stream_resp:
                choices = getattr(chunk, "choices", []) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
        except Exception as exc:
            raise self._handle_exception(exc) from exc

