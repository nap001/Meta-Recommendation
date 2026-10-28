"""Adapter that drives a *caller-supplied* SDK client for the legacy shim.

Legacy MetaRec call sites build their own OpenAI/Azure/Groq client (see
``client.py``) and pass it to ``llm_compat.create_chat_completion``. To route
those calls through the gateway without rewriting call sites, the gateway must
execute the exact wire request the caller prepared, on the exact client the
caller supplied. This adapter does that, and translates SDK failures into the
gateway error taxonomy so the engine — not the SDK — owns every retry:

* an unsupported parameter that ``llm_compat`` knows how to adapt becomes a
  ``CompatibilityError`` (the adaptation is applied to the wire request before
  the engine retries);
* a connection error, timeout, 408/429 or 5xx becomes a ``TransportError``
  (the same failure classes the OpenAI SDK used to retry internally);
* anything else is re-raised unchanged, so the engine stops immediately and
  the shim can re-raise the caller's original exception.

Because many legacy calls run concurrently on one shared engine, the per-call
adapter is bound through a ``ContextVar`` and reached via
``LegacyClientDispatcher``, which is the object actually registered on the
engine. ContextVars are isolated per asyncio task and per thread.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.constants import GatewayErrorCode, HTTP_STATUS_OK
from llm_gateway.contracts import LLMRequestV1, RouteV1
from llm_gateway.errors import CompatibilityError, ProviderError, TransportError

# Provider ID used for routes that execute on a caller-supplied client. It is
# deliberately NOT registered in the ProviderRegistry: the real endpoint behind
# the client is opaque to the gateway, so registry limits/prices do not apply.
LEGACY_PROVIDER_ID = "legacy_sdk_client"

# ``adapt_fn(wire_request, exc)`` mutates ``wire_request`` in place and returns
# the adapted parameter name, or ``None`` when the error is not adaptable.
AdaptFn = Callable[[Dict[str, Any], Exception], Optional[str]]

_HTTP_REQUEST_TIMEOUT = 408
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR_FLOOR = 500


def classify_transport_error(exc: BaseException) -> Optional[str]:
    """Return a transport error code for retryable SDK failures, else ``None``.

    Classification uses the OpenAI SDK exception *types* rather than message
    substrings, so e.g. an auth error whose text happens to contain "connect"
    is never retried.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        return None

    if isinstance(exc, openai.APITimeoutError):
        return GatewayErrorCode.TIMEOUT.value
    if isinstance(exc, openai.APIConnectionError):
        return GatewayErrorCode.CONNECT_ERROR.value
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == _HTTP_TOO_MANY_REQUESTS:
            return GatewayErrorCode.RATE_LIMITED.value
        if status == _HTTP_REQUEST_TIMEOUT:
            return GatewayErrorCode.TIMEOUT.value
        if isinstance(status, int) and status >= _HTTP_SERVER_ERROR_FLOOR:
            return GatewayErrorCode.PROVIDER_5XX.value
    return None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_legacy_response(response: Any, model: str) -> RawAdapterResponse:
    """Tolerantly extract text, tool calls and usage from an SDK-shaped response.

    Legacy callers (and their test doubles) may return SDK objects, plain dicts
    or arbitrary sentinels; none of them may make the gateway raise.
    """
    text: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []

    choices = _field(response, "choices") or []
    if isinstance(choices, (list, tuple)) and choices:
        message = _field(choices[0], "message")
        if message is not None:
            content = _field(message, "content")
            text = content if isinstance(content, str) else None
            for tc in _field(message, "tool_calls") or []:
                function = _field(tc, "function")
                if function is None:
                    continue
                tool_calls.append(
                    {
                        "id": _field(tc, "id") or "call_1",
                        "type": "function",
                        "function": {
                            "name": _field(function, "name") or "",
                            "arguments": _field(function, "arguments") or "{}",
                        },
                    }
                )

    usage = _field(response, "usage")
    prompt_tokens = completion_tokens = total_tokens = None
    if usage is not None:
        prompt_tokens = _optional_int(_field(usage, "prompt_tokens"))
        completion_tokens = _optional_int(_field(usage, "completion_tokens"))
        total_tokens = _optional_int(_field(usage, "total_tokens"))
        if not total_tokens and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

    provider_request_id = _field(response, "id")
    return RawAdapterResponse(
        text=text,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        is_usage_reported=prompt_tokens is not None,
        http_status=HTTP_STATUS_OK,
        provider_request_id=provider_request_id if isinstance(provider_request_id, str) else None,
        model=model,
        raw_response=response,
    )


class LegacyClientAdapter(BaseLLMAdapter):
    """Per-call adapter executing a prepared wire request on a caller's client.

    The engine's own ``normalized_kwargs`` are intentionally ignored: the
    legacy caller already produced the exact wire request (tool-name
    sanitisation, reasoning-model parameter rewrites, ``response_format``,
    provider-specific extras), and rebuilding it from ``LLMRequestV1`` would be
    lossy. Compatibility adaptations are applied to ``wire_request`` itself.
    """

    def __init__(self, client: Any, wire_request: Dict[str, Any], adapt_fn: Optional[AdaptFn] = None) -> None:
        self.client = client
        self.wire_request = wire_request
        self._adapt_fn = adapt_fn
        self.last_exception: Optional[BaseException] = None
        self.last_error_was_adaptation = False
        self.has_response = False
        self.raw_response: Any = None

    def _translate(self, exc: Exception) -> Exception:
        """Map an SDK exception to the gateway taxonomy (or return it unchanged)."""
        self.last_exception = exc
        self.last_error_was_adaptation = False

        if self._adapt_fn is not None:
            adapted = self._adapt_fn(self.wire_request, exc)
            if adapted:
                self.last_error_was_adaptation = True
                # The engine keys its compatibility retry on the parameter name
                # appearing in the message ("temperature" / "max_tokens").
                return CompatibilityError(
                    GatewayErrorCode.UNSUPPORTED_PARAMETER.value,
                    f"Provider rejected parameter '{adapted}'; request adapted for retry",
                )

        code = classify_transport_error(exc)
        if code is not None:
            return TransportError(code, f"Transport failure ({type(exc).__name__})", retryable=True)
        return exc

    def _accept(self, response: Any, route: RouteV1) -> RawAdapterResponse:
        self.has_response = True
        self.raw_response = response
        self.last_exception = None
        self.last_error_was_adaptation = False
        return parse_legacy_response(response, route.model_id)

    def execute(self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]) -> RawAdapterResponse:
        try:
            response = self.client.chat.completions.create(**self.wire_request)
        except Exception as exc:
            translated = self._translate(exc)
            if translated is exc:
                raise
            raise translated from exc
        return self._accept(response, route)

    async def execute_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        try:
            response = await self.client.chat.completions.create(**self.wire_request)
        except Exception as exc:
            translated = self._translate(exc)
            if translated is exc:
                raise
            raise translated from exc
        return self._accept(response, route)


_ACTIVE_LEGACY_ADAPTER: contextvars.ContextVar[Optional[LegacyClientAdapter]] = contextvars.ContextVar(
    "llm_gateway_active_legacy_adapter", default=None
)


@contextmanager
def bind_legacy_adapter(adapter: LegacyClientAdapter) -> Iterator[LegacyClientAdapter]:
    """Bind ``adapter`` as the target of ``LegacyClientDispatcher`` for this context."""
    token = _ACTIVE_LEGACY_ADAPTER.set(adapter)
    try:
        yield adapter
    finally:
        _ACTIVE_LEGACY_ADAPTER.reset(token)


class LegacyClientDispatcher(BaseLLMAdapter):
    """Engine-registered adapter that forwards to the context-bound legacy adapter."""

    @staticmethod
    def _active() -> LegacyClientAdapter:
        adapter = _ACTIVE_LEGACY_ADAPTER.get()
        if adapter is None:
            raise ProviderError(
                GatewayErrorCode.CONFIGURATION_ERROR.value,
                f"No caller-supplied SDK client is bound for provider '{LEGACY_PROVIDER_ID}'",
            )
        return adapter

    def execute(self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]) -> RawAdapterResponse:
        return self._active().execute(request, route, normalized_kwargs)

    async def execute_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        return await self._active().execute_async(request, route, normalized_kwargs)
