"""Engine factories and top-level execution helpers for LLM Gateway."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from llm_gateway.contracts import LLMRequestV1, LLMResultV1
from llm_gateway.registry import ProviderRegistry
from llm_gateway.engine.core import GatewayEngine


def create_engine(
    registry: Optional[ProviderRegistry] = None,
    validator_hook: Optional[Callable[..., Any]] = None,
) -> GatewayEngine:
    """Create a fresh, isolated GatewayEngine instance."""
    return GatewayEngine(registry=registry, validator_hook=validator_hook)


def create_engine_for_registry(
    registry: ProviderRegistry,
    validator_hook: Optional[Callable[..., Any]] = None,
) -> GatewayEngine:
    """Convenience alias for ``create_engine(registry=registry)``."""
    return GatewayEngine(registry=registry, validator_hook=validator_hook)


# Global engine singleton
_GLOBAL_ENGINE = GatewayEngine()


def get_gateway_engine() -> GatewayEngine:
    """Return the global GatewayEngine singleton."""
    return _GLOBAL_ENGINE


def invoke_llm(
    request: LLMRequestV1,
    validator_hook: Optional[Callable[..., Any]] = None,
) -> LLMResultV1:
    """Public synchronous entry point for LLM Gateway."""
    return _GLOBAL_ENGINE.invoke_llm(request, validator_hook=validator_hook)


async def invoke_llm_async(
    request: LLMRequestV1,
    validator_hook: Optional[Callable[..., Any]] = None,
) -> LLMResultV1:
    """Public asynchronous entry point for LLM Gateway."""
    return await _GLOBAL_ENGINE.invoke_llm_async(request, validator_hook=validator_hook)


async def invoke_llm_in_executor(
    request: LLMRequestV1,
    validator_hook: Optional[Callable[..., Any]] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> LLMResultV1:
    """Run the global sync engine safely from within an async context."""
    return await _GLOBAL_ENGINE.invoke_llm_in_executor(request, validator_hook=validator_hook, executor=executor)


def invoke_llm_stream(request: LLMRequestV1):
    """Public synchronous streaming entry point for LLM Gateway."""
    return _GLOBAL_ENGINE.invoke_llm_stream(request)


async def invoke_llm_stream_async(request: LLMRequestV1):
    """Public asynchronous streaming entry point for LLM Gateway."""
    async for chunk in _GLOBAL_ENGINE.invoke_llm_stream_async(request):
        yield chunk
