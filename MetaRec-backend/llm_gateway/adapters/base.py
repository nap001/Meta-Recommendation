"""Base adapter contract for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from llm_gateway.contracts import LLMRequestV1, RouteV1


class RawAdapterResponse(BaseModel):
    text: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    is_usage_reported: bool = False
    http_status: int = 200
    provider_request_id: Optional[str] = None
    model: str = ""
    raw_response: Optional[Any] = None
    model_config = ConfigDict(extra="forbid")


class BaseLLMAdapter(ABC):
    """Abstract interface that all provider adapters must implement."""

    @abstractmethod
    def execute(self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]) -> RawAdapterResponse:
        """Execute synchronous request to provider."""
        raise NotImplementedError

    @abstractmethod
    async def execute_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ) -> RawAdapterResponse:
        """Execute asynchronous request to provider."""
        raise NotImplementedError

    def execute_stream(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute synchronous streaming request yielding string delta tokens."""
        raise NotImplementedError(f"Streaming not implemented for adapter {type(self).__name__}")

    async def execute_stream_async(
        self, request: LLMRequestV1, route: RouteV1, normalized_kwargs: Dict[str, Any]
    ):
        """Execute asynchronous streaming request yielding string delta tokens."""
        raise NotImplementedError(f"Async streaming not implemented for adapter {type(self).__name__}")
