"""Error taxonomy and exception classes for the LLM Gateway layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


from llm_gateway.constants import GatewayErrorCode


class ErrorClass(str, Enum):
    REQUEST = "request"
    POLICY = "policy"
    BUDGET = "budget"
    TRANSPORT = "transport"
    COMPATIBILITY = "compatibility"
    VALIDATION = "validation"
    PROVIDER = "provider"
    INTERNAL = "internal"


class LLMErrorV1(Exception):
    """Structured error emitted by the LLM Gateway."""

    def __init__(
        self,
        code: str,
        error_class: ErrorClass,
        message: str,
        *,
        retryable: bool = False,
        details: Optional[dict[str, Any]] = None,
        last_attempt_id: Optional[str] = None,
    ) -> None:
        super().__init__(f"[{error_class.value}:{code}] {message}")
        self.code = code
        self.error_class = error_class
        self.message = message
        self.retryable = retryable
        self.details = details or {}
        self.last_attempt_id = last_attempt_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "error_class": self.error_class.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
            "last_attempt_id": self.last_attempt_id,
        }


# Specialized Exceptions
class RequestValidationError(LLMErrorV1):
    def __init__(self, code: str, message: str, **kwargs: Any) -> None:
        super().__init__(code, ErrorClass.REQUEST, message, retryable=False, **kwargs)


class PolicyViolationError(LLMErrorV1):
    def __init__(self, code: str, message: str, **kwargs: Any) -> None:
        super().__init__(code, ErrorClass.POLICY, message, retryable=False, **kwargs)


class BudgetExhaustedError(LLMErrorV1):
    def __init__(self, code: str, message: str, **kwargs: Any) -> None:
        super().__init__(code, ErrorClass.BUDGET, message, retryable=False, **kwargs)


class TransportError(LLMErrorV1):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = True,
        retry_after_seconds: Optional[float] = None,
        http_status: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, ErrorClass.TRANSPORT, message, retryable=retryable, **kwargs)
        self.retry_after_seconds = retry_after_seconds
        self.http_status = http_status
        if retry_after_seconds is not None:
            self.details.setdefault("retry_after_seconds", retry_after_seconds)
        if http_status is not None:
            self.details.setdefault("http_status", http_status)


class CompatibilityError(LLMErrorV1):
    def __init__(self, code: str, message: str, *, retryable: bool = True, **kwargs: Any) -> None:
        super().__init__(code, ErrorClass.COMPATIBILITY, message, retryable=retryable, **kwargs)


class OutputValidationError(LLMErrorV1):
    def __init__(self, code: str, message: str, *, retryable: bool = True, **kwargs: Any) -> None:
        super().__init__(code, ErrorClass.VALIDATION, message, retryable=retryable, **kwargs)


class ProviderError(LLMErrorV1):
    def __init__(self, code: str, message: str, *, retryable: bool = False, **kwargs: Any) -> None:
        super().__init__(code, ErrorClass.PROVIDER, message, retryable=retryable, **kwargs)


class ConfigurationError(ProviderError):
    """Raised when a required environment variable or endpoint configuration is missing."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(GatewayErrorCode.CONFIGURATION_ERROR.value, message, retryable=False, **kwargs)


class InternalGatewayError(LLMErrorV1):
    def __init__(self, code: str, message: str, **kwargs: Any) -> None:
        super().__init__(code, ErrorClass.INTERNAL, message, retryable=False, **kwargs)

