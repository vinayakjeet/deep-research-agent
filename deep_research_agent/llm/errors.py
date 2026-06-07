"""LLM client exceptions."""

from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """Base error for LLM operations."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause


class LLMClientError(LLMError):
    """HTTP or API failure from the LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "gemini",
        status_code: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.provider = provider
        self.status_code = status_code


class PlanningError(LLMError):
    """Failed to parse or validate a planning response."""
