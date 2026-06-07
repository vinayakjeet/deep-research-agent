"""Ingestion-layer exceptions."""

from __future__ import annotations

from typing import Any, Optional


class IngestionError(Exception):
    """Base error for ingestion pipeline failures."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause


class SearchProviderError(IngestionError):
    """Search API request failed after retries or returned an invalid payload."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.provider = provider
        self.status_code = status_code


class FetchError(IngestionError):
    """Page fetch failed."""

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.url = url
        self.status_code = status_code


class ParseError(IngestionError):
    """HTML extraction failed."""

    def __init__(
        self,
        message: str,
        *,
        url: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.url = url
