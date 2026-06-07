"""Models for async page fetch results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FetchStatus(str, Enum):
    """Outcome of a single page fetch attempt."""

    OK = "ok"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"


@dataclass(slots=True)
class FetchResult:
    """Result of fetching one URL."""

    url: str
    status: FetchStatus
    status_code: Optional[int] = None
    html: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.status == FetchStatus.OK and bool(self.html)
