"""Exponential backoff retry helper for HTTP operations."""

from __future__ import annotations

import time
from typing import Callable, Optional, Set, TypeVar

T = TypeVar("T")

DEFAULT_RETRY_STATUSES: Set[int] = {429, 500, 502, 503, 504}


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
    retry_on_status: Optional[Set[int]] = None,
    get_status_code: Optional[Callable[[Exception], Optional[int]]] = None,
) -> T:
    """
    Call fn(), retrying on retryable HTTP status codes with exponential backoff.

    Raises the last exception if all attempts fail.
    """
    statuses = retry_on_status if retry_on_status is not None else DEFAULT_RETRY_STATUSES
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            status: Optional[int] = None
            if get_status_code is not None:
                status = get_status_code(exc)
            elif hasattr(exc, "response") and getattr(exc.response, "status_code", None):
                status = int(exc.response.status_code)

            if status not in statuses or attempt >= max_retries:
                raise

            delay = min(base_delay * (2**attempt), max_delay)
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc
