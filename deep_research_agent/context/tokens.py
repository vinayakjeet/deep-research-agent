"""Heuristic token counting for context budget enforcement."""

from __future__ import annotations

import json
from typing import Any


def estimate_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    """Approximate token count from character length."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_object_tokens(value: Any, *, chars_per_token: float = 4.0) -> int:
    """Estimate tokens for a JSON-serializable object."""
    return estimate_tokens(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        chars_per_token=chars_per_token,
    )
