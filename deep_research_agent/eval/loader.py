"""Load and filter evaluation dataset cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load dataset JSON and return parsed content."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "test_cases" not in data:
        raise ValueError("dataset must contain 'test_cases'")
    return data


def filter_cases(
    cases: list[dict[str, Any]],
    *,
    ids: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Filter test cases by id, label intersection, and optional limit."""
    filtered = cases
    if ids:
        id_set = {i.strip() for i in ids if i.strip()}
        filtered = [c for c in filtered if c.get("id") in id_set]
    if labels:
        label_set = {l.strip() for l in labels if l.strip()}
        filtered = [
            c
            for c in filtered
            if label_set.intersection(set(c.get("labels") or []))
        ]
    if limit is not None and limit > 0:
        filtered = filtered[:limit]
    return filtered
