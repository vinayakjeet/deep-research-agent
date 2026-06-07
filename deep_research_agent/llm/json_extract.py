"""Extract and repair JSON from LLM text responses."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)


def repair_json(text: str) -> str:
    """Apply light repairs for common model JSON mistakes."""
    cleaned = text.strip()
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    return cleaned


def _try_parse(candidate: str) -> dict[str, Any] | None:
    for raw in (candidate, repair_json(candidate)):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _extract_braced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Parse the first JSON object from model output.

    Tries: full text, fenced code blocks, then brace-delimited scan.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty text cannot be parsed as JSON")

    direct = _try_parse(stripped)
    if direct is not None:
        return direct

    for match in _JSON_FENCE_RE.finditer(stripped):
        block = match.group(1).strip()
        parsed = _try_parse(block)
        if parsed is not None:
            return parsed

    braced = _extract_braced_object(stripped)
    if braced:
        parsed = _try_parse(braced)
        if parsed is not None:
            return parsed

    raise ValueError("No valid JSON object found in text")
