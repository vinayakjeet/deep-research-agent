"""Split sanitized text into paragraph-sized blocks."""

from __future__ import annotations

import re

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str, *, min_block_chars: int = 40) -> list[str]:
    """Split text on blank lines into non-empty paragraph blocks."""
    if not text.strip():
        return []

    blocks: list[str] = []
    for part in _PARAGRAPH_SPLIT_RE.split(text.strip()):
        block = part.strip()
        if len(block) >= min_block_chars:
            blocks.append(block)

    if blocks:
        return blocks

    return split_by_max_chars(text, max_chars=800, min_block_chars=min_block_chars)


def split_by_max_chars(
    text: str,
    *,
    max_chars: int = 800,
    overlap: int = 100,
    min_block_chars: int = 40,
) -> list[str]:
    """Slice long continuous text into fixed-size windows."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned] if len(cleaned) >= min_block_chars else []

    blocks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        block = cleaned[start:end].strip()
        if len(block) >= min_block_chars:
            blocks.append(block)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)

    return blocks
