"""Streaming telemetry validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PHASE_ALIASES = {
    "acquiring": "fetching",
    "answering": "generating",
}


def normalize_phase(phase: str) -> str:
    """Map agent phase names to dataset expected names."""
    p = phase.strip().lower()
    return PHASE_ALIASES.get(p, p)


def extract_streamed_phases(events: list[dict[str, Any]]) -> list[str]:
    """Collect normalized phase names from phase_start events."""
    phases: list[str] = []
    for event in events:
        if event.get("event") != "phase_start":
            continue
        phase = event.get("phase", "")
        if not phase:
            continue
        normalized = normalize_phase(str(phase))
        if normalized not in phases:
            phases.append(normalized)
    return phases


@dataclass(slots=True)
class StreamingScore:
    passed: bool
    required: list[str]
    observed: list[str]
    missing: list[str]


def score_streaming(
    events: list[dict[str, Any]],
    required_events: list[str],
) -> StreamingScore:
    """Check that all required streaming phases were emitted."""
    observed = extract_streamed_phases(events)
    required = [normalize_phase(r) for r in required_events]
    missing = [r for r in required if r not in observed]
    return StreamingScore(
        passed=len(missing) == 0,
        required=required,
        observed=observed,
        missing=missing,
    )
