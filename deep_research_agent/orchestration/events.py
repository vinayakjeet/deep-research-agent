"""Operational status events for streaming research progress."""

from __future__ import annotations

from typing import Any, Optional

from deep_research_agent.state.enums import AgentPhase

_PHASE_LABELS = {
    AgentPhase.PLANNING: "Planning",
    AgentPhase.SEARCHING: "Searching",
    AgentPhase.ACQUIRING: "Fetching sources",
    AgentPhase.ANSWERING: "Generating answer",
    AgentPhase.COMPLETE: "Complete",
    AgentPhase.ERROR: "Error",
}


def phase_start(phase: AgentPhase, *, message: Optional[str] = None) -> dict[str, Any]:
    """Emit when a workflow phase begins."""
    label = _PHASE_LABELS.get(phase, phase.value)
    return {
        "event": "phase_start",
        "phase": phase.value,
        "status": label,
        "message": message or f"{label}...",
        "details": {},
    }


def phase_complete(
    phase: AgentPhase,
    *,
    details: Optional[dict[str, Any]] = None,
    message: Optional[str] = None,
) -> dict[str, Any]:
    """Emit when a workflow phase finishes successfully."""
    label = _PHASE_LABELS.get(phase, phase.value)
    return {
        "event": "phase_complete",
        "phase": phase.value,
        "status": label,
        "message": message or f"{label} finished.",
        "details": details or {},
    }


def no_evidence_event(*, reason: str, user_query: str) -> dict[str, Any]:
    """Emit when the turn ends without sufficient evidence."""
    return {
        "event": "no_evidence",
        "phase": "no_evidence",
        "status": "NoEvidence",
        "message": "No verifiable evidence could be located.",
        "details": {"reason": reason, "user_query": user_query},
    }


def error_event(message: str, *, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Emit on unrecoverable errors."""
    return {
        "event": "error",
        "phase": AgentPhase.ERROR.value,
        "status": "Error",
        "message": message,
        "details": details or {},
    }


def terminal_event(
    *,
    final_answer: str,
    session_id: str,
    turn_id: int,
    phases_executed: list[str],
    citation_report: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Final event with the completed turn payload."""
    details: dict[str, Any] = {
        "session_id": session_id,
        "turn_id": turn_id,
        "phases_executed": phases_executed,
    }
    if citation_report is not None:
        details["citation_report"] = citation_report
    return {
        "event": "complete",
        "phase": AgentPhase.COMPLETE.value,
        "status": "Complete",
        "message": "Research turn finished.",
        "details": details,
        "final_answer": final_answer,
    }
