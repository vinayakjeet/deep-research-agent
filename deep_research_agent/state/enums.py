"""Workflow and message role enumerations."""

from __future__ import annotations

from enum import Enum


class AgentPhase(str, Enum):
    """Deterministic phases for the native orchestration loop."""

    START = "start"
    PLANNING = "planning"
    SEARCHING = "searching"
    ACQUIRING = "acquiring"
    ANSWERING = "answering"
    COMPLETE = "complete"
    ERROR = "error"

    def is_terminal(self) -> bool:
        return self in {AgentPhase.COMPLETE, AgentPhase.ERROR}


class MessageRole(str, Enum):
    """Roles stored in conversation history."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    @classmethod
    def from_str(cls, value: str) -> MessageRole:
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Invalid message role: {value}") from exc
