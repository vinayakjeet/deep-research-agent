"""JSON serialization helpers for state dataclasses."""

from __future__ import annotations

import json
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Type, TypeVar, get_args, get_origin, get_type_hints

from deep_research_agent.state.schema import AgentState

T = TypeVar("T")


def _encode_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_dict(value)
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_value(item) for key, item in value.items()}
    return value


def to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)!r}")
    return {field.name: _encode_value(getattr(obj, field.name)) for field in fields(obj)}


def _unwrap_optional(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is types.UnionType:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    if origin is not None:
        # typing.Optional, typing.Union
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return field_type


def _decode_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(field_type)
    if origin is list:
        inner_type = get_args(field_type)[0]
        return [_decode_value(inner_type, item) for item in value]

    if origin is dict:
        return value

    resolved = _unwrap_optional(field_type)
    if resolved is not field_type:
        return _decode_value(resolved, value)

    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type(value)

    if isinstance(field_type, type) and is_dataclass(field_type):
        if isinstance(value, dict):
            return from_dict(field_type, value)
        return value

    return value


def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls!r}")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        field_type = hints.get(field.name, field.type)
        kwargs[field.name] = _decode_value(field_type, data[field.name])
    return cls(**kwargs)


def to_json(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(to_dict(obj), ensure_ascii=False, indent=indent)


def from_json(cls: Type[T], payload: str | dict[str, Any]) -> T:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise TypeError("JSON payload must decode to an object")
    return from_dict(cls, data)


def agent_state_to_json(state: AgentState, *, indent: int | None = None) -> str:
    return to_json(state, indent=indent)


def agent_state_from_json(payload: str | dict[str, Any]) -> AgentState:
    return from_json(AgentState, payload)
