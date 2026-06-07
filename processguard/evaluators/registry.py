from __future__ import annotations

from typing import Any, Optional

from .base import Evaluator


_REGISTRY: dict[str, type[Evaluator]] = {}


class AssertionTypeNotRegistered(ValueError):
    """Raised when a case JSONL references an assertion type that hasn't been
    registered. The error message lists every registered type so a fat-finger
    is obvious from the failure."""


def register(cls: type[Evaluator]) -> type[Evaluator]:
    """Class decorator. Registers an Evaluator subclass under its
    `assertion_type` string."""
    if not cls.assertion_type:
        raise ValueError(f"{cls.__name__}.assertion_type must be set")
    if cls.assertion_type in _REGISTRY:
        existing = _REGISTRY[cls.assertion_type].__name__
        raise ValueError(
            f"assertion_type '{cls.assertion_type}' already registered "
            f"to {existing}; cannot also register {cls.__name__}"
        )
    _REGISTRY[cls.assertion_type] = cls
    return cls


def get(assertion_type: str, args: Optional[dict[str, Any]] = None) -> Evaluator:
    """Look up an Evaluator class by its registered name and instantiate it
    from the given args."""
    cls = _REGISTRY.get(assertion_type)
    if cls is None:
        raise AssertionTypeNotRegistered(
            f'"{assertion_type}" — registered: {sorted(_REGISTRY)}'
        )
    return cls.from_args(args)


def registered_names() -> list[str]:
    """Sorted list of registered assertion types — used for introspection
    and error messages."""
    return sorted(_REGISTRY)
