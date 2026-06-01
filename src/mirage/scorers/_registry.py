"""Decorator-based scorer registry so new scorers self-register on import."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from mirage.scorers.base import AbstractScorer

S = TypeVar("S", bound=AbstractScorer)

_REGISTRY: dict[str, type[AbstractScorer]] = {}


def register(name: str) -> Callable[[type[S]], type[S]]:
    """Register a scorer class under `name`."""

    def wrap(cls: type[S]) -> type[S]:
        if name in _REGISTRY:
            raise ValueError(f"Scorer '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_scorer(name: str, **kwargs: Any) -> AbstractScorer:
    """Instantiate the scorer registered under `name` with `kwargs`."""
    if name not in _REGISTRY:
        raise KeyError(f"No scorer named '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_scorers() -> list[str]:
    return sorted(_REGISTRY)
