"""Decorator-based loader registry, parallel to the scorer registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from mirage.scorers.base import BenchmarkExample


class AbstractLoader(ABC):
    """A loader yields a stream of BenchmarkExamples from a data source."""

    name: str = ""

    @abstractmethod
    def load(self) -> Iterator[BenchmarkExample]: ...


L = TypeVar("L", bound=AbstractLoader)

_REGISTRY: dict[str, type[AbstractLoader]] = {}


def register_loader(name: str) -> Callable[[type[L]], type[L]]:
    """Register a loader class under `name`."""

    def wrap(cls: type[L]) -> type[L]:
        if name in _REGISTRY:
            raise ValueError(f"Loader '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_loader(name: str, **kwargs: Any) -> AbstractLoader:
    if name not in _REGISTRY:
        raise KeyError(f"No loader named '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_loaders() -> list[str]:
    return sorted(_REGISTRY)
