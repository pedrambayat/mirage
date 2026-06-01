"""Benchmark framework. Importing this module registers built-in loaders."""

# Side-effect imports — each module registers its loader via @register_loader.
import mirage.benchmark.avida
import mirage.benchmark.epcam_killing
import mirage.benchmark.loaders
import mirage.benchmark.sabdab  # noqa: F401
from mirage.benchmark._registry import (
    AbstractLoader,
    get_loader,
    list_loaders,
    register_loader,
)

__all__ = [
    "AbstractLoader",
    "get_loader",
    "list_loaders",
    "register_loader",
]
