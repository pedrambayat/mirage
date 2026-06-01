"""Scoring framework. Importing this module registers built-in scorers."""

# Side-effect imports — register built-in scorers via their @register decorators.
import mirage.scorers.af2m_confidence as af2m_confidence  # noqa: F401
import mirage.scorers.length as length  # noqa: F401
import mirage.scorers.rmsd_to_crystal as rmsd_to_crystal  # noqa: F401
import mirage.scorers.structural_interface as structural_interface  # noqa: F401
from mirage.scorers._registry import get_scorer, list_scorers, register
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score

__all__ = [
    "AbstractScorer",
    "BenchmarkExample",
    "Score",
    "get_scorer",
    "list_scorers",
    "register",
]
