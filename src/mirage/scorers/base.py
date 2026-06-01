"""Core abstractions shared by every loader and scorer.

`BenchmarkExample` is one (binder, target) pair to be scored. `Score` is one
scorer's output for one example. `AbstractScorer` is the contract every
concrete scorer fulfils. These three types are the only API that loaders
and scorers need to agree on; the rest of the pipeline reads them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkExample:
    """One (binder, target) pair to be scored.

    Permissive enough to carry VHH, scFv, Fab, minibinder, peptide, or any
    other binder format. Multi-chain binders (e.g. Fab heavy + light) use
    a tuple with more than one entry. Anything a specific loader wants to
    preserve but the abstraction does not name goes in `metadata`.
    """

    id: str
    label: str
    binder_chains: tuple[str, ...]
    binder_format: str
    target_chains: tuple[str, ...]
    target_name: str
    source: str
    target_pdb_id: str | None = None
    complex_pdb_path: Path | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    """One scorer's output for one example."""

    example_id: str
    scorer_name: str
    value: float
    extras: dict[str, float | str] = field(default_factory=dict)


class AbstractScorer(ABC):
    """Interface every scorer implements."""

    name: str = ""

    @abstractmethod
    def score(self, example: BenchmarkExample) -> Score: ...

    def score_batch(self, examples: Iterable[BenchmarkExample]) -> Iterator[Score]:
        for example in examples:
            yield self.score(example)

    def nan_score(self, example: BenchmarkExample, **extras: str) -> Score:
        """A NaN-valued score carrying diagnostic ``extras`` (e.g. ``missing=…``).

        Lets a scorer emit a CSV row for an example it cannot score (missing
        input, count mismatch, parse failure) without aborting the batch.
        """
        return Score(
            example_id=example.id,
            scorer_name=self.name,
            value=float("nan"),
            extras=dict(extras),
        )
