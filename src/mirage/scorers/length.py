"""Trivial scorer returning the length of the binder's first chain.

Used to smoke-test the loader -> scorer -> output pipeline without any
model downloads or GPU dependencies.
"""

from __future__ import annotations

from mirage.scorers._registry import register
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score


@register("length")
class LengthScorer(AbstractScorer):
    def score(self, example: BenchmarkExample) -> Score:
        if not example.binder_chains:
            raise ValueError(f"Example {example.id} has no binder chains")
        binder_len = len(example.binder_chains[0])
        return Score(
            example_id=example.id,
            scorer_name=self.name,
            value=float(binder_len),
            extras={"binder_chain_count": str(len(example.binder_chains))},
        )
