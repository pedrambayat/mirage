from __future__ import annotations

import pytest

from mirage.scorers import (
    AbstractScorer,
    BenchmarkExample,
    Score,
    get_scorer,
    list_scorers,
    register,
)


def test_length_registered_by_default() -> None:
    assert "length" in list_scorers()


def test_get_scorer_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_scorer("nonexistent")


def test_get_scorer_returns_instance() -> None:
    scorer = get_scorer("length")
    assert isinstance(scorer, AbstractScorer)
    assert scorer.name == "length"


def test_duplicate_register_raises() -> None:
    @register("dup-test-scorer")
    class _DupA(AbstractScorer):
        def score(self, example: BenchmarkExample) -> Score:
            return Score(example_id=example.id, scorer_name=self.name, value=0.0)

    with pytest.raises(ValueError):

        @register("dup-test-scorer")
        class _DupB(AbstractScorer):
            def score(self, example: BenchmarkExample) -> Score:
                return Score(example_id=example.id, scorer_name=self.name, value=0.0)
