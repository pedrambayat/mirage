from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mirage.scorers import AbstractScorer, BenchmarkExample, Score


def _make_example(**overrides: object) -> BenchmarkExample:
    defaults: dict[str, object] = {
        "id": "ex-1",
        "label": "POS",
        "binder_chains": ("QVQ",),
        "binder_format": "vhh",
        "target_chains": ("MKL",),
        "target_name": "Test",
        "source": "unit-test",
    }
    defaults.update(overrides)
    return BenchmarkExample(**defaults)  # type: ignore[arg-type]


def test_benchmark_example_is_frozen() -> None:
    example = _make_example()
    with pytest.raises(FrozenInstanceError):
        example.id = "other"  # type: ignore[misc]


def test_benchmark_example_optional_fields_default_to_none() -> None:
    example = _make_example()
    assert example.target_pdb_id is None
    assert example.complex_pdb_path is None
    assert example.split is None
    assert example.metadata == {}


def test_score_is_frozen() -> None:
    score = Score(example_id="ex-1", scorer_name="length", value=120.0)
    with pytest.raises(FrozenInstanceError):
        score.value = 121.0  # type: ignore[misc]


def test_abstract_scorer_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        AbstractScorer()  # type: ignore[abstract]


def test_score_batch_iterates_each_example() -> None:
    class _Constant(AbstractScorer):
        name = "constant"

        def score(self, example: BenchmarkExample) -> Score:
            return Score(example_id=example.id, scorer_name=self.name, value=1.0)

    examples = [_make_example(id=f"ex-{i}") for i in range(3)]
    scores = list(_Constant().score_batch(examples))
    assert [s.example_id for s in scores] == ["ex-0", "ex-1", "ex-2"]
    assert all(s.value == 1.0 for s in scores)
