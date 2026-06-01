from __future__ import annotations

import pytest

from mirage.scorers import BenchmarkExample, get_scorer


def test_length_scorer_returns_first_chain_length() -> None:
    example = BenchmarkExample(
        id="ex-1",
        label="POS",
        binder_chains=("QVQLQESGGGLV",),
        binder_format="vhh",
        target_chains=("MKL",),
        target_name="Test",
        source="unit-test",
    )
    scorer = get_scorer("length")
    score = scorer.score(example)
    assert score.example_id == "ex-1"
    assert score.scorer_name == "length"
    assert score.value == 12.0
    assert score.extras["binder_chain_count"] == "1"


def test_length_scorer_uses_first_chain_only() -> None:
    example = BenchmarkExample(
        id="ex-fab",
        label="POS",
        binder_chains=("AAAA", "BBBBBBBB"),
        binder_format="fab",
        target_chains=("MKL",),
        target_name="Test",
        source="unit-test",
    )
    score = get_scorer("length").score(example)
    assert score.value == 4.0
    assert score.extras["binder_chain_count"] == "2"


def test_length_scorer_rejects_empty_binder() -> None:
    example = BenchmarkExample(
        id="ex-bad",
        label="POS",
        binder_chains=(),
        binder_format="vhh",
        target_chains=("MKL",),
        target_name="Test",
        source="unit-test",
    )
    with pytest.raises(ValueError):
        get_scorer("length").score(example)
