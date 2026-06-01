from __future__ import annotations

import numpy as np
import pytest

from mirage.eval.orthogonal import evaluate_frozen_gate, features_for_examples
from mirage.model.ms import train_ms
from mirage.scorers.base import BenchmarkExample


def _example(binder: str, target: str, label: str) -> BenchmarkExample:
    return BenchmarkExample(
        id=f"{binder}-{label}",
        label=label,
        binder_chains=(binder,),
        binder_format="vhh",
        target_chains=(target,),
        target_name="X",
        source="test",
    )


def test_features_for_examples_shape() -> None:
    examples = [_example("KKKK", "DDDD", "BIND"), _example("DDDD", "KKKK", "NONBIND")]
    x, y, names = features_for_examples(examples, positive_label="BIND")
    assert x.shape[0] == 2
    assert list(y) == [1, 0]
    assert "binder_length" in names


def test_features_for_examples_rejects_empty_stream() -> None:
    with pytest.raises(ValueError, match="no examples"):
        features_for_examples([], positive_label="BIND")


def test_evaluate_frozen_gate_returns_metrics() -> None:
    # train a trivial model on synthetic separable features, then evaluate frozen
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.3, 40), rng.normal(2, 0.3, 40)])[:, None]
    y = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)
    model, _ = train_ms(x, y, feature_names=["binder_length"], l2=1.0, target_precision=0.9, seed=0)
    # reuse the same x as "orthogonal" features for a smoke check
    result = evaluate_frozen_gate(model, x, y, n_boot=50, seed=1)
    assert "recall" in result["metrics"]
    assert "recall_ci" in result
    assert len(result["recall_ci"]) == 2
