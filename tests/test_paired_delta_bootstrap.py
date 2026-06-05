from __future__ import annotations

import numpy as np

from mirage.eval.gate import auroc, paired_delta_bootstrap


def _precision_at(threshold: float):
    from mirage.eval.gate import metrics_at_threshold

    def stat(scores: np.ndarray, labels: np.ndarray) -> float:
        return metrics_at_threshold(scores, labels, threshold=threshold)["precision"]

    return stat


def test_paired_delta_positive_when_a_strictly_dominates() -> None:
    # 40 pos / 40 neg. scores_a separates perfectly; scores_b is pure noise.
    rng = np.random.default_rng(0)
    labels = np.array([1] * 40 + [0] * 40)
    scores_a = np.concatenate([rng.uniform(0.6, 1.0, 40), rng.uniform(0.0, 0.4, 40)])
    scores_b = rng.uniform(0.0, 1.0, 80)
    point, lo, hi = paired_delta_bootstrap(
        scores_a, scores_b, labels, statistic=auroc, n_boot=500, seed=1
    )
    assert point > 0.3  # A's AUROC ~1.0, B's ~0.5
    assert lo > 0.0  # CI excludes zero -> the delta "counts"
    assert lo <= point <= hi


def test_paired_delta_brackets_zero_for_identical_scores() -> None:
    rng = np.random.default_rng(2)
    labels = np.array([1] * 30 + [0] * 30)
    scores = rng.uniform(0.0, 1.0, 60)
    point, lo, hi = paired_delta_bootstrap(
        scores, scores.copy(), labels, statistic=auroc, n_boot=500, seed=3
    )
    assert point == 0.0
    assert lo <= 0.0 <= hi


def test_paired_delta_identical_arrays_collapse_ci_to_zero() -> None:
    # True pairing proof: identical arrays make every replicate's delta exactly 0,
    # so the CI collapses to [0, 0]. Independent (unpaired) resampling would NOT.
    rng = np.random.default_rng(4)
    labels = np.array([1] * 25 + [0] * 25)
    a = rng.uniform(0, 1, 50)
    point, lo, hi = paired_delta_bootstrap(a, a.copy(), labels, statistic=auroc, n_boot=200, seed=7)
    assert point == 0.0
    assert lo == 0.0 and hi == 0.0


def test_paired_delta_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    labels = np.array([1] * 25 + [0] * 25)
    a = rng.uniform(0, 1, 50)
    b = rng.uniform(0, 1, 50)
    r1 = paired_delta_bootstrap(a, b, labels, statistic=auroc, n_boot=200, seed=7)
    r2 = paired_delta_bootstrap(a, b, labels, statistic=auroc, n_boot=200, seed=7)
    assert r1 == r2


def test_paired_delta_precision_statistic_runs() -> None:
    rng = np.random.default_rng(5)
    labels = np.array([1] * 20 + [0] * 20)
    a = np.concatenate([rng.uniform(0.5, 1.0, 20), rng.uniform(0.0, 0.5, 20)])
    b = rng.uniform(0.0, 1.0, 40)
    point, lo, hi = paired_delta_bootstrap(
        a, b, labels, statistic=_precision_at(0.5), n_boot=300, seed=9
    )
    assert lo <= point <= hi
    assert point > 0.0  # arm a separates, arm b is noise -> positive precision delta
