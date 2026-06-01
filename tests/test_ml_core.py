from __future__ import annotations

import numpy as np

from mirage.ml import core


def test_auroc_perfect_and_inverted() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert core.auroc(scores, labels) == 1.0
    assert core.auroc(-scores, labels) == 0.0


def test_average_precision_matches_known_value() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    labels = np.asarray([1, 0, 1, 0])
    ap = core.average_precision(scores, labels)
    assert abs(ap - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_assign_folds_no_group_leakage_and_deterministic() -> None:
    groups = np.asarray([f"g{i // 4}" for i in range(40)])
    folds = core.assign_folds(groups, n_splits=5, seed=1)
    for g in np.unique(groups):
        assert np.unique(folds[groups == g]).size == 1
    again = core.assign_folds(groups, n_splits=5, seed=1)
    assert np.array_equal(folds, again)


def test_oof_logistic_recovers_separable_signal() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, 0.5, 30), rng.normal(3, 0.5, 30)])[:, None]
    y = np.concatenate([np.zeros(30), np.ones(30)])
    folds = np.tile(np.arange(5), 12)
    scores = core.oof_logistic_scores(x, y, folds, l2=1.0)
    assert np.isfinite(scores).all()
    assert core.auroc(scores, y.astype(int)) > 0.95


def test_fit_predict_roundtrip() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-2, 0.5, 40), rng.normal(2, 0.5, 40)])[:, None]
    y = np.concatenate([np.zeros(40), np.ones(40)])
    mean, std = core.standardizer(x)
    intercept, coef = core.fit_logistic_regression(core.apply_standardizer(x, mean, std), y, l2=1.0)
    logits = intercept + core.apply_standardizer(x, mean, std) @ coef
    assert core.auroc(logits, y.astype(int)) > 0.95
