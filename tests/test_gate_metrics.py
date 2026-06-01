from __future__ import annotations

import numpy as np

from mirage.eval import gate


def test_confusion_at_threshold() -> None:
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    m = gate.metrics_at_threshold(scores, labels, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0
    assert m["fpr"] == 0.0


def test_choose_threshold_for_precision_hits_target() -> None:
    # one negative scores above one positive; precision 1.0 only above that neg
    scores = np.asarray([0.2, 0.5, 0.55, 0.7, 0.9])
    labels = np.asarray([0, 1, 0, 1, 1])
    thr = gate.choose_threshold_for_precision(scores, labels, target_precision=1.0)
    m = gate.metrics_at_threshold(scores, labels, threshold=thr)
    assert m["precision"] >= 1.0 - 1e-9


def test_recall_at_precision_perfect_separation() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert gate.recall_at_precision(scores, labels, target_precision=0.9) == 1.0


def test_ppv_at_prevalence_bayes() -> None:
    # recall=0.9, specificity=0.9, prevalence=0.5 -> PPV=0.9
    assert abs(gate.ppv_at_prevalence(recall=0.9, specificity=0.9, prevalence=0.5) - 0.9) < 1e-9
    # at prevalence 1e-4 the same gate is nearly worthless
    low = gate.ppv_at_prevalence(recall=0.9, specificity=0.9, prevalence=1e-4)
    assert low < 0.01


def test_ppv_sweep_is_monotone_decreasing_as_rarer() -> None:
    sweep = gate.ppv_prevalence_sweep(
        recall=0.9, specificity=0.99, prevalences=(0.5, 0.1, 1e-3, 1e-4)
    )
    values = [row["ppv"] for row in sweep]
    assert values == sorted(values, reverse=True)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    lo, hi = gate.bootstrap_ci(
        lambda s, y: gate.metrics_at_threshold(s, y, threshold=1.5)["recall"],
        scores,
        labels,
        n_boot=200,
        seed=1,
    )
    point = gate.metrics_at_threshold(scores, labels, threshold=1.5)["recall"]
    assert lo <= point <= hi
