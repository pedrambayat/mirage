"""Gate metrics for an FP-costly binder/non-binder gate.

The mirage gate fixes a high-precision operating point and reports the
sensitivity/specificity achieved there, plus the deployed PPV across a
prevalence sweep (Champloo's ~1:105 ratio badly overstates real-screen PPV).
All functions are numpy-only and take 1-D score/label arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def metrics_at_threshold(
    scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], *, threshold: float
) -> dict[str, float]:
    """Predict positive when score >= threshold; return gate metrics."""
    pred = scores >= threshold
    y = labels.astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    fpr = fp / (fp + tn) if (fp + tn) else math.nan
    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
    }


def _average_ranks(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """1-based ranks with ties resolved to their average (scipy rankdata 'average')."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    n = values.size
    ranks_sorted = np.arange(1, n + 1, dtype=float)
    start = 0
    for end in range(1, n + 1):
        if end == n or sorted_vals[end] != sorted_vals[start]:
            if end - start > 1:
                ranks_sorted[start:end] = (start + 1 + end) / 2.0
            start = end
    out = np.empty(n, dtype=float)
    out[order] = ranks_sorted
    return out


def auroc(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Threshold-free AUROC via the tie-aware Mann-Whitney U statistic.

    Returns NaN if either class is absent; non-finite scores are dropped."""
    finite = np.isfinite(scores)
    s = scores[finite]
    y = labels[finite].astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _average_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _candidate_thresholds(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    uniq = np.unique(scores[np.isfinite(scores)])
    # one threshold just above each distinct score, plus a floor below the min
    bumped = np.nextafter(uniq, np.inf)
    return (
        np.concatenate([[np.nextafter(uniq.min(), -np.inf)], bumped])
        if uniq.size
        else np.array([0.0])
    )


def choose_threshold_for_precision(
    scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], *, target_precision: float
) -> float:
    """Lowest threshold whose precision >= target (maximizing recall at target).

    If no threshold reaches the target, return the threshold with the highest
    precision observed (so the caller still gets a defined operating point).
    """
    best_thr = math.inf
    best_recall = -1.0
    fallback_thr = 0.0
    fallback_prec = -1.0
    for thr in _candidate_thresholds(scores):
        m = metrics_at_threshold(scores, labels, threshold=thr)
        prec = m["precision"]
        if math.isnan(prec):
            continue
        if prec > fallback_prec:
            fallback_prec, fallback_thr = prec, thr
        if prec >= target_precision and m["recall"] > best_recall:
            best_recall, best_thr = m["recall"], thr
    return float(best_thr if math.isfinite(best_thr) else fallback_thr)


def recall_at_precision(
    scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], *, target_precision: float
) -> float:
    thr = choose_threshold_for_precision(scores, labels, target_precision=target_precision)
    return metrics_at_threshold(scores, labels, threshold=thr)["recall"]


def ppv_at_prevalence(*, recall: float, specificity: float, prevalence: float) -> float:
    """Bayesian PPV translation: sensitivity & specificity are prevalence-free,
    so deployed PPV is recomputed at the target prevalence."""
    tpr = recall * prevalence
    fpr_mass = (1.0 - specificity) * (1.0 - prevalence)
    denom = tpr + fpr_mass
    return float(tpr / denom) if denom > 0 else math.nan


def ppv_prevalence_sweep(
    *, recall: float, specificity: float, prevalences: Sequence[float]
) -> list[dict[str, float]]:
    return [
        {
            "prevalence": float(p),
            "ppv": ppv_at_prevalence(recall=recall, specificity=specificity, prevalence=p),
        }
        for p in prevalences
    ]


def bootstrap_ci(
    statistic: Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float],
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Stratified bootstrap CI: resample positives and negatives separately so
    small cohorts keep both classes present in every replicate."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    vals: list[float] = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        bn = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([bp, bn])
        v = statistic(scores[idx], labels[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (math.nan, math.nan)
    arr = np.asarray(vals)
    return (float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2)))


# default prevalence grid: from balanced down to a realistic in-silico screen rate
DEFAULT_PREVALENCES: tuple[float, ...] = (0.5, 0.1, 0.01, 1e-3, 1e-4)


def summary_dict(
    scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], *, threshold: float
) -> dict[str, Any]:
    """Convenience bundle: metrics at threshold + PPV sweep at that operating point."""
    m = metrics_at_threshold(scores, labels, threshold=threshold)
    sweep = ppv_prevalence_sweep(
        recall=m["recall"], specificity=m["specificity"], prevalences=DEFAULT_PREVALENCES
    )
    return {"metrics": m, "ppv_sweep": sweep}
