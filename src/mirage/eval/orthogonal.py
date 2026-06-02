"""Apply a Champloo-frozen M-S gate to an orthogonal dataset, unchanged.

The headline mirage test: a threshold chosen on Champloo must hold its precision
on data it never saw. This module builds Tier-S features for any stream of
BenchmarkExamples and evaluates the frozen gate with bootstrap CIs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from mirage.eval.gate import bootstrap_ci, metrics_at_threshold
from mirage.features.normalize import normalize_antigen, normalize_binder
from mirage.features.sequence import FEATURE_NAMES, sequence_features
from mirage.model.ms import MsModel
from mirage.scorers.base import BenchmarkExample


def features_for_examples(
    examples: Iterable[BenchmarkExample], *, positive_label: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], tuple[str, ...]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for ex in examples:
        feats = sequence_features(
            normalize_binder(ex.binder_chains[0]),
            normalize_antigen(ex.target_chains[0]),
        )
        rows.append([feats[name] for name in FEATURE_NAMES])
        labels.append(1 if ex.label == positive_label else 0)
    if not rows:
        raise ValueError("features_for_examples received no examples to featurize")
    x = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    return x, y, FEATURE_NAMES


def evaluate_frozen_gate(
    model: MsModel,
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score features with the frozen model + threshold; return metrics + CIs."""
    scores = model.predict_logit(x)
    metrics = metrics_at_threshold(scores, y, threshold=model.threshold)
    thr = model.threshold

    def _recall(s: np.ndarray[Any, Any], yy: np.ndarray[Any, Any]) -> float:
        return metrics_at_threshold(s, yy, threshold=thr)["recall"]

    def _specificity(s: np.ndarray[Any, Any], yy: np.ndarray[Any, Any]) -> float:
        return metrics_at_threshold(s, yy, threshold=thr)["specificity"]

    def _precision(s: np.ndarray[Any, Any], yy: np.ndarray[Any, Any]) -> float:
        return metrics_at_threshold(s, yy, threshold=thr)["precision"]

    has_both = int((y == 1).sum()) > 0 and int((y == 0).sum()) > 0
    return {
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "metrics": metrics,
        "recall_ci": bootstrap_ci(_recall, scores, y, n_boot=n_boot, seed=seed)
        if (y == 1).sum()
        else (float("nan"), float("nan")),
        "specificity_ci": bootstrap_ci(_specificity, scores, y, n_boot=n_boot, seed=seed)
        if (y == 0).sum()
        else (float("nan"), float("nan")),
        "precision_ci": bootstrap_ci(_precision, scores, y, n_boot=n_boot, seed=seed)
        if has_both
        else (float("nan"), float("nan")),
    }


def features_for_examples_embedding(
    examples: Iterable[BenchmarkExample],
    cache: dict[str, np.ndarray[Any, Any]],
    *,
    positive_label: str,
    layout: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build embedding paired features for a stream of examples, normalizing each
    sequence to its mature domain first (so lookups hit the cache, which is keyed
    on normalized sequences). Raises KeyError if a sequence was not embedded."""
    from mirage.features.embeddings import paired_matrix

    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    for ex in examples:
        binder = normalize_binder(ex.binder_chains[0])
        antigen = ":".join(normalize_antigen(c) for c in ex.target_chains)
        pairs.append((binder, antigen))
        labels.append(1 if ex.label == positive_label else 0)
    if not pairs:
        raise ValueError("features_for_examples_embedding received no examples")
    x = paired_matrix(pairs, cache, layout=layout)
    y = np.array(labels, dtype=int)
    return x, y
