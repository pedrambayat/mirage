"""Numpy-only logistic regression, grouped folds, and ranking metrics.

Single source of truth: the Champloo analysis script and all mirage Phase A
code import these rather than re-implementing them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def auroc(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.size * neg.size))


def average_precision(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    n_pos = int((labels == 1).sum())
    if scores.size == 0 or n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(float)
    precision = np.cumsum(labels_sorted) / np.arange(1, labels_sorted.size + 1)
    return float((precision * labels_sorted).sum() / n_pos)


def standardizer(
    x_train: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return (mean, std) with zero-variance columns forced to std=1."""
    mean: np.ndarray[Any, Any] = x_train.mean(axis=0)
    std: np.ndarray[Any, Any] = x_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(
    x: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
    std: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Apply a fitted ``(mean, std)`` standardizer to ``x``."""
    result: np.ndarray[Any, Any] = (x - mean) / std
    return result


def fit_logistic_regression(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    l2: float,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> tuple[float, np.ndarray[Any, Any]]:
    beta: np.ndarray[Any, Any] = np.zeros(x.shape[1] + 1, dtype=float)
    design: np.ndarray[Any, Any] = np.column_stack([np.ones(x.shape[0], dtype=float), x])
    penalty: np.ndarray[Any, Any] = np.diag(
        np.concatenate([[0.0], np.full(x.shape[1], l2, dtype=float)])
    )
    prev_loss = math.inf
    for _ in range(max_iter):
        logits = np.clip(design @ beta, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(pred * (1.0 - pred), 1e-9)
        gradient = design.T @ (pred - y) + penalty @ beta
        hessian = (design.T * weights) @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        loss = float(
            -np.sum(y * np.log(pred + 1e-12) + (1.0 - y) * np.log(1.0 - pred + 1e-12))
            + 0.5 * float(beta @ penalty @ beta)
        )
        if abs(prev_loss - loss) < tolerance or float(np.linalg.norm(step)) < tolerance:
            break
        prev_loss = loss
    return float(beta[0]), beta[1:]


def assign_folds(groups: np.ndarray[Any, Any], *, n_splits: int, seed: int) -> np.ndarray[Any, Any]:
    """Assign each row to a fold by its group label.

    All rows sharing a group go to the same fold (no group spans train/test),
    so passing antigen or VHH PDB ids gives leakage-controlled grouped K-fold,
    while passing per-row ids reduces to ordinary K-fold. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    shuffled = rng.permutation(unique.size)
    fold_of_group = {g: int(shuffled[i] % n_splits) for i, g in enumerate(unique)}
    return np.asarray([fold_of_group[g] for g in groups], dtype=int)


def oof_logistic_scores(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    folds: np.ndarray[Any, Any],
    *,
    l2: float,
) -> np.ndarray[Any, Any]:
    out: np.ndarray[Any, Any] = np.full(y.shape, math.nan, dtype=float)
    for fold in np.unique(folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        y_train = y[train_mask]
        if y_train.size < 2 or np.unique(y_train).size < 2:
            continue
        mean, std = standardizer(x[train_mask])
        x_train = apply_standardizer(x[train_mask], mean, std)
        x_test = apply_standardizer(x[test_mask], mean, std)
        intercept, coef = fit_logistic_regression(x_train, y_train, l2=l2)
        out[test_mask] = intercept + x_test @ coef
    return out
