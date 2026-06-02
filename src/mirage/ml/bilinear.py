"""numpy low-rank bilinear logistic model: score = (P_a e_a) . (P_g e_g) + b.

The minimal model that can separate cognate from shuffled pairs, where an
additive model is provably ~chance. Trained by gradient descent on the logistic
loss with L2 on the projections. Pure numpy — mirage stays torch-free."""

from __future__ import annotations

from typing import Any

import numpy as np

from mirage.ml.core import apply_standardizer, standardizer


def _sigmoid(z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    result: np.ndarray[Any, Any] = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    return result


def predict_bilinear(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    proj_a: np.ndarray[Any, Any],
    proj_g: np.ndarray[Any, Any],
    intercept: float,
) -> np.ndarray[Any, Any]:
    ua = xa @ proj_a.T
    ug = xg @ proj_g.T
    out: np.ndarray[Any, Any] = np.sum(ua * ug, axis=1) + intercept
    return out


def fit_bilinear(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    rank: int,
    l2: float,
    lr: float,
    n_iter: int,
    seed: int,
    max_grad_norm: float = 1.0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    """Fit projections (P_a, P_g) and bias by full-batch gradient descent.

    Inputs are assumed already standardized by the caller. Returns
    (proj_a [rank x d_a], proj_g [rank x d_g], intercept).

    The two-tower bilinear loss is non-convex and each tower's gradient scales
    with the *other* tower's magnitude, so plain GD can grow the projections
    geometrically and overflow for unlucky inits at high feature dimension.
    Clipping each projection's gradient norm to ``max_grad_norm`` bounds the
    per-step growth; combined with the L2 restoring term ``l2 * proj`` it keeps
    ``||proj||`` bounded (≈ ``max_grad_norm / l2``), so training stays finite
    regardless of seed. Set ``max_grad_norm`` to ``inf`` to disable clipping."""
    rng = np.random.default_rng(seed)
    n, da = xa.shape
    dg = xg.shape[1]
    proj_a = rng.normal(0.0, 1.0 / np.sqrt(da), size=(rank, da))
    proj_g = rng.normal(0.0, 1.0 / np.sqrt(dg), size=(rank, dg))
    intercept = 0.0
    yf = y.astype(float)
    for _ in range(n_iter):
        ua = xa @ proj_a.T  # n x rank
        ug = xg @ proj_g.T  # n x rank
        logits = np.sum(ua * ug, axis=1) + intercept
        resid = _sigmoid(logits) - yf  # n
        g_proj_a = (resid[:, None] * ug).T @ xa / n + l2 * proj_a
        g_proj_g = (resid[:, None] * ua).T @ xg / n + l2 * proj_g
        if np.isfinite(max_grad_norm):
            norm_a = float(np.linalg.norm(g_proj_a))
            norm_g = float(np.linalg.norm(g_proj_g))
            if norm_a > max_grad_norm:
                g_proj_a *= max_grad_norm / norm_a
            if norm_g > max_grad_norm:
                g_proj_g *= max_grad_norm / norm_g
        proj_a -= lr * g_proj_a
        proj_g -= lr * g_proj_g
        intercept -= lr * float(resid.mean())
    return proj_a, proj_g, float(intercept)


def bilinear_oof_scores(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    folds: np.ndarray[Any, Any],
    *,
    rank: int,
    l2: float,
    lr: float,
    n_iter: int,
    seed: int,
    max_grad_norm: float = 1.0,
) -> np.ndarray[Any, Any]:
    """Out-of-fold bilinear logits, standardizing on each fold's train rows so
    held-out scores never see test statistics."""
    out: np.ndarray[Any, Any] = np.full(y.shape, np.nan, dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        if np.unique(y[train]).size < 2:
            continue
        ma, sa = standardizer(xa[train])
        mg, sg = standardizer(xg[train])
        pa, pg, b = fit_bilinear(
            apply_standardizer(xa[train], ma, sa),
            apply_standardizer(xg[train], mg, sg),
            y[train],
            rank=rank,
            l2=l2,
            lr=lr,
            n_iter=n_iter,
            seed=seed,
            max_grad_norm=max_grad_norm,
        )
        out[test] = predict_bilinear(
            apply_standardizer(xa[test], ma, sa),
            apply_standardizer(xg[test], mg, sg),
            pa,
            pg,
            b,
        )
    return out
