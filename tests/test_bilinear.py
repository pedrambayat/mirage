from __future__ import annotations

import numpy as np

from mirage.eval.gate import auroc
from mirage.ml.bilinear import bilinear_oof_scores, fit_bilinear, predict_bilinear
from mirage.ml.core import apply_standardizer, fit_logistic_regression, standardizer


def _planted_interaction(n=400, d=8, seed=0):
    rng = np.random.default_rng(seed)
    xa = rng.normal(size=(n, d))
    xg = rng.normal(size=(n, d))
    m = rng.normal(size=(d, d))
    score = np.sum((xa @ m) * xg, axis=1)
    y = (score > np.median(score)).astype(int)
    return xa, xg, y


def test_bilinear_recovers_planted_interaction():
    xa, xg, y = _planted_interaction()
    pa, pg, b = fit_bilinear(xa, xg, y, rank=8, l2=1e-3, lr=0.1, n_iter=3000, seed=1)
    logits = predict_bilinear(xa, xg, pa, pg, b)
    assert auroc(logits, y) > 0.8


def test_additive_logistic_is_chance_on_pure_interaction():
    # Use a larger n so the finite-sample Newton solver cannot exploit noise
    # correlations: with n=400 the IRLS fit finds ~0.62 AUROC due to
    # finite-sample coincidences; at n=2000 it converges to near-chance.
    xa, xg, y = _planted_interaction(n=2000)
    x = np.concatenate([xa, xg], axis=1)
    mean, std = standardizer(x)
    xs = apply_standardizer(x, mean, std)
    ic, coef = fit_logistic_regression(xs, y.astype(float), l2=1.0)
    logits = ic + xs @ coef
    assert abs(auroc(logits, y) - 0.5) < 0.1


def test_fit_bilinear_is_deterministic():
    xa, xg, y = _planted_interaction()
    a = fit_bilinear(xa, xg, y, rank=4, l2=1.0, lr=0.05, n_iter=200, seed=2)
    b = fit_bilinear(xa, xg, y, rank=4, l2=1.0, lr=0.05, n_iter=200, seed=2)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and a[2] == b[2]


def test_oof_scores_are_finite_and_grouped():
    xa, xg, y = _planted_interaction()
    folds = np.array([i % 5 for i in range(y.size)])
    oof = bilinear_oof_scores(xa, xg, y, folds, rank=8, l2=1e-3, lr=0.1, n_iter=1000, seed=1)
    assert np.isfinite(oof).all()
    assert auroc(oof, y) > 0.7


def test_gradient_clipping_keeps_fit_finite_on_amplifying_structure():
    # Each tower's gradient scales with the OTHER tower's magnitude, so on inputs
    # with exploitable bilinear structure plain GD grows the projections
    # geometrically and overflows to nan/inf (this is what the 1280-d SAbDab ESM
    # embeddings triggered). Gradient clipping (on by default) bounds ||proj||
    # (~max_grad_norm / l2) and keeps the fit finite regardless of seed.
    rng = np.random.default_rng(0)
    n, d, r = 400, 256, 16
    u = rng.normal(size=(d, r))
    v = rng.normal(size=(d, r))
    xa = rng.normal(size=(n, d)) * 3.0
    xg = rng.normal(size=(n, d)) * 3.0
    score = np.sum((xa @ u) * (xg @ v), axis=1)
    y = (score > np.median(score)).astype(int)
    # teeth: without clipping it diverges, confirming the clip is load-bearing
    pa_u, pg_u, _ = fit_bilinear(
        xa, xg, y, rank=r, l2=1e-4, lr=1.0, n_iter=2000, seed=7, max_grad_norm=float("inf")
    )
    assert not (np.isfinite(pa_u).all() and np.isfinite(pg_u).all())
    # with clipping (default) it stays finite
    pa, pg, b = fit_bilinear(xa, xg, y, rank=r, l2=1e-2, lr=1.0, n_iter=2000, seed=7)
    assert np.isfinite(pa).all() and np.isfinite(pg).all() and np.isfinite(b)
