from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_sabdab_baseline", REPO / "scripts" / "analyze_sabdab_baseline.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_sabdab_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def _interaction_data(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    xa = rng.normal(size=(n, d))
    xg = rng.normal(size=(n, d))
    m = rng.normal(size=(d, d))
    y = (np.sum((xa @ m) * xg, axis=1) > 0).astype(int)
    folds = np.array([i % 5 for i in range(n)])
    return xa, xg, y, folds


def test_linear_rung_returns_summary_and_model():
    mod = _load()
    xa, xg, y, folds = _interaction_data()
    x = np.concatenate([xa, xg], axis=1)
    summary, model = mod.run_linear_rung(x, y, folds, l2=1.0, target_precision=0.9, seed=1)
    assert "auroc" in summary and "metrics" in summary
    assert hasattr(model, "predict_logit")


def test_bilinear_rung_beats_linear_on_interaction():
    mod = _load()
    xa, xg, y, folds = _interaction_data()
    x = np.concatenate([xa, xg], axis=1)
    lin_summary, _ = mod.run_linear_rung(x, y, folds, l2=1.0, target_precision=0.9, seed=1)
    bil_summary, _bil_model = mod.run_bilinear_rung(
        xa,
        xg,
        y,
        folds,
        rank=6,
        l2=1e-3,
        lr=0.1,
        n_iter=1500,
        target_precision=0.9,
        seed=1,
    )
    assert bil_summary["auroc"] > lin_summary["auroc"]
    assert bil_summary["auroc"] > 0.7
