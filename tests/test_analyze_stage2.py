from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from mirage.eval.gate import auroc

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_stage2", REPO / "scripts" / "analyze_stage2.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_stage2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_analyze_scores_matches_gate_auroc():
    mod = _load()
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = (scores + rng.normal(scale=0.5, size=200) > 0).astype(int)
    out = mod.analyze_scores(scores, labels, target_precision=0.9, seed=1)
    assert abs(out["auroc"] - auroc(scores, labels)) < 1e-9
    assert "metrics" in out and "auroc_ci" in out
    assert out["auroc_ci"][0] <= out["auroc"] <= out["auroc_ci"][1]
