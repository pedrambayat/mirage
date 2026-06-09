from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def test_negatives_inherit_parent_fold():
    mod = _load()
    pair_ids = ["p0", "p0__neg0", "p0__neg1", "p1", "p1__neg0", "p2"]
    folds = mod.random_folds_by_positive(pair_ids, n_splits=5, seed=1)
    assert folds[0] == folds[1] == folds[2]  # p0 and its negatives
    assert folds[3] == folds[4]  # p1 and its negative


def test_random_folds_deterministic_and_in_range():
    mod = _load()
    pair_ids = [f"p{i}" for i in range(50)] + [f"p{i}__neg0" for i in range(50)]
    a = mod.random_folds_by_positive(pair_ids, n_splits=5, seed=7)
    b = mod.random_folds_by_positive(pair_ids, n_splits=5, seed=7)
    assert (a == b).all()
    assert a.min() >= 0 and a.max() <= 4
    # different seed gives a different assignment (overwhelmingly likely)
    c = mod.random_folds_by_positive(pair_ids, n_splits=5, seed=8)
    assert not (a == c).all()
