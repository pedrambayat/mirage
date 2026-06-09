from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_stage2", REPO / "scripts" / "train_stage2.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_stage2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_load_perres_roundtrip(tmp_path):
    mod = _load()
    a = np.arange(6, dtype=np.float16).reshape(3, 2)
    b = np.arange(4, dtype=np.float16).reshape(2, 2)
    np.savez(tmp_path / "p.npz", **{"0": a, "1": b})
    (tmp_path / "k.txt").write_text("AAA\nCC\n")
    cache = mod.load_perres(tmp_path / "p.npz", tmp_path / "k.txt")
    assert set(cache) == {"AAA", "CC"}
    assert cache["AAA"].shape == (3, 2) and np.allclose(cache["CC"], b)


def test_oof_folds_disjoint_and_exhaustive():
    mod = _load()
    folds = np.array([0, 0, 1, 2, 2, 1])
    seen = np.zeros(6, dtype=bool)
    n = 0
    for test, train in mod.oof_folds(folds):
        assert not (test & train).any()
        assert (test | train).all()
        seen |= test
        n += 1
    assert n == 3 and seen.all()
