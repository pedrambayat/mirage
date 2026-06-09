from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "embed_perresidue", REPO / "scripts" / "embed_perresidue.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embed_perresidue"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_iter_windows_non_overlapping():
    mod = _load()
    assert mod.iter_windows(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert mod.iter_windows(3, 1022) == [(0, 3)]
