from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "embed_sequences", REPO / "scripts" / "embed_sequences.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embed_sequences"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_windows_cover_sequence_without_overlap():
    mod = _load()
    assert mod.iter_windows(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_short_sequence_single_window():
    mod = _load()
    assert mod.iter_windows(3, 1022) == [(0, 3)]


def test_exact_multiple():
    mod = _load()
    assert mod.iter_windows(8, 4) == [(0, 4), (4, 8)]
