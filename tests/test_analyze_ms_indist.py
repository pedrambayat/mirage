from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def _load(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_feature_matrix_parses_columns() -> None:
    analyze = _load("analyze_ms_indist.py")
    rows = [
        {
            "pair_id": "a",
            "label": "1",
            "iptm": "0.9",
            "antigen_pdb": "AAAA",
            "binder_length": "120.0",
            "binder_net_charge": "2.0",
        },
        {
            "pair_id": "b",
            "label": "0",
            "iptm": "0.1",
            "antigen_pdb": "BBBB",
            "binder_length": "118.0",
            "binder_net_charge": "-1.0",
        },
    ]
    x, y, iptm, groups, names = analyze.load_feature_matrix(
        rows, feature_names=("binder_length", "binder_net_charge")
    )
    assert x.shape == (2, 2)
    assert list(y) == [1, 0]
    assert np.allclose(iptm, [0.9, 0.1])
    assert list(groups) == ["AAAA", "BBBB"]
    assert names == ("binder_length", "binder_net_charge")
