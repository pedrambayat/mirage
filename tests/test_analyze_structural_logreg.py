from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def _load_script() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_structural_logreg.py"
    spec = importlib.util.spec_from_file_location("analyze_structural_logreg", script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_logistic_loo_scores_separate_simple_signal() -> None:
    module = _load_script()
    x = np.asarray([[-4.0], [-3.0], [-2.0], [2.0], [3.0], [4.0]])
    y = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    scores = module._logistic_loo_scores(x, y, l2=1.0)

    assert np.isfinite(scores).all()
    assert module._auroc(scores, y.astype(int)) == 1.0


def test_logistic_row_flips_lower_is_better_features() -> None:
    module = _load_script()
    rows = [
        {"value": "", "__extras": {"atom_clashes_2a": 10.0}},
        {"value": "", "__extras": {"atom_clashes_2a": 9.0}},
        {"value": "", "__extras": {"atom_clashes_2a": 2.0}},
        {"value": "", "__extras": {"atom_clashes_2a": 1.0}},
    ]
    labels = np.asarray([0, 0, 1, 1])

    row = module._logistic_row(
        "toy",
        rows,
        labels,
        feature_names=("atom_clashes_2a",),
        l2=1.0,
    )

    assert row["n"] == 4
    assert row["auroc"] == 1.0
