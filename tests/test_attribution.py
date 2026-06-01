from __future__ import annotations

import numpy as np

from mirage.eval.attribution import standardized_contributions


def test_dominant_feature_ranks_first() -> None:
    # feature 0 carries all signal; feature 1 is noise
    rng = np.random.default_rng(0)
    n = 200
    x0 = np.concatenate([rng.normal(-2, 0.3, n), rng.normal(2, 0.3, n)])
    x1 = rng.normal(0, 1, 2 * n)
    x = np.column_stack([x0, x1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    ranked = standardized_contributions(x, y, feature_names=["signal", "noise"], l2=1.0)
    assert ranked[0]["feature"] == "signal"
    assert abs(ranked[0]["abs_contribution"]) > abs(ranked[1]["abs_contribution"])
