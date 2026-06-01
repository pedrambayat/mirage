"""Feature attribution for the linear gate.

For an L2-logistic model on standardized features, the standardized coefficient
is the per-feature contribution to the log-odds — the linear-model analogue of
a global SHAP value. This is the numpy stand-in for the spec's SHAP guard
(detecting whether the gate keys on trivial sequence mismatches); `shap` is not
a project dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mirage.ml.core import apply_standardizer, fit_logistic_regression, standardizer


def standardized_contributions(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    feature_names: list[str],
    l2: float,
) -> list[dict[str, float | str]]:
    """Fit on standardized features; return features ranked by |coefficient|."""
    mean, std = standardizer(x)
    xs = apply_standardizer(x, mean, std)
    _intercept, coef = fit_logistic_regression(xs, y, l2=l2)
    rows: list[dict[str, float | str]] = [
        {"feature": name, "coefficient": float(c), "abs_contribution": float(abs(c))}
        for name, c in zip(feature_names, coef, strict=True)
    ]
    rows.sort(key=lambda r: float(r["abs_contribution"]), reverse=True)
    return rows
