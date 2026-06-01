"""M-S: the sequence-only mirage gate.

`train_ms` fits an L2-logistic model on Tier-S features, computes leakage-aware
out-of-fold scores for honest in-distribution metrics, fits a final model on all
rows, and picks the operating threshold at a target precision. The frozen
`MsModel` (standardization + coefficients + threshold) serializes to JSON and is
applied unchanged to orthogonal datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.gate import choose_threshold_for_precision
from mirage.ml.core import (
    apply_standardizer,
    assign_folds,
    fit_logistic_regression,
    oof_logistic_scores,
    standardizer,
)


@dataclass(frozen=True)
class MsModel:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    intercept: float
    coef: list[float]
    threshold: float
    target_precision: float

    def predict_logit(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        xs = apply_standardizer(x, mean, std)
        result: np.ndarray[Any, Any] = self.intercept + xs @ np.asarray(self.coef)
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> MsModel:
        data: dict[str, Any] = json.loads(path.read_text())
        return cls(**data)


def train_ms(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    feature_names: list[str],
    l2: float,
    target_precision: float,
    seed: int,
    groups: np.ndarray[Any, Any] | None = None,
    n_splits: int = 5,
) -> tuple[MsModel, np.ndarray[Any, Any]]:
    """Return (frozen model, out-of-fold scores).

    `groups` drives the leakage-controlled OOF split (e.g. antigen PDB). If
    None, an ordinary K-fold over rows is used.

    The returned OOF scores are for honest held-out *reporting only*. The frozen
    model's operating threshold is chosen on the **full-fit model's own logits**
    so it is on the same scale the frozen model emits (`predict_logit`) — the
    orthogonal harness applies that threshold unchanged and must reproduce the
    target operating point. (Picking the threshold on OOF scores and applying it
    to the full-fit model would be a scale mismatch: each OOF fold has its own
    standardizer + intercept.)
    """
    y = y.astype(float)
    if groups is None:
        groups = np.arange(x.shape[0]).astype(str)

    # OOF scores: purely for the honest in-distribution report downstream.
    folds = assign_folds(groups, n_splits=n_splits, seed=seed)
    oof = oof_logistic_scores(x, y, folds, l2=l2)

    # Final full-data fit — this is what freezes into the artifact.
    mean, std = standardizer(x)
    xs = apply_standardizer(x, mean, std)
    intercept, coef = fit_logistic_regression(xs, y, l2=l2)

    # Threshold on the full-fit model's own logits (same scale as predict_logit).
    full_logits: np.ndarray[Any, Any] = intercept + xs @ coef
    finite = np.isfinite(full_logits)
    threshold = choose_threshold_for_precision(
        full_logits[finite], y[finite].astype(int), target_precision=target_precision
    )

    model = MsModel(
        feature_names=list(feature_names),
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
        intercept=float(intercept),
        coef=[float(v) for v in coef],
        threshold=float(threshold),
        target_precision=float(target_precision),
    )
    return model, oof
