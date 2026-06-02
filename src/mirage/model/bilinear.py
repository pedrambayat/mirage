"""Frozen low-rank bilinear gate artifact.

Mirrors model/ms.py::MsModel: standardizer stats + projections + threshold,
serialized to JSON, applied unchanged to held-out sets. ``predict_logit`` takes
the concat layout x = [e_a | e_g] (raw), so the existing orthogonal harness
(eval/orthogonal.py::evaluate_frozen_gate) works without modification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mirage.ml.bilinear import predict_bilinear
from mirage.ml.core import apply_standardizer


@dataclass(frozen=True)
class BilinearModel:
    feature_dim: int
    rank: int
    mean_a: list[float]
    std_a: list[float]
    mean_g: list[float]
    std_g: list[float]
    proj_a: list[list[float]]
    proj_g: list[list[float]]
    intercept: float
    threshold: float
    target_precision: float

    def predict_logit(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        d = self.feature_dim
        xa = apply_standardizer(x[:, :d], np.asarray(self.mean_a), np.asarray(self.std_a))
        xg = apply_standardizer(x[:, d:], np.asarray(self.mean_g), np.asarray(self.std_g))
        return predict_bilinear(
            xa, xg, np.asarray(self.proj_a), np.asarray(self.proj_g), self.intercept
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> BilinearModel:
        data: dict[str, Any] = json.loads(path.read_text())
        return cls(**data)
