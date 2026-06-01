from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mirage.model.ms import MsModel, train_ms


def _separable_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.4, 60), rng.normal(2, 0.4, 60)])[:, None]
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(int)
    return x, y, ["f0"]


def test_train_ms_produces_usable_model() -> None:
    x, y, names = _separable_data()
    model, oof = train_ms(x, y, feature_names=names, l2=1.0, target_precision=0.9, seed=1)
    assert isinstance(model, MsModel)
    assert np.isfinite(oof).all()
    # frozen model separates the classes
    logits = model.predict_logit(x)
    assert ((logits >= model.threshold).astype(int) == y).mean() > 0.9


def test_threshold_is_on_the_frozen_model_scale() -> None:
    # The shipped threshold must hit the target precision on the frozen model's
    # OWN logits (predict_logit) — not on the OOF scores. This is the scale-
    # consistency guarantee the orthogonal harness depends on.
    from mirage.eval.gate import metrics_at_threshold

    x, y, names = _separable_data()
    model, _ = train_ms(x, y, feature_names=names, l2=1.0, target_precision=0.9, seed=1)
    logits = model.predict_logit(x)
    m = metrics_at_threshold(logits, y, threshold=model.threshold)
    assert m["precision"] >= 0.9 - 1e-9


def test_ms_model_save_load_roundtrip(tmp_path: Path) -> None:
    x, y, names = _separable_data()
    model, _ = train_ms(x, y, feature_names=names, l2=1.0, target_precision=0.9, seed=1)
    path = tmp_path / "ms.json"
    model.save(path)
    loaded = MsModel.load(path)
    assert loaded.feature_names == model.feature_names
    assert abs(loaded.threshold - model.threshold) < 1e-12
    assert np.allclose(loaded.predict_logit(x), model.predict_logit(x))
    # artifact is human-readable JSON
    json.loads(path.read_text())
