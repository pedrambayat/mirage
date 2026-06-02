from __future__ import annotations

from pathlib import Path

import numpy as np

from mirage.ml.bilinear import fit_bilinear, predict_bilinear
from mirage.ml.core import apply_standardizer, standardizer
from mirage.model.bilinear import BilinearModel


def _fit_small():
    rng = np.random.default_rng(0)
    n, d = 200, 6
    xa = rng.normal(size=(n, d))
    xg = rng.normal(size=(n, d))
    m = rng.normal(size=(d, d))
    y = (np.sum((xa @ m) * xg, axis=1) > 0).astype(int)
    ma, sa = standardizer(xa)
    mg, sg = standardizer(xg)
    pa, pg, b = fit_bilinear(
        apply_standardizer(xa, ma, sa),
        apply_standardizer(xg, mg, sg),
        y,
        rank=d,
        l2=1e-3,
        lr=0.1,
        n_iter=500,
        seed=1,
    )
    model = BilinearModel(
        feature_dim=d,
        rank=d,
        mean_a=ma.tolist(),
        std_a=sa.tolist(),
        mean_g=mg.tolist(),
        std_g=sg.tolist(),
        proj_a=pa.tolist(),
        proj_g=pg.tolist(),
        intercept=b,
        threshold=0.0,
        target_precision=0.9,
    )
    return model, xa, xg, (ma, sa, mg, sg, pa, pg, b)


def test_predict_logit_matches_manual_pipeline():
    model, xa, xg, (ma, sa, mg, sg, pa, pg, b) = _fit_small()
    x = np.concatenate([xa, xg], axis=1)
    manual = predict_bilinear(
        apply_standardizer(xa, ma, sa), apply_standardizer(xg, mg, sg), pa, pg, b
    )
    assert np.allclose(model.predict_logit(x), manual)


def test_save_load_roundtrip(tmp_path: Path):
    model, xa, xg, _ = _fit_small()
    x = np.concatenate([xa, xg], axis=1)
    path = tmp_path / "bilinear.json"
    model.save(path)
    loaded = BilinearModel.load(path)
    assert loaded.feature_dim == model.feature_dim
    assert np.allclose(loaded.predict_logit(x), model.predict_logit(x))
