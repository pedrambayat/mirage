"""Train the four-rung sequence-only ladder on SAbDab and report gate metrics.

Rungs: 0 additive Tier-S, 1 additive ESM-concat, 2 diagonal bilinear (Hadamard,
reusing the existing logistic), 3 low-rank bilinear two-tower. OOF scores use the
pre-assigned antigen-cluster folds; the head-to-head AUROC / recall@precision /
PPV-sweep table is written to JSON and the best interaction rung is frozen.

Use::

    uv run python scripts/analyze_sabdab_baseline.py \\
        --pairs data/staged/sabdab/sabdab_pairs.csv \\
        --embeddings data/staged/sabdab/embeddings.npy \\
        --keys data/staged/sabdab/keys.txt \\
        --output results/published/sabdab_baseline.json \\
        --model-out results/published/sabdab_bilinear_model.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.gate import auroc, choose_threshold_for_precision, summary_dict
from mirage.features.embeddings import load_embedding_cache, paired_matrix
from mirage.features.sequence import FEATURE_NAMES, sequence_features
from mirage.ml.bilinear import bilinear_oof_scores, fit_bilinear, predict_bilinear
from mirage.ml.core import apply_standardizer, standardizer
from mirage.model.bilinear import BilinearModel
from mirage.model.ms import MsModel, train_ms


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def run_linear_rung(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    folds: np.ndarray[Any, Any],
    *,
    l2: float,
    target_precision: float,
    seed: int,
) -> tuple[dict[str, Any], MsModel]:
    """Fit an additive/diagonal logistic rung. Passing the pre-assigned fold
    column as ``groups`` (with n_splits = #folds) makes train_ms's grouped OOF
    reproduce exactly those antigen-cluster folds."""
    names = [f"f{i}" for i in range(x.shape[1])]
    n_splits = int(np.unique(folds).size)
    model, oof = train_ms(
        x,
        y,
        feature_names=names,
        l2=l2,
        target_precision=target_precision,
        seed=seed,
        groups=folds.astype(str),
        n_splits=n_splits,
    )
    finite = np.isfinite(oof)
    thr = choose_threshold_for_precision(oof[finite], y[finite], target_precision=target_precision)
    summary = summary_dict(oof[finite], y[finite], threshold=thr)
    summary["auroc"] = auroc(oof[finite], y[finite])
    return summary, model


def run_bilinear_rung(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    folds: np.ndarray[Any, Any],
    *,
    rank: int,
    l2: float,
    lr: float,
    n_iter: int,
    target_precision: float,
    seed: int,
) -> tuple[dict[str, Any], BilinearModel]:
    oof = bilinear_oof_scores(xa, xg, y, folds, rank=rank, l2=l2, lr=lr, n_iter=n_iter, seed=seed)
    finite = np.isfinite(oof)
    thr_oof = choose_threshold_for_precision(
        oof[finite], y[finite], target_precision=target_precision
    )
    summary = summary_dict(oof[finite], y[finite], threshold=thr_oof)
    summary["auroc"] = auroc(oof[finite], y[finite])

    ma, sa = standardizer(xa)
    mg, sg = standardizer(xg)
    pa, pg, b = fit_bilinear(
        apply_standardizer(xa, ma, sa),
        apply_standardizer(xg, mg, sg),
        y,
        rank=rank,
        l2=l2,
        lr=lr,
        n_iter=n_iter,
        seed=seed,
    )
    full = predict_bilinear(
        apply_standardizer(xa, ma, sa), apply_standardizer(xg, mg, sg), pa, pg, b
    )
    thr_full = choose_threshold_for_precision(full, y, target_precision=target_precision)
    model = BilinearModel(
        feature_dim=int(xa.shape[1]),
        rank=rank,
        mean_a=ma.tolist(),
        std_a=sa.tolist(),
        mean_g=mg.tolist(),
        std_g=sg.tolist(),
        proj_a=pa.tolist(),
        proj_g=pg.tolist(),
        intercept=b,
        threshold=float(thr_full),
        target_precision=target_precision,
    )
    return summary, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--bilinear-l2", type=float, default=1e-2)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    rows = read_csv(args.pairs)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    folds = np.array([int(r["fold"]) for r in rows], dtype=int)
    binders = [r["binder_seq"] for r in rows]
    antigens = [r["antigen_seq"] for r in rows]
    pairs = list(zip(binders, antigens, strict=True))

    cache = load_embedding_cache(args.embeddings, args.keys)
    x_concat = paired_matrix(pairs, cache, layout="concat")
    x_hadamard = paired_matrix(pairs, cache, layout="hadamard")
    d = x_hadamard.shape[1]
    xa, xg = x_concat[:, :d], x_concat[:, d:]

    x_tiers = np.array(
        [[sequence_features(b, a)[name] for name in FEATURE_NAMES] for b, a in pairs],
        dtype=float,
    )

    tp, seed = args.target_precision, args.seed
    results: dict[str, Any] = {
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "target_precision": tp,
    }
    results["rung0_tier_s"], _ = run_linear_rung(
        x_tiers, y, folds, l2=args.l2, target_precision=tp, seed=seed
    )
    results["rung1_esm_concat"], _ = run_linear_rung(
        x_concat, y, folds, l2=args.l2, target_precision=tp, seed=seed
    )
    results["rung2_hadamard"], diag_model = run_linear_rung(
        x_hadamard, y, folds, l2=args.l2, target_precision=tp, seed=seed
    )
    results["rung3_bilinear"], bil_model = run_bilinear_rung(
        xa,
        xg,
        y,
        folds,
        rank=args.rank,
        l2=args.bilinear_l2,
        lr=args.lr,
        n_iter=args.n_iter,
        target_precision=tp,
        seed=seed,
    )

    # Freeze the strongest interaction rung by OOF AUROC.
    if results["rung3_bilinear"]["auroc"] >= results["rung2_hadamard"]["auroc"]:
        results["frozen_rung"] = "rung3_bilinear"
        bil_model.save(args.model_out)
    else:
        results["frozen_rung"] = "rung2_hadamard"
        diag_model.save(args.model_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    for name in (
        "rung0_tier_s",
        "rung1_esm_concat",
        "rung2_hadamard",
        "rung3_bilinear",
    ):
        print(
            f"{name}: AUROC={results[name]['auroc']:.3f}"
            f" recall={results[name]['metrics']['recall']:.3f}"
        )
    print(f"Froze {results['frozen_rung']} to {args.model_out}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
