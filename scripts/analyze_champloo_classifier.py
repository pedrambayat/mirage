"""Phase 1 Champloo/Smorodina classifier baseline.

Reads the staged Champloo pair table (see ``stage_champloo_pairs.py``) and
evaluates, with AUPRC (primary) and AUROC (secondary):

* ``raw_iptm`` -- the released predictor confidence used directly as the score
  (no training; split-invariant).
* ``logistic_iptm`` -- numpy-only L2 logistic regression on ipTM alone.
* ``logistic_iptm_meta`` -- logistic regression on ipTM plus cheap, no-structure
  metadata features (VHH / antigen length, antigen secondary-structure content).

Each learned model is scored with out-of-fold predictions under three required
splits:

* ``random_pair`` -- ordinary K-fold over pairs (leakage-permissive sanity check).
* ``held_out_vhh`` -- grouped K-fold by VHH (row) PDB; no VHH appears in both
  train and test.
* ``held_out_antigen`` -- grouped K-fold by antigen (column) PDB.

The off-diagonal negatives are *constructed shuffled non-cognate* pairings, not
experimentally verified non-binders. The class balance is heavily skewed
(~1% cognate), which is why AUPRC is the headline metric.

Use::

    uv run python scripts/analyze_champloo_classifier.py \\
        --pairs data/staged/champloo/champloo_pairs_af3.csv \\
        --output results/published/champloo_af3_classifier.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mirage.ml.core import (
    assign_folds,
    oof_logistic_scores,
)
from mirage.ml.core import (
    auroc as _auroc,
)
from mirage.ml.core import (
    average_precision as _average_precision,
)

_META_FEATURES = (
    "vhh_length",
    "antigen_length",
    "antigen_helix_content",
    "antigen_sheet_content",
    "antigen_loop_content",
)


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _floats(rows: list[dict[str, str]], key: str) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    for i, row in enumerate(rows):
        value = row.get(key, "")
        try:
            out[i] = float(value) if value not in ("", None) else math.nan
        except (TypeError, ValueError):
            out[i] = math.nan
    return out


def feature_matrix(rows: list[dict[str, str]], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([_floats(rows, name) for name in feature_names])


def _metric_row(
    split: str,
    model: str,
    features: str,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    predictor: str,
) -> dict[str, Any]:
    scored = np.isfinite(scores)
    s = scores[scored]
    y = labels[scored].astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "predictor": predictor,
        "split": split,
        "model": model,
        "features": features,
        "n": int(y.size),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "baseline_ap": float(y.mean()) if y.size else math.nan,
        "ap": _average_precision(s, y) if n_pos and n_neg else math.nan,
        "auroc": _auroc(s, y) if n_pos and n_neg else math.nan,
    }


def evaluate(
    rows: list[dict[str, str]],
    *,
    n_splits: int,
    seed: int,
    l2: float,
) -> list[dict[str, Any]]:
    labels = np.asarray([int(r["label"]) for r in rows], dtype=int)
    predictor = rows[0]["predictor"] if rows else ""
    iptm = feature_matrix(rows, ("iptm",))
    iptm_meta = feature_matrix(rows, ("iptm", *_META_FEATURES))
    vhh_pdb = np.asarray([r["vhh_pdb"] for r in rows])
    antigen_pdb = np.asarray([r["antigen_pdb"] for r in rows])
    pair_id = np.asarray([r["pair_id"] for r in rows])

    out: list[dict[str, Any]] = []
    # Raw predictor confidence: no training, split-invariant.
    out.append(_metric_row("all", "raw_iptm", "iptm", iptm[:, 0], labels, predictor=predictor))

    split_groups = {
        "random_pair": pair_id,
        "held_out_vhh": vhh_pdb,
        "held_out_antigen": antigen_pdb,
    }
    models = {
        "logistic_iptm": (iptm, "iptm"),
        "logistic_iptm_meta": (iptm_meta, ";".join(("iptm", *_META_FEATURES))),
    }
    for split, groups in split_groups.items():
        folds = assign_folds(groups, n_splits=n_splits, seed=seed)
        for model, (x_all, feature_str) in models.items():
            finite = np.isfinite(x_all).all(axis=1)
            scores = np.full(labels.shape, math.nan, dtype=float)
            scores[finite] = oof_logistic_scores(
                x_all[finite], labels[finite].astype(float), folds[finite], l2=l2
            )
            out.append(_metric_row(split, model, feature_str, scores, labels, predictor=predictor))
    return out


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "predictor",
        "split",
        "model",
        "features",
        "n",
        "n_positive",
        "n_negative",
        "baseline_ap",
        "ap",
        "auroc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_report(rows: list[dict[str, Any]]) -> None:
    print(f"{'split':<16} {'model':<20} {'n':>6} {'pos':>4} {'ap':>7} {'auroc':>7} {'base_ap':>8}")
    for row in rows:
        print(
            f"{row['split']:<16} {row['model']:<20} {int(row['n']):6d} "
            f"{int(row['n_positive']):4d} {float(row['ap']):7.3f} "
            f"{float(row['auroc']):7.3f} {float(row['baseline_ap']):8.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    rows = read_pairs(args.pairs)
    metrics = evaluate(rows, n_splits=args.n_splits, seed=args.seed, l2=args.l2)
    write_metrics(args.output, metrics)
    _print_report(metrics)
    print(json.dumps({"pairs": len(rows), "metric_rows": len(metrics)}))
    print(f"Wrote {len(metrics)} metric rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
