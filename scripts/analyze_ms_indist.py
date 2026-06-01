"""Train M-S on Champloo Tier-S features and report in-distribution gate metrics.

Compares the sequence-only gate (M-S, out-of-fold under the held-out-antigen
split) against the raw-ipTM floor at a fixed-precision operating point, writes
the frozen M-S artifact, the PPV-prevalence sweep, and the feature-attribution
ranking (the SHAP guard).

Use::

    uv run python scripts/analyze_ms_indist.py \\
        --features data/staged/champloo/champloo_features_af3.csv \\
        --model-out results/published/ms_model_af3.json \\
        --output results/published/mirage_ms_indist_af3.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.attribution import standardized_contributions
from mirage.eval.gate import (
    DEFAULT_PREVALENCES,
    auroc,
    choose_threshold_for_precision,
    summary_dict,
)
from mirage.features.sequence import FEATURE_NAMES
from mirage.model.ms import train_ms


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_feature_matrix(
    rows: list[dict[str, str]], *, feature_names: tuple[str, ...]
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    tuple[str, ...],
]:
    x = np.array([[_to_float(r[name]) for name in feature_names] for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    iptm = np.array([_to_float(r["iptm"]) for r in rows], dtype=float)
    groups = np.array([r["antigen_pdb"] for r in rows])
    return x, y, iptm, groups, feature_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    rows = read_csv(args.features)
    x, y, iptm, groups, names = load_feature_matrix(rows, feature_names=FEATURE_NAMES)
    finite_rows = np.isfinite(x).all(axis=1)
    x, y, iptm, groups = x[finite_rows], y[finite_rows], iptm[finite_rows], groups[finite_rows]

    model, oof = train_ms(
        x,
        y,
        feature_names=list(names),
        l2=args.l2,
        target_precision=args.target_precision,
        seed=args.seed,
        groups=groups,
    )
    model.save(args.model_out)

    # Honest in-distribution report: the OOF scores live on a different scale
    # than the frozen full-fit model, so the operating point is re-derived on
    # the OOF scores. (model.threshold is the *shipped* full-fit threshold used
    # unchanged by the orthogonal harness — do NOT apply it to OOF scores.)
    finite_oof = np.isfinite(oof)
    oof_thr = choose_threshold_for_precision(
        oof[finite_oof], y[finite_oof], target_precision=args.target_precision
    )
    ms_summary = summary_dict(oof[finite_oof], y[finite_oof], threshold=oof_thr)
    ms_summary["auroc"] = auroc(oof[finite_oof], y[finite_oof])

    iptm_finite = np.isfinite(iptm)
    iptm_thr = choose_threshold_for_precision(
        iptm[iptm_finite], y[iptm_finite], target_precision=args.target_precision
    )
    iptm_summary = summary_dict(iptm[iptm_finite], y[iptm_finite], threshold=iptm_thr)
    iptm_summary["auroc"] = auroc(iptm[iptm_finite], y[iptm_finite])

    attribution = standardized_contributions(
        x, y.astype(float), feature_names=list(names), l2=args.l2
    )

    result = {
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "target_precision": args.target_precision,
        "prevalences": list(DEFAULT_PREVALENCES),
        "oof_threshold": float(oof_thr),
        "shipped_threshold": float(model.threshold),
        "ms": ms_summary,
        "iptm_floor": iptm_summary,
        "attribution": attribution,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "ms_auroc": ms_summary["auroc"],
                "ms_recall": ms_summary["metrics"]["recall"],
                "iptm_auroc": iptm_summary["auroc"],
                "iptm_recall": iptm_summary["metrics"]["recall"],
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output} and model {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
