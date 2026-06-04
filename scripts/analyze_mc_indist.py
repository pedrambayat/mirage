"""M-C in-distribution analysis: the rung ladder vs the M-S 0.496 floor (SAbDab).

For each cumulative rung (0=ipTM, 1=+confidence, 2=+geometry, 3=+CDR) on the SAbDab
Protenix feature CSV, fit the M-S recipe out-of-fold under the *same*
held-out-antigen-cluster folds, and report marginal AUROC + bootstrap CI, gate metrics
at a fixed-precision operating point, the PPV-prevalence sweep, and standardized
coefficients. The headline is the **paired delta** Rung 2/3 vs Rung 0: does interface
geometry (+ CDR) add signal on top of ipTM (paired-bootstrap ΔAUROC CI excludes 0)?
A random-split contrast and the CDR-mapping failure rate are reported as diagnostics.
The Rung-3 model is frozen for cross-regime transfer.

Use::

    uv run python scripts/analyze_mc_indist.py \\
        --features data/staged/mc/sabdab_features.csv \\
        --output results/published/mc_indist.json \\
        --model-out results/published/mc_sabdab_model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.attribution import standardized_contributions
from mirage.eval.gate import (
    auroc,
    bootstrap_ci,
    metrics_at_threshold,
    paired_delta_bootstrap,
    summary_dict,
)
from mirage.features.mc_rungs import (
    cdr_mapping_failure_rate,
    fit_rung_model,
    folds_array,
    labels_array,
    read_feature_csv,
    rung_matrix,
)
from mirage.ml.core import assign_folds

_RUNGS = (0, 1, 2, 3)


def _precision_stat_at(threshold: float):
    def stat(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
        return metrics_at_threshold(scores, labels, threshold=threshold)["precision"]

    return stat


def analyze_indist(
    features_path: Path,
    *,
    l2: float,
    target_precision: float,
    seed: int,
    n_boot: int = 1000,
) -> dict[str, Any]:
    rows = read_feature_csv(features_path)
    y = labels_array(rows)
    folds = folds_array(rows)

    per_rung: dict[str, dict[str, Any]] = {}
    oof_by_rung: dict[int, np.ndarray[Any, Any]] = {}
    thr_by_rung: dict[int, float] = {}
    models: dict[int, Any] = {}

    for k in _RUNGS:
        x, names = rung_matrix(rows, rung=k)
        model, oof = fit_rung_model(
            x, y, folds, feature_names=names, l2=l2, target_precision=target_precision
        )
        oof_by_rung[k] = oof
        thr_by_rung[k] = model.threshold
        models[k] = model
        gate = summary_dict(oof, y, threshold=model.threshold)
        per_rung[f"rung{k}"] = {
            "features": names,
            "auroc": auroc(oof, y),
            "auroc_ci": list(bootstrap_ci(auroc, oof, y, n_boot=n_boot, seed=seed)),
            "gate": gate,
            "coefficients": standardized_contributions(
                x, y.astype(float), feature_names=names, l2=l2
            ),
        }

    def _delta(a: int, b: int) -> dict[str, Any]:
        da_pt, da_lo, da_hi = paired_delta_bootstrap(
            oof_by_rung[a], oof_by_rung[b], y, statistic=auroc, n_boot=n_boot, seed=seed
        )
        dp_pt, dp_lo, dp_hi = paired_delta_bootstrap(
            oof_by_rung[a],
            oof_by_rung[b],
            y,
            statistic=_precision_stat_at(thr_by_rung[a]),
            n_boot=n_boot,
            seed=seed,
        )
        return {
            "delta_auroc": da_pt,
            "delta_auroc_ci": [da_lo, da_hi],
            "delta_precision": dp_pt,
            "delta_precision_ci": [dp_lo, dp_hi],
        }

    paired = {
        "r2_minus_r0": _delta(2, 0),
        "r3_minus_r0": _delta(3, 0),
        "r3_minus_r2": _delta(3, 2),
    }

    # Random-split contrast: ordinary K-fold over per-row ids (not antigen clusters).
    row_ids = np.array([r["pair_id"] for r in rows])
    rand_folds = assign_folds(row_ids, n_splits=5, seed=seed)
    random_split: dict[str, dict[str, float]] = {}
    for k in _RUNGS:
        x, names = rung_matrix(rows, rung=k)
        _, oof_r = fit_rung_model(
            x,
            y,
            rand_folds,
            feature_names=names,
            l2=l2,
            target_precision=target_precision,
        )
        random_split[f"rung{k}"] = {"auroc": auroc(oof_r, y)}

    frozen = models[3]
    return {
        "n": len(rows),
        "n_positive": int((y == 1).sum()),
        "ms_floor_auroc": 0.496,
        "rungs": per_rung,
        "paired_deltas": paired,
        "random_split": random_split,
        "cdr_mapping_failure_rate": cdr_mapping_failure_rate(rows),
        "frozen_rung": "rung3",
        "frozen_model": frozen.__dict__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    result = analyze_indist(
        args.features,
        l2=args.l2,
        target_precision=args.target_precision,
        seed=args.seed,
        n_boot=args.n_boot,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    from mirage.model.ms import MsModel

    MsModel(**result["frozen_model"]).save(args.model_out)

    rungs = result["rungs"]
    print("rung  AUROC   [CI]            delta-AUROC vs R0")
    for k in _RUNGS:
        r = rungs[f"rung{k}"]
        lo, hi = r["auroc_ci"]
        delta = ""
        if k in (2, 3):
            d = result["paired_deltas"][f"r{k}_minus_r0"]
            dlo, dhi = d["delta_auroc_ci"]
            delta = f"  d={d['delta_auroc']:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        print(f"  {k}   {r['auroc']:.3f}  [{lo:.3f}, {hi:.3f}]{delta}")
    print(f"CDR-mapping failure rate: {result['cdr_mapping_failure_rate']:.4f}")
    print(f"Wrote {args.output}; froze rung3 -> {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
