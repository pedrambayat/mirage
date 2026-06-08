"""M-C EpCAM real-negative canary: apply the frozen SAbDab gate to EpCAM, unchanged.

Re-fits the SAbDab rung-0 and rung-3 gates on the committed SAbDab feature CSV
(full-fit; rung-3 reproduces mc_sabdab_model.json), then applies them — never
trained on EpCAM — to the EpCAM feature CSV. Three reads:

  1. PRIMARY (threshold-free): AUROC of rung-0 (ipTM) and rung-3 on
     designed-binders-vs-shuffled, with the paired Delta(R3-R0) bootstrap CI.
  2. CALIBRATION: the frozen rung-3 (and rung-0) gate at its SAbDab P=0.90
     threshold applied unchanged -> precision/recall/specificity + CIs.
  3. SECONDARY (exploratory): among the 14 positives, the rung-3 mirage score for
     the 8 functional vs 6 non-functional. N=14, killing != binding -> descriptive
     only, NO inferential claim.

Use::

    uv run python scripts/analyze_mc_epcam.py \\
        --sabdab-features data/staged/mc/sabdab_features.csv \\
        --epcam-features data/staged/mc/epcam_features.csv \\
        --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \\
        --output results/published/mc_epcam_canary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.gate import auroc, paired_delta_bootstrap
from mirage.eval.orthogonal import evaluate_frozen_gate
from mirage.features.mc_rungs import (
    fit_rung_model,
    folds_array,
    labels_array,
    read_feature_csv,
    rung_matrix,
)
from mirage.model.ms import MsModel

_TRANSFER_RUNG = 3
_CONTRAST_RUNG = 0


def vhh_id_from_pair(pair_id: str) -> str:
    """'epcam-10__epcam' -> '10'."""
    return pair_id.removeprefix("epcam-").split("__")[0]


def killing_label_map(killing_labels: Path) -> dict[str, int]:
    """vhh_id -> 1 (Good/functional) / 0 (Bad/non-functional)."""
    with killing_labels.open(newline="") as fh:
        return {r["vhh_id"]: (1 if r["label"].strip() == "Good" else 0) for r in csv.DictReader(fh)}


def fit_sabdab_rung(
    sab_rows: list[dict[str, str]], *, rung: int, l2: float, target_precision: float
) -> MsModel:
    """Full-fit a frozen SAbDab rung gate (rung-3 reproduces mc_sabdab_model.json)."""
    x, names = rung_matrix(sab_rows, rung=rung)
    y = labels_array(sab_rows)
    folds = folds_array(sab_rows)
    model, _ = fit_rung_model(
        x, y, folds, feature_names=names, l2=l2, target_precision=target_precision
    )
    return model


def analyze_epcam(
    sabdab_features: Path,
    epcam_features: Path,
    killing_labels: Path,
    *,
    l2: float,
    target_precision: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    sab_rows = read_feature_csv(sabdab_features)
    epc_rows = read_feature_csv(epcam_features)
    y = labels_array(epc_rows)

    m0 = fit_sabdab_rung(sab_rows, rung=_CONTRAST_RUNG, l2=l2, target_precision=target_precision)
    m3 = fit_sabdab_rung(sab_rows, rung=_TRANSFER_RUNG, l2=l2, target_precision=target_precision)

    x0, _ = rung_matrix(epc_rows, rung=_CONTRAST_RUNG)
    x3, _ = rung_matrix(epc_rows, rung=_TRANSFER_RUNG)
    s0 = m0.predict_logit(x0)
    s3 = m3.predict_logit(x3)

    # Read 1 — primary, threshold-free
    delta_point, delta_lo, delta_hi = paired_delta_bootstrap(
        s3, s0, y, statistic=auroc, n_boot=n_boot, seed=seed
    )
    primary = {
        "rung0_auroc": auroc(s0, y),
        "rung3_auroc": auroc(s3, y),
        "delta_auroc_r3_minus_r0": {"point": delta_point, "ci": [delta_lo, delta_hi]},
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }

    # Read 2 — calibration (frozen SAbDab P=target_precision threshold, unchanged)
    calibration = {
        "rung3": evaluate_frozen_gate(m3, x3, y, n_boot=n_boot, seed=seed),
        "rung0_contrast": evaluate_frozen_gate(m0, x0, y, n_boot=n_boot, seed=seed),
        "frozen_threshold_rung3": m3.threshold,
        "frozen_threshold_rung0": m0.threshold,
    }

    # Read 3 — secondary, exploratory (functional vs non-functional, N=14)
    kmap = killing_label_map(killing_labels)
    pos_idx = [i for i, r in enumerate(epc_rows) if r["label"] == "1"]
    func = np.array([kmap[vhh_id_from_pair(epc_rows[i]["pair_id"])] for i in pos_idx], dtype=int)
    pos_scores = s3[np.array(pos_idx, dtype=int)]
    secondary = {
        "caveat": "N=14, killing != binding, descriptive only — no inferential claim",
        "n_functional": int((func == 1).sum()),
        "n_nonfunctional": int((func == 0).sum()),
        "killing_auroc_rung3": auroc(pos_scores, func),
        "functional_scores": [float(v) for v in pos_scores[func == 1]],
        "nonfunctional_scores": [float(v) for v in pos_scores[func == 0]],
    }

    return {"primary": primary, "calibration": calibration, "secondary_killing": secondary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sabdab-features", type=Path, required=True)
    parser.add_argument("--epcam-features", type=Path, required=True)
    parser.add_argument("--killing-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    result = analyze_epcam(
        args.sabdab_features,
        args.epcam_features,
        args.killing_labels,
        l2=args.l2,
        target_precision=args.target_precision,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    p = result["primary"]
    print(f"EpCAM n={p['n']} ({p['n_positive']} pos / {p['n_negative']} neg)")
    print(f"  rung0 (ipTM) AUROC: {round(p['rung0_auroc'], 3)}")
    print(f"  rung3 (full)  AUROC: {round(p['rung3_auroc'], 3)}")
    d = p["delta_auroc_r3_minus_r0"]
    print(f"  Delta(R3-R0): {round(d['point'], 3)} CI {[round(c, 3) for c in d['ci']]}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
