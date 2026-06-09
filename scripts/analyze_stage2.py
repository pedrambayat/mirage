"""Compute gate metrics for the stage-2 OOF scores and compare to the floor.

Reads the OOF scores CSV produced by scripts/train_stage2.py (in the esm env)
and computes, per rung, AUROC + a fixed-precision operating point + PPV sweep +
a bootstrap CI on AUROC, reusing mirage's numpy eval/gate.py. Carries the stage-1
bilinear floor (rung3 AUROC) for the head-to-head.

Use:
    uv run python scripts/analyze_stage2.py \\
      --scores data/staged/sabdab/stage2_oof_scores.csv \\
      --baseline results/published/sabdab_baseline.json \\
      --output results/published/sabdab_stage2.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.gate import auroc, bootstrap_ci, choose_threshold_for_precision, summary_dict


def analyze_scores(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    target_precision: float,
    seed: int,
) -> dict[str, Any]:
    finite = np.isfinite(scores)
    s, y = scores[finite], labels[finite]
    thr = choose_threshold_for_precision(s, y, target_precision=target_precision)
    out = summary_dict(s, y, threshold=thr)
    out["auroc"] = auroc(s, y)
    out["auroc_ci"] = bootstrap_ci(auroc, s, y, n_boot=1000, seed=seed)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    with args.scores.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows], dtype=int)

    result: dict[str, Any] = {"n": int(y.size), "n_positive": int((y == 1).sum())}
    for rung in ("score_xattn", "score_mlp"):
        s = np.array([float(r[rung]) for r in rows], dtype=float)
        result[rung] = analyze_scores(s, y, target_precision=args.target_precision, seed=args.seed)

    floor = json.loads(args.baseline.read_text())
    result["bilinear_floor_auroc"] = floor["rung3_bilinear"]["auroc"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for rung in ("score_xattn", "score_mlp"):
        m = result[rung]
        print(f"{rung}: AUROC={m['auroc']:.3f} CI={tuple(round(v, 3) for v in m['auroc_ci'])}")
    print(f"bilinear floor AUROC={result['bilinear_floor_auroc']:.3f}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
