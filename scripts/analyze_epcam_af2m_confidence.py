"""Analyze AF2-M confidence on the SNAP EpCAM designed-binder benchmark.

The EpCAM set has no crystal pose for per-example RMSD. This script instead
tests whether AF2-M confidence separates known EpCAM binders (POS) from the two
negative sets already in SNAP:

- SCR: CDR-scrambled variants of EpCAM binders.
- OFF: literature VHHs whose real target is not EpCAM.

Use::

    uv run python scripts/analyze_epcam_af2m_confidence.py \\
        --confidence results/published/epcam_af2m_confidence.csv \\
        --output results/published/epcam_af2m_confidence_label_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

_CONFIDENCE_METRICS = (
    "iptm",
    "ptm",
    "ranking_confidence",
    "iptm_over_ptm",
    "plddt_full_mean",
    "plddt_binder_mean",
    "plddt_target_mean",
    "plddt_interface_mean",
    "pae_interchain_mean",
    "pae_interchain_max",
    "pae_interface_mean",
)

_LOWER_IS_BETTER = frozenset({"pae_interchain_mean", "pae_interchain_max", "pae_interface_mean"})
_BOOTSTRAP_SEED = 0


def _read_confidence(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                row["__extras"] = json.loads(row.get("extras_json", "") or "{}")
            except json.JSONDecodeError:
                row["__extras"] = {}
            rows.append(row)
    return rows


def _value_or_extra(row: dict[str, Any], key: str) -> float:
    value = row.get("value") if key == "iptm" else row["__extras"].get(key)
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman_rho(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size < 2:
        return math.nan
    rx = _average_ranks(scores)
    ry = _average_ranks(labels)
    sx = rx - rx.mean()
    sy = ry - ry.mean()
    denom = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    if denom == 0.0:
        return math.nan
    return float((sx * sy).sum() / denom)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.size * neg.size))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(float)
    precision = np.cumsum(labels_sorted) / np.arange(1, labels_sorted.size + 1)
    return float((precision * labels_sorted).sum() / n_pos)


def _stratified_bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: Any,
    *,
    n_bootstrap: int,
) -> tuple[float, float]:
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if n_bootstrap <= 0 or pos_idx.size == 0 or neg_idx.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    vals = np.full(n_bootstrap, math.nan)
    for i in range(n_bootstrap):
        sample_idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        vals[i] = metric_fn(scores[sample_idx], labels[sample_idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan, math.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _label_specs(rows: list[dict[str, Any]]) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    return (
        ("POS_vs_all_negatives", labels == "POS", np.isin(labels, ["SCR", "OFF"])),
        ("POS_vs_SCR", labels == "POS", labels == "SCR"),
        ("POS_vs_OFF", labels == "POS", labels == "OFF"),
    )


def _metric_rows(
    rows: list[dict[str, Any]], *, n_bootstrap: int
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for comparison, pos_mask, neg_mask in _label_specs(rows):
        keep = pos_mask | neg_mask
        labels = pos_mask[keep].astype(int)
        baseline_ap = float(labels.mean()) if labels.size else math.nan
        for metric in _CONFIDENCE_METRICS:
            values = np.asarray([_value_or_extra(row, metric) for row in rows], dtype=float)[keep]
            finite = np.isfinite(values)
            n_pos = int((labels[finite] == 1).sum())
            n_neg = int((labels[finite] == 0).sum())
            result: dict[str, float | int | str] = {
                "comparison": comparison,
                "metric": metric,
                "n": int(finite.sum()),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "baseline_ap": baseline_ap,
                "spearman_rho": math.nan,
                "auroc": math.nan,
                "auroc_ci_low": math.nan,
                "auroc_ci_high": math.nan,
                "ap": math.nan,
                "ap_ci_low": math.nan,
                "ap_ci_high": math.nan,
            }
            if n_pos and n_neg:
                direction = -1.0 if metric in _LOWER_IS_BETTER else 1.0
                scores = direction * values[finite]
                lab = labels[finite]
                result["spearman_rho"] = _spearman_rho(scores, lab.astype(float))
                result["auroc"] = _auroc(scores, lab)
                result["ap"] = _average_precision(scores, lab)
                auroc_low, auroc_high = _stratified_bootstrap_ci(
                    scores, lab, _auroc, n_bootstrap=n_bootstrap
                )
                ap_low, ap_high = _stratified_bootstrap_ci(
                    scores, lab, _average_precision, n_bootstrap=n_bootstrap
                )
                result["auroc_ci_low"] = auroc_low
                result["auroc_ci_high"] = auroc_high
                result["ap_ci_low"] = ap_low
                result["ap_ci_high"] = ap_high
            out.append(result)
    return out


def _write_csv(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_report(rows: list[dict[str, float | int | str]]) -> None:
    print(
        f"{'comparison':<20} {'metric':<28} {'n':>4} {'pos':>4} {'neg':>4} {'auroc':>8} {'ap':>8}"
    )
    for comparison in ("POS_vs_all_negatives", "POS_vs_SCR", "POS_vs_OFF"):
        subset = [r for r in rows if r["comparison"] == comparison]
        ranked = sorted(subset, key=lambda r: float(r["ap"]), reverse=True)
        for row in ranked[:5]:
            print(
                f"{row['comparison']:<20} {row['metric']:<28} {row['n']:>4} "
                f"{row['n_positive']:>4} {row['n_negative']:>4} "
                f"{float(row['auroc']):>8.3f} {float(row['ap']):>8.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    rows = _read_confidence(args.confidence)
    if not rows:
        raise SystemExit(f"no rows in {args.confidence}")
    metric_rows = _metric_rows(rows, n_bootstrap=args.bootstrap)
    _write_csv(args.output, metric_rows)
    _print_report(metric_rows)
    print(f"\nWrote {len(metric_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
