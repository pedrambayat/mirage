"""Analyze a dependency-free multi-feature structural baseline.

This script combines predictor-agnostic ``structural_interface`` features with
a small numpy-only L2 logistic regression. It mirrors the AF2-M confidence
logistic baseline but reads only published structural score CSVs and existing
RMSD labels.

Use::

    uv run python scripts/analyze_structural_logreg.py \\
        --real-structural results/published/sabdab_structural_interface_n200.csv \\
        --real-rmsd results/published/sabdab_af2m_rmsd_n200.csv \\
        --scramble-structural results/published/sabdab_structural_interface_scrambles.csv \\
        --wrong-target-structural results/published/sabdab_structural_interface_wrong_targets.csv \\
        --epcam-structural results/published/epcam_structural_interface.csv \\
        --output results/published/structural_interface_logreg.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_CONTACT_FEATURES = (
    "contacts_per_binder_residue_4a",
    "contacts_per_binder_residue_6a",
    "contacts_per_binder_residue_8a",
    "contacts_per_target_residue_4a",
    "contacts_per_target_residue_6a",
    "contacts_per_target_residue_8a",
    "binder_interface_fraction",
    "target_interface_fraction",
    "atom_contact_fraction_5a",
)

_CLASH_FEATURES = (
    "atom_close_contact_fraction_3a",
    "atom_clash_fraction_2a",
    "atom_clashes_2a",
)

_CHEMISTRY_FEATURES = (
    "hydrophobic_contact_fraction_6a",
    "aromatic_contact_fraction_6a",
    "polar_contact_fraction_6a",
    "charged_contact_fraction_6a",
    "opposite_charge_contact_fraction_6a",
    "same_charge_contact_fraction_6a",
)

_INTERFACE_DISTANCE_FEATURES = (
    "mean_interface_residue_distance_8a",
    "median_interface_residue_distance_8a",
    "std_interface_residue_distance_8a",
    "close_residue_contact_fraction_4a_within_8a",
    "min_interchain_heavy_atom_distance",
    "mean_interchain_heavy_atom_distance",
)

_EXPOSURE_FEATURES = (
    "buried_sasa_proxy_per_binder_atom",
    "buried_sasa_proxy_per_target_atom",
    "buried_sasa_proxy_per_interface_residue",
    "buried_sasa_balance",
    "mean_binder_atom_exposure_loss",
    "mean_target_atom_exposure_loss",
)

_PACKING_FEATURES = (
    "atom_packing_fraction_0_1a_gap",
    "binder_atom_packing_coverage_0_1a_gap",
    "target_atom_packing_coverage_0_1a_gap",
    "atom_packing_complementarity_score",
    "mean_abs_nearest_surface_gap",
    "shape_complementarity_proxy",
)

_DEFAULT_FEATURES = (
    *_CONTACT_FEATURES,
    *_CLASH_FEATURES,
    *_CHEMISTRY_FEATURES,
    *_INTERFACE_DISTANCE_FEATURES,
    *_EXPOSURE_FEATURES,
    *_PACKING_FEATURES,
)

_LOWER_IS_BETTER = frozenset(
    {
        "atom_close_contact_fraction_3a",
        "atom_clash_fraction_2a",
        "atom_clashes_2a",
        "mean_interface_residue_distance_8a",
        "median_interface_residue_distance_8a",
        "std_interface_residue_distance_8a",
        "min_interchain_heavy_atom_distance",
        "mean_interchain_heavy_atom_distance",
        "mean_abs_nearest_surface_gap",
    }
)


def _read_scores(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                row["__extras"] = json.loads(row.get("extras_json", "") or "{}")
            except json.JSONDecodeError:
                row["__extras"] = {}
            rows.append(row)
    return rows


def _read_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fh:
        return {row["example_id"]: row for row in csv.DictReader(fh)}


def _value_or_extra(row: dict[str, Any], metric: str) -> float:
    value = row.get("value") if metric == "value" else row["__extras"].get(metric)
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _rmsd_value(row: dict[str, str]) -> float:
    try:
        return float(row["value"])
    except (TypeError, ValueError):
        return math.nan


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
    if scores.size == 0 or n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(float)
    precision = np.cumsum(labels_sorted) / np.arange(1, labels_sorted.size + 1)
    return float((precision * labels_sorted).sum() / n_pos)


def _feature_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for metric in feature_names:
        direction = -1.0 if metric in _LOWER_IS_BETTER else 1.0
        values = np.asarray([_value_or_extra(row, metric) for row in rows], dtype=float)
        columns.append(direction * values)
    return np.column_stack(columns)


def _fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> tuple[float, np.ndarray]:
    beta = np.zeros(x.shape[1] + 1, dtype=float)
    design = np.column_stack([np.ones(x.shape[0], dtype=float), x])
    penalty = np.diag(np.concatenate([[0.0], np.full(x.shape[1], l2, dtype=float)]))
    prev_loss = math.inf
    for _ in range(max_iter):
        logits = np.clip(design @ beta, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(pred * (1.0 - pred), 1e-9)
        gradient = design.T @ (pred - y) + penalty @ beta
        hessian = (design.T * weights) @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        loss = float(
            -np.sum(y * np.log(pred + 1e-12) + (1.0 - y) * np.log(1.0 - pred + 1e-12))
            + 0.5 * float(beta @ penalty @ beta)
        )
        if abs(prev_loss - loss) < tolerance or float(np.linalg.norm(step)) < tolerance:
            break
        prev_loss = loss
    return float(beta[0]), beta[1:]


def _standardize_train_apply(
    x_train: np.ndarray, x_apply: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (x_train - mean) / std, (x_apply - mean) / std, mean, std


def _logistic_loo_scores(x: np.ndarray, y: np.ndarray, *, l2: float) -> np.ndarray:
    out = np.full(y.shape, math.nan, dtype=float)
    for i in range(y.size):
        train_mask = np.ones(y.size, dtype=bool)
        train_mask[i] = False
        y_train = y[train_mask]
        if y_train.size < 2 or np.unique(y_train).size < 2:
            continue
        x_train, x_test, _, _ = _standardize_train_apply(x[train_mask], x[i : i + 1])
        intercept, coef = _fit_logistic_regression(x_train, y_train, l2=l2)
        out[i] = float(intercept + x_test[0] @ coef)
    return out


def _logistic_row(
    comparison: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    l2: float,
) -> dict[str, Any]:
    x_all = _feature_matrix(rows, feature_names)
    finite = np.isfinite(x_all).all(axis=1)
    x = x_all[finite]
    y = labels[finite].astype(float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out: dict[str, Any] = {
        "comparison": comparison,
        "model": "structural_logistic_regression_loo",
        "features": ";".join(feature_names),
        "n": int(y.size),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "baseline_ap": float(y.mean()) if y.size else math.nan,
        "auroc": math.nan,
        "ap": math.nan,
        "l2": l2,
        "intercept": math.nan,
        "coefficients_json": "{}",
    }
    if y.size < 3 or n_pos == 0 or n_neg == 0:
        return out

    scores = _logistic_loo_scores(x, y, l2=l2)
    scored = np.isfinite(scores)
    if scored.sum() >= 2:
        out["auroc"] = _auroc(scores[scored], y[scored].astype(int))
        out["ap"] = _average_precision(scores[scored], y[scored].astype(int))

    x_train, _, mean, std = _standardize_train_apply(x, x)
    intercept, coef = _fit_logistic_regression(x_train, y, l2=l2)
    raw_coef = coef / std
    out["intercept"] = intercept - float((mean / std) @ coef)
    out["coefficients_json"] = json.dumps(
        {m: float(c) for m, c in zip(feature_names, raw_coef, strict=True)},
        sort_keys=True,
    )
    return out


def _pose_rows(
    structural_rows: list[dict[str, Any]],
    rmsd_rows: dict[str, dict[str, str]],
    *,
    feature_names: tuple[str, ...],
    l2: float,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in structural_rows
        if row["example_id"] in rmsd_rows
        and math.isfinite(_rmsd_value(rmsd_rows[row["example_id"]]))
    ]
    specs = (
        ("rmsd<4A", 4.0),
        ("rmsd<8A", 8.0),
    )
    return [
        _logistic_row(
            name,
            rows,
            np.asarray(
                [_rmsd_value(rmsd_rows[row["example_id"]]) < cutoff for row in rows],
                dtype=int,
            ),
            feature_names=feature_names,
            l2=l2,
        )
        for name, cutoff in specs
    ]


def _negative_row(
    comparison: str,
    real_rows: list[dict[str, Any]],
    rmsd_rows: dict[str, dict[str, str]],
    negative_rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    l2: float,
) -> dict[str, Any]:
    positives = [
        row
        for row in real_rows
        if row["example_id"] in rmsd_rows and _rmsd_value(rmsd_rows[row["example_id"]]) < 4.0
    ]
    rows = positives + negative_rows
    labels = np.asarray([1] * len(positives) + [0] * len(negative_rows), dtype=int)
    return _logistic_row(comparison, rows, labels, feature_names=feature_names, l2=l2)


def _epcam_rows(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    l2: float,
) -> list[dict[str, Any]]:
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    specs = (
        ("POS_vs_all_negatives", labels == "POS", np.isin(labels, ["SCR", "OFF"])),
        ("POS_vs_SCR", labels == "POS", labels == "SCR"),
        ("POS_vs_OFF", labels == "POS", labels == "OFF"),
    )
    out: list[dict[str, Any]] = []
    for comparison, pos_mask, neg_mask in specs:
        keep = pos_mask | neg_mask
        kept_rows = [row for row, keep_row in zip(rows, keep, strict=True) if keep_row]
        out.append(
            _logistic_row(
                comparison,
                kept_rows,
                pos_mask[keep].astype(int),
                feature_names=feature_names,
                l2=l2,
            )
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "comparison",
        "model",
        "features",
        "n",
        "n_positive",
        "n_negative",
        "baseline_ap",
        "auroc",
        "ap",
        "l2",
        "intercept",
        "coefficients_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_report(rows: list[dict[str, Any]]) -> None:
    print(f"{'comparison':<40} {'n':>5} {'pos':>5} {'neg':>5} {'auroc':>7} {'ap':>7}")
    for row in rows:
        print(
            f"{row['comparison']:<40} {int(row['n']):5d} "
            f"{int(row['n_positive']):5d} {int(row['n_negative']):5d} "
            f"{float(row['auroc']):7.3f} {float(row['ap']):7.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-structural", type=Path, required=True)
    parser.add_argument("--real-rmsd", type=Path, required=True)
    parser.add_argument("--scramble-structural", type=Path)
    parser.add_argument("--wrong-target-structural", type=Path)
    parser.add_argument("--epcam-structural", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(_DEFAULT_FEATURES),
        choices=_DEFAULT_FEATURES,
        help="Structural metrics to use as logistic-regression features.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1.0,
        help="L2 regularization strength for the logistic-regression baseline.",
    )
    args = parser.parse_args()

    feature_names = tuple(args.features)
    real_rows = _read_scores(args.real_structural)
    rmsd_rows = _read_index(args.real_rmsd)
    rows = _pose_rows(real_rows, rmsd_rows, feature_names=feature_names, l2=args.l2)
    if args.scramble_structural is not None:
        rows.append(
            _negative_row(
                "near_native(rmsd<4A)_vs_scramble",
                real_rows,
                rmsd_rows,
                _read_scores(args.scramble_structural),
                feature_names=feature_names,
                l2=args.l2,
            )
        )
    if args.wrong_target_structural is not None:
        rows.append(
            _negative_row(
                "near_native(rmsd<4A)_vs_wrong_target",
                real_rows,
                rmsd_rows,
                _read_scores(args.wrong_target_structural),
                feature_names=feature_names,
                l2=args.l2,
            )
        )
    if args.epcam_structural is not None:
        rows.extend(
            _epcam_rows(
                _read_scores(args.epcam_structural),
                feature_names=feature_names,
                l2=args.l2,
            )
        )

    _write_csv(args.output, rows)
    _print_report(rows)
    print(f"Wrote {len(rows)} structural logistic-regression rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
