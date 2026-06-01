"""Analyze crystal-independent structural-interface scores.

This is the predictor-agnostic counterpart to the AF2-M confidence analyses:
it reads score CSVs from ``structural_interface`` plus existing labels and
reports AUROC/AP for pose-correctness labels, SAbDab synthetic negatives, and
the EpCAM POS/SCR/OFF transfer check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_STRUCTURAL_METRICS = (
    "value",
    "residue_pair_contacts_4a",
    "residue_pair_contacts_6a",
    "residue_pair_contacts_8a",
    "residue_pair_contact_fraction_4a",
    "residue_pair_contact_fraction_6a",
    "residue_pair_contact_fraction_8a",
    "contacts_per_binder_residue_4a",
    "contacts_per_binder_residue_6a",
    "contacts_per_binder_residue_8a",
    "contacts_per_target_residue_4a",
    "contacts_per_target_residue_6a",
    "contacts_per_target_residue_8a",
    "n_interface_residues_binder",
    "n_interface_residues_target",
    "binder_interface_fraction",
    "target_interface_fraction",
    "atom_contacts_5a",
    "atom_contact_fraction_5a",
    "atom_close_contacts_3a",
    "atom_close_contact_fraction_3a",
    "atom_clashes_2a",
    "atom_clash_fraction_2a",
    "buried_area_proxy_5a",
    "buried_sasa_proxy_a2",
    "buried_sasa_proxy_binder_a2",
    "buried_sasa_proxy_target_a2",
    "buried_sasa_proxy_per_binder_atom",
    "buried_sasa_proxy_per_target_atom",
    "buried_sasa_proxy_per_interface_residue",
    "buried_sasa_balance",
    "mean_binder_atom_exposure_loss",
    "mean_target_atom_exposure_loss",
    "atom_packing_pairs_0_1a_gap",
    "atom_packing_fraction_0_1a_gap",
    "atom_packing_shell_pairs_2a_gap",
    "binder_atom_packing_coverage_0_1a_gap",
    "target_atom_packing_coverage_0_1a_gap",
    "atom_packing_complementarity_score",
    "mean_abs_nearest_surface_gap",
    "shape_complementarity_proxy",
    "hydrophobic_contact_pairs_6a",
    "hydrophobic_contact_fraction_6a",
    "aromatic_contact_pairs_6a",
    "aromatic_contact_fraction_6a",
    "polar_contact_pairs_6a",
    "polar_contact_fraction_6a",
    "charged_contact_pairs_6a",
    "charged_contact_fraction_6a",
    "opposite_charge_contact_pairs_6a",
    "opposite_charge_contact_fraction_6a",
    "same_charge_contact_pairs_6a",
    "same_charge_contact_fraction_6a",
    "mean_interface_residue_distance_8a",
    "median_interface_residue_distance_8a",
    "std_interface_residue_distance_8a",
    "close_residue_contact_fraction_4a_within_8a",
    "min_interchain_heavy_atom_distance",
    "mean_interchain_heavy_atom_distance",
)

_LOWER_IS_BETTER = frozenset(
    {
        "atom_clashes_2a",
        "atom_clash_fraction_2a",
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


def _rmsd_value(row: dict[str, str]) -> float:
    return float(row["value"])


def _dockq_best(row: dict[str, str]) -> float:
    try:
        extras = json.loads(row.get("extras_json", "") or "{}")
    except json.JSONDecodeError:
        return math.nan
    value = extras.get("dockq_best")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _value_or_extra(row: dict[str, Any], metric: str) -> float:
    value = row.get("value") if metric == "value" else row["__extras"].get(metric)
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
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
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(float)
    precision = np.cumsum(labels_sorted) / np.arange(1, labels_sorted.size + 1)
    return float((precision * labels_sorted).sum() / n_pos)


def _metric_rows(
    label_name: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    baseline_ap = float(labels.mean()) if labels.size else math.nan
    for metric in _STRUCTURAL_METRICS:
        values = np.asarray([_value_or_extra(row, metric) for row in rows], dtype=float)
        if metric in _LOWER_IS_BETTER:
            values = -values
        finite = np.isfinite(values)
        finite_labels = labels[finite]
        finite_values = values[finite]
        out.append(
            {
                "comparison": label_name,
                "metric": metric,
                "n": int(finite.sum()),
                "n_positive": int((finite_labels == 1).sum()),
                "n_negative": int((finite_labels == 0).sum()),
                "baseline_ap": baseline_ap,
                "auroc": _auroc(finite_values, finite_labels),
                "ap": _average_precision(finite_values, finite_labels),
            }
        )
    return out


def _pose_rows(
    structural_rows: list[dict[str, Any]],
    rmsd_rows: dict[str, dict[str, str]],
) -> list[dict[str, float | int | str]]:
    rows = [row for row in structural_rows if row["example_id"] in rmsd_rows]
    out: list[dict[str, float | int | str]] = []
    label_specs: tuple[tuple[str, list[int]], ...] = (
        (
            "rmsd<4A",
            [_rmsd_value(rmsd_rows[row["example_id"]]) < 4.0 for row in rows],
        ),
        (
            "rmsd<8A",
            [_rmsd_value(rmsd_rows[row["example_id"]]) < 8.0 for row in rows],
        ),
        (
            "dockq_best>=0.23",
            [_dockq_best(rmsd_rows[row["example_id"]]) >= 0.23 for row in rows],
        ),
    )
    for label_name, labels in label_specs:
        out.extend(_metric_rows(label_name, rows, np.asarray(labels, dtype=int)))
    return out


def _negative_rows(
    label_name: str,
    real_rows: list[dict[str, Any]],
    rmsd_rows: dict[str, dict[str, str]],
    negative_rows: list[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    positives = [
        row
        for row in real_rows
        if row["example_id"] in rmsd_rows and _rmsd_value(rmsd_rows[row["example_id"]]) < 4.0
    ]
    rows = positives + negative_rows
    labels = np.asarray([1] * len(positives) + [0] * len(negative_rows), dtype=int)
    return _metric_rows(label_name, rows, labels)


def _epcam_rows(rows: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    specs = (
        ("POS_vs_all_negatives", labels == "POS", np.isin(labels, ["SCR", "OFF"])),
        ("POS_vs_SCR", labels == "POS", labels == "SCR"),
        ("POS_vs_OFF", labels == "POS", labels == "OFF"),
    )
    out: list[dict[str, float | int | str]] = []
    for label_name, pos_mask, neg_mask in specs:
        keep = pos_mask | neg_mask
        kept_rows = [r for r, k in zip(rows, keep, strict=True) if k]
        out.extend(_metric_rows(label_name, kept_rows, pos_mask[keep].astype(int)))
    return out


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "comparison",
        "metric",
        "n",
        "n_positive",
        "n_negative",
        "baseline_ap",
        "auroc",
        "ap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-structural", type=Path, required=True)
    parser.add_argument("--real-rmsd", type=Path, required=True)
    parser.add_argument("--scramble-structural", type=Path)
    parser.add_argument("--wrong-target-structural", type=Path)
    parser.add_argument("--epcam-structural", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    real_rows = _read_scores(args.real_structural)
    rmsd_rows = _read_index(args.real_rmsd)
    rows = _pose_rows(real_rows, rmsd_rows)
    if args.scramble_structural is not None:
        rows.extend(
            _negative_rows(
                "near_native(rmsd<4A)_vs_scramble",
                real_rows,
                rmsd_rows,
                _read_scores(args.scramble_structural),
            )
        )
    if args.wrong_target_structural is not None:
        rows.extend(
            _negative_rows(
                "near_native(rmsd<4A)_vs_wrong_target",
                real_rows,
                rmsd_rows,
                _read_scores(args.wrong_target_structural),
            )
        )
    if args.epcam_structural is not None:
        rows.extend(_epcam_rows(_read_scores(args.epcam_structural)))
    _write_csv(args.output, rows)
    print(f"Wrote {len(rows)} structural-interface metric rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
