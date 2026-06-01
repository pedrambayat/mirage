"""Phase 2 Champloo/Smorodina classifier: does AF3 geometry beat ipTM alone?

Phase 1 showed raw AF3 ipTM separates cognate from shuffled non-cognate pairs
(AP ~0.197 / AUROC ~0.754 at 1.1% prevalence) and a simple ipTM-only / ipTM+
metadata logistic does not improve on it. Phase 2 adds predictor-agnostic
structural features (clash / interface / contact / exposure / packing) computed
by ``StructuralInterfaceScorer`` on the released AF3 ``model_0`` complexes, and
asks whether combining ipTM with geometry beats confidence alone.

It joins:

* the structural feature CSV (``score_champloo_structures.py`` output), and
* the released AF3 ipTM matrix (per-pair confidence),

by ``example_id = {VHH_PDB}__{ANTIGEN_PDB}``, then evaluates, with AP/AUPRC
(primary) and AUROC (secondary):

* ``raw_iptm`` -- released confidence used directly (no training, split-invariant);
  recomputed on the structure-covered subset for an exact head-to-head.
* ``raw_<clash feature>`` -- the single best clash feature used directly
  (sign-flipped so fewer clashes scores higher), split-invariant.
* ``logistic_iptm`` -- numpy L2 logistic on ipTM alone (Phase 1 sanity check).
* ``logistic_structural`` -- logistic on the structural feature families only.
* ``logistic_combined`` -- logistic on ipTM + structural features.

Learned models use grouped K-fold out-of-fold predictions under three splits:
``random_pair``, ``held_out_vhh``, ``held_out_antigen``.

Use::

    uv run python scripts/analyze_champloo_structural.py \\
        --structural results/published/champloo_af3_structural_interface.csv \\
        --matrix <champloo>/iptm_confidence_scores/iptm_confidence_scores/af3_matrix_clean.csv \\
        --output results/published/champloo_af3_structural_classifier.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Feature families mirror scripts/analyze_structural_logreg.py so the Phase 2
# structural model is the same predictor-agnostic geometry used elsewhere.
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
_STRUCTURAL_FEATURES = (
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


def parse_example_id(example_id: str) -> tuple[str, str]:
    """``{VHH_PDB}__{ANTIGEN_PDB}`` -> ``(vhh_pdb, antigen_pdb)`` (upper-cased)."""
    vhh, _, antigen = example_id.partition("__")
    return vhh.upper(), antigen.upper()


def load_iptm_matrix(path: Path) -> dict[tuple[str, str], float]:
    """Load the ipTM matrix as ``{(vhh_pdb, antigen_pdb): iptm}`` (upper-cased)."""
    out: dict[tuple[str, str], float] = {}
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = [c.strip().upper() for c in header[1:]]
        for raw in reader:
            row_pdb = raw[0].strip().upper()
            for col_pdb, value in zip(cols, raw[1:], strict=True):
                value = value.strip()
                if value == "":
                    continue
                try:
                    iptm = float(value)
                except ValueError:
                    continue
                if not math.isnan(iptm):
                    out[(row_pdb, col_pdb)] = iptm
    return out


def read_structural_scores(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                row["__extras"] = json.loads(row.get("extras_json", "") or "{}")
            except json.JSONDecodeError:
                row["__extras"] = {}
            rows.append(row)
    return rows


def _feature_value(row: dict[str, Any], metric: str) -> float:
    if metric == "iptm":
        value: Any = row.get("iptm")
    else:
        value = row["__extras"].get(metric)
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def feature_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for metric in feature_names:
        direction = -1.0 if metric in _LOWER_IS_BETTER else 1.0
        values = np.asarray([_feature_value(row, metric) for row in rows], dtype=float)
        columns.append(direction * values)
    return np.column_stack(columns)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.size * neg.size))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    if scores.size == 0 or n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(float)
    precision = np.cumsum(labels_sorted) / np.arange(1, labels_sorted.size + 1)
    return float((precision * labels_sorted).sum() / n_pos)


def fit_logistic_regression(
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


def _standardize(x_train: np.ndarray, x_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (x_train - mean) / std, (x_apply - mean) / std


def assign_folds(groups: np.ndarray, *, n_splits: int, seed: int) -> np.ndarray:
    """Assign each row a fold by hashing its group with a fixed seed.

    Grouping by ``pair_id`` (unique) gives ordinary pair-level K-fold; grouping
    by VHH or antigen PDB gives the leakage-controlled splits.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    shuffled = rng.permutation(unique.size)
    fold_of_group = {g: int(shuffled[i] % n_splits) for i, g in enumerate(unique)}
    return np.asarray([fold_of_group[g] for g in groups], dtype=int)


def oof_logistic_scores(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, *, l2: float
) -> np.ndarray:
    out = np.full(y.shape, math.nan, dtype=float)
    for fold in np.unique(folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        y_train = y[train_mask]
        if y_train.size < 2 or np.unique(y_train).size < 2:
            continue
        x_train, x_test = _standardize(x[train_mask], x[test_mask])
        intercept, coef = fit_logistic_regression(x_train, y_train, l2=l2)
        out[test_mask] = intercept + x_test @ coef
    return out


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
        "ap": average_precision(s, y) if n_pos and n_neg else math.nan,
        "auroc": auroc(s, y) if n_pos and n_neg else math.nan,
    }


def evaluate(
    rows: list[dict[str, Any]],
    *,
    predictor: str,
    n_splits: int,
    seed: int,
    l2: float,
) -> list[dict[str, Any]]:
    labels = np.asarray([int(r["label_int"]) for r in rows], dtype=int)
    vhh_pdb = np.asarray([r["vhh_pdb"] for r in rows])
    antigen_pdb = np.asarray([r["antigen_pdb"] for r in rows])
    pair_id = np.asarray([r["example_id"] for r in rows])

    iptm = feature_matrix(rows, ("iptm",))
    structural = feature_matrix(rows, _STRUCTURAL_FEATURES)
    combined = feature_matrix(rows, ("iptm", *_STRUCTURAL_FEATURES))

    out: list[dict[str, Any]] = []
    # Split-invariant single-feature baselines.
    out.append(_metric_row("all", "raw_iptm", "iptm", iptm[:, 0], labels, predictor=predictor))
    for clash in _CLASH_FEATURES:
        col = feature_matrix(rows, (clash,))[:, 0]
        out.append(_metric_row("all", f"raw_{clash}", clash, col, labels, predictor=predictor))

    split_groups = {
        "random_pair": pair_id,
        "held_out_vhh": vhh_pdb,
        "held_out_antigen": antigen_pdb,
    }
    models = {
        "logistic_iptm": (iptm, "iptm"),
        "logistic_structural": (structural, ";".join(_STRUCTURAL_FEATURES)),
        "logistic_combined": (combined, ";".join(("iptm", *_STRUCTURAL_FEATURES))),
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


def build_rows(
    structural_rows: list[dict[str, Any]],
    iptm: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Join structural rows with ipTM; drop missing structures / missing ipTM."""
    rows: list[dict[str, Any]] = []
    for row in structural_rows:
        if row["__extras"].get("missing") == "prediction":
            continue
        example_id = row["example_id"]
        vhh_pdb, antigen_pdb = parse_example_id(example_id)
        key = (vhh_pdb, antigen_pdb)
        if key not in iptm:
            continue
        rows.append(
            {
                "example_id": example_id,
                "vhh_pdb": vhh_pdb,
                "antigen_pdb": antigen_pdb,
                "label_int": 1 if str(row.get("label", "")).strip().upper() == "COGNATE" else 0,
                "iptm": iptm[key],
                "__extras": row["__extras"],
            }
        )
    return rows


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
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
    print(f"{'split':<16} {'model':<22} {'n':>6} {'pos':>4} {'ap':>7} {'auroc':>7} {'base_ap':>8}")
    for row in rows:
        print(
            f"{row['split']:<16} {row['model']:<22} {int(row['n']):6d} "
            f"{int(row['n_positive']):4d} {float(row['ap']):7.3f} "
            f"{float(row['auroc']):7.3f} {float(row['baseline_ap']):8.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictor", type=str, default="af3")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    structural_rows = read_structural_scores(args.structural)
    iptm = load_iptm_matrix(args.matrix)
    rows = build_rows(structural_rows, iptm)
    if not rows:
        raise SystemExit("no joined rows: check structural CSV and matrix paths")

    metrics = evaluate(
        rows, predictor=args.predictor, n_splits=args.n_splits, seed=args.seed, l2=args.l2
    )
    _write_metrics(args.output, metrics)
    _print_report(metrics)
    n_pos = sum(r["label_int"] for r in rows)
    print(json.dumps({"joined_pairs": len(rows), "positives": n_pos, "metric_rows": len(metrics)}))
    print(f"Wrote {len(metrics)} metric rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
