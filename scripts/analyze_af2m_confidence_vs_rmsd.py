"""Joins published AF2-M confidence + RMSD + DockQ CSVs and computes
Spearman rho, AUROC, and AP of each confidence metric vs three binary
near-native labels (RMSD<4Å, RMSD<8Å, dockq_best≥0.23). Also emits
RMSD-bucket-wise summary tables (mean ± std of each confidence metric
per pose-quality bucket) for direct comparison with SNAP's benchmark
table.

When ``--scramble-confidence`` or ``--wrong-target-confidence`` points
at a sibling confidence CSV from negative controls, an additional
analysis is run: AUROC + AP for
``{near-native real binder (rmsd<cutoff)} vs {negative control}`` on
each confidence metric. Smorodina et al. 2026 (the mirage thesis paper)
argues for PR-AUC under class imbalance; both AUROC and AP are reported
so the two can be compared directly.

When ``--binder-format-stratify`` is set, the per-label table is sliced
by binder_format too (VHH / Fab / scFv), addressing the open-item from
the N=200 baseline that the aggregated AUROC may hide format-specific
behavior — and matching the GPCR-vs-soluble stratification finding in
Harvey et al. 2026.

Use::

    uv run python scripts/analyze_af2m_confidence_vs_rmsd.py \\
        --confidence results/published/sabdab_af2m_confidence_n200.csv \\
        --rmsd       results/published/sabdab_af2m_rmsd_n200.csv \\
        --dockq      results/published/sabdab_af2m_rmsd_n200_dockq.csv \\
        --output     results/published/sabdab_af2m_confidence_vs_rmsd_n200.csv \\
        --scramble-confidence results/published/sabdab_af2m_confidence_scrambles.csv \\
        --wrong-target-confidence results/published/sabdab_af2m_confidence_wrong_targets.csv \\
        --binder-format-stratify

Stats hand-rolled in numpy — mirage deliberately avoids scipy / sklearn
in the core env. Spearman uses average-rank tie handling; AUROC uses
the Mann-Whitney U identity with a 0.5 weight for ties; AP uses the
scikit-learn-equivalent step-precision integral. The optional logistic
regression baseline is also numpy-only and reports leave-one-out scores.
The optional logistic ablation output reports the same leave-one-out
metric for single-feature models and PAE+pLDDT two-feature candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

# For PAE metrics the "good" direction is lower (smaller predicted aligned
# error). Flip the sign before correlating / computing AUROC so that the
# directionality is consistent across all metrics ("higher is better").
_LOWER_IS_BETTER = frozenset({"pae_interchain_mean", "pae_interchain_max", "pae_interface_mean"})

_DEFAULT_LOGISTIC_FEATURES = (
    "iptm",
    "ptm",
    "ranking_confidence",
    "iptm_over_ptm",
    "plddt_binder_mean",
    "plddt_interface_mean",
    "pae_interchain_mean",
    "pae_interface_mean",
    "pae_interchain_max",
)

_PLDDT_FEATURES = (
    "plddt_full_mean",
    "plddt_binder_mean",
    "plddt_target_mean",
    "plddt_interface_mean",
)

_PAE_FEATURES = (
    "pae_interchain_mean",
    "pae_interchain_max",
    "pae_interface_mean",
)

_BOOTSTRAP_SEED = 0


def _read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                r["__extras"] = json.loads(r.get("extras_json", "") or "{}")
            except json.JSONDecodeError:
                r["__extras"] = {}
            rows.append(r)
    return rows


def _value_or_extra(row: dict[str, Any], key: str) -> float:
    if key == "_value":
        v = row.get("value")
    else:
        v = row["__extras"].get(key)
    try:
        if v is None or v == "":
            return math.nan
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with average-rank tie handling. Returns NaN if undefined."""
    if x.size < 2 or y.size < 2:
        return math.nan
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    sx = rx - rx.mean()
    sy = ry - ry.mean()
    denom = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    if denom == 0:
        return math.nan
    return float((sx * sy).sum() / denom)


def _average_ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    i = 0
    n = a.size
    sorted_vals = a[order]
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for a higher-score-means-more-positive convention. Mann-Whitney U."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    # For each pos, count negs with strictly lower score + 0.5 * ties.
    n_pos = pos.size
    n_neg = neg.size
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (PR-AUC) under the higher-score-means-more-positive convention.

    Matches sklearn.metrics.average_precision_score: a sum of
    precision-at-each-positive over the total number of positives, which
    is the step-interpolated PR-AUC.

    Returns NaN if there are no positives.
    """
    if scores.size == 0:
        return math.nan
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(float)
    cum_tp = np.cumsum(labels_sorted)
    cum_fp = np.cumsum(1.0 - labels_sorted)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    # Sum precision only at positions where we hit a true positive,
    # divide by total positives. Equivalent to ∫ P dR with step interp.
    return float((precision * labels_sorted).sum() / n_pos)


def _stratified_bootstrap_metric_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: Any,
    *,
    n_bootstrap: int,
    seed: int = _BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI from class-stratified bootstrap resamples."""
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if n_bootstrap <= 0 or pos_idx.size == 0 or neg_idx.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = np.full(n_bootstrap, math.nan, dtype=float)
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
    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1.0 - alpha / 2.0))
    return lo, hi


def _join(
    confidence: list[dict[str, Any]],
    rmsd: list[dict[str, Any]],
    dockq: list[dict[str, Any]],
) -> tuple[
    list[str],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return (ids, conf_by_metric, rmsd, dockq_best, binder_format) arrays.

    Joins on ``example_id``. Drops rows missing the headline RMSD or
    confidence value. Rows missing dockq_best get NaN in that array.
    ``binder_format`` mirrors the column on the confidence CSV (fall
    back to the rmsd CSV if absent) — used for format-stratified
    reporting.
    """
    by_id_conf = {r["example_id"]: r for r in confidence}
    by_id_rmsd = {r["example_id"]: r for r in rmsd}
    by_id_dockq = {r["example_id"]: r for r in dockq}

    ids: list[str] = []
    conf_vals: dict[str, list[float]] = {m: [] for m in _CONFIDENCE_METRICS}
    rmsd_vals: list[float] = []
    dockq_vals: list[float] = []
    fmt_vals: list[str] = []
    for eid in by_id_conf:
        cr = by_id_conf[eid]
        rr = by_id_rmsd.get(eid)
        if rr is None:
            continue
        rmsd_v = _value_or_extra(rr, "_value")
        if math.isnan(rmsd_v):
            continue
        # Confidence must at least have the headline iPTM (value).
        if math.isnan(_value_or_extra(cr, "_value")):
            continue
        ids.append(eid)
        for m in _CONFIDENCE_METRICS:
            conf_vals[m].append(_value_or_extra(cr, m))
        rmsd_vals.append(rmsd_v)
        if (dr := by_id_dockq.get(eid)) is not None:
            dockq_vals.append(_value_or_extra(dr, "dockq_best"))
        else:
            dockq_vals.append(math.nan)
        fmt_vals.append(str(cr.get("binder_format") or rr.get("binder_format") or ""))

    return (
        ids,
        {m: np.asarray(conf_vals[m], dtype=float) for m in _CONFIDENCE_METRICS},
        np.asarray(rmsd_vals, dtype=float),
        np.asarray(dockq_vals, dtype=float),
        np.asarray(fmt_vals, dtype=object),
    )


def _per_label_table(
    confidence: dict[str, np.ndarray],
    label_name: str,
    labels: np.ndarray,
    stratum: str = "all",
    bootstrap: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    valid_mask = ~np.isnan(labels)
    labels_valid = labels[valid_mask].astype(int)
    n_pos = int((labels_valid == 1).sum())
    n_neg = int((labels_valid == 0).sum())
    for m in _CONFIDENCE_METRICS:
        vals = confidence[m][valid_mask]
        finite = ~np.isnan(vals)
        if finite.sum() < 2 or n_pos == 0 or n_neg == 0:
            out.append(
                {
                    "stratum": stratum,
                    "label": label_name,
                    "metric": m,
                    "n": int(finite.sum()),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "spearman_rho": math.nan,
                    "auroc": math.nan,
                    "auroc_ci_low": math.nan,
                    "auroc_ci_high": math.nan,
                    "ap": math.nan,
                    "ap_ci_low": math.nan,
                    "ap_ci_high": math.nan,
                }
            )
            continue
        v = vals[finite]
        lab = labels_valid[finite]
        # Direction: for PAE metrics ("lower is better"), flip sign so
        # higher == more positive across the whole table. Spearman rho is
        # signed; AUROC and AP are direction-aware.
        direction = -1.0 if m in _LOWER_IS_BETTER else 1.0
        scored = direction * v
        auroc = _auroc(scored, lab)
        ap = _average_precision(scored, lab)
        auroc_ci_low, auroc_ci_high = _stratified_bootstrap_metric_ci(
            scored, lab, _auroc, n_bootstrap=bootstrap
        )
        ap_ci_low, ap_ci_high = _stratified_bootstrap_metric_ci(
            scored, lab, _average_precision, n_bootstrap=bootstrap
        )
        out.append(
            {
                "stratum": stratum,
                "label": label_name,
                "metric": m,
                "n": int(finite.sum()),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "spearman_rho": _spearman_rho(scored, lab.astype(float)),
                "auroc": auroc,
                "auroc_ci_low": auroc_ci_low,
                "auroc_ci_high": auroc_ci_high,
                "ap": ap,
                "ap_ci_low": ap_ci_low,
                "ap_ci_high": ap_ci_high,
            }
        )
    return out


def _per_label_per_format_tables(
    confidence: dict[str, np.ndarray],
    label_name: str,
    labels: np.ndarray,
    binder_format: np.ndarray,
    bootstrap: int = 0,
) -> list[dict[str, Any]]:
    """Run _per_label_table once per binder_format stratum.

    Mirrors the Harvey-et-al 2026 finding that AF-M confidence varies by
    target class — mirage's cheapest proxy for that is binder_format.
    """
    out: list[dict[str, Any]] = []
    for fmt in sorted({str(f) for f in binder_format if f}):
        mask = binder_format == fmt
        if not mask.any():
            continue
        sub_conf = {m: confidence[m][mask] for m in confidence}
        sub_labels = labels[mask]
        out.extend(
            _per_label_table(sub_conf, label_name, sub_labels, stratum=fmt, bootstrap=bootstrap)
        )
    return out


def _feature_matrix(
    confidence: dict[str, np.ndarray], feature_names: tuple[str, ...]
) -> np.ndarray:
    cols: list[np.ndarray] = []
    for m in feature_names:
        vals = confidence[m]
        direction = -1.0 if m in _LOWER_IS_BETTER else 1.0
        cols.append(direction * vals)
    return np.column_stack(cols)


def _fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Fit a small L2-regularized logistic regression with numpy.

    Returns ``(intercept, coefficients)``. The intercept is not penalized.
    """
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


def _logistic_table(
    confidence: dict[str, np.ndarray],
    label_name: str,
    labels: np.ndarray,
    binder_format: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    l2: float,
    stratum: str = "all",
) -> list[dict[str, Any]]:
    valid = ~np.isnan(labels)
    if stratum != "all":
        valid &= binder_format == stratum
    x_all = _feature_matrix(confidence, feature_names)
    finite = valid & np.isfinite(x_all).all(axis=1)
    y = labels[finite].astype(float)
    x = x_all[finite]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    row: dict[str, Any] = {
        "stratum": stratum,
        "label": label_name,
        "model": "logistic_regression_loo",
        "features": ";".join(feature_names),
        "n": int(y.size),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "l2": l2,
        "spearman_rho": math.nan,
        "auroc": math.nan,
        "ap": math.nan,
        "intercept": math.nan,
        "coefficients_json": "{}",
    }
    if y.size < 3 or n_pos == 0 or n_neg == 0:
        return [row]
    loo_scores = _logistic_loo_scores(x, y, l2=l2)
    scored = np.isfinite(loo_scores)
    if scored.sum() >= 2:
        row["spearman_rho"] = _spearman_rho(loo_scores[scored], y[scored])
        row["auroc"] = _auroc(loo_scores[scored], y[scored].astype(int))
        row["ap"] = _average_precision(loo_scores[scored], y[scored].astype(int))

    x_train, _, mean, std = _standardize_train_apply(x, x)
    intercept, coef = _fit_logistic_regression(x_train, y, l2=l2)
    row["intercept"] = intercept
    # Convert back to the feature scale used in the CSV for easier inspection.
    raw_coef = coef / std
    raw_intercept = intercept - float((mean / std) @ coef)
    row["intercept"] = raw_intercept
    row["coefficients_json"] = json.dumps(
        {m: float(c) for m, c in zip(feature_names, raw_coef, strict=True)},
        sort_keys=True,
    )
    return [row]


def _logistic_tables(
    confidence: dict[str, np.ndarray],
    label_name: str,
    labels: np.ndarray,
    binder_format: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    l2: float,
    stratify: bool,
) -> list[dict[str, Any]]:
    rows = _logistic_table(
        confidence,
        label_name,
        labels,
        binder_format,
        feature_names=feature_names,
        l2=l2,
    )
    if stratify:
        for fmt in sorted({str(f) for f in binder_format if f}):
            rows.extend(
                _logistic_table(
                    confidence,
                    label_name,
                    labels,
                    binder_format,
                    feature_names=feature_names,
                    l2=l2,
                    stratum=fmt,
                )
            )
    return rows


def _logistic_ablation_feature_sets() -> tuple[tuple[str, ...], ...]:
    singletons = tuple((m,) for m in _CONFIDENCE_METRICS)
    pae_plddt_pairs = tuple((pae, plddt) for pae in _PAE_FEATURES for plddt in _PLDDT_FEATURES)
    return singletons + pae_plddt_pairs + (_DEFAULT_LOGISTIC_FEATURES,)


def _logistic_ablation_tables(
    confidence: dict[str, np.ndarray],
    label_name: str,
    labels: np.ndarray,
    binder_format: np.ndarray,
    *,
    l2: float,
    stratify: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_names in _logistic_ablation_feature_sets():
        feature_set_type = "single" if len(feature_names) == 1 else "pae_plddt_pair"
        if feature_names == _DEFAULT_LOGISTIC_FEATURES:
            feature_set_type = "default_all"
        for row in _logistic_tables(
            confidence,
            label_name,
            labels,
            binder_format,
            feature_names=feature_names,
            l2=l2,
            stratify=stratify,
        ):
            rows.append({"feature_set_type": feature_set_type, **row})
    return rows


def _read_negative_confidence(
    negative_path: Path,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    """Return (ids, conf_by_metric, binder_format) for negative-control rows.

    Drops rows where the headline iPTM ``value`` is NaN.
    """
    rows = _read_csv(negative_path)
    ids: list[str] = []
    conf_vals: dict[str, list[float]] = {m: [] for m in _CONFIDENCE_METRICS}
    fmt_vals: list[str] = []
    for r in rows:
        if math.isnan(_value_or_extra(r, "_value")):
            continue
        ids.append(r["example_id"])
        for m in _CONFIDENCE_METRICS:
            conf_vals[m].append(_value_or_extra(r, m))
        fmt_vals.append(str(r.get("binder_format") or ""))
    return (
        ids,
        {m: np.asarray(conf_vals[m], dtype=float) for m in _CONFIDENCE_METRICS},
        np.asarray(fmt_vals, dtype=object),
    )


def _negative_vs_real_label(
    real_rmsd: np.ndarray, n_negative: int, rmsd_cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build a binary label vector for {near-native real} vs {negative control}.

    ``real_rmsd`` lines up with the joined real-binder rows. Negative
    controls come after them. Returns (labels, keep_mask) where
    keep_mask drops real-binder rows that are *neither* near-native nor
    far enough to confidently call non-positive — for the simplest
    first pass we keep only `rmsd < cutoff` reals as positives and
    treat ALL reals with `rmsd >= cutoff` as non-data (not negatives,
    since they're real binders). Negative controls are all negatives.
    """
    real_labels = np.where(real_rmsd < rmsd_cutoff, 1.0, math.nan)
    negative_labels = np.zeros(n_negative, dtype=float)
    labels = np.concatenate([real_labels, negative_labels])
    keep = ~np.isnan(labels)
    return labels, keep


def _negative_vs_real_rows(
    *,
    label_name: str,
    negative_name: str,
    negative_path: Path,
    conf_by_metric: dict[str, np.ndarray],
    rmsd_arr: np.ndarray,
    fmt_arr: np.ndarray,
    rmsd_cutoff: float,
    bootstrap: int,
    stratify: bool,
    logistic_rows: list[dict[str, Any]],
    logistic_ablation_rows: list[dict[str, Any]],
    logistic_output_enabled: bool,
    logistic_ablation_enabled: bool,
    logistic_features: tuple[str, ...],
    logistic_l2: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    negative_ids, negative_conf, negative_fmt = _read_negative_confidence(negative_path)
    if not negative_ids:
        print(
            f"warning: no usable rows in {negative_path}; "
            f"skipping {negative_name}-vs-real analysis",
            file=sys.stderr,
        )
        return rows

    combined_conf = {
        m: np.concatenate([conf_by_metric[m], negative_conf[m]]) for m in _CONFIDENCE_METRICS
    }
    combined_fmt = np.concatenate([fmt_arr, negative_fmt])
    labels, _ = _negative_vs_real_label(
        rmsd_arr, n_negative=len(negative_ids), rmsd_cutoff=rmsd_cutoff
    )
    rows.extend(_per_label_table(combined_conf, label_name, labels, bootstrap=bootstrap))
    if stratify:
        rows.extend(
            _per_label_per_format_tables(
                combined_conf, label_name, labels, combined_fmt, bootstrap=bootstrap
            )
        )
    if logistic_output_enabled:
        logistic_rows.extend(
            _logistic_tables(
                combined_conf,
                label_name,
                labels,
                combined_fmt,
                feature_names=logistic_features,
                l2=logistic_l2,
                stratify=stratify,
            )
        )
    if logistic_ablation_enabled:
        logistic_ablation_rows.extend(
            _logistic_ablation_tables(
                combined_conf,
                label_name,
                labels,
                combined_fmt,
                l2=logistic_l2,
                stratify=stratify,
            )
        )
    print(f"\n== {negative_name}-vs-real (n_{negative_name}={len(negative_ids)}) ==")
    _print_report(
        n_total=len(rmsd_arr) + len(negative_ids),
        per_label={label_name: rows},
        buckets=[],
    )
    return rows


def _bucket_summary(
    confidence: dict[str, np.ndarray],
    rmsd: np.ndarray,
    buckets: tuple[tuple[str, float, float], ...] = (
        ("<=4 Å", -math.inf, 4.0),
        ("4-8 Å", 4.0, 8.0),
        (">8 Å", 8.0, math.inf),
    ),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, lo, hi in buckets:
        mask = (rmsd > lo) & (rmsd <= hi)
        n = int(mask.sum())
        row: dict[str, Any] = {"bucket": name, "n": n}
        for m in _CONFIDENCE_METRICS:
            vals = confidence[m][mask]
            vals = vals[~np.isnan(vals)]
            if vals.size:
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=0))
            else:
                row[f"{m}_mean"] = math.nan
                row[f"{m}_std"] = math.nan
        out.append(row)
    return out


def _print_report(
    n_total: int,
    per_label: dict[str, list[dict[str, Any]]],
    buckets: list[dict[str, Any]],
) -> None:
    print(f"Joined N={n_total} examples")
    for label, rows in per_label.items():
        print(f"\n== Confidence vs near-native, label='{label}' ==")
        print(
            f"{'stratum':<8} {'metric':<28} {'n':>4} {'pos':>4} {'neg':>4} "
            f"{'spearman_rho':>14} {'auroc':>8} {'ap':>8}"
        )
        for r in rows:
            print(
                f"{r['stratum']:<8} {r['metric']:<28} {r['n']:>4} "
                f"{r['n_positive']:>4} {r['n_negative']:>4} "
                f"{r['spearman_rho']:>14.3f} {r['auroc']:>8.3f} {r['ap']:>8.3f}"
            )
    if not buckets:
        return
    print("\n== RMSD-bucket-wise confidence (mean ± std) ==")
    headline_metrics = (
        "iptm",
        "ptm",
        "iptm_over_ptm",
        "ranking_confidence",
        "plddt_full_mean",
        "plddt_interface_mean",
        "pae_interchain_mean",
        "pae_interface_mean",
    )
    header = "bucket  " + "  ".join(f"{m:>22}" for m in headline_metrics) + "    n"
    print(header)
    for b in buckets:
        parts = [f"{b['bucket']:<7}"]
        for m in headline_metrics:
            parts.append(f"  {b[f'{m}_mean']:>10.3f} ± {b[f'{m}_std']:>7.3f}")
        parts.append(f"  {b['n']:>4}")
        print(" ".join(parts))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--rmsd", type=Path, required=True)
    parser.add_argument("--dockq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scramble-confidence",
        type=Path,
        default=None,
        help=(
            "Optional confidence CSV from CDR-scrambled negative-control "
            "predictions. When set, a {near-native real} vs {scramble} "
            "AUROC + AP table is emitted alongside the within-real-binder "
            "tables, written to <output stem>_scrambles.csv."
        ),
    )
    parser.add_argument(
        "--wrong-target-confidence",
        type=Path,
        default=None,
        help=(
            "Optional confidence CSV from wrong-target re-pairing negative-control "
            "predictions. When set, a {near-native real} vs {wrong target} "
            "AUROC + AP table is emitted alongside the within-real-binder "
            "tables, written to <output stem>_wrong_targets.csv."
        ),
    )
    parser.add_argument(
        "--scramble-rmsd-cutoff",
        type=float,
        default=4.0,
        help=(
            "RMSD cutoff (Å) defining the 'near-native real binder' "
            "positive class for the scramble-vs-real label. Real binders "
            "with rmsd >= cutoff are *excluded* (neither positive nor "
            "negative) so the comparison is real-good vs scramble-by-"
            "construction. Default 4.0 Å."
        ),
    )
    parser.add_argument(
        "--scramble-bootstrap",
        type=int,
        default=1000,
        help=(
            "Number of deterministic class-stratified bootstrap resamples "
            "for AUROC/AP confidence intervals in the scramble-vs-real table. "
            "Set to 0 to disable. Default 1000."
        ),
    )
    parser.add_argument(
        "--binder-format-stratify",
        action="store_true",
        help=(
            "Emit per-binder-format AUROC + AP tables alongside the "
            "aggregated ones. Stratifies on the binder_format column "
            "(VHH / Fab / scFv)."
        ),
    )
    parser.add_argument(
        "--logistic-output",
        type=Path,
        default=None,
        help=(
            "Optional CSV path for a numpy-only multi-feature logistic-regression "
            "baseline. Reported AUROC/AP use leave-one-out scores; coefficients "
            "are from a fit on all finite rows for inspection."
        ),
    )
    parser.add_argument(
        "--logistic-ablation-output",
        type=Path,
        default=None,
        help=(
            "Optional CSV path for leave-one-out logistic-regression feature ablations. "
            "Emits all single-feature models, all PAE+pLDDT two-feature candidates, "
            "and the default all-feature model."
        ),
    )
    parser.add_argument(
        "--logistic-features",
        nargs="+",
        default=list(_DEFAULT_LOGISTIC_FEATURES),
        choices=_CONFIDENCE_METRICS,
        help="Confidence metrics to use as logistic-regression features.",
    )
    parser.add_argument(
        "--logistic-l2",
        type=float,
        default=1.0,
        help="L2 regularization strength for the logistic-regression baseline.",
    )
    args = parser.parse_args()
    logistic_features = tuple(args.logistic_features)

    confidence = _read_csv(args.confidence)
    rmsd_rows = _read_csv(args.rmsd)
    dockq_rows = _read_csv(args.dockq)

    ids, conf_by_metric, rmsd_arr, dockq_arr, fmt_arr = _join(confidence, rmsd_rows, dockq_rows)
    n = len(ids)
    if n == 0:
        print("no rows joined cleanly; nothing to analyze", file=sys.stderr)
        return 1

    labels = {
        "rmsd<4A": (rmsd_arr < 4.0).astype(float),
        "rmsd<8A": (rmsd_arr < 8.0).astype(float),
        "dockq_best>=0.23": np.where(
            np.isnan(dockq_arr), math.nan, (dockq_arr >= 0.23).astype(float)
        ),
    }

    per_label: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    logistic_rows: list[dict[str, Any]] = []
    logistic_ablation_rows: list[dict[str, Any]] = []
    for label_name, lab in labels.items():
        rows = _per_label_table(conf_by_metric, label_name, lab)
        per_label[label_name] = rows
        summary_rows.extend(rows)
        if args.binder_format_stratify:
            fmt_rows = _per_label_per_format_tables(conf_by_metric, label_name, lab, fmt_arr)
            per_label[label_name].extend(fmt_rows)
            summary_rows.extend(fmt_rows)
        if args.logistic_output is not None:
            logistic_rows.extend(
                _logistic_tables(
                    conf_by_metric,
                    label_name,
                    lab,
                    fmt_arr,
                    feature_names=logistic_features,
                    l2=args.logistic_l2,
                    stratify=args.binder_format_stratify,
                )
            )
        if args.logistic_ablation_output is not None:
            logistic_ablation_rows.extend(
                _logistic_ablation_tables(
                    conf_by_metric,
                    label_name,
                    lab,
                    fmt_arr,
                    l2=args.logistic_l2,
                    stratify=args.binder_format_stratify,
                )
            )

    buckets = _bucket_summary(conf_by_metric, rmsd_arr)
    _print_report(n, per_label, buckets)
    _write_csv(args.output, summary_rows)

    if args.scramble_confidence is not None:
        label_name = f"near_native(rmsd<{args.scramble_rmsd_cutoff:g}A)_vs_scramble"
        scramble_rows = _negative_vs_real_rows(
            label_name=label_name,
            negative_name="scramble",
            negative_path=args.scramble_confidence,
            conf_by_metric=conf_by_metric,
            rmsd_arr=rmsd_arr,
            fmt_arr=fmt_arr,
            rmsd_cutoff=args.scramble_rmsd_cutoff,
            bootstrap=args.scramble_bootstrap,
            stratify=args.binder_format_stratify,
            logistic_rows=logistic_rows,
            logistic_ablation_rows=logistic_ablation_rows,
            logistic_output_enabled=args.logistic_output is not None,
            logistic_ablation_enabled=args.logistic_ablation_output is not None,
            logistic_features=logistic_features,
            logistic_l2=args.logistic_l2,
        )
        scramble_csv = args.output.with_name(args.output.stem + "_scrambles.csv")
        _write_csv(scramble_csv, scramble_rows)
        print(f"Wrote {len(scramble_rows)} scramble-vs-real rows to {scramble_csv}")

    if args.wrong_target_confidence is not None:
        label_name = f"near_native(rmsd<{args.scramble_rmsd_cutoff:g}A)_vs_wrong_target"
        wrong_target_rows = _negative_vs_real_rows(
            label_name=label_name,
            negative_name="wrong_target",
            negative_path=args.wrong_target_confidence,
            conf_by_metric=conf_by_metric,
            rmsd_arr=rmsd_arr,
            fmt_arr=fmt_arr,
            rmsd_cutoff=args.scramble_rmsd_cutoff,
            bootstrap=args.scramble_bootstrap,
            stratify=args.binder_format_stratify,
            logistic_rows=logistic_rows,
            logistic_ablation_rows=logistic_ablation_rows,
            logistic_output_enabled=args.logistic_output is not None,
            logistic_ablation_enabled=args.logistic_ablation_output is not None,
            logistic_features=logistic_features,
            logistic_l2=args.logistic_l2,
        )
        wrong_target_csv = args.output.with_name(args.output.stem + "_wrong_targets.csv")
        _write_csv(wrong_target_csv, wrong_target_rows)
        print(f"Wrote {len(wrong_target_rows)} wrong-target-vs-real rows to {wrong_target_csv}")

    bucket_csv = args.output.with_name(args.output.stem + "_buckets.csv")
    _write_csv(bucket_csv, buckets)
    if args.logistic_output is not None:
        _write_csv(args.logistic_output, logistic_rows)
        print(f"Wrote {len(logistic_rows)} logistic-regression rows to {args.logistic_output}")
    if args.logistic_ablation_output is not None:
        _write_csv(args.logistic_ablation_output, logistic_ablation_rows)
        print(
            f"Wrote {len(logistic_ablation_rows)} logistic-ablation rows "
            f"to {args.logistic_ablation_output}"
        )

    print(f"\nWrote {len(summary_rows)} (metric, label) rows to {args.output}")
    print(f"Wrote {len(buckets)} RMSD-bucket rows to {bucket_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
