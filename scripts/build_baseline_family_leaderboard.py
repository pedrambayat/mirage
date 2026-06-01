"""Build a compact family-level leaderboard from published metric tables.

The input tables are already-scored analyses. This script does not rescore
predictions; it selects the best row within each baseline family for each label
comparison, ranking by average precision first and AUROC second.

Use::

    uv run python scripts/build_baseline_family_leaderboard.py \\
        --output results/published/baseline_family_leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_AF2M_CONFIDENCE_METRICS = frozenset(
    {
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
    }
)

_STRUCTURAL_CLASH_METRICS = frozenset(
    {
        "atom_clashes_2a",
        "atom_clash_fraction_2a",
    }
)

_STRUCTURAL_EXPOSURE_METRICS = frozenset(
    {
        "buried_sasa_proxy_a2",
        "buried_sasa_proxy_binder_a2",
        "buried_sasa_proxy_target_a2",
        "buried_sasa_proxy_per_binder_atom",
        "buried_sasa_proxy_per_target_atom",
        "buried_sasa_proxy_per_interface_residue",
        "buried_sasa_balance",
        "mean_binder_atom_exposure_loss",
        "mean_target_atom_exposure_loss",
    }
)

_STRUCTURAL_PACKING_METRICS = frozenset(
    {
        "atom_packing_pairs_0_1a_gap",
        "atom_packing_fraction_0_1a_gap",
        "atom_packing_shell_pairs_2a_gap",
        "binder_atom_packing_coverage_0_1a_gap",
        "target_atom_packing_coverage_0_1a_gap",
        "atom_packing_complementarity_score",
        "mean_abs_nearest_surface_gap",
        "shape_complementarity_proxy",
    }
)

_DEFAULT_PUBLISHED_DIR = Path("results/published")
_DEFAULT_AF2M_TABLES = (
    _DEFAULT_PUBLISHED_DIR / "sabdab_af2m_confidence_vs_rmsd_n200.csv",
    _DEFAULT_PUBLISHED_DIR / "sabdab_af2m_confidence_vs_rmsd_n200_scrambles.csv",
    _DEFAULT_PUBLISHED_DIR / "sabdab_af2m_confidence_vs_rmsd_n200_wrong_targets.csv",
    _DEFAULT_PUBLISHED_DIR / "epcam_af2m_confidence_label_metrics.csv",
)
_DEFAULT_STRUCTURAL_TABLE = _DEFAULT_PUBLISHED_DIR / "structural_interface_label_metrics.csv"
_DEFAULT_STRUCTURAL_LOGREG_TABLE = _DEFAULT_PUBLISHED_DIR / "structural_interface_logreg.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _float_value(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int_value(row: dict[str, Any], key: str) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _comparison_name(row: dict[str, str]) -> str:
    return row.get("comparison") or row.get("label") or ""


def _family_rows(
    rows: Iterable[dict[str, str]],
    *,
    family: str,
    metrics: frozenset[str],
    source_table: Path,
    require_all_stratum: bool,
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for row in rows:
        if require_all_stratum and row.get("stratum", "all") != "all":
            continue
        metric = row.get("metric") or row.get("model") or ""
        comparison = _comparison_name(row)
        if metric not in metrics or not comparison:
            continue
        auroc = _float_value(row, "auroc")
        ap = _float_value(row, "ap")
        if not (math.isfinite(auroc) and math.isfinite(ap)):
            continue
        n = _int_value(row, "n")
        n_positive = _int_value(row, "n_positive")
        n_negative = _int_value(row, "n_negative")
        baseline_ap = _float_value(row, "baseline_ap")
        if not math.isfinite(baseline_ap) and n > 0:
            baseline_ap = n_positive / n
        out.append(
            {
                "comparison": comparison,
                "family": family,
                "source_table": source_table.name,
                "best_metric": metric,
                "n": n,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "baseline_ap": baseline_ap,
                "auroc": auroc,
                "ap": ap,
            }
        )
    return out


def _best_key(row: dict[str, float | int | str]) -> tuple[float, float, str]:
    ap = row["ap"]
    auroc = row["auroc"]
    return (
        float(ap) if isinstance(ap, float) and math.isfinite(ap) else -math.inf,
        float(auroc) if isinstance(auroc, float) and math.isfinite(auroc) else -math.inf,
        str(row["best_metric"]),
    )


def build_leaderboard(
    *,
    af2m_tables: Iterable[Path],
    structural_table: Path,
    structural_logreg_table: Path | None = None,
) -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = []
    for path in af2m_tables:
        candidates.extend(
            _family_rows(
                _read_csv(path),
                family="af2m_confidence",
                metrics=_AF2M_CONFIDENCE_METRICS,
                source_table=path,
                require_all_stratum=True,
            )
        )

    structural_rows = _read_csv(structural_table)
    candidates.extend(
        _family_rows(
            structural_rows,
            family="structural_clash",
            metrics=_STRUCTURAL_CLASH_METRICS,
            source_table=structural_table,
            require_all_stratum=False,
        )
    )
    candidates.extend(
        _family_rows(
            structural_rows,
            family="structural_exposure",
            metrics=_STRUCTURAL_EXPOSURE_METRICS,
            source_table=structural_table,
            require_all_stratum=False,
        )
    )
    candidates.extend(
        _family_rows(
            structural_rows,
            family="structural_packing",
            metrics=_STRUCTURAL_PACKING_METRICS,
            source_table=structural_table,
            require_all_stratum=False,
        )
    )
    if structural_logreg_table is not None and structural_logreg_table.exists():
        candidates.extend(
            _family_rows(
                _read_csv(structural_logreg_table),
                family="structural_logreg",
                metrics=frozenset({"structural_logistic_regression_loo"}),
                source_table=structural_logreg_table,
                require_all_stratum=False,
            )
        )

    best: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for row in candidates:
        key = (str(row["comparison"]), str(row["family"]))
        if key not in best or _best_key(row) > _best_key(best[key]):
            best[key] = row

    return sorted(best.values(), key=lambda r: (str(r["comparison"]), str(r["family"])))


def _write_csv(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "comparison",
        "family",
        "source_table",
        "best_metric",
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


def _print_report(rows: list[dict[str, float | int | str]]) -> None:
    print(f"{'comparison':<40} {'family':<20} {'best_metric':<34} {'auroc':>7} {'ap':>7}")
    for row in rows:
        print(
            f"{row['comparison']!s:<40} {row['family']!s:<20} "
            f"{row['best_metric']!s:<34} {float(row['auroc']):7.3f} "
            f"{float(row['ap']):7.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--af2m-table",
        type=Path,
        action="append",
        dest="af2m_tables",
        help="AF2-M metric table. May be passed multiple times.",
    )
    parser.add_argument(
        "--structural-table",
        type=Path,
        default=_DEFAULT_STRUCTURAL_TABLE,
        help="Structural-interface metric table.",
    )
    parser.add_argument(
        "--structural-logreg-table",
        type=Path,
        default=_DEFAULT_STRUCTURAL_LOGREG_TABLE,
        help=(
            "Optional structural logistic-regression metric table. Included when the file exists."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_leaderboard(
        af2m_tables=args.af2m_tables or _DEFAULT_AF2M_TABLES,
        structural_table=args.structural_table,
        structural_logreg_table=args.structural_logreg_table,
    )
    _write_csv(args.output, rows)
    _print_report(rows)
    print(f"Wrote {len(rows)} family leaderboard rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
