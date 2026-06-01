"""Combine per-predictor Champloo classifier metrics into one comparison table.

Reads the published per-predictor outputs of ``analyze_champloo_classifier.py``
(each carrying a ``predictor`` column) and concatenates them into a single
cross-predictor comparison, ordered by predictor then split then model. This is
the Phase 1 AF3-vs-Boltz-2-vs-Chai-1 headline artifact.

Use::

    uv run python scripts/compare_champloo_predictors.py \\
        --inputs results/published/champloo_af3_classifier.csv \\
                 results/published/champloo_boltz2_classifier.csv \\
                 results/published/champloo_chai1_classifier.csv \\
        --output results/published/champloo_predictor_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_FIELDNAMES = [
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

_SPLIT_ORDER = {"all": 0, "random_pair": 1, "held_out_vhh": 2, "held_out_antigen": 3}
_MODEL_ORDER = {"raw_iptm": 0, "logistic_iptm": 1, "logistic_iptm_meta": 2}


def combine_rows(tables: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    rows = [row for table in tables for row in table]
    rows.sort(
        key=lambda r: (
            r.get("predictor", ""),
            _SPLIT_ORDER.get(r.get("split", ""), 99),
            _MODEL_ORDER.get(r.get("model", ""), 99),
        )
    )
    return rows


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = combine_rows([_read(p) for p in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'predictor':<10} {'split':<16} {'model':<20} {'ap':>7} {'auroc':>7}")
    for row in rows:
        ap = float(row["ap"]) if row["ap"] not in ("", "nan") else float("nan")
        auroc = float(row["auroc"]) if row["auroc"] not in ("", "nan") else float("nan")
        print(
            f"{row['predictor']:<10} {row['split']:<16} {row['model']:<20} {ap:7.3f} {auroc:7.3f}"
        )
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
