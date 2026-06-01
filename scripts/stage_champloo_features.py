"""Join Champloo supplementary-table sequences onto staged pairs and emit a
Tier-S feature CSV for the mirage sequence-only gate (M-S).

Use::

    uv run python scripts/stage_champloo_features.py \\
        --pairs data/staged/champloo/champloo_pairs_af3.csv \\
        --supp ../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \\  # noqa: E501
        --output data/staged/champloo/champloo_features_af3.csv
"""  # noqa: E501

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mirage.features.sequence import FEATURE_NAMES, sequence_features

_BASE_COLUMNS = ("pair_id", "vhh_pdb", "antigen_pdb", "label", "iptm")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def sequences_by_pdb(supp_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map pdb_id -> {vhh_sequence, antigen_sequence}; first row per pdb wins."""
    out: dict[str, dict[str, str]] = {}
    for row in supp_rows:
        pdb = row["pdb_id"]
        if pdb not in out:
            out[pdb] = {
                "vhh_sequence": row.get("vhh_sequence", ""),
                "antigen_sequence": row.get("antigen_sequence", ""),
            }
    return out


def build_feature_rows(
    pairs: list[dict[str, str]], seqs: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in pairs:
        vhh = seqs.get(p["vhh_pdb"])
        ant = seqs.get(p["antigen_pdb"])
        if vhh is None or ant is None:
            continue
        binder_seq = vhh["vhh_sequence"]
        target_seq = ant["antigen_sequence"]
        if not binder_seq or not target_seq:
            continue
        feats = sequence_features(binder_seq, target_seq)
        row: dict[str, str] = {col: p.get(col, "") for col in _BASE_COLUMNS}
        for name in FEATURE_NAMES:
            row[name] = repr(feats[name])
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*_BASE_COLUMNS, *FEATURE_NAMES]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--supp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = read_csv(args.pairs)
    supp = read_csv(args.supp)
    rows = build_feature_rows(pairs, sequences_by_pdb(supp))
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} feature rows to {args.output} (from {len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
