"""Normalize raw AVIDa-hIL6 files into one staged CSV the loader consumes.

Use::

    uv run python scripts/stage_avida.py \\
        --records ../abdisc-data/avida/raw/AVIDa-hIL6.csv \\
        --antigens ../abdisc-data/avida/raw/antigen_sequences.csv \\
        --output ../abdisc-data/avida/avida_staged.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mirage.features.normalize import normalize_antigen, normalize_binder


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def antigen_map(antigen_rows: list[dict[str, str]]) -> dict[str, str]:
    """Map antigen label -> sequence. Tolerates the two common column namings."""
    out: dict[str, str] = {}
    for row in antigen_rows:
        label = row.get("Ag_label") or row.get("antigen_label") or row.get("label", "")
        seq = row.get("antigen_sequence") or row.get("Ag_sequence") or row.get("sequence", "")
        if label and seq:
            out[label] = seq
    return out


def build_rows(records: list[dict[str, str]], antigens: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, rec in enumerate(records):
        label = rec.get("label", "")
        ag_label = rec.get("Ag_label", "")
        seq = normalize_binder(rec.get("VHH_sequence", ""))
        antigen_seq = normalize_antigen(antigens.get(ag_label, ""))
        if not seq or not antigen_seq or label not in ("0", "1"):
            continue
        rows.append(
            {
                "vhh_id": f"avida-{i}",
                "vhh_sequence": seq,
                "antigen_label": ag_label,
                "antigen_sequence": antigen_seq,
                "label": label,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--antigens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_csv(args.records)
    antigens = antigen_map(read_csv(args.antigens))
    rows = build_rows(records, antigens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["vhh_id", "vhh_sequence", "antigen_label", "antigen_sequence", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} AVIDa rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
