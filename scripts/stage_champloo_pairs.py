"""Stage the Champloo/Smorodina VHH-antigen pair label table for Phase 1.

This builds a flat pair table by joining the Champloo experimental-system
metadata (``Supplementary_Table_1_final_experimental_vhh_ag_systems.csv``) with
one released predictor's ipTM score matrix (preferably AF3). It is the Phase 1
"get any classifier working" staging step from the 2026-05-26 Champloo plan.

Important data-shape note (do not silently lose this):

* The plan describes a conceptual 106 x 106 system matrix. There are 106
  non-excluded systems, but only 91 *unique* ``pdb_id`` values: 15 PDB entries
  host more than one VHH-antigen system. The released ipTM matrices are keyed
  by ``pdb_id`` (91 x 91), so duplicate-PDB systems collapse to a single cell
  and cannot be told apart from the released scores. This staging therefore
  operates at ``pdb_id`` granularity: 91 cognate diagonal positives and the
  off-diagonal non-cognate pairs that have a finite released ipTM.
* Off-diagonal pairs are *constructed shuffled non-cognate* pairings, NOT
  experimentally verified non-binders.

Use::

    uv run python scripts/stage_champloo_pairs.py \\
        --metadata <champloo>/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \\
        --matrix <champloo>/iptm_confidence_scores/iptm_confidence_scores/af3_matrix_clean.csv \\
        --predictor af3 \\
        --output data/staged/champloo/champloo_pairs_af3.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

# Per-PDB metadata columns carried onto each pair for the downstream classifier.
_VHH_META = ("vhh_length", "resolution")
_ANTIGEN_META = (
    "antigen_length",
    "antigen_helix_content",
    "antigen_sheet_content",
    "antigen_loop_content",
)


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def filter_included(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only rows where ``is_excluded`` is FALSE (case-insensitive)."""
    return [r for r in rows if r.get("is_excluded", "").strip().upper() == "FALSE"]


def metadata_by_pdb(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index included metadata by ``pdb_id``.

    The released matrices are keyed by ``pdb_id``, so duplicate-PDB systems are
    indistinguishable downstream. The first occurrence wins; the count of
    collapsed systems is preserved as ``_n_systems`` for transparency.
    """
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        pdb = row["pdb_id"]
        if pdb not in out:
            out[pdb] = {**row, "_n_systems": "1"}
        else:
            out[pdb]["_n_systems"] = str(int(out[pdb]["_n_systems"]) + 1)
    return out


def load_score_matrix(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Load a square ipTM matrix keyed by ``pdb_id``.

    Returns ``(pdb_ids, matrix)`` where ``matrix[vhh_pdb][antigen_pdb]`` is the
    ipTM for pairing the VHH from the row PDB with the antigen from the column
    PDB. Empty / non-numeric cells become NaN.
    """
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col_pdbs = header[1:]
        matrix: dict[str, dict[str, float]] = {}
        for raw in reader:
            row_pdb = raw[0]
            cells: dict[str, float] = {}
            for col_pdb, value in zip(col_pdbs, raw[1:], strict=True):
                try:
                    cells[col_pdb] = float(value) if value.strip() != "" else math.nan
                except ValueError:
                    cells[col_pdb] = math.nan
            matrix[row_pdb] = cells
    return col_pdbs, matrix


def build_pair_table(
    meta_by_pdb: dict[str, dict[str, str]],
    pdb_ids: list[str],
    matrix: dict[str, dict[str, float]],
    *,
    predictor: str,
) -> list[dict[str, Any]]:
    """Build the pair label table at ``pdb_id`` granularity.

    * positive (label 1): cognate diagonal pair (vhh_pdb == antigen_pdb).
    * negative (label 0): shuffled non-cognate off-diagonal pair.

    Only PDBs present in both the matrix and the included metadata are kept, and
    pairs with a missing (NaN) ipTM are dropped.
    """
    usable = [p for p in pdb_ids if p in meta_by_pdb]
    rows: list[dict[str, Any]] = []
    for vhh_pdb in usable:
        for antigen_pdb in usable:
            iptm = matrix.get(vhh_pdb, {}).get(antigen_pdb, math.nan)
            if math.isnan(iptm):
                continue
            vhh_meta = meta_by_pdb[vhh_pdb]
            antigen_meta = meta_by_pdb[antigen_pdb]
            row: dict[str, Any] = {
                "pair_id": f"{vhh_pdb}__{antigen_pdb}",
                "vhh_pdb": vhh_pdb,
                "antigen_pdb": antigen_pdb,
                "label": 1 if vhh_pdb == antigen_pdb else 0,
                "predictor": predictor,
                "iptm": iptm,
            }
            for col in _VHH_META:
                row[f"vhh_{col}" if not col.startswith("vhh_") else col] = vhh_meta.get(col, "")
            for col in _ANTIGEN_META:
                row[col] = antigen_meta.get(col, "")
            rows.append(row)
    return rows


def _fieldnames() -> list[str]:
    base = ["pair_id", "vhh_pdb", "antigen_pdb", "label", "predictor", "iptm"]
    vhh = [c if c.startswith("vhh_") else f"vhh_{c}" for c in _VHH_META]
    return base + vhh + list(_ANTIGEN_META)


def write_pairs(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--predictor", type=str, default="af3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_meta = load_metadata(args.metadata)
    included = filter_included(all_meta)
    meta_by_pdb = metadata_by_pdb(included)
    pdb_ids, matrix = load_score_matrix(args.matrix)
    rows = build_pair_table(meta_by_pdb, pdb_ids, matrix, predictor=args.predictor)

    n_pos = sum(r["label"] for r in rows)
    n_neg = len(rows) - n_pos
    write_pairs(args.output, rows)
    print(
        f"metadata rows={len(all_meta)} included={len(included)} "
        f"unique_pdbs={len(meta_by_pdb)} matrix_pdbs={len(pdb_ids)}"
    )
    print(f"pairs={len(rows)} positives(cognate)={n_pos} negatives(non-cognate)={n_neg}")
    print(f"Wrote {len(rows)} {args.predictor} pair rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
