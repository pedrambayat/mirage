"""Build the Champloo pairs CSV for the M-C structure track.

Constructs 91 cognate diagonal positives (one per unique pdb_id in the
Champloo table, after excluding is_excluded=TRUE rows) plus k=5 distribution-
matched, cross-cluster, fold-consistent shuffled negatives per positive.
Reuses the SAbDab pair-building machinery from stage_sabdab_pairs.py directly.

The pair_id convention follows the prediction-directory layout used elsewhere:
``{pdb_id}__{pdb_id}`` for cognate positives and ``{pdb_id}__neg{j}`` for the
j-th negative of that positive (added automatically by build_pairs).

Use::

    uv run python scripts/stage_champloo_protenix_pairs.py \\
        --champloo-table ../abdisc-data/champloo/Supplementary_Table_1_*.csv \\
        --output data/staged/champloo/champloo_protenix_pairs.csv
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from mirage.features.clustering import cluster_antigens
from mirage.features.normalize import normalize_antigen, normalize_binder
from mirage.ml.core import assign_folds

# ---------------------------------------------------------------------------
# Import build_pairs, Positive, and _FIELDNAMES from stage_sabdab_pairs.py.
# These live in scripts/ (not an installable package), so we use importlib.
# ---------------------------------------------------------------------------
_SABDAB_SCRIPT = Path(__file__).resolve().parent / "stage_sabdab_pairs.py"


def _load_sabdab_module():  # type: ignore[return]
    spec = importlib.util.spec_from_file_location("stage_sabdab_pairs", _SABDAB_SCRIPT)
    assert spec and spec.loader
    if "stage_sabdab_pairs" in sys.modules:
        return sys.modules["stage_sabdab_pairs"]
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_sabdab_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_champloo_positives(
    table_path: Path,
    *,
    n_splits: int = 5,
    seed: int,
) -> list:  # list[Positive] — typed at runtime from the sabdab module
    """Build the list of Positive objects from the Champloo metadata table.

    Reads the Supplementary Table 1 CSV, drops rows where is_excluded is
    "TRUE", deduplicates to unique pdb_ids (the released ipTM matrices are
    keyed at pdb_id granularity — duplicate-PDB rows collapse to the first
    occurrence), normalizes sequences, clusters antigens, and assigns folds.

    Returns a list of Positive(pair_id, binder_seq, antigen_seq,
    antigen_cluster, fold) with pair_id = "{pdb_id}__{pdb_id}".
    """
    mod = _load_sabdab_module()
    positive_cls = mod.Positive

    with table_path.open(newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    # Drop excluded rows
    included = [r for r in all_rows if r.get("is_excluded", "").strip().upper() != "TRUE"]

    # Deduplicate to unique pdb_id — first occurrence wins (mirrors stage_champloo_pairs.py)
    seen_pdbs: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in included:
        pdb = row["pdb_id"]
        if pdb not in seen_pdbs:
            seen_pdbs.add(pdb)
            unique_rows.append(row)

    # Normalize sequences
    raw: list[tuple[str, str, str]] = []  # (pdb_id, binder_seq, antigen_seq)
    for row in unique_rows:
        binder = normalize_binder(row["vhh_sequence"])
        antigen = normalize_antigen(row["antigen_sequence"])
        if not binder or not antigen:
            continue
        raw.append((row["pdb_id"], binder, antigen))

    # Cluster unique antigens then map each row's antigen to its cluster id
    unique_ag = sorted({a for _, _, a in raw})
    ag_cluster_ids = cluster_antigens(unique_ag)
    ag_cluster: dict[str, int] = dict(zip(unique_ag, ag_cluster_ids, strict=True))

    # Assign folds (grouped by antigen cluster — whole clusters stay together)
    clusters: np.ndarray[int, np.dtype[np.int_]] = np.array([ag_cluster[a] for _, _, a in raw])
    folds: np.ndarray[int, np.dtype[np.int_]] = assign_folds(clusters, n_splits=n_splits, seed=seed)

    return [
        positive_cls(
            pair_id=f"{pdb}__{pdb}",
            binder_seq=b,
            antigen_seq=a,
            antigen_cluster=int(ag_cluster[a]),
            fold=int(f),
        )
        for (pdb, b, a), f in zip(raw, folds, strict=True)
    ]


app = typer.Typer(help="Build the Champloo protenix pairs CSV.")


@app.command()
def main(
    champloo_table: Annotated[
        Path,
        typer.Option("--champloo-table", help="Path to Supplementary_Table_1 CSV."),
    ],
    output: Annotated[Path, typer.Option(help="Output pairs CSV path.")],
    k: Annotated[int, typer.Option(help="Number of negatives per positive.")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 20260601,
    n_splits: Annotated[int, typer.Option("--n-splits", help="Number of CV folds.")] = 5,
) -> None:
    """Build Champloo positives and k cross-cluster negatives; write to --output."""
    mod = _load_sabdab_module()

    positives = build_champloo_positives(champloo_table, n_splits=n_splits, seed=seed)
    rows = mod.build_pairs(positives, k=k, seed=seed)

    n_pos = sum(1 for r in rows if r["label"] == "1")
    n_neg = sum(1 for r in rows if r["label"] == "0")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=mod._FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    typer.echo(f"positives={n_pos} negatives={n_neg} total={len(rows)}")


if __name__ == "__main__":
    app()
