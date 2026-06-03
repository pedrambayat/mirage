"""Fault-tolerant M-C feature assembler.

Turns predicted (VHH, antigen) complexes into one cached feature row per pair
by composing three already-built pieces:

1. Geometry   — StructuralInterfaceScorer (sequence-mode, score_with_chains)
2. Confidence — ProtenixConfidenceScorer (pure-JSON; iPTM / PAE / pLDDT)
3. CDR        — cdr_engagement_features (ANARCI IMGT mapping; row-preserving)

Usage::

    uv run python scripts/extract_mc_features.py \\
        --pairs data/staged/champloo/champloo_protenix_pairs.csv \\
        --predictions-root data/staged/champloo/protenix \\
        --output results/mc_features.csv \\
        --failed-log results/mc_features_failed.txt
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from mirage.features.cdr_engagement import CDR_FEATURE_NAMES, cdr_engagement_features
from mirage.scorers._structure import (
    chain_residues,
    interface_residue_indices,
    load_structure,
    read_chain_roles_json,
    resolve_chain_roles_by_sequence,
)
from mirage.scorers.base import BenchmarkExample
from mirage.scorers.protenix_confidence import ProtenixConfidenceScorer
from mirage.scorers.structural_interface import StructuralInterfaceScorer

# ---------------------------------------------------------------------------
# Stable output schema
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "pair_id",
    "label",
    "antigen_cluster",
    "fold",
    "prediction_present",
    # Confidence (Protenix)
    "iptm",
    "ptm",
    "interface_pae",
    "min_interface_pae",
    "interface_plddt",
    "mean_plddt",
    # Geometry (StructuralInterfaceScorer)
    "n_interface_residues_binder",
    "n_interface_residues_target",
    "buried_sasa_proxy_a2",
    "atom_contacts_5a",
    "shape_complementarity_proxy",
    "atom_clash_fraction_2a",
    # CDR engagement
    "cdr_contact_fraction",
    "cdr1_contact_fraction",
    "cdr2_contact_fraction",
    "cdr3_contact_fraction",
    "cdr_mapping_ok",
]

_CONFIDENCE_KEYS = (
    "iptm",
    "ptm",
    "interface_pae",
    "min_interface_pae",
    "interface_plddt",
    "mean_plddt",
)
_GEOMETRY_KEYS = (
    "n_interface_residues_binder",
    "n_interface_residues_target",
    "buried_sasa_proxy_a2",
    "atom_contacts_5a",
    "shape_complementarity_proxy",
    "atom_clash_fraction_2a",
)

# ---------------------------------------------------------------------------
# Structure discovery (mirrors Task 2 sequence-mode logic)
# ---------------------------------------------------------------------------


def _find_structure(pred_dir: Path) -> Path | None:
    """Locate a predicted structure file under ``pred_dir``.

    Search order: ``rank1.cif``, ``rank1.pdb``, first ``*sample_0*.cif``
    found recursively (covers nested ``seed_0/predictions/`` layout and flat
    fixture layout).
    """
    for name in ("rank1.cif", "rank1.pdb"):
        candidate = pred_dir / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(pred_dir.rglob("*sample_0*.cif")):
        return candidate
    return None


# ---------------------------------------------------------------------------
# Core row assembler
# ---------------------------------------------------------------------------


def assemble_row(row: dict[str, str], predictions_root: Path) -> dict[str, str]:
    """Assemble a single feature row for one (binder, antigen) pair.

    Parameters
    ----------
    row:
        Dict with keys: pair_id, binder_seq, antigen_seq, label,
        antigen_cluster, fold.
    predictions_root:
        Root directory under which per-pair prediction directories live.

    Returns
    -------
    dict[str, str]
        All FEATURE_COLUMNS as strings. If no prediction is found,
        ``prediction_present`` is ``"0"`` and every feature column is ``""``
        (never fabricated). If a prediction is found, feature values are
        ``str(value)`` (NaN → ``"nan"``). May raise on corrupt predictions;
        wrapping is done by ``build_rows``.
    """
    pair_id = row["pair_id"]
    passthrough = {
        "pair_id": pair_id,
        "label": row["label"],
        "antigen_cluster": row["antigen_cluster"],
        "fold": row["fold"],
    }

    pred_dir = predictions_root / pair_id
    structure_path = _find_structure(pred_dir)

    if structure_path is None:
        # No prediction on disk — return sentinel with blank feature cols
        out = dict(passthrough)
        out["prediction_present"] = "0"
        for col in FEATURE_COLUMNS:
            if col not in out:
                out[col] = ""
        return out

    # Build a BenchmarkExample (binder_chains / target_chains carry sequences for
    # chain-role resolution by sequence identity)
    example = BenchmarkExample(
        id=pair_id,
        label=row["label"],
        binder_chains=(row["binder_seq"],),
        binder_format="vhh",
        target_chains=(row["antigen_seq"],),
        target_name="",
        source="pairs_csv",
    )

    # ------------------------------------------------------------------
    # 1. Confidence (pure JSON — never touches structure file)
    # ------------------------------------------------------------------
    conf = ProtenixConfidenceScorer(predictions_root).score(example)
    if conf.extras.get("missing") == "prediction":
        # JSON absent despite structure existing — treat as missing confidence
        conf_vals: dict[str, float] = {k: float("nan") for k in _CONFIDENCE_KEYS}
    else:
        conf_vals = {
            k: float(conf.extras[k]) if k in conf.extras else float("nan") for k in _CONFIDENCE_KEYS
        }

    # ------------------------------------------------------------------
    # 2. Geometry: load structure, resolve chain roles, compute geometry
    # ------------------------------------------------------------------
    pred = load_structure(structure_path)

    roles = read_chain_roles_json(structure_path.parent / "chain_roles.json")
    if roles is not None:
        binder_ids, target_ids = roles
    else:
        binder_ids, target_ids = resolve_chain_roles_by_sequence(
            pred,
            binder_seqs=example.binder_chains,
            target_seqs=example.target_chains,
        )

    geom = StructuralInterfaceScorer(
        predictions_root, chain_resolution="sequence"
    ).score_with_chains(example, pred, binder_ids, target_ids)

    geom_vals: dict[str, float] = {
        k: float(geom.extras[k]) if k in geom.extras else float("nan") for k in _GEOMETRY_KEYS
    }

    # ------------------------------------------------------------------
    # 3. CDR engagement — needs binder interface residue indices
    # ------------------------------------------------------------------
    binder_residues_per_chain = [chain_residues(pred, c) for c in binder_ids]
    target_residues_per_chain = [chain_residues(pred, c) for c in target_ids]
    iface = interface_residue_indices(binder_residues_per_chain, target_residues_per_chain)
    # Single binder chain assumption (VHH): ci is 0 for all entries
    binder_iface_idx = np.array([ri for (_ci, ri) in iface], dtype=int)

    cdr = cdr_engagement_features(
        row["binder_seq"],
        binder_iface_idx,
        n_binder_residues=len(row["binder_seq"]),
    )

    # ------------------------------------------------------------------
    # Assemble output dict
    # ------------------------------------------------------------------
    out = dict(passthrough)
    out["prediction_present"] = "1"
    for k in _CONFIDENCE_KEYS:
        out[k] = str(conf_vals[k])
    for k in _GEOMETRY_KEYS:
        out[k] = str(geom_vals[k])
    for k in CDR_FEATURE_NAMES:
        out[k] = str(cdr[k])
    return out


# ---------------------------------------------------------------------------
# Batch runner with fault tolerance
# ---------------------------------------------------------------------------


def build_rows(
    rows: Iterable[dict[str, str]],
    predictions_root: Path,
    failed_log: Path,
) -> list[dict[str, str]]:
    """Run ``assemble_row`` for every row; catch failures and log them.

    On any exception for a single row, the pair_id is appended (one per line)
    to ``failed_log`` and processing continues. Returns the list of successful
    result dicts.
    """
    results: list[dict[str, str]] = []
    n_failed = 0
    for row in rows:
        try:
            results.append(assemble_row(row, predictions_root))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, Exception):
            n_failed += 1
            with failed_log.open("a") as fh:
                fh.write(row["pair_id"] + "\n")
    if n_failed:
        print(f"  {n_failed} pair(s) failed — see {failed_log}")
    return results


# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pairs: Annotated[Path, typer.Option("--pairs", help="Pairs CSV (pair_id, binder_seq, …)")],
    predictions_root: Annotated[
        Path, typer.Option("--predictions-root", help="Root of per-pair prediction directories")
    ],
    output: Annotated[Path, typer.Option("--output", help="Output CSV path")],
    failed_log: Annotated[
        Path,
        typer.Option("--failed-log", help="Path to write failed pair_ids (one per line)"),
    ],
) -> None:
    """Extract M-C structure-track features for all pairs in PAIRS CSV."""
    with pairs.open(newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    output.parent.mkdir(parents=True, exist_ok=True)
    failed_log.parent.mkdir(parents=True, exist_ok=True)

    result_rows = build_rows(all_rows, predictions_root, failed_log)
    n_built = len(result_rows)
    n_failed = len(all_rows) - n_built

    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"built={n_built} failed={n_failed}")


if __name__ == "__main__":
    app()
