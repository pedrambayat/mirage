#!/usr/bin/env python3
"""Stage wrong-target SAbDab re-pairings for AF2-M.

Builds synthetic negative controls by taking a staged SAbDab binder from the
published N=200 AF2-M union manifest and pairing it with a different staged
SAbDab antigen. This keeps the binder sequence intact and changes only the
intended target, which is a harder specificity negative than CDR scrambling.

Default policy:

* pick binders from the existing N=200 rows, prioritizing rmsd<8A parents;
* generate two wrong-targets for each of 15 binders (30 predictions total);
* require the donor target sequence and target name to differ from the binder
  parent's target;
* write AF2-M FASTAs plus the standard three-column manifest;
* write a metadata CSV mapping each synthetic example to binder parent and
  wrong-target donor.

Use::

    uv run python scripts/stage_wrong_target_repairings.py

Does NOT submit SLURM. Prints the sbatch command for review and submission.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from mirage.pose_predictors.af2m import af2m_from_env
from mirage.pose_predictors.base import StagedManifest
from mirage.scorers.base import BenchmarkExample

logger = logging.getLogger("stage_wrong_target_repairings")

_NEAR_NATIVE_RMSD_A = 8.0


@dataclass(frozen=True)
class _StagedParent:
    example_id: str
    binder_format: str
    target_name: str
    rmsd: float
    binder_chains: tuple[str, ...]
    target_chains: tuple[str, ...]


@dataclass(frozen=True)
class _Pairing:
    example: BenchmarkExample
    binder_parent_id: str
    target_donor_id: str
    binder_parent_rmsd: float
    target_donor_name: str


def _read_union_manifest(path: Path) -> list[str]:
    with path.open(newline="") as fh:
        return [r["example_id"] for r in csv.DictReader(fh, delimiter="\t")]


def _read_rmsd_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fh:
        return {r["example_id"]: r for r in csv.DictReader(fh)}


def _parse_colabfold_fasta(path: Path) -> tuple[str, ...]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"FASTA too short: {path}")
    return tuple(lines[1].split(":"))


def _split_chains(
    chains: tuple[str, ...], binder_format: str, example_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    n_binder = 2 if binder_format == "fab" else 1
    binder = chains[:n_binder]
    target = chains[n_binder:]
    if not binder or not target:
        raise ValueError(
            f"{example_id}: cannot split FASTA chains for binder_format={binder_format!r}"
        )
    return binder, target


def _load_staged_parents(
    union_ids: Iterable[str],
    rmsd_rows: dict[str, dict[str, str]],
    fasta_dir: Path,
) -> list[_StagedParent]:
    parents: list[_StagedParent] = []
    for example_id in union_ids:
        row = rmsd_rows.get(example_id)
        if row is None:
            logger.warning("missing RMSD row for %s", example_id)
            continue
        binder_format = row.get("binder_format", "")
        fasta_path = fasta_dir / f"{example_id}.fasta"
        if not fasta_path.is_file():
            logger.warning("missing staged FASTA for %s: %s", example_id, fasta_path)
            continue
        try:
            rmsd = float(row["value"])
        except (KeyError, TypeError, ValueError):
            rmsd = float("nan")
        chains = _parse_colabfold_fasta(fasta_path)
        binder_chains, target_chains = _split_chains(chains, binder_format, example_id)
        parents.append(
            _StagedParent(
                example_id=example_id,
                binder_format=binder_format,
                target_name=row.get("target_name", ""),
                rmsd=rmsd,
                binder_chains=binder_chains,
                target_chains=target_chains,
            )
        )
    return parents


def _target_key(parent: _StagedParent) -> tuple[str, tuple[str, ...]]:
    return parent.target_name.lower(), parent.target_chains


def _select_binders(parents: list[_StagedParent], n_binders: int) -> list[_StagedParent]:
    near = sorted(
        (p for p in parents if p.rmsd < _NEAR_NATIVE_RMSD_A),
        key=lambda p: (p.rmsd, p.example_id),
    )
    far = sorted(
        (p for p in parents if not (p.rmsd < _NEAR_NATIVE_RMSD_A)),
        key=lambda p: (p.rmsd, p.example_id),
    )
    return (near + far)[:n_binders]


def _build_pairings(
    parents: list[_StagedParent],
    *,
    n_binders: int,
    targets_per_binder: int,
    seed: int,
) -> list[_Pairing]:
    binders = _select_binders(parents, n_binders)
    rng = np.random.default_rng(seed)
    pairings: list[_Pairing] = []
    for binder_parent in binders:
        candidates = [
            donor
            for donor in parents
            if donor.example_id != binder_parent.example_id
            and _target_key(donor) != _target_key(binder_parent)
        ]
        if len(candidates) < targets_per_binder:
            logger.warning(
                "%s: only %d eligible wrong-target donors; requested %d",
                binder_parent.example_id,
                len(candidates),
                targets_per_binder,
            )
        if not candidates:
            continue
        order = rng.permutation(len(candidates))
        chosen = [candidates[i] for i in order[:targets_per_binder]]
        for idx, donor in enumerate(chosen, start=1):
            example_id = f"{binder_parent.example_id}-wrongtarget{idx:02d}"
            metadata = {
                "wrong_target_binder_parent_id": binder_parent.example_id,
                "wrong_target_donor_id": donor.example_id,
                "wrong_target_binder_parent_rmsd_to_crystal": binder_parent.rmsd,
                "wrong_target_donor_target_name": donor.target_name,
                "binder_format": binder_parent.binder_format,
            }
            example = BenchmarkExample(
                id=example_id,
                label="WRONG_TARGET",
                binder_chains=binder_parent.binder_chains,
                binder_format=binder_parent.binder_format,
                target_chains=donor.target_chains,
                target_name=donor.target_name,
                source="sabdab-wrong-target",
                target_pdb_id=None,
                complex_pdb_path=None,
                metadata=metadata,
            )
            pairings.append(
                _Pairing(
                    example=example,
                    binder_parent_id=binder_parent.example_id,
                    target_donor_id=donor.example_id,
                    binder_parent_rmsd=binder_parent.rmsd,
                    target_donor_name=donor.target_name,
                )
            )
    return pairings


def _write_pairing_metadata(path: Path, pairings: list[_Pairing]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "example_id",
                "label",
                "binder_parent_id",
                "target_donor_id",
                "binder_parent_rmsd_to_crystal",
                "binder_format",
                "target_name",
            ]
        )
        for pairing in pairings:
            writer.writerow(
                [
                    pairing.example.id,
                    pairing.example.label,
                    pairing.binder_parent_id,
                    pairing.target_donor_id,
                    pairing.binder_parent_rmsd,
                    pairing.example.binder_format,
                    pairing.target_donor_name,
                ]
            )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--union-manifest",
        type=Path,
        default=Path("data/staged/af2m/manifest_20260513_n200_union.tsv"),
    )
    parser.add_argument(
        "--rmsd-csv",
        type=Path,
        default=Path("results/published/sabdab_af2m_rmsd_n200.csv"),
    )
    parser.add_argument(
        "--fasta-dir",
        type=Path,
        default=Path("data/staged/af2m/fasta"),
        help="Directory holding the existing per-parent ColabFold FASTAs.",
    )
    parser.add_argument("--n-binders", type=int, default=15)
    parser.add_argument("--targets-per-binder", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional path for the wrong-target pairing metadata CSV.",
    )
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()

    parents = _load_staged_parents(
        _read_union_manifest(args.union_manifest),
        _read_rmsd_rows(args.rmsd_csv),
        args.fasta_dir,
    )
    if not parents:
        logger.error("no staged parents loaded; aborting")
        return 1
    pairings = _build_pairings(
        parents,
        n_binders=args.n_binders,
        targets_per_binder=args.targets_per_binder,
        seed=args.seed,
    )
    if not pairings:
        logger.error("no wrong-target pairings generated; aborting")
        return 1

    unique_binder_ids = {p.binder_parent_id for p in pairings}
    near_native_binder_ids = {
        p.binder_parent_id for p in pairings if p.binder_parent_rmsd < _NEAR_NATIVE_RMSD_A
    }
    logger.info(
        "generated %d wrong-target pairings from %d binders; near_native_binders=%d",
        len(pairings),
        len(unique_binder_ids),
        len(near_native_binder_ids),
    )
    for pairing in pairings[:5]:
        logger.info(
            "  example: %s  binder=%s  target_donor=%s",
            pairing.example.id,
            pairing.binder_parent_id,
            pairing.target_donor_id,
        )
    if len(pairings) > 5:
        logger.info("  ... (%d more)", len(pairings) - 5)

    predictor = af2m_from_env()
    manifest = predictor.stage([p.example for p in pairings])
    date = datetime.now().strftime("%Y%m%d")
    new_path = manifest.path.with_name(f"manifest_{date}_wrong_targets.tsv")
    manifest.path.rename(new_path)
    manifest = StagedManifest(
        path=new_path,
        n_rows=manifest.n_rows,
        n_already_cached=manifest.n_already_cached,
    )

    metadata_path = args.metadata_output
    if metadata_path is None:
        metadata_path = manifest.path.with_name(f"manifest_{date}_wrong_targets_metadata.csv")
    _write_pairing_metadata(metadata_path, pairings)

    if manifest.n_rows == 0:
        logger.info("nothing to submit (all cached); manifest=%s", manifest.path)
        logger.info("metadata=%s", metadata_path)
        return 0

    cmd = predictor.sbatch_command(
        manifest, chunk_size=args.chunk_size, max_concurrent=args.max_concurrent
    )
    n_tasks = (manifest.n_rows + args.chunk_size - 1) // args.chunk_size
    print()
    print("=" * 60)
    print("Wrong-target re-pairings staged.")
    print(f"  manifest:        {manifest.path}")
    print(f"  metadata:        {metadata_path}")
    print(f"  examples:        {manifest.n_rows}")
    print(f"  cached:          {manifest.n_already_cached}")
    print(f"  chunk_size:      {args.chunk_size}")
    print(f"  array tasks:     {n_tasks} (indices 0-{n_tasks - 1})")
    print(f"  max concurrent:  {args.max_concurrent}")
    print()
    print("To submit:")
    print("  " + " ".join(cmd))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
