#!/usr/bin/env python3
"""Stage CDR-scrambled negative controls for AF2-M.

Selects N parent binders from the published N=200 SAbDab union manifest
(biased toward near-native rmsd<8Å rows, format-stratified for the
remainder), reads each parent's binder + target sequences from the
already-staged AF2-M FASTAs, runs ANARCI per binder chain to identify
Chothia CDR boundaries (handles VHH single-domain, Fab heavy+light,
scFv multi-domain on a single chain), shuffles residues *within* each
CDR independently with a fixed numpy seed (framework untouched), and
emits new FASTAs + a manifest TSV ready for the existing AF2-M SLURM
pipeline.

Output::

    data/staged/af2m/fasta/<parent_id>-scramble<k>.fasta
    data/staged/af2m/manifest_<date>_scrambles.tsv

Decisions locked 2026-05-13 session:

* 30 parents x 1 scramble = 30 predictions (~3.5 h on b200-mig45).
* Format-stratified across VHH/Fab/scFv, biased toward near-native
  parents (all rmsd<8Å rows first, format-stratify the remainder).
* All CDRs shuffled (H1+H2+H3 on heavy; L1+L2+L3 also for Fab/scFv).
* Within-CDR position permutation (preserves AA composition, breaks
  order) — matches SNAP's scramble methodology.

Use::

    uv run python scripts/stage_cdr_scrambles.py \\
        --n-parents 30 --n-scrambles 1 --seed 20260514

Does NOT submit SLURM. Prints the sbatch command at the end for
manual review + submission, mirroring stage_af2m_pilot.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mirage.benchmark.sabdab import _prepend_path, _resolve_hmmer_bin
from mirage.pose_predictors.af2m import af2m_from_env
from mirage.pose_predictors.base import StagedManifest
from mirage.scorers.base import BenchmarkExample

logger = logging.getLogger("stage_cdr_scrambles")

# Chothia CDR position ranges (inclusive). Insertion-coded positions
# (e.g. 100A, 100B in H3) inside [lo, hi] are picked up automatically by
# the ANARCI walk below.
_CHOTHIA_CDR_RANGES_H = {"H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}
_CHOTHIA_CDR_RANGES_L = {"L1": (24, 34), "L2": (50, 56), "L3": (89, 97)}

_NEAR_NATIVE_RMSD_A = 8.0


def _read_union_manifest(path: Path) -> list[str]:
    with path.open() as fh:
        return [r["example_id"] for r in csv.DictReader(fh, delimiter="\t")]


def _read_rmsd_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                rmsd = float(r["value"])
            except (TypeError, ValueError):
                rmsd = float("nan")
            rows.append(
                {
                    "example_id": r["example_id"],
                    "rmsd": rmsd,
                    "binder_format": r.get("binder_format", ""),
                    "target_name": r.get("target_name", ""),
                }
            )
    return rows


def _select_parents(
    rmsd_rows: list[dict[str, Any]],
    ids_in_scope: set[str],
    n_parents: int,
) -> list[dict[str, Any]]:
    """Pick N parents biased toward near-native, format-stratified remainder.

    Step 1: take every rmsd<8Å row in scope (deterministic, sorted by
    example_id). Step 2: fill the remainder by format-stratifying the
    rmsd≥8Å pool proportional to its format mix, again deterministic.
    """
    in_scope = [r for r in rmsd_rows if r["example_id"] in ids_in_scope]
    near = sorted(
        (r for r in in_scope if r["rmsd"] < _NEAR_NATIVE_RMSD_A),
        key=lambda r: r["example_id"],
    )
    far = sorted(
        (r for r in in_scope if not (r["rmsd"] < _NEAR_NATIVE_RMSD_A)),
        key=lambda r: r["example_id"],
    )
    selected = near[:n_parents]
    remaining = n_parents - len(selected)
    if remaining <= 0:
        return selected[:n_parents]

    by_fmt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in far:
        by_fmt[r["binder_format"]].append(r)
    total = sum(len(v) for v in by_fmt.values())
    if total == 0:
        return selected

    quotas = {
        fmt: max(1 if vs else 0, round(remaining * len(vs) / total)) for fmt, vs in by_fmt.items()
    }
    while sum(quotas.values()) > remaining:
        biggest = max(quotas, key=lambda k: quotas[k])
        quotas[biggest] -= 1
    while sum(quotas.values()) < remaining:
        biggest = max(by_fmt, key=lambda k: len(by_fmt[k]))
        quotas[biggest] = quotas.get(biggest, 0) + 1

    for fmt, q in quotas.items():
        selected.extend(by_fmt.get(fmt, [])[:q])
    return selected[:n_parents]


def _parse_fasta_chains(path: Path) -> tuple[str, ...]:
    """Return chain sequences from a ColabFold multimer FASTA (`a:b:c` form)."""
    text_lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(text_lines) < 2:
        raise ValueError(f"FASTA too short: {path}")
    return tuple(text_lines[1].split(":"))


def _split_binder_target(
    chains: tuple[str, ...], binder_format: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reverse _example_to_fasta: binder chains first, then target chains.

    VHH and scFv have one binder chain; Fab has two (heavy + light).
    Format strings come from SAbDabLoader._binder_format.
    """
    n_binder = 2 if binder_format == "fab" else 1
    return chains[:n_binder], chains[n_binder:]


def _anarci_cdr_indices(seq: str, hmmer_bin: str) -> list[tuple[str, list[int]]]:
    """For one binder chain, return [(loop_name, sequence_indices)] per CDR.

    Handles single-domain chains (VHH heavy, Fab heavy, Fab light) and
    multi-domain chains (scFv with V_H + V_L on one chain). Returns an
    empty list if ANARCI cannot annotate the sequence.
    """
    from anarci import anarci  # type: ignore[import-untyped]

    with _prepend_path(hmmer_bin):
        results = anarci([("q", seq)], scheme="chothia", hmmerpath=hmmer_bin)
    numbering_list, details_list, _ = results
    if not numbering_list or numbering_list[0] is None:
        return []

    out: list[tuple[str, list[int]]] = []
    for domain_idx, (domain_numbering, seq_start, _seq_end) in enumerate(numbering_list[0]):
        chain_type = ""
        if details_list and details_list[0] and domain_idx < len(details_list[0]):
            chain_type = str(details_list[0][domain_idx].get("chain_type", ""))
        if chain_type == "H":
            ranges = _CHOTHIA_CDR_RANGES_H
        elif chain_type in ("K", "L"):
            ranges = _CHOTHIA_CDR_RANGES_L
        else:
            continue
        per_loop: dict[str, list[int]] = defaultdict(list)
        seq_idx = seq_start
        for (pos, _ins), aa in domain_numbering:
            if aa == "-":
                continue
            for name, (lo, hi) in ranges.items():
                if lo <= pos <= hi:
                    per_loop[name].append(seq_idx)
            seq_idx += 1
        for name, idxs in per_loop.items():
            if idxs:
                out.append((f"{chain_type}{name[1:]}", idxs))
    return out


def _shuffle_within_cdrs(
    seq: str, cdrs: list[tuple[str, list[int]]], rng: np.random.Generator
) -> str:
    """Permute residues within each CDR independently. Framework untouched."""
    chars = list(seq)
    for _name, idxs in cdrs:
        residues = [chars[i] for i in idxs]
        permuted = rng.permutation(len(residues))
        for dst, src in zip(idxs, permuted, strict=True):
            chars[dst] = residues[src]
    return "".join(chars)


def _build_scramble(
    parent_row: dict[str, Any],
    fasta_dir: Path,
    hmmer_bin: str,
    rng: np.random.Generator,
    scramble_idx: int,
) -> BenchmarkExample | None:
    """Construct one scramble example from a parent_row. Returns None on failure."""
    parent_id = parent_row["example_id"]
    binder_format = parent_row["binder_format"]
    fasta_path = fasta_dir / f"{parent_id}.fasta"
    if not fasta_path.is_file():
        logger.warning("missing parent FASTA: %s", fasta_path)
        return None

    chains = _parse_fasta_chains(fasta_path)
    binder_chains, target_chains = _split_binder_target(chains, binder_format)
    if not binder_chains or not target_chains:
        logger.warning("%s: empty binder/target after split (fmt=%s)", parent_id, binder_format)
        return None

    scrambled_chains: list[str] = []
    cdrs_per_chain: list[list[tuple[str, list[int]]]] = []
    for chain_seq in binder_chains:
        cdrs = _anarci_cdr_indices(chain_seq, hmmer_bin)
        if not cdrs:
            logger.warning("%s: ANARCI found no CDRs on binder chain (skipping parent)", parent_id)
            return None
        scrambled_chains.append(_shuffle_within_cdrs(chain_seq, cdrs, rng))
        cdrs_per_chain.append(cdrs)

    new_id = f"{parent_id}-scramble{scramble_idx:02d}"
    metadata: dict[str, Any] = {
        "scramble_parent_id": parent_id,
        "scramble_index": scramble_idx,
        "scramble_cdr_indices": [
            {name: idxs for name, idxs in per_chain} for per_chain in cdrs_per_chain
        ],
        "scramble_parent_rmsd_to_crystal": parent_row.get("rmsd"),
        "scramble_parent_near_native": (
            float(parent_row.get("rmsd", float("nan"))) < _NEAR_NATIVE_RMSD_A
        ),
        "binder_format": binder_format,
        "target_name": parent_row.get("target_name", ""),
    }
    return BenchmarkExample(
        id=new_id,
        label="SCRAMBLE",
        binder_chains=tuple(scrambled_chains),
        binder_format=binder_format,
        target_chains=target_chains,
        target_name=parent_row.get("target_name", ""),
        source="sabdab-cdr-scramble",
        target_pdb_id=None,
        complex_pdb_path=None,
        metadata=metadata,
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
    parser.add_argument("--n-parents", type=int, default=30)
    parser.add_argument(
        "--n-scrambles", type=int, default=1, help="Scrambled variants per parent (default 1)."
    )
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument(
        "--hmmer-bin",
        type=str,
        default=None,
        help=(
            "Override directory containing hmmscan "
            "(defaults to the mber conda env per SAbDab loader convention)."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()

    union_ids = set(_read_union_manifest(args.union_manifest))
    rmsd_rows = _read_rmsd_csv(args.rmsd_csv)
    parent_rows = _select_parents(rmsd_rows, union_ids, args.n_parents)
    logger.info(
        "selected %d parents (target=%d): formats=%s, near_native=%d",
        len(parent_rows),
        args.n_parents,
        dict(
            (fmt, sum(1 for r in parent_rows if r["binder_format"] == fmt))
            for fmt in sorted({r["binder_format"] for r in parent_rows})
        ),
        sum(1 for r in parent_rows if r["rmsd"] < _NEAR_NATIVE_RMSD_A),
    )

    hmmer_bin = _resolve_hmmer_bin(args.hmmer_bin)
    rng = np.random.default_rng(args.seed)

    scrambles: list[BenchmarkExample] = []
    for parent_row in parent_rows:
        for k in range(1, args.n_scrambles + 1):
            ex = _build_scramble(parent_row, args.fasta_dir, hmmer_bin, rng, k)
            if ex is not None:
                scrambles.append(ex)

    if not scrambles:
        logger.error("no scrambles generated; aborting")
        return 1

    logger.info("generated %d scramble examples", len(scrambles))
    for ex in scrambles[:5]:
        logger.info(
            "  example: %s  fmt=%s  H_len=%d",
            ex.id,
            ex.binder_format,
            len(ex.binder_chains[0]),
        )
    if len(scrambles) > 5:
        logger.info("  ... (%d more)", len(scrambles) - 5)

    predictor = af2m_from_env()
    manifest = predictor.stage(scrambles)

    new_path = manifest.path.with_name(
        f"manifest_{datetime.now().strftime('%Y%m%d')}_scrambles.tsv"
    )
    manifest.path.rename(new_path)
    manifest = StagedManifest(
        path=new_path,
        n_rows=manifest.n_rows,
        n_already_cached=manifest.n_already_cached,
    )

    if manifest.n_rows == 0:
        logger.info("nothing to submit (all cached); manifest=%s", manifest.path)
        return 0

    cmd = predictor.sbatch_command(
        manifest, chunk_size=args.chunk_size, max_concurrent=args.max_concurrent
    )
    n_tasks = (manifest.n_rows + args.chunk_size - 1) // args.chunk_size
    print()
    print("=" * 60)
    print("Scrambles staged.")
    print(f"  manifest:        {manifest.path}")
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
