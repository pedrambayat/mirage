#!/usr/bin/env python3
"""Precompute one ColabFold MSA per UNIQUE sequence for the Protenix campaign.

Runs inside the ``protenix`` conda env: **stdlib + subprocess only** (no mirage,
no BioPython), so it can execute wherever ``protenix`` is on PATH.

Mechanism (option a — ``protenix msa``)::

    protenix msa -i <shard.fasta> -o <out_dir> -m colabfold

queries the ColabFold MMseqs2 server (host from ``MMSEQS_SERVICE_HOST_URL``) and
writes ``<out_dir>/<i>/<i>/non_pairing.a3m`` — the per-chain unpaired MSA that a
Protenix input references via ``proteinChain.unpairedMsaPath``.

Protenix **reorders** sequences internally, so the output index ``<i>`` is NOT the
FASTA order. We therefore map each a3m back to its sequence by reading the a3m's
own **query line** (the first record's sequence, which is the verbatim input), then
symlink it into the shared cache at ``<cache>/<sha1(seq)[:16]>.a3m`` — exactly the
path the staged per-pair JSONs already point at.

Idempotent + shardable: sequences already in the cache are skipped, and
``--shard-index/--n-shards`` partitions the remaining work for parallel execution
(SLURM array or background processes).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


def sequence_hash(seq: str) -> str:
    """SHA-1 hex, first 16 chars.

    MUST stay identical to ``mirage.pose_predictors.protenix.sequence_hash`` so the
    cache keys here line up with the ``unpairedMsaPath`` references written at
    staging time. ``tests/test_precompute_msa.py`` asserts the two agree.
    """
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def cache_path_for(cache_dir: Path, seq: str) -> Path:
    return cache_dir / f"{sequence_hash(seq)}.a3m"


def read_unique_manifest(path: Path) -> list[str]:
    """Read a ``unique_seqs.txt`` (one ``"<seq>\\t<hash>"`` line per sequence);
    return the sequences in file order."""
    seqs: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        seqs.append(line.split("\t")[0])
    return seqs


def unmapped_sequences(seqs: list[str], cache_dir: Path) -> list[str]:
    """Sequences whose ``<cache>/<hash>.a3m`` does not yet exist (skip cached)."""
    return [s for s in seqs if not cache_path_for(cache_dir, s).exists()]


def shard(seqs: list[str], shard_index: int, n_shards: int) -> list[str]:
    """Round-robin partition: shard ``i`` gets indices ``i, i+n, i+2n, ...``."""
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    return [s for i, s in enumerate(seqs) if i % n_shards == shard_index]


def write_fasta(seqs: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">seq{i}\n{s}\n" for i, s in enumerate(seqs)))


def a3m_query_sequence(a3m: Path) -> str | None:
    """The query sequence of an a3m: the line after its first ``>`` header."""
    lines = a3m.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(">"):
            return lines[i + 1].strip() if i + 1 < len(lines) else None
    return None


def harvest(msa_out_dir: Path, cache_dir: Path) -> int:
    """Symlink every ``non_pairing.a3m`` under ``msa_out_dir`` into the cache,
    keyed by the a3m's own query-sequence hash. Returns the number harvested.
    Robust to Protenix's index reordering; safe to call repeatedly."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for a3m in sorted(msa_out_dir.rglob("non_pairing.a3m")):
        query = a3m_query_sequence(a3m)
        if not query:
            continue
        dest = cache_path_for(cache_dir, query)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(a3m.resolve())
        n += 1
    return n


def run_protenix_msa(fasta: Path, out_dir: Path) -> int:
    """Shell out to ``protenix msa`` (ColabFold mode). Env vars + conda activation
    are the caller's responsibility (the SLURM wrapper / launch shell)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["protenix", "msa", "-i", str(fasta), "-o", str(out_dir), "-m", "colabfold"]
    print(f"[precompute_msa] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unique-manifest", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument(
        "--dry-run", action="store_true", help="write the shard FASTA but do not call protenix"
    )
    args = ap.parse_args(argv)

    seqs = read_unique_manifest(args.unique_manifest)
    todo = unmapped_sequences(seqs, args.cache_dir)
    mine = shard(todo, args.shard_index, args.n_shards)
    print(
        f"[precompute_msa] total={len(seqs)} todo={len(todo)} "
        f"shard={args.shard_index}/{args.n_shards} mine={len(mine)}",
        flush=True,
    )
    if not mine:
        return 0

    fasta = args.work_dir / f"shard_{args.shard_index}.fasta"
    out_dir = args.work_dir / f"shard_{args.shard_index}_out"
    write_fasta(mine, fasta)
    if args.dry_run:
        print(
            f"[precompute_msa] DRY-RUN: would run protenix msa -i {fasta} -o {out_dir}", flush=True
        )
        return 0

    rc = run_protenix_msa(fasta, out_dir)
    harvested = harvest(out_dir, args.cache_dir)
    print(
        f"[precompute_msa] protenix_rc={rc} harvested={harvested} -> {args.cache_dir}", flush=True
    )
    # Success if we cached something even when protenix returned nonzero on a few queries.
    return 0 if harvested > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
