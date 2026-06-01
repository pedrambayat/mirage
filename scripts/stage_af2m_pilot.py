#!/usr/bin/env python3
"""Stage a stratified AF2-M pilot from the SAbDab loader.

Loads 909 SAbDab examples, picks a deterministic mix across VHH/Fab/scFv,
writes per-example FASTAs and a manifest TSV, and prints the sbatch
command that would submit them. Does NOT submit — submission is a manual
follow-up step gated on user review.

Usage::

    uv run python scripts/stage_af2m_pilot.py [--n 20] [--chunk-size 5] [--max-concurrent 4]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

from mirage.benchmark import get_loader
from mirage.pose_predictors.af2m import af2m_from_env
from mirage.scorers.base import BenchmarkExample

logger = logging.getLogger("stage_af2m_pilot")

_STRATA_QUOTAS_20 = {"vhh": 8, "fab": 8, "scfv": 4}


def _stratify(examples: list[BenchmarkExample], n: int) -> list[BenchmarkExample]:
    """Return a deterministic, format-stratified subset of the first N examples."""
    by_format: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for ex in examples:
        by_format[ex.binder_format].append(ex)

    if n == 20:
        quotas = _STRATA_QUOTAS_20
    else:
        # For non-20 N: proportional to global format mix, biased to keep
        # at least 1 in each stratum where available.
        total = sum(len(v) for v in by_format.values())
        quotas = {
            fmt: max(1 if vs else 0, round(n * len(vs) / total)) for fmt, vs in by_format.items()
        }
        # Trim to exactly n: shave from the largest stratum.
        while sum(quotas.values()) > n:
            biggest = max(quotas, key=lambda k: quotas[k])
            quotas[biggest] -= 1
        while sum(quotas.values()) < n:
            biggest = max(by_format, key=lambda k: len(by_format[k]))
            quotas[biggest] = quotas.get(biggest, 0) + 1

    picked: list[BenchmarkExample] = []
    for fmt, quota in quotas.items():
        picked.extend(by_format.get(fmt, [])[:quota])
    return picked


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()

    logger.info("loading SAbDab (this includes the ANARCI gate ~9 min on full TSV)")
    loader = get_loader("sabdab")
    examples = list(loader.load())
    logger.info("SAbDab yielded %d examples total", len(examples))

    pilot = _stratify(examples, args.n)
    logger.info(
        "pilot composition: %s",
        {fmt: sum(1 for e in pilot if e.binder_format == fmt) for fmt in ("vhh", "fab", "scfv")},
    )
    for ex in pilot:
        logger.info(
            "  %s  fmt=%s  H=%d  target=%d  pdb=%s",
            ex.id,
            ex.binder_format,
            len(ex.binder_chains[0]),
            sum(len(t) for t in ex.target_chains),
            ex.target_pdb_id,
        )

    predictor = af2m_from_env()
    logger.info("output_root=%s", predictor.output_root)
    logger.info("staged_root=%s", predictor.staged_root)
    logger.info("slurm_script=%s", predictor.slurm_script)

    manifest = predictor.stage(pilot)
    logger.info(
        "manifest: %s  (n_rows=%d, n_already_cached=%d)",
        manifest.path,
        manifest.n_rows,
        manifest.n_already_cached,
    )

    if manifest.n_rows == 0:
        logger.info("nothing to submit — all pilot examples already have rank1.pdb")
        return 0

    cmd = predictor.sbatch_command(
        manifest,
        chunk_size=args.chunk_size,
        max_concurrent=args.max_concurrent,
    )
    n_tasks = (manifest.n_rows + args.chunk_size - 1) // args.chunk_size
    print()
    print("============================================================")
    print("Pilot staged.")
    print(f"  manifest:        {manifest.path}")
    print(f"  examples:        {manifest.n_rows}")
    print(f"  chunk_size:      {args.chunk_size}")
    print(f"  array tasks:     {n_tasks} (indices 0-{n_tasks - 1})")
    print(f"  max concurrent:  {args.max_concurrent}")
    print()
    print("To submit:")
    print("  " + " ".join(cmd))
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
