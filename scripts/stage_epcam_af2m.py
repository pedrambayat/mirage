#!/usr/bin/env python3
"""Stage the SNAP EpCAM designed-binder dataset for AF2-M.

Loads the 153-example EpCAM benchmark through ``EpCAMLoader``, writes
ColabFold FASTAs plus the standard AF2-M manifest, writes a small metadata CSV,
and prints the sbatch command for manual review. Does NOT submit SLURM.

Use::

    uv run python scripts/stage_epcam_af2m.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from mirage.benchmark import get_loader
from mirage.pose_predictors.af2m import af2m_from_env
from mirage.pose_predictors.base import StagedManifest
from mirage.scorers.base import BenchmarkExample

logger = logging.getLogger("stage_epcam_af2m")


def _label_counts(examples: Iterable[BenchmarkExample]) -> dict[str, int]:
    return dict(Counter(ex.label for ex in examples))


def _write_metadata(path: Path, examples: list[BenchmarkExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "example_id",
                "label",
                "binder_format",
                "target_name",
                "source",
                "target_pdb_id",
                "vhh_no",
                "source_vhh_no",
                "scramble_variant_idx",
                "real_target",
            ]
        )
        for example in examples:
            writer.writerow(
                [
                    example.id,
                    example.label,
                    example.binder_format,
                    example.target_name,
                    example.source,
                    example.target_pdb_id or "",
                    example.metadata.get("vhh_no", ""),
                    example.metadata.get("source_vhh_no", ""),
                    example.metadata.get("scramble_variant_idx", ""),
                    example.metadata.get("real_target", ""),
                ]
            )


def _rename_manifest(manifest: StagedManifest, name: str) -> StagedManifest:
    if manifest.n_rows == 0:
        return manifest
    new_path = manifest.path.with_name(name)
    manifest.path.rename(new_path)
    return StagedManifest(
        path=new_path,
        n_rows=manifest.n_rows,
        n_already_cached=manifest.n_already_cached,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="SNAP binder-discrimination/data directory; defaults to MIRAGE_EPCAM_DATA.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional path for the EpCAM manifest metadata CSV.",
    )
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()

    loader_kwargs: dict[str, Path] = {}
    if args.data_dir is not None:
        loader_kwargs["data_dir"] = args.data_dir
    examples = list(get_loader("epcam", **loader_kwargs).load())
    if not examples:
        logger.error("EpCAM loader yielded no examples; aborting")
        return 1

    logger.info("loaded %d EpCAM examples; labels=%s", len(examples), _label_counts(examples))
    for example in examples[:5]:
        logger.info("  %s  label=%s  target=%s", example.id, example.label, example.target_name)
    if len(examples) > 5:
        logger.info("  ... (%d more)", len(examples) - 5)

    predictor = af2m_from_env()
    logger.info("output_root=%s", predictor.output_root)
    logger.info("staged_root=%s", predictor.staged_root)
    logger.info("slurm_script=%s", predictor.slurm_script)

    manifest = predictor.stage(examples)
    date = datetime.now().strftime("%Y%m%d")
    manifest = _rename_manifest(manifest, f"manifest_{date}_epcam.tsv")

    metadata_path = args.metadata_output
    if metadata_path is None:
        metadata_path = predictor.staged_root / f"manifest_{date}_epcam_metadata.csv"
    _write_metadata(metadata_path, examples)

    print()
    print("=" * 60)
    print("EpCAM AF2-M inputs staged.")
    print(f"  metadata:        {metadata_path}")
    print(f"  total examples:  {len(examples)}")
    print(f"  label counts:    {_label_counts(examples)}")
    print(f"  queued examples: {manifest.n_rows}")
    print(f"  cached:          {manifest.n_already_cached}")

    if manifest.n_rows == 0:
        print("  manifest:        <none; all examples already cached>")
        print()
        print("Nothing to submit.")
        print("=" * 60)
        return 0

    cmd = predictor.sbatch_command(
        manifest, chunk_size=args.chunk_size, max_concurrent=args.max_concurrent
    )
    n_tasks = (manifest.n_rows + args.chunk_size - 1) // args.chunk_size
    print(f"  manifest:        {manifest.path}")
    print(f"  chunk_size:      {args.chunk_size}")
    print(f"  array tasks:     {n_tasks} (indices 0-{n_tasks - 1})")
    print(f"  max concurrent:  {args.max_concurrent}")
    print()
    print("To submit after approval:")
    print("  " + " ".join(cmd))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
