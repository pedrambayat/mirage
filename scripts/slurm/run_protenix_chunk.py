#!/usr/bin/env python3
"""Run one chunk of a Protenix manifest on a SLURM array task — BATCHED.

Pure stdlib — runs inside the ``protenix`` conda env which has no mirage
package and no BioPython. Shells out to ``protenix pred`` ONCE per chunk with a
combined multi-job input, so the ~40 s model load is amortized across the whole
chunk (a single complex is ~40 s load + ~30-300 s inference; batching N keeps the
load fixed). Post-processing (symlink ``rank1.cif`` per complex) is done here.

Usage::

    run_protenix_chunk.py <manifest.tsv> <array_task_id> <chunk_size>

Manifest columns (header + rows): ``example_id``, ``input_path``, ``out_dir``.
The chunk is rows ``[array_task_id * chunk_size, (array_task_id + 1) * chunk_size)``.
Each ``input_path`` is a single-job Protenix input (a JSON list with one job);
this runner merges the chunk's not-yet-done jobs into one combined input and runs
``protenix pred --input combined.json --out_dir <output_root>``. Protenix writes
``<output_root>/<example_id>/seed_0/predictions/<example_id>_sample_0.cif`` per job,
and is per-job fault tolerant (a job that fails logs a WARNING and the rest
continue), so one bad complex never sinks the chunk.

The three Blackwell env vars (``PROTENIX_ROOT_DIR``, ``LAYERNORM_TYPE``,
``MMSEQS_SERVICE_HOST_URL``) and ``conda activate protenix`` are set by
``predict_protenix.slurm``; this runner assumes it is already inside the env.

2026-05-28 Blackwell lock-in flags (do NOT change):
  --trimul_kernel torch --triatt_kernel torch --enable_fusion false
  avoids cuequivariance kernel hangs on B200 (cu128 + native LayerNorm).
  --need_atom_confidence true is REQUIRED to write *_full_data_*.json with PAE.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# 2026-05-28 lock-in. Keep in sync with ProtenixPosePredictor docstring.
PROTENIX_FLAGS: tuple[str, ...] = (
    "--seeds", "0",
    "--model_name", "protenix_base_default_v1.0.0",
    "--use_msa", "true",
    "--msa_server_mode", "colabfold",
    "--use_template", "false",
    "--use_default_params", "true",
    "--trimul_kernel", "torch",
    "--triatt_kernel", "torch",
    "--enable_fusion", "false",
    "--need_atom_confidence", "true",
)  # fmt: skip


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(f"usage: {argv[0]} <manifest.tsv> <array_task_id> <chunk_size>", file=sys.stderr)
        return 2
    manifest_path = Path(argv[1])
    task_id = int(argv[2])
    chunk_size = int(argv[3])

    rows = _read_manifest(manifest_path)
    start = task_id * chunk_size
    chunk = rows[start : start + chunk_size]
    print(
        f"[run_protenix_chunk] task={task_id} chunk_size={chunk_size} "
        f"rows=[{start}:{start + chunk_size}] selected={len(chunk)}/{len(rows)}",
        flush=True,
    )
    if not chunk:
        return 0

    # Skip already-done complexes; merge the rest into one combined input.
    combined_jobs: list[dict[str, Any]] = []
    pending: list[tuple[str, Path]] = []  # (example_id, out_dir)
    n_skipped = 0
    for row in chunk:
        out_dir = Path(row["out_dir"])
        if (out_dir / "rank1.cif").is_file():
            print(f"[run_protenix_chunk] SKIP {row['example_id']} (rank1.cif exists)", flush=True)
            n_skipped += 1
            continue
        jobs = json.loads(Path(row["input_path"]).read_text())
        combined_jobs.extend(jobs)  # each per-pair input is a 1-element list
        pending.append((row["example_id"], out_dir))

    if not pending:
        print(f"[run_protenix_chunk] task={task_id} nothing to do ({n_skipped} cached)", flush=True)
        return 0

    # All out_dirs share one parent (the dataset output root); Protenix writes
    # <output_root>/<name>/... per job when given --out_dir <output_root>.
    output_root = pending[0][1].parent
    # Work dir unique per (array job, task) so two arrays sharing an output root
    # never clobber each other's combined_input.json.
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "0"))
    work = output_root / f"_chunk_{job_id}_{task_id}"
    work.mkdir(parents=True, exist_ok=True)
    combined_path = work / "combined_input.json"
    combined_path.write_text(json.dumps(combined_jobs))

    t0 = time.time()
    rc = _run_batch(combined_path, output_root, work / "pred.log")
    elapsed = time.time() - t0
    print(
        f"[run_protenix_chunk] protenix pred rc={rc} for {len(pending)} jobs ({elapsed:.0f}s)",
        flush=True,
    )

    # Symlink rank1.cif per complex; a missing CIF = that job failed (continue).
    n_done = 0
    n_failed = 0
    for example_id, out_dir in pending:
        try:
            _post_process(out_dir, example_id)
            n_done += 1
        except Exception as exc:
            print(f"[run_protenix_chunk] FAIL {example_id}: {exc}", flush=True)
            n_failed += 1

    shutil.rmtree(work, ignore_errors=True)  # combined_input + pred.log no longer needed
    print(
        f"[run_protenix_chunk] summary task={task_id} done={n_done} "
        f"skipped={n_skipped} failed={n_failed}",
        flush=True,
    )
    return 1 if n_failed and n_done == 0 else 0


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _run_batch(combined_input: Path, output_root: Path, log_path: Path) -> int:
    cmd: list[str] = [
        "protenix", "pred",
        "--input", str(combined_input),
        "--out_dir", str(output_root),
        *PROTENIX_FLAGS,
    ]  # fmt: skip
    with log_path.open("w") as log_fh:
        log_fh.write(f"$ {' '.join(cmd)}\n")
        log_fh.flush()
        result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    return result.returncode


def _post_process(out_dir: Path, example_id: str) -> None:
    """Symlink ``rank1.cif`` to the sample_0 CIF Protenix wrote under out_dir.

    No BioPython / CIF parsing — chain resolution happens later in the mirage env.
    """
    cif_candidates = sorted(Path(out_dir).rglob("*sample_0*.cif"))
    if not cif_candidates:
        raise FileNotFoundError(f"no sample_0 CIF under {out_dir}")
    sample0_cif = cif_candidates[0]
    symlink = out_dir / "rank1.cif"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sample0_cif.relative_to(out_dir))
    # Drop diffusion samples 1-4 (each carries a ~7 MB full_data JSON); only the
    # top-ranked sample_0 is used for features. Keeps the on-disk footprint ~5x
    # smaller so the shared project quota survives the full campaign.
    for extra in Path(out_dir).rglob("*_sample_[1-9]*"):
        extra.unlink()
    print(f"[run_protenix_chunk] rank1.cif → {sample0_cif.relative_to(out_dir)}", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
