#!/usr/bin/env python3
"""Run one chunk of a Protenix manifest on a SLURM array task.

Pure stdlib — runs inside the ``protenix`` conda env which has no mirage
package and no BioPython. Shells out to ``protenix pred`` for the actual GPU
work; post-processing (symlink ``rank1.cif``) is done here in Python.

Usage::

    run_protenix_chunk.py <manifest.tsv> <array_task_id> <chunk_size>

Manifest columns (header + rows): ``example_id``, ``input_path``, ``out_dir``.
The chunk is rows ``[array_task_id * chunk_size, (array_task_id + 1) * chunk_size)``.

The three Blackwell env vars (``PROTENIX_ROOT_DIR``, ``LAYERNORM_TYPE``,
``MMSEQS_SERVICE_HOST_URL``) and ``conda activate protenix`` are set by
``predict_protenix.slurm``; this runner assumes it is already inside the
activated env.

2026-05-28 Blackwell lock-in flags (do NOT change):
  --trimul_kernel torch --triatt_kernel torch --enable_fusion false
  avoids cuequivariance kernel hangs on B200 (cu128 + native LayerNorm).
  --need_atom_confidence true is REQUIRED to write *_full_data_*.json with PAE.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

# 2026-05-28 lock-in. Keep in sync with ProtenixPosePredictor docstring.
# These flags do NOT include --input / --out_dir (supplied per row).
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

MIRAGE_VERSION_KEY = "mirage_predict_protenix_version"
MIRAGE_VERSION = "1"


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            f"usage: {argv[0]} <manifest.tsv> <array_task_id> <chunk_size>",
            file=sys.stderr,
        )
        return 2
    manifest_path = Path(argv[1])
    task_id = int(argv[2])
    chunk_size = int(argv[3])

    rows = _read_manifest(manifest_path)
    start = task_id * chunk_size
    end = start + chunk_size
    chunk = rows[start:end]
    print(
        f"[run_protenix_chunk] task={task_id} chunk_size={chunk_size} "
        f"rows=[{start}:{end}] selected={len(chunk)}/{len(rows)}",
        flush=True,
    )

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for row in chunk:
        example_id = row["example_id"]
        input_path = Path(row["input_path"])
        out_dir = Path(row["out_dir"])

        if (out_dir / "rank1.cif").is_file():
            print(
                f"[run_protenix_chunk] SKIP {example_id} (rank1.cif exists)",
                flush=True,
            )
            n_skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ok = _run_one(input_path, out_dir)
        elapsed = time.time() - t0
        if not ok:
            print(
                f"[run_protenix_chunk] FAIL {example_id} ({elapsed:.1f}s)",
                flush=True,
            )
            n_failed += 1
            continue

        try:
            _post_process(out_dir, example_id)
        except Exception as exc:
            print(
                f"[run_protenix_chunk] POSTPROCESS FAIL {example_id}: {exc}",
                flush=True,
            )
            n_failed += 1
            continue

        print(
            f"[run_protenix_chunk] DONE {example_id} ({elapsed:.1f}s)",
            flush=True,
        )
        n_done += 1

    print(
        f"[run_protenix_chunk] summary task={task_id} done={n_done} "
        f"skipped={n_skipped} failed={n_failed}",
        flush=True,
    )
    return 1 if n_failed and n_done == 0 else 0


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _run_one(input_path: Path, out_dir: Path) -> bool:
    cmd: list[str] = [
        "protenix",
        "pred",
        "--input", str(input_path),
        "--out_dir", str(out_dir),
        *PROTENIX_FLAGS,
    ]  # fmt: skip
    log_path = out_dir / "log.txt"
    with log_path.open("w") as log_fh:
        log_fh.write(f"$ {' '.join(cmd)}\n")
        log_fh.flush()
        result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    return result.returncode == 0


def _post_process(out_dir: Path, example_id: str) -> None:
    """Symlink ``rank1.cif`` to the sample_0 CIF written by Protenix.

    No BioPython / CIF parsing here — chain resolution happens later in the
    mirage env. We only create the stable alias that ``results_for`` checks.
    """
    cif_candidates = list(Path(out_dir).rglob("*sample_0*.cif"))
    if not cif_candidates:
        raise FileNotFoundError(f"no sample_0 CIF found under {out_dir} for {example_id}")
    # Pick the first (usually only) match; rglob order is implementation-defined
    # but deterministic within a run.
    sample0_cif = cif_candidates[0]
    symlink = out_dir / "rank1.cif"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    # Relative symlink so the output dir is relocatable
    symlink.symlink_to(sample0_cif.relative_to(out_dir))
    print(
        f"[run_protenix_chunk] rank1.cif → {sample0_cif.relative_to(out_dir)}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
