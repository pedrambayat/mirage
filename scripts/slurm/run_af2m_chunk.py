#!/usr/bin/env python3
"""Run one chunk of a ColabFold AF2-Multimer manifest.

Invoked from the SLURM array task. Pure stdlib so it can run under either
the mirage uv env or a bare system Python. Shells out to
``colabfold_batch`` for the actual GPU work; does post-processing
(symlink ``rank1.pdb``, write consolidated ``scores.json``) in Python.

Usage::

    run_af2m_chunk.py <manifest.tsv> <array_task_id> <chunk_size>

Manifest columns (header + rows): ``example_id``, ``fasta_path``, ``out_dir``.
The chunk is rows ``[array_task_id * chunk_size, (array_task_id + 1) * chunk_size)``.

Environment overrides:

* ``MIRAGE_COLABFOLD_BIN`` — path to ``colabfold_batch`` (default: workspace localcolabfold)
* ``MIRAGE_AF2M_EXTRA_FLAGS`` — extra flags appended to the colabfold_batch call
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_COLABFOLD_BIN = (
    "/vast/projects/dbgoodma/goodman-laboratory/pbayat/"
    "localcolabfold/.pixi/envs/default/bin/colabfold_batch"
)

# 2026-05-12 lock-in. Keep in sync with src/mirage/pose_predictors/af2m.py docstring.
COLABFOLD_FLAGS: tuple[str, ...] = (
    "--model-type", "alphafold2_multimer_v3",
    "--num-models", "5",
    "--num-recycle", "3",
    "--num-seeds", "1",
    "--msa-mode", "mmseqs2_uniref_env",
    "--pair-mode", "unpaired_paired",
    "--random-seed", "0",
)  # fmt: skip

MIRAGE_VERSION_KEY = "mirage_predict_af2m_version"
MIRAGE_VERSION = "1"


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(f"usage: {argv[0]} <manifest.tsv> <array_task_id> <chunk_size>", file=sys.stderr)
        return 2
    manifest_path = Path(argv[1])
    task_id = int(argv[2])
    chunk_size = int(argv[3])

    rows = _read_manifest(manifest_path)
    start = task_id * chunk_size
    end = start + chunk_size
    chunk = rows[start:end]
    print(
        f"[run_af2m_chunk] task={task_id} chunk_size={chunk_size} "
        f"rows=[{start}:{end}] selected={len(chunk)}/{len(rows)}",
        flush=True,
    )

    colabfold_bin = os.environ.get("MIRAGE_COLABFOLD_BIN", DEFAULT_COLABFOLD_BIN)
    if not Path(colabfold_bin).is_file():
        print(f"[run_af2m_chunk] FATAL colabfold_batch not at {colabfold_bin}", file=sys.stderr)
        return 1

    extra_flags = tuple(f for f in os.environ.get("MIRAGE_AF2M_EXTRA_FLAGS", "").split() if f)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for row in chunk:
        example_id = row["example_id"]
        fasta_path = Path(row["fasta_path"])
        out_dir = Path(row["out_dir"])

        if (out_dir / "rank1.pdb").is_file():
            print(f"[run_af2m_chunk] SKIP {example_id} (rank1.pdb exists)", flush=True)
            n_skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ok = _run_one(colabfold_bin, fasta_path, out_dir, extra_flags)
        elapsed = time.time() - t0
        if not ok:
            print(f"[run_af2m_chunk] FAIL {example_id} ({elapsed:.1f}s)", flush=True)
            n_failed += 1
            continue

        try:
            _post_process(out_dir, example_id)
        except Exception as exc:
            print(f"[run_af2m_chunk] POSTPROCESS FAIL {example_id}: {exc}", flush=True)
            n_failed += 1
            continue

        print(f"[run_af2m_chunk] DONE {example_id} ({elapsed:.1f}s)", flush=True)
        n_done += 1

    print(
        f"[run_af2m_chunk] summary task={task_id} done={n_done} "
        f"skipped={n_skipped} failed={n_failed}",
        flush=True,
    )
    return 1 if n_failed and n_done == 0 else 0


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _run_one(
    colabfold_bin: str,
    fasta_path: Path,
    out_dir: Path,
    extra_flags: tuple[str, ...],
) -> bool:
    cmd: list[str] = [colabfold_bin, *COLABFOLD_FLAGS, *extra_flags, str(fasta_path), str(out_dir)]
    log_path = out_dir / "log.txt"
    with log_path.open("w") as log_fh:
        log_fh.write(f"$ {' '.join(cmd)}\n")
        log_fh.flush()
        result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    return result.returncode == 0


_RANK_001_PDB_RE = re.compile(r"_unrelaxed_rank_001_.*\.pdb$")
_RANK_001_JSON_RE = re.compile(r"_scores_rank_001_.*\.json$")
_RANKED_JSON_RE = re.compile(r"_scores_rank_(\d+)_.*\.json$")


def _post_process(out_dir: Path, example_id: str) -> None:
    """Symlink the rank-1 unrelaxed PDB and write a flat scores.json.

    `scores.json` carries the headline metrics every downstream scorer
    needs (iPTM, pTM, mean pLDDT) plus per-rank breakdowns and the
    settings used. Heavy raw outputs (PAE matrix, per-residue pLDDT) stay
    in the ColabFold files in the same directory.
    """
    pdb_candidates = [p for p in out_dir.iterdir() if _RANK_001_PDB_RE.search(p.name)]
    if not pdb_candidates:
        raise FileNotFoundError(f"no rank_001 unrelaxed PDB in {out_dir}")
    rank1_pdb = pdb_candidates[0]
    symlink = out_dir / "rank1.pdb"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(rank1_pdb.name)

    per_rank: list[dict[str, float | int | str]] = []
    for path in sorted(out_dir.iterdir()):
        match = _RANKED_JSON_RE.search(path.name)
        if not match:
            continue
        with path.open() as fh:
            payload = json.load(fh)
        per_rank.append(
            {
                "rank": int(match.group(1)),
                "pdb": _matching_pdb_for_json(out_dir, path).name,
                "iptm": float(payload.get("iptm", float("nan"))),
                "ptm": float(payload.get("ptm", float("nan"))),
                "mean_plddt": _mean(payload.get("plddt", [])),
                "max_pae": float(payload.get("max_pae", float("nan"))),
            }
        )
    per_rank.sort(key=lambda d: int(d["rank"]))

    top = per_rank[0] if per_rank else {}
    scores = {
        "example_id": example_id,
        "model_type": "alphafold2_multimer_v3",
        "settings": {
            "num_models": 5,
            "num_recycle": 3,
            "num_seeds": 1,
            "msa_mode": "mmseqs2_uniref_env",
            "pair_mode": "unpaired_paired",
            "random_seed": 0,
            "templates": False,
            "amber": False,
        },
        "rank1": {k: top.get(k) for k in ("iptm", "ptm", "mean_plddt", "max_pae", "pdb")},
        "per_rank": per_rank,
        MIRAGE_VERSION_KEY: MIRAGE_VERSION,
    }
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True))


def _matching_pdb_for_json(out_dir: Path, json_path: Path) -> Path:
    pdb_name = json_path.name.replace("_scores_", "_unrelaxed_").replace(".json", ".pdb")
    candidate = out_dir / pdb_name
    if candidate.is_file():
        return candidate
    return json_path


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
