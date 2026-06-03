"""Protenix pose predictor via SLURM array.

Wraps a SLURM array job that runs ``protenix pred`` on chunks of a manifest.
Defaults match the 2026-05-28 Blackwell smoke run lock-in:

* ``--model_name protenix_base_default_v1.0.0``
* ``--seeds 0``
* ``--use_msa true`` with ColabFold server (``--msa_server_mode colabfold``)
* ``--use_template false`` (avoids crystal-template leakage on SAbDab)
* ``--trimul_kernel torch --triatt_kernel torch`` (Blackwell/cu128 workaround)
* ``--enable_fusion false`` (Blackwell workaround)
* ``--need_atom_confidence true`` (writes ``*_full_data_*.json`` with PAE matrix)

MSA reuse: when ``unpairedMsaPath`` in the per-chain JSON points to a cached
``.a3m`` file, Protenix skips the MSA server for that chain. The MSA cache is
populated once per unique sequence by a separate task; this wrapper only writes
the references.

Output layout (one directory per example)::

    <output_root>/<example_id>/
        rank1.cif                          # symlink to sample_0 CIF
        <protenix nested raw output ...>

Staging layout::

    <staged_root>/
        inputs/<example_id>.json
        manifest_<timestamp>.tsv

Manifest TSV columns (header + one row per example): ``example_id``,
``input_path``, ``out_dir``. The SLURM array task uses ``$SLURM_ARRAY_TASK_ID``
and ``chunk_size`` to pick its slice of rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from mirage._paths import default_protenix_predictions_root, repo_root
from mirage.pose_predictors.base import AbstractPosePredictor, StagedManifest
from mirage.scorers.base import BenchmarkExample

logger = logging.getLogger(__name__)

_DEFAULT_ACCOUNT = "dbgoodma-goodman-laboratory"
_DEFAULT_PARTITION = "dgx-b200"
_DEFAULT_QOS = "dgx"
_DEFAULT_TIME = "08:00:00"
_SLURM_SCRIPT_DEFAULT_RELPATH = ("scripts", "slurm", "predict_protenix.slurm")
_RUNNER_SCRIPT_NAME = "run_protenix_chunk.py"

_AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")


def sequence_hash(seq: str) -> str:
    """Return the first 16 hex characters of the SHA-1 digest of ``seq``.

    Canonical home for this helper. Imported by ``scripts/stage_protenix_pairs.py``
    (which re-exports it for backward compatibility) and any downstream task that
    needs deterministic per-sequence filenames.
    """
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def _clean_sequence(seq: str) -> str:
    """Uppercase, strip whitespace, and drop non-standard residue letters.

    Protenix expects the 20 canonical amino acids only.
    """
    return "".join(c for c in seq.upper() if c in _AA_ALLOWED)


_SBATCH_RE = re.compile(r"Submitted batch job (\d+)")


def _parse_sbatch_job_id(stdout: str) -> str:
    match = _SBATCH_RE.search(stdout)
    if not match:
        raise RuntimeError(f"could not parse sbatch job id from: {stdout!r}")
    return match.group(1)


class ProtenixPosePredictor(AbstractPosePredictor):
    """Protenix pose predictor via a SLURM array job."""

    name = "protenix"

    def __init__(
        self,
        output_root: str | Path,
        staged_root: str | Path,
        *,
        msa_cache_dir: str | Path | None = None,
        slurm_script: str | Path | None = None,
        account: str = _DEFAULT_ACCOUNT,
        partition: str = _DEFAULT_PARTITION,
        qos: str = _DEFAULT_QOS,
        time_limit: str = _DEFAULT_TIME,
    ) -> None:
        self.output_root = Path(output_root)
        self.staged_root = Path(staged_root)
        if msa_cache_dir is None:
            msa_cache_dir = repo_root() / "data" / "staged" / "protenix" / "msa_cache"
        self.msa_cache_dir = Path(msa_cache_dir)
        if slurm_script is None:
            slurm_script = repo_root().joinpath(*_SLURM_SCRIPT_DEFAULT_RELPATH)
        self.slurm_script = Path(slurm_script)
        self.account = account
        self.partition = partition
        self.qos = qos
        self.time_limit = time_limit

    def results_for(self, example: BenchmarkExample) -> Path | None:
        candidate = self.output_root / example.id / "rank1.cif"
        return candidate if candidate.is_file() else None

    def stage(self, examples: Iterable[BenchmarkExample]) -> StagedManifest:
        inputs_dir = self.staged_root / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = self.staged_root / f"manifest_{timestamp}.tsv"

        n_rows = 0
        n_cached = 0
        with manifest_path.open("w") as fh:
            fh.write("example_id\tinput_path\tout_dir\n")
            for example in examples:
                if self.results_for(example) is not None:
                    n_cached += 1
                    continue

                input_path = inputs_dir / f"{example.id}.json"
                out_dir = self.output_root / example.id
                self._write_input_json(example, input_path)
                fh.write(f"{example.id}\t{input_path}\t{out_dir}\n")
                n_rows += 1

        if n_rows == 0:
            manifest_path.unlink()
            logger.info("Protenix stage: nothing to do (%d already cached)", n_cached)
        else:
            logger.info(
                "Protenix stage: %d new rows queued, %d already cached, manifest=%s",
                n_rows,
                n_cached,
                manifest_path,
            )
        return StagedManifest(path=manifest_path, n_rows=n_rows, n_already_cached=n_cached)

    def _write_input_json(self, example: BenchmarkExample, path: Path) -> None:
        """Write the Protenix per-pair input JSON for ``example``."""
        sequences: list[dict[str, Any]] = []
        for seq in example.binder_chains:
            clean = _clean_sequence(seq)
            sequences.append(
                {
                    "proteinChain": {
                        "sequence": clean,
                        "count": 1,
                        "unpairedMsaPath": str(self.msa_cache_dir / f"{sequence_hash(clean)}.a3m"),
                    }
                }
            )
        for seq in example.target_chains:
            clean = _clean_sequence(seq)
            sequences.append(
                {
                    "proteinChain": {
                        "sequence": clean,
                        "count": 1,
                        "unpairedMsaPath": str(self.msa_cache_dir / f"{sequence_hash(clean)}.a3m"),
                    }
                }
            )
        payload: dict[str, Any] = {"name": example.id, "sequences": sequences}
        path.write_text(json.dumps(payload, indent=2))

    def submit(
        self,
        manifest: StagedManifest,
        *,
        chunk_size: int = 8,
        max_concurrent: int = 16,
        dry_run: bool = False,
    ) -> str | None:
        if manifest.n_rows == 0:
            logger.info("Protenix submit: manifest is empty, nothing to submit")
            return None
        if not self.slurm_script.is_file():
            raise FileNotFoundError(f"SLURM script not found: {self.slurm_script}")

        cmd = self.sbatch_command(manifest, chunk_size=chunk_size, max_concurrent=max_concurrent)
        n_tasks = (manifest.n_rows + chunk_size - 1) // chunk_size
        (self.output_root / "_logs").mkdir(parents=True, exist_ok=True)
        if dry_run:
            logger.info("Protenix submit (dry-run): %s", " ".join(cmd))
            return None

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = _parse_sbatch_job_id(result.stdout)
        logger.info(
            "Protenix submit: job=%s, tasks=%d, chunk=%d",
            job_id,
            n_tasks,
            chunk_size,
        )
        return job_id

    def sbatch_command(
        self,
        manifest: StagedManifest,
        *,
        chunk_size: int = 8,
        max_concurrent: int = 16,
    ) -> list[str]:
        """Build the sbatch command without submitting (for inspection / tests)."""
        n_tasks = (manifest.n_rows + chunk_size - 1) // chunk_size
        log_dir = self.output_root / "_logs"
        runner_path = self.slurm_script.parent / _RUNNER_SCRIPT_NAME
        return [
            "sbatch",
            f"--account={self.account}",
            f"--partition={self.partition}",
            f"--qos={self.qos}",
            "--gres=gpu:1",
            f"--time={self.time_limit}",
            f"--array=0-{n_tasks - 1}%{max_concurrent}",
            f"--output={log_dir}/protenix_%A_%a.out",
            f"--error={log_dir}/protenix_%A_%a.err",
            str(self.slurm_script),
            str(manifest.path),
            str(chunk_size),
            str(runner_path),
        ]


def protenix_from_env(
    *,
    output_root: str | Path | None = None,
    staged_root: str | Path | None = None,
    **kwargs: Any,
) -> ProtenixPosePredictor:
    """Build a ProtenixPosePredictor from env vars or sensible defaults.

    Env vars: ``MIRAGE_PROTENIX_OUTPUT_ROOT``, ``MIRAGE_PROTENIX_STAGED_ROOT``.
    Falls back to ``<repo_root>/data/raw/predictions/protenix`` and
    ``<repo_root>/data/staged/protenix``.
    """
    out = output_root or os.environ.get("MIRAGE_PROTENIX_OUTPUT_ROOT")
    stg = staged_root or os.environ.get("MIRAGE_PROTENIX_STAGED_ROOT")
    if out is None:
        out = default_protenix_predictions_root()
    if stg is None:
        stg = repo_root() / "data" / "staged" / "protenix"
    return ProtenixPosePredictor(output_root=out, staged_root=stg, **kwargs)
