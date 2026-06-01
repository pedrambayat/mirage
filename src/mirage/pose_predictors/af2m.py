"""AF2-Multimer pose predictor via ColabFold.

Wraps a SLURM array job that runs `colabfold_batch` on chunks of a manifest.
Defaults match the 2026-05-12 lock-in:

* ``--model-type alphafold2_multimer_v3``
* ``--num-models 5``
* ``--num-recycle 3``
* ``--num-seeds 1``
* ``--msa-mode mmseqs2_uniref_env``
* ``--pair-mode unpaired_paired``
* ``--random-seed 0``
* no ``--templates`` (avoids crystal-template leakage on SAbDab)
* no ``--amber`` (relaxation is optional and roughly doubles GPU time)

Output layout (one directory per example)::

    <output_root>/<example_id>/
        rank1.pdb            # symlink to the rank_001 unrelaxed PDB
        scores.json          # iPTM, pTM, mean pLDDT per model + settings
        <colabfold raw output ...>

Staging layout::

    <staged_root>/
        fasta/<example_id>.fasta
        manifest_<timestamp>.tsv

Manifest TSV columns (header line + one row per example): ``example_id``,
``fasta_path``, ``out_dir``. The SLURM array task uses ``$SLURM_ARRAY_TASK_ID``
and ``chunk_size`` to pick its slice of rows.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from mirage._paths import default_af2m_predictions_root, repo_root
from mirage.pose_predictors.base import AbstractPosePredictor, StagedManifest
from mirage.scorers.base import BenchmarkExample

logger = logging.getLogger(__name__)

_DEFAULT_ACCOUNT = "dbgoodma-goodman-laboratory"
_DEFAULT_PARTITION = "b200-mig45"
_DEFAULT_TIME = "12:00:00"
_SLURM_SCRIPT_DEFAULT_RELPATH = ("scripts", "slurm", "predict_af2m.slurm")
_RUNNER_SCRIPT_NAME = "run_af2m_chunk.py"


class AF2MPosePredictor(AbstractPosePredictor):
    """ColabFold AF2-Multimer pose predictor."""

    name = "af2m"

    def __init__(
        self,
        output_root: str | Path,
        staged_root: str | Path,
        *,
        slurm_script: str | Path | None = None,
        account: str = _DEFAULT_ACCOUNT,
        partition: str = _DEFAULT_PARTITION,
        time_limit: str = _DEFAULT_TIME,
    ) -> None:
        self.output_root = Path(output_root)
        self.staged_root = Path(staged_root)
        if slurm_script is None:
            slurm_script = repo_root().joinpath(*_SLURM_SCRIPT_DEFAULT_RELPATH)
        self.slurm_script = Path(slurm_script)
        self.account = account
        self.partition = partition
        self.time_limit = time_limit

    def results_for(self, example: BenchmarkExample) -> Path | None:
        candidate = self.output_root / example.id / "rank1.pdb"
        return candidate if candidate.is_file() else None

    def stage(self, examples: Iterable[BenchmarkExample]) -> StagedManifest:
        fasta_dir = self.staged_root / "fasta"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = self.staged_root / f"manifest_{timestamp}.tsv"

        n_rows = 0
        n_cached = 0
        with manifest_path.open("w") as fh:
            fh.write("example_id\tfasta_path\tout_dir\n")
            for example in examples:
                if self.results_for(example) is not None:
                    n_cached += 1
                    continue
                fasta_path = fasta_dir / f"{example.id}.fasta"
                fasta_path.write_text(_example_to_fasta(example))
                out_dir = self.output_root / example.id
                fh.write(f"{example.id}\t{fasta_path}\t{out_dir}\n")
                n_rows += 1

        if n_rows == 0:
            manifest_path.unlink()
            logger.info("AF2M stage: nothing to do (%d already cached)", n_cached)
        else:
            logger.info(
                "AF2M stage: %d new rows queued, %d already cached, manifest=%s",
                n_rows,
                n_cached,
                manifest_path,
            )
        return StagedManifest(path=manifest_path, n_rows=n_rows, n_already_cached=n_cached)

    def submit(
        self,
        manifest: StagedManifest,
        *,
        chunk_size: int = 5,
        max_concurrent: int = 4,
        dry_run: bool = False,
    ) -> str | None:
        if manifest.n_rows == 0:
            logger.info("AF2M submit: manifest is empty, nothing to submit")
            return None
        if not self.slurm_script.is_file():
            raise FileNotFoundError(f"SLURM script not found: {self.slurm_script}")

        cmd = self.sbatch_command(manifest, chunk_size=chunk_size, max_concurrent=max_concurrent)
        n_tasks = (manifest.n_rows + chunk_size - 1) // chunk_size
        (self.output_root / "_logs").mkdir(parents=True, exist_ok=True)
        if dry_run:
            logger.info("AF2M submit (dry-run): %s", " ".join(cmd))
            return None

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = _parse_sbatch_job_id(result.stdout)
        logger.info("AF2M submit: job=%s, tasks=%d, chunk=%d", job_id, n_tasks, chunk_size)
        return job_id

    def sbatch_command(
        self,
        manifest: StagedManifest,
        *,
        chunk_size: int = 5,
        max_concurrent: int = 4,
    ) -> list[str]:
        """Build the sbatch command without submitting (for inspection / tests)."""
        n_tasks = (manifest.n_rows + chunk_size - 1) // chunk_size
        log_dir = self.output_root / "_logs"
        runner_path = self.slurm_script.parent / _RUNNER_SCRIPT_NAME
        return [
            "sbatch",
            f"--account={self.account}",
            f"--partition={self.partition}",
            "--gres=gpu:1",
            f"--time={self.time_limit}",
            f"--array=0-{n_tasks - 1}%{max_concurrent}",
            f"--output={log_dir}/af2m_%A_%a.out",
            f"--error={log_dir}/af2m_%A_%a.err",
            str(self.slurm_script),
            str(manifest.path),
            str(chunk_size),
            str(runner_path),
        ]


def _example_to_fasta(example: BenchmarkExample) -> str:
    """Serialise an example into ColabFold multimer FASTA format.

    ColabFold multimer expects one record per complex, with chains
    separated by ``:``. Order: binder chains first, then target chains
    (matches the SNAP convention so binder/target chain identification in
    the predicted PDB is consistent across the pipeline).
    """
    chains: list[str] = []
    chains.extend(_clean_sequence(s) for s in example.binder_chains)
    chains.extend(_clean_sequence(s) for s in example.target_chains)
    if not chains:
        raise ValueError(f"example {example.id} has no chains to predict")
    if any(not c for c in chains):
        raise ValueError(f"example {example.id} has an empty chain after cleaning")
    return f">{example.id}\n{':'.join(chains)}\n"


_AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")


def _clean_sequence(seq: str) -> str:
    """Uppercase, strip whitespace, and drop non-standard letters.

    ColabFold's MSA + AF2-M expect the 20 canonical amino acids only.
    """
    return "".join(c for c in seq.upper() if c in _AA_ALLOWED)


_SBATCH_RE = re.compile(r"Submitted batch job (\d+)")


def _parse_sbatch_job_id(stdout: str) -> str:
    match = _SBATCH_RE.search(stdout)
    if not match:
        raise RuntimeError(f"could not parse sbatch job id from: {stdout!r}")
    return match.group(1)


def af2m_from_env(
    *,
    output_root: str | Path | None = None,
    staged_root: str | Path | None = None,
    **kwargs: Any,
) -> AF2MPosePredictor:
    """Build an AF2MPosePredictor from env vars or sensible defaults.

    Env vars: ``MIRAGE_AF2M_OUTPUT_ROOT``, ``MIRAGE_AF2M_STAGED_ROOT``.
    Falls back to ``<repo_root>/data/raw/predictions/af2m`` and
    ``<repo_root>/data/staged/af2m``.
    """
    out = output_root or os.environ.get("MIRAGE_AF2M_OUTPUT_ROOT")
    stg = staged_root or os.environ.get("MIRAGE_AF2M_STAGED_ROOT")
    if out is None:
        out = default_af2m_predictions_root()
    if stg is None:
        stg = repo_root() / "data" / "staged" / "af2m"
    return AF2MPosePredictor(output_root=out, staged_root=stg, **kwargs)
