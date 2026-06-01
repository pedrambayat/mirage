"""Abstract pose-predictor interface.

The three-phase lifecycle mirrors how cluster GPU jobs actually work:

1. `stage(examples)` — write per-example inputs (FASTAs, manifest TSV) onto
   shared storage. Idempotent: examples whose predictions are already
   cached are skipped.
2. `submit(manifest)` — dispatch a SLURM array over the manifest. Returns
   the SLURM job id so the caller can poll or wait via `squeue`/`sacct`.
3. `results_for(example)` — once jobs finish, return the cached predicted
   PDB path (or None if the prediction has not landed yet).

Concrete predictors (`AF2MPosePredictor`, future `ProtenixPosePredictor`,
…) implement the three methods. The wrapper does not import GPU libraries;
the SLURM script does, in its own conda/pixi env.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mirage.scorers.base import BenchmarkExample


@dataclass(frozen=True)
class StagedManifest:
    """Pointer to a freshly staged batch of examples, ready for submission.

    `path` — TSV with columns (example_id, fasta_path, out_dir).
    `n_rows` — examples in the manifest (i.e. needing prediction).
    `n_already_cached` — examples skipped because results already exist.
    """

    path: Path
    n_rows: int
    n_already_cached: int


class AbstractPosePredictor(ABC):
    """Predictor contract: stage → submit → results_for."""

    name: str = ""

    @abstractmethod
    def results_for(self, example: BenchmarkExample) -> Path | None:
        """Return the cached predicted-complex PDB path, or None."""

    @abstractmethod
    def stage(self, examples: Iterable[BenchmarkExample]) -> StagedManifest:
        """Write per-example inputs and a manifest TSV; skip cached ones."""

    @abstractmethod
    def submit(
        self,
        manifest: StagedManifest,
        *,
        chunk_size: int = 5,
        max_concurrent: int = 4,
        dry_run: bool = False,
    ) -> str | None:
        """Submit `manifest` as a SLURM array; return the job id or None on dry-run."""
