"""AVIDa-hIL6 loader: sequence-only VHH / IL-6-family binding labels with REAL
assay-based negatives. Held-out orthogonal test for the mirage gate."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from mirage.benchmark._registry import AbstractLoader, register_loader
from mirage.scorers.base import BenchmarkExample


@register_loader("avida")
class AvidaLoader(AbstractLoader):
    """Reads the staged CSV produced by ``scripts/stage_avida.py``.

    Label mapping: ``1`` -> ``BIND``, ``0`` -> ``NONBIND`` (real assay negatives).
    """

    def __init__(self, staged_csv: str | Path) -> None:
        self.staged_csv = Path(staged_csv)
        if not self.staged_csv.is_file():
            raise FileNotFoundError(f"AVIDa staged CSV not found: {self.staged_csv}")

    def load(self) -> Iterator[BenchmarkExample]:
        with self.staged_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                yield BenchmarkExample(
                    id=row["vhh_id"],
                    label="BIND" if row["label"] == "1" else "NONBIND",
                    binder_chains=(row["vhh_sequence"],),
                    binder_format="vhh",
                    target_chains=(row["antigen_sequence"],),
                    target_name=row.get("antigen_label", "IL6-family"),
                    source="avida-hil6",
                    metadata={"raw_label": row["label"]},
                )
