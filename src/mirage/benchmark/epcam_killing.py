"""Labeled-EpCAM loader: designed VHHs with REAL CAR-T killing labels vs AsPC1
(EpCAM+). Held-out orthogonal test in the designed-binder deployment regime.

Label = functional killing (one step downstream of binding): ``Good`` -> ``BIND``,
``Bad`` -> ``NONBIND``. N is tiny (14) — test-only, report with wide CIs.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from mirage.benchmark._registry import AbstractLoader, register_loader
from mirage.benchmark.targets import EPCAM_ECD
from mirage.scorers.base import BenchmarkExample

_LABEL_MAP = {"Good": "BIND", "Bad": "NONBIND"}


@register_loader("epcam_killing")
class EpCAMKillingLoader(AbstractLoader):
    """Reads an owner-authored ``epcam_killing_labels.csv`` (vhh_id, vhh_sequence, label)."""

    def __init__(self, labels_csv: str | Path) -> None:
        self.labels_csv = Path(labels_csv)
        if not self.labels_csv.is_file():
            raise FileNotFoundError(f"EpCAM killing labels CSV not found: {self.labels_csv}")

    def load(self) -> Iterator[BenchmarkExample]:
        with self.labels_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                raw = row["label"].strip()
                if raw not in _LABEL_MAP:
                    raise ValueError(f"Unexpected EpCAM killing label {raw!r}; expected Good/Bad")
                yield BenchmarkExample(
                    id=f"epcam-kill-{row['vhh_id']}",
                    label=_LABEL_MAP[raw],
                    binder_chains=(row["vhh_sequence"],),
                    binder_format="vhh",
                    target_chains=(EPCAM_ECD,),
                    target_name="EpCAM",
                    source="epcam-killing",
                    target_pdb_id="4MZV",
                    metadata={"assay": "cart_killing_aspc1", "vhh_id": row["vhh_id"]},
                )
