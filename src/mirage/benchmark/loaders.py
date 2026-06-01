"""Concrete loaders. New loaders register themselves via @register_loader."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from pathlib import Path

from mirage.benchmark._registry import AbstractLoader, register_loader
from mirage.benchmark.targets import EPCAM_ECD
from mirage.scorers.base import BenchmarkExample


@register_loader("epcam")
class EpCAMLoader(AbstractLoader):
    """SNAP EpCAM VHH dataset: 43 POS, 86 SCR, 24 OFF binders against EpCAM ECD.

    Reads the three CSVs that live in SNAP's `binder-discrimination/data/`
    directory. All examples share `target_name='EpCAM'` and target sequence
    EPCAM_ECD; the literature negatives carry their real (non-EpCAM) targets
    in metadata for reference.
    """

    POS_FILE = "epcam_positives_filtered.csv"
    SCR_FILE = "epcam_scrambled_negatives.csv"
    OFF_FILE = "epcam_literature_negatives.csv"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        resolved = data_dir if data_dir is not None else os.environ.get("MIRAGE_EPCAM_DATA")
        if resolved is None:
            raise ValueError(
                "EpCAMLoader needs data_dir or MIRAGE_EPCAM_DATA env var "
                "pointing at SNAP's binder-discrimination/data/ directory."
            )
        self.data_dir = Path(resolved)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

    def load(self) -> Iterator[BenchmarkExample]:
        yield from self._load_positives()
        yield from self._load_scrambled()
        yield from self._load_off_target()

    def _load_positives(self) -> Iterator[BenchmarkExample]:
        path = self.data_dir / self.POS_FILE
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                yield BenchmarkExample(
                    id=f"epcam-pos-{row['vhh_no']}",
                    label="POS",
                    binder_chains=(row["vhh_sequence"],),
                    binder_format="vhh",
                    target_chains=(EPCAM_ECD,),
                    target_name="EpCAM",
                    source="epcam-snap",
                    target_pdb_id="4MZV",
                    metadata={
                        "vhh_no": row["vhh_no"],
                        "cdr3_sequence": row.get("cdr3_sequence", ""),
                        "cluster_id": row.get("cluster_id", ""),
                    },
                )

    def _load_scrambled(self) -> Iterator[BenchmarkExample]:
        path = self.data_dir / self.SCR_FILE
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                yield BenchmarkExample(
                    id=row["vhh_id"],
                    label="SCR",
                    binder_chains=(row["vhh_sequence"],),
                    binder_format="vhh",
                    target_chains=(EPCAM_ECD,),
                    target_name="EpCAM",
                    source="epcam-snap",
                    target_pdb_id="4MZV",
                    metadata={
                        "source_vhh_no": row.get("source_vhh_no", ""),
                        "scramble_variant_idx": row.get("scramble_variant_idx", ""),
                        "negative_type": row.get("negative_type", ""),
                    },
                )

    def _load_off_target(self) -> Iterator[BenchmarkExample]:
        path = self.data_dir / self.OFF_FILE
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                yield BenchmarkExample(
                    id=f"epcam-off-{row['vhh_id']}",
                    label="OFF",
                    binder_chains=(row["sequence"],),
                    binder_format="vhh",
                    target_chains=(EPCAM_ECD,),
                    target_name="EpCAM",
                    source="epcam-snap",
                    target_pdb_id="4MZV",
                    metadata={
                        "real_target": row.get("target", ""),
                        "literature_source": row.get("source", ""),
                    },
                )
