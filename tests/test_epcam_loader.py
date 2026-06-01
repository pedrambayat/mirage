from __future__ import annotations

from pathlib import Path

import pytest

from mirage.benchmark import get_loader, list_loaders
from mirage.benchmark.targets import EPCAM_ECD


def test_epcam_registered_by_default() -> None:
    assert "epcam" in list_loaders()


def test_epcam_loader_requires_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIRAGE_EPCAM_DATA", raising=False)
    with pytest.raises(ValueError, match="data_dir"):
        get_loader("epcam")


def test_epcam_loader_rejects_missing_dir() -> None:
    with pytest.raises(FileNotFoundError):
        get_loader("epcam", data_dir=Path("/nonexistent/path/mirage-test"))


def test_epcam_loader_yields_expected_counts(epcam_data_dir: Path) -> None:
    loader = get_loader("epcam", data_dir=epcam_data_dir)
    examples = list(loader.load())
    by_label: dict[str, int] = {}
    for ex in examples:
        by_label[ex.label] = by_label.get(ex.label, 0) + 1
    assert by_label == {"POS": 43, "SCR": 86, "OFF": 24}


def test_epcam_loader_examples_shape(epcam_data_dir: Path) -> None:
    loader = get_loader("epcam", data_dir=epcam_data_dir)
    examples = list(loader.load())
    pos = next(ex for ex in examples if ex.label == "POS")
    assert pos.binder_format == "vhh"
    assert len(pos.binder_chains) == 1
    assert pos.binder_chains[0].startswith("QVQ")  # VHH framework signature
    assert pos.target_chains == (EPCAM_ECD,)
    assert pos.target_name == "EpCAM"
    assert pos.source == "epcam-snap"
    assert pos.target_pdb_id == "4MZV"

    off = next(ex for ex in examples if ex.label == "OFF")
    assert off.metadata["real_target"]  # literature negatives carry their real target
