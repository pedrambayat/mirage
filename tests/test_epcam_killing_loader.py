from __future__ import annotations

import csv
from pathlib import Path

from mirage.benchmark._registry import get_loader


def _write_labels(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["vhh_id", "vhh_sequence", "label"])
        writer.writeheader()
        writer.writerow({"vhh_id": "10", "vhh_sequence": "QVQLAAAA", "label": "Good"})
        writer.writerow({"vhh_id": "14", "vhh_sequence": "EVQLBBBB", "label": "Bad"})


def test_epcam_killing_maps_good_bad_to_bind_nonbind(tmp_path: Path) -> None:
    labels = tmp_path / "epcam_killing_labels.csv"
    _write_labels(labels)
    loader = get_loader("epcam_killing", labels_csv=labels)
    examples = {e.id: e for e in loader.load()}
    assert examples["epcam-kill-10"].label == "BIND"
    assert examples["epcam-kill-14"].label == "NONBIND"
    assert examples["epcam-kill-10"].target_name == "EpCAM"
    assert examples["epcam-kill-10"].binder_format == "vhh"
    assert examples["epcam-kill-10"].metadata["assay"] == "cart_killing_aspc1"


def test_epcam_killing_requires_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    try:
        get_loader("epcam_killing", labels_csv=missing)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
