from __future__ import annotations

import os
from pathlib import Path

import pytest

SNAP_EPCAM_FALLBACKS: tuple[Path, ...] = (
    # PARCC: SNAP clone on Betty project storage.
    Path(
        "/vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/snap/outputs/data"
    ),
    # Laptop: SNAP clone inside the Obsidian Google Drive vault.
    Path(
        "/Users/pedrambayat/Library/CloudStorage/GoogleDrive-pbayat123@gmail.com/My Drive/"
        "second brain/2 - Source Material/Research/SNAP/snap/binder-discrimination/data"
    ),
)


@pytest.fixture
def epcam_data_dir() -> Path:
    override = os.environ.get("MIRAGE_EPCAM_DATA")
    if override:
        candidate = Path(override)
        if candidate.is_dir():
            return candidate
        pytest.skip(f"EpCAM data dir not available: {candidate}")
    for fallback in SNAP_EPCAM_FALLBACKS:
        if fallback.is_dir():
            return fallback
    pytest.skip("EpCAM data dir not available on any known fallback")


SABDAB_DEFAULT = Path(
    "/vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/abdisc-data/sabdab"
)


@pytest.fixture
def sabdab_data_dir() -> Path:
    override = os.environ.get("MIRAGE_SABDAB_DATA")
    candidate = Path(override) if override else SABDAB_DEFAULT
    if not candidate.is_dir():
        pytest.skip(f"SAbDab data dir not available: {candidate}")
    if not (candidate / "summary.tsv").is_file():
        pytest.skip(f"SAbDab summary.tsv missing in {candidate}")
    return candidate
