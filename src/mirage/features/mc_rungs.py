"""Nested rung feature matrices for the M-C structure-track discriminator.

Reads B1's per-pair Protenix feature CSV (data/staged/mc/*_features.csv) and builds
the cumulative rung feature matrices: 0=ipTM, 1=+confidence internals, 2=+interface
geometry, 3=+CDR engagement. The raw ``interface_plddt`` column (74% NaN) is replaced
by a derived ``interface_plddt_missing`` 0/1 flag; ``cdr_mapping_ok`` is never a
feature (near-constant; its failure rate is reported elsewhere as a diagnostic).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

# Single source of truth for which columns each cumulative rung uses.
RUNG_COLUMNS: dict[int, list[str]] = {
    0: ["iptm"],
    1: [
        "iptm",
        "ptm",
        "interface_pae",
        "min_interface_pae",
        "mean_plddt",
        "interface_plddt_missing",
    ],
    2: [
        "iptm",
        "ptm",
        "interface_pae",
        "min_interface_pae",
        "mean_plddt",
        "interface_plddt_missing",
        "n_interface_residues_binder",
        "n_interface_residues_target",
        "buried_sasa_proxy_a2",
        "atom_contacts_5a",
        "shape_complementarity_proxy",
        "atom_clash_fraction_2a",
    ],
    3: [
        "iptm",
        "ptm",
        "interface_pae",
        "min_interface_pae",
        "mean_plddt",
        "interface_plddt_missing",
        "n_interface_residues_binder",
        "n_interface_residues_target",
        "buried_sasa_proxy_a2",
        "atom_contacts_5a",
        "shape_complementarity_proxy",
        "atom_clash_fraction_2a",
        "cdr_contact_fraction",
        "cdr1_contact_fraction",
        "cdr2_contact_fraction",
        "cdr3_contact_fraction",
    ],
}

_MISSING_TOKENS = frozenset({"", "nan", "none"})


def read_feature_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def derive_missing_flag(raw_interface_plddt: str) -> float:
    """1.0 when interface_plddt is absent (empty / NaN), else 0.0.

    The missingness is itself discriminative: most non-cognate pairs have no
    high-confidence cross-chain contacts to average pLDDT over.
    """
    token = raw_interface_plddt.strip().lower()
    if token in _MISSING_TOKENS:
        return 1.0
    try:
        return 1.0 if math.isnan(float(raw_interface_plddt)) else 0.0
    except ValueError:
        return 1.0


def _cell(row: dict[str, str], column: str) -> float:
    if column == "interface_plddt_missing":
        return derive_missing_flag(row.get("interface_plddt", ""))
    return float(row[column])


def rung_matrix(rows: list[dict[str, str]], *, rung: int) -> tuple[np.ndarray[Any, Any], list[str]]:
    """Build the cumulative feature matrix for ``rung`` (0-3).

    Returns ``(x, feature_names)`` with ``x`` shape ``(len(rows), len(names))`` and
    no non-finite entries (the only NaN-prone source, interface_plddt, is consumed
    as the missingness flag, never as a continuous value).
    """
    names = RUNG_COLUMNS[rung]
    x = np.array([[_cell(r, c) for c in names] for r in rows], dtype=float)
    return x, list(names)


def labels_array(rows: list[dict[str, str]]) -> np.ndarray[Any, Any]:
    return np.array([int(r["label"]) for r in rows], dtype=int)


def folds_array(rows: list[dict[str, str]]) -> np.ndarray[Any, Any]:
    """The M-S held-out-antigen-cluster fold assignment, read straight from the CSV."""
    return np.array([int(r["fold"]) for r in rows], dtype=int)


def cdr_mapping_failure_rate(rows: list[dict[str, str]]) -> float:
    """Fraction of rows where ANARCI failed to map the CDR onto the binder chain."""
    if not rows:
        return math.nan
    ok = sum(1 for r in rows if float(r["cdr_mapping_ok"]) >= 0.5)
    return 1.0 - ok / len(rows)
