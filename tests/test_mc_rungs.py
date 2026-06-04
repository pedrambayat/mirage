from __future__ import annotations

import math

import numpy as np

from mirage.features.mc_rungs import (
    RUNG_COLUMNS,
    derive_missing_flag,
    rung_matrix,
)


def _row(**over: str) -> dict[str, str]:
    base = {
        "pair_id": "p",
        "label": "1",
        "antigen_cluster": "0",
        "fold": "0",
        "prediction_present": "1",
        "iptm": "0.5",
        "ptm": "0.6",
        "interface_pae": "12.0",
        "min_interface_pae": "5.0",
        "interface_plddt": "70.0",
        "mean_plddt": "80.0",
        "n_interface_residues_binder": "10",
        "n_interface_residues_target": "12",
        "buried_sasa_proxy_a2": "900.0",
        "atom_contacts_5a": "40",
        "shape_complementarity_proxy": "0.6",
        "atom_clash_fraction_2a": "0.01",
        "cdr_contact_fraction": "0.7",
        "cdr1_contact_fraction": "0.2",
        "cdr2_contact_fraction": "0.2",
        "cdr3_contact_fraction": "0.3",
        "cdr_mapping_ok": "1.0",
    }
    base.update(over)
    return base


def test_rung_columns_are_nested_and_sized() -> None:
    assert RUNG_COLUMNS[0] == ["iptm"]
    assert len(RUNG_COLUMNS[1]) == 6
    assert len(RUNG_COLUMNS[2]) == 12
    assert len(RUNG_COLUMNS[3]) == 16
    # nested: each rung is a prefix-superset of the previous
    for k in (1, 2, 3):
        assert RUNG_COLUMNS[k][: len(RUNG_COLUMNS[k - 1])] == RUNG_COLUMNS[k - 1]


def test_derive_missing_flag_marks_empty_and_nan() -> None:
    assert derive_missing_flag("") == 1.0
    assert derive_missing_flag("nan") == 1.0
    assert derive_missing_flag("NaN") == 1.0
    assert derive_missing_flag("70.0") == 0.0


def test_rung_matrix_builds_missing_flag_column() -> None:
    rows = [_row(interface_plddt=""), _row(interface_plddt="70.0")]
    x, names = rung_matrix(rows, rung=1)
    assert names == RUNG_COLUMNS[1]
    flag_col = names.index("interface_plddt_missing")
    assert x[0, flag_col] == 1.0
    assert x[1, flag_col] == 0.0
    # raw interface_plddt is never a feature column
    assert "interface_plddt" not in names
    # every value finite (no NaN leaks into the design matrix)
    assert np.isfinite(x).all()


def test_rung_matrix_rung0_is_single_iptm_column() -> None:
    rows = [_row(iptm="0.42")]
    x, names = rung_matrix(rows, rung=0)
    assert names == ["iptm"]
    assert x.shape == (1, 1)
    assert math.isclose(x[0, 0], 0.42)
