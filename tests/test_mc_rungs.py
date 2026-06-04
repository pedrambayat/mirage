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


def test_derive_missing_flag_marks_empty_nan_and_inf() -> None:
    assert derive_missing_flag("") == 1.0
    assert derive_missing_flag("nan") == 1.0
    assert derive_missing_flag("NaN") == 1.0
    assert derive_missing_flag("none") == 1.0
    assert derive_missing_flag("inf") == 1.0
    assert derive_missing_flag("-inf") == 1.0
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


def test_fit_rung_model_reproduces_ms_recipe() -> None:
    # Separable toy: iptm high for positives. Folds given explicitly (as the CSV
    # column would supply). The frozen model must (a) be an MsModel with the rung's
    # feature names, (b) carry OOF scores matching a direct oof_logistic_scores call.
    import numpy as np

    from mirage.features.mc_rungs import fit_rung_model, rung_matrix
    from mirage.ml.core import oof_logistic_scores
    from mirage.model.ms import MsModel

    rows = []
    for i in range(40):
        lab = 1 if i % 2 == 0 else 0
        iptm = 0.7 if lab == 1 else 0.3
        rows.append(_row(label=str(lab), fold=str(i % 5), iptm=str(iptm)))
    x, names = rung_matrix(rows, rung=0)
    y = np.array([int(r["label"]) for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])

    model, oof = fit_rung_model(x, y, folds, feature_names=names, l2=1.0, target_precision=0.9)
    assert isinstance(model, MsModel)
    assert model.feature_names == names
    expected_oof = oof_logistic_scores(x, y.astype(float), folds, l2=1.0)
    np.testing.assert_allclose(oof, expected_oof, equal_nan=True)
    # threshold is chosen on the full-fit model's OWN logits, not the OOF scores:
    # re-deriving the threshold from the full-fit logits must reproduce it exactly.
    import math

    from mirage.eval.gate import choose_threshold_for_precision

    full_logits = model.predict_logit(x)
    assert np.isfinite(full_logits).all()
    recomputed = choose_threshold_for_precision(full_logits, y.astype(int), target_precision=0.9)
    assert math.isclose(model.threshold, recomputed)
