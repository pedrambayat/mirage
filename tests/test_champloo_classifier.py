from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def _load(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- staging: filtering + label generation --------------------------------


def test_filter_included_keeps_only_false() -> None:
    stage = _load("stage_champloo_pairs.py")
    rows = [
        {"pdb_id": "AAAA", "is_excluded": "FALSE"},
        {"pdb_id": "BBBB", "is_excluded": "TRUE"},
        {"pdb_id": "CCCC", "is_excluded": "false"},
        {"pdb_id": "DDDD", "is_excluded": "True"},
    ]
    kept = stage.filter_included(rows)
    assert {r["pdb_id"] for r in kept} == {"AAAA", "CCCC"}


def test_metadata_by_pdb_collapses_duplicates() -> None:
    stage = _load("stage_champloo_pairs.py")
    rows = [
        {"pdb_id": "AAAA", "vhh_length": "120"},
        {"pdb_id": "AAAA", "vhh_length": "121"},
        {"pdb_id": "BBBB", "vhh_length": "130"},
    ]
    by_pdb = stage.metadata_by_pdb(rows)
    assert set(by_pdb) == {"AAAA", "BBBB"}
    assert by_pdb["AAAA"]["_n_systems"] == "2"  # first wins, count preserved
    assert by_pdb["AAAA"]["vhh_length"] == "120"


def test_build_pair_table_labels_and_drops_missing() -> None:
    stage = _load("stage_champloo_pairs.py")
    meta = {
        "AAAA": {"vhh_length": "120", "antigen_length": "200", "resolution": "2.0"},
        "BBBB": {"vhh_length": "118", "antigen_length": "150", "resolution": "2.5"},
    }
    pdb_ids = ["AAAA", "BBBB"]
    nan = float("nan")
    matrix = {
        "AAAA": {"AAAA": 0.9, "BBBB": 0.2},
        "BBBB": {"AAAA": nan, "BBBB": 0.8},  # one missing off-diagonal cell
    }
    rows = stage.build_pair_table(meta, pdb_ids, matrix, predictor="af3")
    by_id = {r["pair_id"]: r for r in rows}
    # cognate diagonal pairs are positive
    assert by_id["AAAA__AAAA"]["label"] == 1
    assert by_id["BBBB__BBBB"]["label"] == 1
    # off-diagonal is negative
    assert by_id["AAAA__BBBB"]["label"] == 0
    # missing-ipTM pair dropped
    assert "BBBB__AAAA" not in by_id
    assert len(rows) == 3
    # metadata joined from the correct side (VHH from row PDB, antigen from col PDB)
    assert by_id["AAAA__BBBB"]["vhh_length"] == "120"
    assert by_id["AAAA__BBBB"]["antigen_length"] == "150"


# --- split behavior --------------------------------------------------------


def test_held_out_group_split_has_no_group_leakage() -> None:
    analyze = _load("analyze_champloo_classifier.py")
    groups = np.asarray([f"g{i // 4}" for i in range(40)])
    folds = analyze.assign_folds(groups, n_splits=5, seed=1)
    # every member of a group lands in exactly one fold
    for g in np.unique(groups):
        assert np.unique(folds[groups == g]).size == 1


def test_assign_folds_is_deterministic() -> None:
    analyze = _load("analyze_champloo_classifier.py")
    groups = np.asarray([f"g{i}" for i in range(20)])
    a = analyze.assign_folds(groups, n_splits=4, seed=7)
    b = analyze.assign_folds(groups, n_splits=4, seed=7)
    assert np.array_equal(a, b)


def test_oof_logistic_recovers_separable_signal() -> None:
    analyze = _load("analyze_champloo_classifier.py")
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, 0.5, 30), rng.normal(3, 0.5, 30)])[:, None]
    y = np.concatenate([np.zeros(30), np.ones(30)])
    folds = np.tile(np.arange(5), 12)
    scores = analyze.oof_logistic_scores(x, y, folds, l2=1.0)
    assert np.isfinite(scores).all()
    assert analyze._auroc(scores, y.astype(int)) > 0.95


# --- metric calculation ----------------------------------------------------


def test_auroc_perfect_and_inverted() -> None:
    analyze = _load("analyze_champloo_classifier.py")
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert analyze._auroc(scores, labels) == 1.0
    assert analyze._auroc(-scores, labels) == 0.0


def test_average_precision_matches_known_value() -> None:
    analyze = _load("analyze_champloo_classifier.py")
    # ranking: pos, neg, pos, neg -> AP = (1/1 + 2/3) / 2
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    labels = np.asarray([1, 0, 1, 0])
    ap = analyze._average_precision(scores, labels)
    assert abs(ap - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_combine_rows_orders_by_predictor_split_model() -> None:
    compare = _load("compare_champloo_predictors.py")
    chai = [
        {"predictor": "chai1", "split": "random_pair", "model": "logistic_iptm"},
        {"predictor": "chai1", "split": "all", "model": "raw_iptm"},
    ]
    af3 = [
        {"predictor": "af3", "split": "all", "model": "raw_iptm"},
        {"predictor": "af3", "split": "held_out_vhh", "model": "logistic_iptm_meta"},
    ]
    combined = compare.combine_rows([chai, af3])
    assert [(r["predictor"], r["split"], r["model"]) for r in combined] == [
        ("af3", "all", "raw_iptm"),
        ("af3", "held_out_vhh", "logistic_iptm_meta"),
        ("chai1", "all", "raw_iptm"),
        ("chai1", "random_pair", "logistic_iptm"),
    ]


def test_average_precision_perfect_ranking_is_one() -> None:
    analyze = _load("analyze_champloo_classifier.py")
    labels = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    # scores rank all positives strictly above all negatives -> AP == 1.0
    scores = np.where(labels == 1, 1.0, 0.0)
    assert abs(analyze._average_precision(scores, labels) - 1.0) < 1e-9
