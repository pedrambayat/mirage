from __future__ import annotations

import csv
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


# --- stage_champloo_structures.py ------------------------------------------


def test_parse_member_name_valid() -> None:
    stage = _load("stage_champloo_structures.py")
    parsed = stage.parse_member_name("system_2_3_3OGO_3RJQ_model_0.pdb")
    assert parsed == (2, 3, "3OGO", "3RJQ", 0)


def test_parse_member_name_rejects_other() -> None:
    stage = _load("stage_champloo_structures.py")
    assert stage.parse_member_name("readme.txt") is None
    assert stage.parse_member_name("system_2_3_3OGO_3RJQ.pdb") is None


def test_example_id_uppercases() -> None:
    stage = _load("stage_champloo_structures.py")
    assert stage.example_id_for("3ogo", "3rjq") == "3OGO__3RJQ"


def test_index_model_entries_filters_and_first_wins() -> None:
    stage = _load("stage_champloo_structures.py")
    entries = [
        {"name": "system_2_2_3OGO_3OGO_model_0.pdb", "compress_size": 1},
        {"name": "system_2_2_3OGO_3OGO_model_1.pdb", "compress_size": 2},
        # duplicate-PDB system collapsing to the same (vhh,ag) cell, model_0:
        {"name": "system_9_9_3OGO_3OGO_model_0.pdb", "compress_size": 3},
        {"name": "system_2_3_3ogo_3rjq_model_0.pdb", "compress_size": 4},
    ]
    index = stage.index_model_entries(entries, model=0)
    assert set(index) == {("3OGO", "3OGO"), ("3OGO", "3RJQ")}
    # first occurrence wins for the collapsed cell
    assert index[("3OGO", "3OGO")]["name"] == "system_2_2_3OGO_3OGO_model_0.pdb"


def test_matrix_pairs_labels_and_drops_empty(tmp_path: Path) -> None:
    stage = _load("stage_champloo_structures.py")
    matrix = tmp_path / "m.csv"
    with matrix.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_1", "AAAA", "BBBB"])
        w.writerow(["AAAA", "0.9", ""])  # empty cell dropped
        w.writerow(["BBBB", "0.1", "0.8"])
    pairs = stage.matrix_pairs(matrix)
    assert ("AAAA", "AAAA", 1) in pairs
    assert ("BBBB", "AAAA", 0) in pairs
    assert ("BBBB", "BBBB", 1) in pairs
    # (AAAA, BBBB) was empty -> dropped
    assert all(not (v == "AAAA" and a == "BBBB") for v, a, _ in pairs)
    assert len(pairs) == 3


# --- score_champloo_structures.py ------------------------------------------


def test_manifest_to_examples_chain_and_label() -> None:
    score = _load("score_champloo_structures.py")
    rows = [
        {"example_id": "AAAA__AAAA", "vhh_pdb": "AAAA", "antigen_pdb": "AAAA", "label": "1"},
        {"example_id": "AAAA__BBBB", "vhh_pdb": "AAAA", "antigen_pdb": "BBBB", "label": "0"},
    ]
    examples = list(score.manifest_to_examples(rows, source="champloo_af3"))
    assert [e.label for e in examples] == ["COGNATE", "SHUFFLED"]
    for e in examples:
        assert e.binder_chains == ("A",)
        assert e.target_chains == ("B",)
        assert e.binder_format == "vhh"
        assert e.source == "champloo_af3"
    assert examples[1].target_name == "BBBB"


# --- analyze_champloo_structural.py ----------------------------------------


def test_parse_example_id() -> None:
    analyze = _load("analyze_champloo_structural.py")
    assert analyze.parse_example_id("3ogo__3rjq") == ("3OGO", "3RJQ")


def test_load_iptm_matrix_drops_empty(tmp_path: Path) -> None:
    analyze = _load("analyze_champloo_structural.py")
    matrix = tmp_path / "m.csv"
    with matrix.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_1", "AAAA", "BBBB"])
        w.writerow(["AAAA", "0.9", ""])
        w.writerow(["BBBB", "0.1", "0.8"])
    iptm = analyze.load_iptm_matrix(matrix)
    assert iptm[("AAAA", "AAAA")] == 0.9
    assert ("AAAA", "BBBB") not in iptm
    assert iptm[("BBBB", "BBBB")] == 0.8


def test_feature_matrix_sign_flip() -> None:
    analyze = _load("analyze_champloo_structural.py")
    rows = [
        {"iptm": 0.9, "__extras": {"atom_clashes_2a": 5.0}},
        {"iptm": 0.1, "__extras": {"atom_clashes_2a": 0.0}},
    ]
    # atom_clashes_2a is lower-is-better -> sign-flipped
    x = analyze.feature_matrix(rows, ("atom_clashes_2a",))
    assert x[0, 0] == -5.0
    assert x[1, 0] == -0.0


def test_auroc_and_ap_known_values() -> None:
    analyze = _load("analyze_champloo_structural.py")
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])
    # perfect would be 1.0; here one positive ranks below a negative -> 0.75
    assert abs(analyze.auroc(scores, labels) - 0.75) < 1e-9
    # AP: ranks 0.9(P),0.8(N),0.2(P),0.1(N): (1/1 + 2/3)/2 = 0.8333...
    assert abs(analyze.average_precision(scores, labels) - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_assign_folds_deterministic_and_grouped() -> None:
    analyze = _load("analyze_champloo_structural.py")
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    f1 = analyze.assign_folds(groups, n_splits=3, seed=7)
    f2 = analyze.assign_folds(groups, n_splits=3, seed=7)
    assert np.array_equal(f1, f2)
    # rows sharing a group share a fold (no leakage)
    assert f1[0] == f1[1]
    assert f1[2] == f1[3]
    assert f1[4] == f1[5]


def test_oof_logistic_recovers_separable_signal() -> None:
    analyze = _load("analyze_champloo_structural.py")
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(3.0, 0.5, 40), rng.normal(-3.0, 0.5, 40)])[:, None]
    y = np.array([1.0] * 40 + [0.0] * 40)
    folds = np.array(([0, 1, 2, 3, 4] * 16)[:80])
    scores = analyze.oof_logistic_scores(x, y, folds, l2=1.0)
    assert analyze.auroc(scores, y.astype(int)) > 0.95


def test_build_rows_joins_and_filters() -> None:
    analyze = _load("analyze_champloo_structural.py")
    structural_rows = [
        {"example_id": "AAAA__AAAA", "label": "COGNATE", "__extras": {"atom_clashes_2a": 0.0}},
        {"example_id": "AAAA__BBBB", "label": "SHUFFLED", "__extras": {"atom_clashes_2a": 3.0}},
        # missing structure -> dropped
        {"example_id": "AAAA__CCCC", "label": "SHUFFLED", "__extras": {"missing": "prediction"}},
        # no ipTM for this pair -> dropped
        {"example_id": "DDDD__DDDD", "label": "COGNATE", "__extras": {"atom_clashes_2a": 0.0}},
    ]
    iptm = {("AAAA", "AAAA"): 0.9, ("AAAA", "BBBB"): 0.2}
    rows = analyze.build_rows(structural_rows, iptm)
    assert [r["example_id"] for r in rows] == ["AAAA__AAAA", "AAAA__BBBB"]
    assert rows[0]["label_int"] == 1
    assert rows[1]["label_int"] == 0
    assert rows[0]["iptm"] == 0.9


def test_evaluate_smoke() -> None:
    analyze = _load("analyze_champloo_structural.py")
    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []
    feats = analyze._STRUCTURAL_FEATURES
    for i in range(60):
        label = 1 if i < 12 else 0
        base = 1.0 if label else 0.0
        extras = {f: float(base + rng.normal(0, 0.3)) for f in feats}
        rows.append(
            {
                "example_id": f"P{i:03d}__Q{i:03d}",
                "vhh_pdb": f"P{i:03d}",
                "antigen_pdb": f"Q{i:03d}",
                "label_int": label,
                "iptm": float(0.8 * base + rng.normal(0, 0.1)),
                "__extras": extras,
            }
        )
    metrics = analyze.evaluate(rows, predictor="af3", n_splits=5, seed=3, l2=1.0)
    models = {m["model"] for m in metrics}
    assert "raw_iptm" in models
    assert "logistic_combined" in models
    # the raw_iptm row is split-invariant ("all")
    raw = next(m for m in metrics if m["model"] == "raw_iptm")
    assert raw["split"] == "all"
    assert raw["n_positive"] == 12
