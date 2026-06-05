from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

_FEATURE_COLS = [
    "pair_id",
    "label",
    "antigen_cluster",
    "fold",
    "prediction_present",
    "iptm",
    "ptm",
    "interface_pae",
    "min_interface_pae",
    "interface_plddt",
    "mean_plddt",
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
    "cdr_mapping_ok",
]


def _load(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pairs(path: Path, antigens: dict[str, str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["pair_id", "binder_seq", "antigen_seq", "label", "antigen_cluster", "fold"],
        )
        w.writeheader()
        for pid, ag in antigens.items():
            w.writerow(
                {
                    "pair_id": pid,
                    "binder_seq": "QVQ",
                    "antigen_seq": ag,
                    "label": "1",
                    "antigen_cluster": "0",
                    "fold": "0",
                }
            )


def _write_feature_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a feature CSV with the full M-C schema."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FEATURE_COLS)
        w.writeheader()
        w.writerows(rows)


def _feature_row(
    *,
    pair_id: str,
    label: int,
    antigen_cluster: int,
    fold: int,
    is_positive: bool,
) -> dict[str, str]:
    """Return one feature-CSV row with planted signal: positives have high iptm
    and a filled interface_plddt; negatives have low iptm and blank interface_plddt
    (missingness signal)."""
    if is_positive:
        iptm = "0.72"
        ptm = "0.80"
        interface_pae = "10.0"
        min_interface_pae = "5.0"
        interface_plddt = "82.0"
        mean_plddt = "85.0"
        n_binder = "40"
        n_target = "45"
        buried_sasa = "2000.0"
        atom_contacts = "500.0"
        shape_comp = "0.65"
        clash = "0.0001"
        cdr_contact = "0.80"
        cdr1 = "0.20"
        cdr2 = "0.20"
        cdr3 = "0.40"
    else:
        iptm = "0.18"
        ptm = "0.55"
        interface_pae = "28.0"
        min_interface_pae = "18.0"
        interface_plddt = ""  # missing for negatives → missingness flag = 1
        mean_plddt = "70.0"
        n_binder = "5"
        n_target = "4"
        buried_sasa = "50.0"
        atom_contacts = "20.0"
        shape_comp = "0.15"
        clash = "0.05"
        cdr_contact = "0.10"
        cdr1 = "0.02"
        cdr2 = "0.03"
        cdr3 = "0.05"
    return {
        "pair_id": pair_id,
        "label": str(label),
        "antigen_cluster": str(antigen_cluster),
        "fold": str(fold),
        "prediction_present": "1",
        "iptm": iptm,
        "ptm": ptm,
        "interface_pae": interface_pae,
        "min_interface_pae": min_interface_pae,
        "interface_plddt": interface_plddt,
        "mean_plddt": mean_plddt,
        "n_interface_residues_binder": n_binder,
        "n_interface_residues_target": n_target,
        "buried_sasa_proxy_a2": buried_sasa,
        "atom_contacts_5a": atom_contacts,
        "shape_complementarity_proxy": shape_comp,
        "atom_clash_fraction_2a": clash,
        "cdr_contact_fraction": cdr_contact,
        "cdr1_contact_fraction": cdr1,
        "cdr2_contact_fraction": cdr2,
        "cdr3_contact_fraction": cdr3,
        "cdr_mapping_ok": "1",
    }


def _write_pairs_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a pairs CSV (pair_id, binder_seq, antigen_seq, label, antigen_cluster, fold)."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["pair_id", "binder_seq", "antigen_seq", "label", "antigen_cluster", "fold"],
        )
        w.writeheader()
        w.writerows(rows)


def test_champloo_antigen_overlap_flags_shared_antigens(tmp_path: Path) -> None:
    shared = "M" + "A" * 60
    distinct = "M" + "W" * 60
    sab = tmp_path / "sab.csv"
    cha = tmp_path / "cha.csv"
    _pairs(sab, {"s1": shared})
    _pairs(cha, {"c1": shared, "c2": distinct})

    mod = _load("analyze_mc_cross_regime.py")
    overlapping = mod.champloo_antigen_overlap(cha, sab, max_identity=0.9)
    assert "c1" in overlapping  # same antigen as a SAbDab pair -> leakage
    assert "c2" not in overlapping


def test_champloo_antigen_overlap_max_identity_one_flags_exact_match(tmp_path: Path) -> None:
    # At max_identity=1.0, an exactly-identical antigen must still be flagged
    # (guards against a threshold-inversion bug in identity_to_jaccard_threshold).
    seq = "M" + "A" * 60
    other = "M" + "W" * 60
    sab = tmp_path / "sab.csv"
    cha = tmp_path / "cha.csv"
    _pairs(sab, {"s1": seq})
    _pairs(cha, {"c1": seq, "c2": other})

    mod = _load("analyze_mc_cross_regime.py")
    overlapping = mod.champloo_antigen_overlap(cha, sab, max_identity=1.0)
    assert "c1" in overlapping
    assert "c2" not in overlapping


def test_analyze_cross_regime_runs_both_directions(tmp_path: Path) -> None:
    # -----------------------------------------------------------------
    # SAbDab: antigens from "M"+"A"*60 family (two variants), two folds,
    # two antigen clusters, balanced classes (8 pos / 8 neg).
    # Champloo: antigens mostly from "M"+"W"*60 family (distinct from SAbDab)
    # so dedup keeps them, plus 2 overlap rows ("M"+"A"*60) that get dropped.
    # Both kept classes present in both train/test sets after dedup.
    # -----------------------------------------------------------------

    # SAbDab antigen sequences (two flavours to give two clusters)
    sab_ag_0 = "M" + "A" * 60  # cluster 0
    sab_ag_1 = "M" + "A" * 59 + "C"  # cluster 1 (one substitution, distinct enough)

    # Champloo antigen sequences — mostly W-family (kept after dedup)
    cha_ag_keep = "M" + "W" * 60  # cluster 2, completely distinct
    cha_ag_keep2 = "M" + "W" * 59 + "V"  # cluster 3, distinct variant
    # Overlapping antigen (same as sab_ag_0) → will be dropped by dedup
    cha_ag_overlap = sab_ag_0

    # ------ SAbDab feature + pairs rows --------------------------------
    sab_feat_rows: list[dict[str, str]] = []
    sab_pair_rows: list[dict[str, str]] = []

    # 8 positives: 4 in fold 0 (cluster 0 antigen), 4 in fold 1 (cluster 1 antigen)
    for i in range(4):
        pid = f"sab_pos0_{i}"
        sab_feat_rows.append(
            _feature_row(pair_id=pid, label=1, antigen_cluster=0, fold=0, is_positive=True)
        )
        sab_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": sab_ag_0,
                "label": "1",
                "antigen_cluster": "0",
                "fold": "0",
            }
        )
    for i in range(4):
        pid = f"sab_pos1_{i}"
        sab_feat_rows.append(
            _feature_row(pair_id=pid, label=1, antigen_cluster=1, fold=1, is_positive=True)
        )
        sab_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": sab_ag_1,
                "label": "1",
                "antigen_cluster": "1",
                "fold": "1",
            }
        )

    # 8 negatives: 4 in fold 0 (cluster 0), 4 in fold 1 (cluster 1)
    for i in range(4):
        pid = f"sab_neg0_{i}"
        sab_feat_rows.append(
            _feature_row(pair_id=pid, label=0, antigen_cluster=0, fold=0, is_positive=False)
        )
        sab_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": sab_ag_0,
                "label": "0",
                "antigen_cluster": "0",
                "fold": "0",
            }
        )
    for i in range(4):
        pid = f"sab_neg1_{i}"
        sab_feat_rows.append(
            _feature_row(pair_id=pid, label=0, antigen_cluster=1, fold=1, is_positive=False)
        )
        sab_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": sab_ag_1,
                "label": "0",
                "antigen_cluster": "1",
                "fold": "1",
            }
        )

    # ------ Champloo feature + pairs rows --------------------------------
    # Kept rows: 6 pos + 6 neg across two distinct antigen families (clusters 2 & 3)
    # across 2 folds, so after dedup both classes remain.
    cha_feat_rows: list[dict[str, str]] = []
    cha_pair_rows: list[dict[str, str]] = []

    for i in range(3):
        pid = f"cha_pos2_{i}"
        cha_feat_rows.append(
            _feature_row(pair_id=pid, label=1, antigen_cluster=2, fold=0, is_positive=True)
        )
        cha_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": cha_ag_keep,
                "label": "1",
                "antigen_cluster": "2",
                "fold": "0",
            }
        )
    for i in range(3):
        pid = f"cha_pos3_{i}"
        cha_feat_rows.append(
            _feature_row(pair_id=pid, label=1, antigen_cluster=3, fold=1, is_positive=True)
        )
        cha_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": cha_ag_keep2,
                "label": "1",
                "antigen_cluster": "3",
                "fold": "1",
            }
        )
    for i in range(3):
        pid = f"cha_neg2_{i}"
        cha_feat_rows.append(
            _feature_row(pair_id=pid, label=0, antigen_cluster=2, fold=0, is_positive=False)
        )
        cha_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": cha_ag_keep,
                "label": "0",
                "antigen_cluster": "2",
                "fold": "0",
            }
        )
    for i in range(3):
        pid = f"cha_neg3_{i}"
        cha_feat_rows.append(
            _feature_row(pair_id=pid, label=0, antigen_cluster=3, fold=1, is_positive=False)
        )
        cha_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": cha_ag_keep2,
                "label": "0",
                "antigen_cluster": "3",
                "fold": "1",
            }
        )

    # 2 overlap rows (will be dropped by dedup): one pos, one neg
    for lbl, is_pos, suffix in [(1, True, "ovlp_pos"), (0, False, "ovlp_neg")]:
        pid = f"cha_{suffix}"
        cha_feat_rows.append(
            _feature_row(pair_id=pid, label=lbl, antigen_cluster=0, fold=0, is_positive=is_pos)
        )
        cha_pair_rows.append(
            {
                "pair_id": pid,
                "binder_seq": "QVQ",
                "antigen_seq": cha_ag_overlap,
                "label": str(lbl),
                "antigen_cluster": "0",
                "fold": "0",
            }
        )

    # ------ Write to disk ------------------------------------------------
    sab_feat = tmp_path / "sab_feat.csv"
    cha_feat = tmp_path / "cha_feat.csv"
    sab_pairs = tmp_path / "sab_pairs.csv"
    cha_pairs = tmp_path / "cha_pairs.csv"

    _write_feature_csv(sab_feat, sab_feat_rows)
    _write_feature_csv(cha_feat, cha_feat_rows)
    _write_pairs_csv(sab_pairs, sab_pair_rows)
    _write_pairs_csv(cha_pairs, cha_pair_rows)

    # ------ Run pipeline -------------------------------------------------
    mod = _load("analyze_mc_cross_regime.py")
    result = mod.analyze_cross_regime(
        sab_feat,
        cha_feat,
        sab_pairs,
        cha_pairs,
        max_identity=0.9,
        l2=1.0,
        target_precision=0.9,
        n_boot=50,
        seed=0,
    )

    # dedup bookkeeping is internally consistent
    d = result["dedup"]
    assert d["champloo_total"] == d["champloo_kept"] + d["champloo_dropped_as_overlap"]
    assert d["champloo_kept"] > 0

    # both directions, both rungs present with an AUROC and evaluate_frozen_gate keys
    for direction in ("sabdab_to_champloo_primary", "champloo_to_sabdab_caveated"):
        for rung in ("rung3", "rung0_contrast"):
            r = result[direction][rung]
            assert "auroc" in r and "metrics" in r
            assert r["n"] == r["n_positive"] + r["n_negative"]

    # output is JSON-serializable (Task 7 writes it to disk)
    json.dumps(result, default=str)
