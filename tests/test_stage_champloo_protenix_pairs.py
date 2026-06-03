"""Tests for scripts/stage_champloo_protenix_pairs.py."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic Champloo-shaped data.
#
# We need ≥2 distinct antigen clusters per fold after assign_folds so that
# build_pairs can always find a cross-cluster negative.
#
# Strategy: 10 rows with homopolymeric antigens (e.g. poly-A, poly-C, …) that
# share no 5-mers with each other — each lands in its own greedy cluster (10
# clusters total). With n_splits=5 and seed=0, assign_folds maps exactly 2
# clusters to every fold, so all 10 positives get 5 negatives each and the
# strong invariant n_neg == 5 * n_pos holds.
#
# We also include one is_excluded=TRUE row (PDB11) that must be dropped.
# ---------------------------------------------------------------------------

# Homopolymeric antigens — no shared 5-mers across distinct amino acids.
_SYNTHETIC_ROWS = [
    # pdb_id, vhh_sequence, antigen_sequence (≥37 residues), is_excluded
    ("PDB1", "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYA", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "FALSE"),
    ("PDB2", "EVQLVESGGGLVQPGGSLRLSCAASGYIFSSYS", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "FALSE"),
    ("PDB3", "QVQLLESGGGLVQPGGSLRLSCAASGFNIKDTY", "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD", "FALSE"),
    ("PDB4", "EVQLLESGGGLVQPGGSLRLSCAASGFRISDTS", "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE", "FALSE"),
    (
        "PDB5",
        "QVQLVESGGALVQAGGSLRLSCAASGRTFSDYA",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FALSE",
    ),
    ("PDB6", "EVQLVESGGALVQAGGSLRLSCAASGYAFSDYS", "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", "FALSE"),
    ("PDB7", "QVQLQESGGGLVQAGGSLRLSCAASGRTISNYA", "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN", "FALSE"),
    ("PDB8", "EVQLQESGGGLVQAGGSLRLSCAASGRTFNNYA", "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", "FALSE"),
    ("PDB9", "QVQLLESGGGLVQPGGSLRLSCAASGYSISTYA", "KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK", "FALSE"),
    (
        "PDB10",
        "EVQLLESGGGLVQPGGSLRLSCAASGYGISTYS",
        "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
        "FALSE",
    ),
    # Excluded row — must NOT appear in output
    ("PDB11", "QVQLLESGGGLVQPGGSLRLSCAASGYTFTSSS", "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM", "TRUE"),
]

_FIELDNAMES = ["pdb_id", "vhh_sequence", "antigen_sequence", "is_excluded"]


def _write_champloo_csv(path: Path, rows=_SYNTHETIC_ROWS) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*_FIELDNAMES, "extra_col"])
        writer.writeheader()
        for pdb, vhh, ag, excl in rows:
            writer.writerow(
                {
                    "pdb_id": pdb,
                    "vhh_sequence": vhh,
                    "antigen_sequence": ag,
                    "is_excluded": excl,
                    "extra_col": "ignored",
                }
            )


def test_excluded_row_is_dropped(tmp_path):
    mod = _load_module("stage_champloo_protenix_pairs")
    csv_path = tmp_path / "champloo.csv"
    _write_champloo_csv(csv_path)

    positives = mod.build_champloo_positives(csv_path, seed=0)

    pdb_ids = [p.pair_id.split("__")[0] for p in positives]
    assert "PDB11" not in pdb_ids
    assert len(positives) == 10  # 11 rows - 1 excluded


def test_pair_id_is_pdb_double(tmp_path):
    mod = _load_module("stage_champloo_protenix_pairs")
    csv_path = tmp_path / "champloo.csv"
    _write_champloo_csv(csv_path)

    positives = mod.build_champloo_positives(csv_path, seed=0)

    for p in positives:
        parts = p.pair_id.split("__")
        assert len(parts) == 2
        assert parts[0] == parts[1], f"pair_id not diagonal: {p.pair_id}"


def test_antigen_cluster_is_int(tmp_path):
    mod = _load_module("stage_champloo_protenix_pairs")
    csv_path = tmp_path / "champloo.csv"
    _write_champloo_csv(csv_path)

    positives = mod.build_champloo_positives(csv_path, seed=0)

    for p in positives:
        assert isinstance(p.antigen_cluster, int)


def test_fold_in_range(tmp_path):
    mod = _load_module("stage_champloo_protenix_pairs")
    csv_path = tmp_path / "champloo.csv"
    _write_champloo_csv(csv_path)

    n_splits = 5
    positives = mod.build_champloo_positives(csv_path, n_splits=n_splits, seed=0)

    for p in positives:
        assert 0 <= p.fold < n_splits, f"fold {p.fold} out of range [0, {n_splits})"


def test_build_pairs_negatives_are_cross_cluster(tmp_path):
    """All negatives must come from a different antigen cluster than their positive."""
    mod_champloo = _load_module("stage_champloo_protenix_pairs")
    # Load stage_sabdab_pairs for build_pairs
    spec = importlib.util.spec_from_file_location(
        "stage_sabdab_pairs", REPO / "scripts" / "stage_sabdab_pairs.py"
    )
    assert spec and spec.loader
    mod_sab = importlib.util.module_from_spec(spec)
    sys.modules["stage_sabdab_pairs"] = mod_sab
    spec.loader.exec_module(mod_sab)

    csv_path = tmp_path / "champloo.csv"
    _write_champloo_csv(csv_path)

    positives = mod_champloo.build_champloo_positives(csv_path, seed=0)
    rows = mod_sab.build_pairs(positives, k=5, seed=0)

    pos_by_id = {p.pair_id: p for p in positives}
    for r in rows:
        if r["label"] == "0":
            parent_id = r["pair_id"].split("__neg")[0]
            parent = pos_by_id[parent_id]
            assert int(r["antigen_cluster"]) != parent.antigen_cluster


def test_build_pairs_neg_count_equals_5x(tmp_path):
    """n_negatives == 5 * n_positives (strong form — every fold has ≥2 antigen clusters).

    The synthetic data uses 10 homopolymeric antigens that all land in distinct
    greedy clusters. With n_splits=5 and seed=0, assign_folds maps exactly 2
    clusters per fold, so every positive gets exactly k=5 negatives.
    """
    mod_champloo = _load_module("stage_champloo_protenix_pairs")
    spec = importlib.util.spec_from_file_location(
        "stage_sabdab_pairs", REPO / "scripts" / "stage_sabdab_pairs.py"
    )
    assert spec and spec.loader
    mod_sab = importlib.util.module_from_spec(spec)
    sys.modules["stage_sabdab_pairs"] = mod_sab
    spec.loader.exec_module(mod_sab)

    csv_path = tmp_path / "champloo.csv"
    _write_champloo_csv(csv_path)

    positives = mod_champloo.build_champloo_positives(csv_path, seed=0)
    rows = mod_sab.build_pairs(positives, k=5, seed=0)

    n_pos = sum(1 for r in rows if r["label"] == "1")
    n_neg = sum(1 for r in rows if r["label"] == "0")

    assert n_pos == len(positives)
    # Strong invariant: each fold has ≥2 clusters so every positive gets exactly k negatives.
    assert n_neg == 5 * n_pos
