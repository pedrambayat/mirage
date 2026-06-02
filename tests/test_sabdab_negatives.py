from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location(
        "stage_sabdab_pairs", REPO / "scripts" / "stage_sabdab_pairs.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_sabdab_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


def _uniform_positives(mod, n_clusters=4, per_cluster=25):
    pos = []
    for c in range(n_clusters):
        for j in range(per_cluster):
            ag = f"AG_{c}_{j}"  # distinct antigen per (cluster, j)
            pos.append(
                mod.Positive(
                    pair_id=f"p_{c}_{j}",
                    binder_seq=f"VHH_{c}_{j}",
                    antigen_seq=ag,
                    antigen_cluster=c,
                    fold=0,
                )
            )
    return pos


def test_negatives_are_cross_cluster_and_fold_consistent():
    mod = _load_module()
    pos = _uniform_positives(mod)
    rows = mod.build_pairs(pos, k=5, seed=7)
    by_id = {p.pair_id: p for p in pos}
    for r in rows:
        if r["label"] == "0":
            parent = by_id[r["pair_id"].split("__neg")[0]]
            assert int(r["antigen_cluster"]) != parent.antigen_cluster
            assert int(r["fold"]) == parent.fold


def test_one_positive_plus_k_negatives_each():
    mod = _load_module()
    pos = _uniform_positives(mod)
    rows = mod.build_pairs(pos, k=5, seed=7)
    assert sum(1 for r in rows if r["label"] == "1") == len(pos)
    assert sum(1 for r in rows if r["label"] == "0") == len(pos) * 5


def test_negative_cluster_marginal_matches_positive_marginal():
    mod = _load_module()
    pos = _uniform_positives(mod)
    rows = mod.build_pairs(pos, k=40, seed=11)
    neg_clusters = [int(r["antigen_cluster"]) for r in rows if r["label"] == "0"]
    counts = Counter(neg_clusters)
    total = sum(counts.values())
    for c in range(4):
        assert abs(counts[c] / total - 0.25) < 0.05


def test_build_pairs_is_deterministic():
    mod = _load_module()
    pos = _uniform_positives(mod)
    assert mod.build_pairs(pos, k=5, seed=3) == mod.build_pairs(pos, k=5, seed=3)
