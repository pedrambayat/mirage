import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "stage_epcam_protenix_pairs.py"


def _load():
    spec = importlib.util.spec_from_file_location("stage_epcam_protenix_pairs", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_epcam_protenix_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_negative_pool_excludes_epcam_cluster_antigens():
    mod = _load()
    epcam = "MARTGYHIKLPQRSTWVYACDEFGHIKLMNPQ"  # stand-in EpCAM ECD
    near_epcam = epcam[:-1] + "A"  # ~identical -> same cluster, must be excluded
    far1 = "WWWWWWWWWWQQQQQQQQQQYYYYYYYYYYEE"
    far2 = "CCCCCCCCCCDDDDDDDDDDFFFFFFFFFFGG"
    pool = mod.epcam_antigen_negative_pool([near_epcam, far1, far2], epcam, max_identity=0.9)
    assert near_epcam not in pool
    assert far1 in pool and far2 in pool


def test_build_epcam_pairs_shape_and_ids():
    mod = _load()
    positives = [
        ("10", "QVQLVESGGG", "EPCAMSEQ", "functional"),
        ("14", "EVQLVESGGA", "EPCAMSEQ", "nonfunctional"),
    ]
    pool = [f"WRONGANTIGEN{i}" for i in range(20)]
    rows = mod.build_epcam_pairs(positives, pool, k=5, seed=7)
    # 2 positives + 2*5 negatives
    assert len(rows) == 12
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    assert len(pos) == 2 and len(neg) == 10
    # positive pair_id convention + antigen is EpCAM
    assert {r["pair_id"] for r in pos} == {"epcam-10__epcam", "epcam-14__epcam"}
    assert all(r["antigen_seq"] == "EPCAMSEQ" for r in pos)
    # negative pair_id convention + antigen drawn from the pool (never EpCAM)
    assert all(r["pair_id"].endswith(tuple(f"__neg{j}" for j in range(5))) for r in neg)
    assert all(r["antigen_seq"] in pool for r in neg)
    # schema columns present and exact
    assert all(set(r) == set(mod._FIELDNAMES) for r in rows)


def test_build_epcam_pairs_is_deterministic():
    mod = _load()
    positives = [("10", "QVQLVESGGG", "EPCAMSEQ", "functional")]
    pool = [f"A{i}" for i in range(50)]
    a = mod.build_epcam_pairs(positives, pool, k=5, seed=7)
    b = mod.build_epcam_pairs(positives, pool, k=5, seed=7)
    assert [r["antigen_seq"] for r in a] == [r["antigen_seq"] for r in b]


def test_load_epcam_positives_maps_label_and_normalizes(tmp_path):
    mod = _load()
    csv_path = tmp_path / "killing.csv"
    csv_path.write_text(
        "vhh_id,vhh_sequence,label\n"
        "10,QVQLVESGGGLVQPGGSLRLSCAAS,Good\n"
        "14,EVQLVESGGGLVQPGGSLRLSCAAS,Bad\n"
    )
    pos = mod.load_epcam_positives(csv_path)
    assert [p[0] for p in pos] == ["10", "14"]
    assert [p[3] for p in pos] == ["functional", "nonfunctional"]
    # antigen is the EpCAM ECD constant (normalized), identical across positives
    assert pos[0][2] == pos[1][2] and len(pos[0][2]) > 0
