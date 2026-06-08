import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_mc_epcam.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_mc_epcam", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_mc_epcam"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vhh_id_from_pair():
    mod = _load()
    assert mod.vhh_id_from_pair("epcam-10__epcam") == "10"
    assert mod.vhh_id_from_pair("epcam-74__epcam") == "74"


def test_killing_label_map(tmp_path):
    mod = _load()
    p = tmp_path / "k.csv"
    p.write_text("vhh_id,vhh_sequence,label\n10,AAAA,Good\n14,CCCC,Bad\n")
    m = mod.killing_label_map(p)
    assert m == {"10": 1, "14": 0}


def _feature_row(pair_id, label, iptm, fold="0"):
    # Minimal row matching extract_mc_features FEATURE_COLUMNS. Geometry/CDR are
    # constant so only ipTM carries signal; interface_plddt blank -> missing flag.
    return {
        "pair_id": pair_id, "label": label, "antigen_cluster": "0", "fold": fold,
        "prediction_present": "1",
        "iptm": str(iptm), "ptm": "0.5", "interface_pae": "15.0",
        "min_interface_pae": "8.0", "interface_plddt": "", "mean_plddt": "70.0",
        "n_interface_residues_binder": "10", "n_interface_residues_target": "12",
        "buried_sasa_proxy_a2": "500.0", "atom_contacts_5a": "40.0",
        "shape_complementarity_proxy": "0.5", "atom_clash_fraction_2a": "0.01",
        "cdr_contact_fraction": "0.6", "cdr1_contact_fraction": "0.2",
        "cdr2_contact_fraction": "0.2", "cdr3_contact_fraction": "0.2",
        "cdr_mapping_ok": "1",
    }


def _write_features(path, rows):
    import csv as _csv
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_analyze_epcam_end_to_end(tmp_path):
    mod = _load()
    rng = __import__("numpy").random.default_rng(0)
    # SAbDab train: 40 positives (high ipTM) + 40 negatives (low ipTM), 5 folds.
    sab = []
    for i in range(40):
        sab.append(_feature_row(f"s{i}__s{i}", "1", 0.6 + 0.1 * rng.random(), fold=str(i % 5)))
        sab.append(_feature_row(f"s{i}__neg0", "0", 0.2 + 0.1 * rng.random(), fold=str(i % 5)))
    sab_path = tmp_path / "sab.csv"
    _write_features(sab_path, sab)
    # EpCAM test: 4 designed positives + 8 shuffled negatives.
    epc = []
    for vid in ("10", "25", "14", "16"):
        epc.append(_feature_row(f"epcam-{vid}__epcam", "1", 0.55 + 0.1 * rng.random()))
        for j in range(2):
            epc.append(_feature_row(f"epcam-{vid}__neg{j}", "0", 0.2 + 0.1 * rng.random()))
    epc_path = tmp_path / "epc.csv"
    _write_features(epc_path, epc)
    kill = tmp_path / "k.csv"
    kill.write_text("vhh_id,vhh_sequence,label\n10,AA,Good\n25,AA,Good\n14,AA,Bad\n16,AA,Bad\n")

    res = mod.analyze_epcam(
        sab_path, epc_path, kill, l2=1.0, target_precision=0.9, n_boot=200, seed=1
    )
    assert res["primary"]["n"] == 12 and res["primary"]["n_positive"] == 4
    assert res["primary"]["rung0_auroc"] > 0.8  # planted ipTM signal transfers
    assert len(res["primary"]["delta_auroc_r3_minus_r0"]["ci"]) == 2
    assert res["calibration"]["rung3"]["n"] == 12
    assert res["secondary_killing"]["n_functional"] == 2
    assert res["secondary_killing"]["n_nonfunctional"] == 2
