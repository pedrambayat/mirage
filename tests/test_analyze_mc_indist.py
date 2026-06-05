from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def _load(script_name: str) -> Any:
    # Scripts live outside the importable package; load by path like the existing
    # tests/test_analyze_ms_indist.py does (there is no `scripts` package).
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, n: int = 120) -> None:
    rng = np.random.default_rng(0)
    fields = [
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
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            lab = i % 2
            # planted: positives get higher iptm AND more cdr contacts
            iptm = 0.6 + 0.2 * rng.random() if lab else 0.2 + 0.2 * rng.random()
            cdr = 0.7 if lab else 0.2
            w.writerow(
                {
                    "pair_id": f"p{i}",
                    "label": str(lab),
                    "antigen_cluster": str(i % 10),
                    "fold": str(i % 5),
                    "prediction_present": "1",
                    "iptm": f"{iptm:.3f}",
                    "ptm": "0.6",
                    "interface_pae": "12.0",
                    "min_interface_pae": "5.0",
                    "interface_plddt": ("" if lab == 0 else "70.0"),  # missingness signal
                    "mean_plddt": "80.0",
                    "n_interface_residues_binder": "10",
                    "n_interface_residues_target": "12",
                    "buried_sasa_proxy_a2": "900.0",
                    "atom_contacts_5a": "40",
                    "shape_complementarity_proxy": "0.6",
                    "atom_clash_fraction_2a": "0.01",
                    "cdr_contact_fraction": f"{cdr}",
                    "cdr1_contact_fraction": "0.2",
                    "cdr2_contact_fraction": "0.2",
                    "cdr3_contact_fraction": "0.3",
                    "cdr_mapping_ok": "1.0",
                }
            )


def test_analyze_indist_reports_all_rungs_and_paired_deltas(tmp_path: Path) -> None:
    csv_path = tmp_path / "feat.csv"
    _write_csv(csv_path)
    mod = _load("analyze_mc_indist.py")
    result = mod.analyze_indist(csv_path, l2=1.0, target_precision=0.9, seed=0, n_boot=200)

    # one entry per rung, each with an AUROC and a CI
    for k in (0, 1, 2, 3):
        assert f"rung{k}" in result["rungs"]
        assert "auroc" in result["rungs"][f"rung{k}"]
        assert "auroc_ci" in result["rungs"][f"rung{k}"]
        assert "coefficients" in result["rungs"][f"rung{k}"]

    # paired deltas present with point + CI
    for key in ("r2_minus_r0", "r3_minus_r0", "r3_minus_r2"):
        d = result["paired_deltas"][key]
        assert {"delta_auroc", "delta_auroc_ci", "delta_precision", "delta_precision_ci"} <= set(d)

    # random-split contrast present
    assert "random_split" in result
    for k in (0, 1, 2, 3):
        assert "auroc" in result["random_split"][f"rung{k}"]

    # CDR-mapping failure-rate diagnostic present
    assert "cdr_mapping_failure_rate" in result

    # the frozen model is the rung-3 MsModel dict
    assert result["frozen_rung"] == "rung3"
    assert len(result["frozen_model"]["feature_names"]) == 16

    # value sanity on the planted-signal synthetic data: rung 0 (ipTM) beats chance
    # and its CI is ordered. (Real numbers come from the Task 5 run on SAbDab.)
    assert result["rungs"]["rung0"]["auroc"] > 0.6
    lo, hi = result["rungs"]["rung0"]["auroc_ci"]
    assert lo <= hi
