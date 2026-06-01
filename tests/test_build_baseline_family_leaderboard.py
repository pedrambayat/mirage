from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_script() -> Any:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_baseline_family_leaderboard.py"
    )
    spec = importlib.util.spec_from_file_location("build_baseline_family_leaderboard", script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules["build_baseline_family_leaderboard"] = module
    spec.loader.exec_module(module)
    return module


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_leaderboard_selects_best_metric_per_family(tmp_path: Path) -> None:
    module = _load_script()
    af2m = tmp_path / "af2m.csv"
    structural = tmp_path / "structural.csv"
    structural_logreg = tmp_path / "structural_logreg.csv"
    _write_rows(
        af2m,
        [
            {
                "stratum": "all",
                "label": "rmsd<4A",
                "metric": "iptm",
                "n": "4",
                "n_positive": "2",
                "n_negative": "2",
                "baseline_ap": "0.5",
                "auroc": "0.75",
                "ap": "0.7",
            },
            {
                "stratum": "all",
                "label": "rmsd<4A",
                "metric": "pae_interface_mean",
                "n": "4",
                "n_positive": "2",
                "n_negative": "2",
                "baseline_ap": "0.5",
                "auroc": "1.0",
                "ap": "0.9",
            },
            {
                "stratum": "VHH",
                "label": "rmsd<4A",
                "metric": "plddt_binder_mean",
                "n": "2",
                "n_positive": "1",
                "n_negative": "1",
                "baseline_ap": "0.5",
                "auroc": "1.0",
                "ap": "1.0",
            },
        ],
    )
    _write_rows(
        structural,
        [
            {
                "comparison": "rmsd<4A",
                "metric": "atom_clashes_2a",
                "n": "4",
                "n_positive": "2",
                "n_negative": "2",
                "baseline_ap": "0.5",
                "auroc": "0.8",
                "ap": "0.6",
            },
            {
                "comparison": "rmsd<4A",
                "metric": "buried_sasa_proxy_per_interface_residue",
                "n": "4",
                "n_positive": "2",
                "n_negative": "2",
                "baseline_ap": "0.5",
                "auroc": "0.65",
                "ap": "0.55",
            },
            {
                "comparison": "rmsd<4A",
                "metric": "atom_packing_complementarity_score",
                "n": "4",
                "n_positive": "2",
                "n_negative": "2",
                "baseline_ap": "0.5",
                "auroc": "1.0",
                "ap": "1.0",
            },
        ],
    )
    _write_rows(
        structural_logreg,
        [
            {
                "comparison": "rmsd<4A",
                "model": "structural_logistic_regression_loo",
                "features": "atom_clashes_2a;contacts_per_binder_residue_6a",
                "n": "4",
                "n_positive": "2",
                "n_negative": "2",
                "baseline_ap": "0.5",
                "auroc": "0.95",
                "ap": "0.95",
            }
        ],
    )

    rows = module.build_leaderboard(
        af2m_tables=[af2m],
        structural_table=structural,
        structural_logreg_table=structural_logreg,
    )
    by_family = {row["family"]: row for row in rows}

    assert by_family["af2m_confidence"]["best_metric"] == "pae_interface_mean"
    assert by_family["structural_clash"]["best_metric"] == "atom_clashes_2a"
    assert (
        by_family["structural_exposure"]["best_metric"] == "buried_sasa_proxy_per_interface_residue"
    )
    assert by_family["structural_logreg"]["best_metric"] == "structural_logistic_regression_loo"
    assert by_family["structural_packing"]["best_metric"] == "atom_packing_complementarity_score"
    assert len(rows) == 5
