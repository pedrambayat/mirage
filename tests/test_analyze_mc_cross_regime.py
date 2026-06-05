from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any


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
