from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_script() -> Any:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "stage_wrong_target_repairings.py"
    )
    spec = importlib.util.spec_from_file_location("stage_wrong_target_repairings", script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules["stage_wrong_target_repairings"] = module
    spec.loader.exec_module(module)
    return module


def test_build_pairings_prioritizes_near_native_and_changes_target() -> None:
    module = _load_script()
    parents = [
        module._StagedParent(
            example_id="sabdab-near1-H-A",
            binder_format="vhh",
            target_name="target-a",
            rmsd=2.0,
            binder_chains=("BINDER1",),
            target_chains=("TARGETA",),
        ),
        module._StagedParent(
            example_id="sabdab-near2-H-B",
            binder_format="vhh",
            target_name="target-b",
            rmsd=3.0,
            binder_chains=("BINDER2",),
            target_chains=("TARGETB",),
        ),
        module._StagedParent(
            example_id="sabdab-far-H-C",
            binder_format="vhh",
            target_name="target-c",
            rmsd=30.0,
            binder_chains=("BINDER3",),
            target_chains=("TARGETC",),
        ),
    ]

    pairings = module._build_pairings(parents, n_binders=2, targets_per_binder=1, seed=20260525)

    assert [p.binder_parent_id for p in pairings] == [
        "sabdab-near1-H-A",
        "sabdab-near2-H-B",
    ]
    assert all(p.example.label == "WRONG_TARGET" for p in pairings)
    assert all(p.binder_parent_id != p.target_donor_id for p in pairings)
    for pairing in pairings:
        parent = next(p for p in parents if p.example_id == pairing.binder_parent_id)
        assert pairing.example.binder_chains == parent.binder_chains
        assert pairing.example.target_chains != parent.target_chains


def test_load_staged_parents_splits_fab_chains(tmp_path: Path) -> None:
    module = _load_script()
    fasta_dir = tmp_path / "fasta"
    fasta_dir.mkdir()
    (fasta_dir / "sabdab-fab-H-A.fasta").write_text(">sabdab-fab-H-A\nHHHH:LLLL:TTTT\n")

    rows = {
        "sabdab-fab-H-A": {
            "example_id": "sabdab-fab-H-A",
            "value": "5.5",
            "binder_format": "fab",
            "target_name": "target",
        }
    }
    parents = module._load_staged_parents(["sabdab-fab-H-A"], rows, fasta_dir)

    assert len(parents) == 1
    assert parents[0].binder_chains == ("HHHH", "LLLL")
    assert parents[0].target_chains == ("TTTT",)
