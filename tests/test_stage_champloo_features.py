from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sequences_by_pdb_picks_first() -> None:
    stage = _load("stage_champloo_features.py")
    supp = [
        {"pdb_id": "AAAA", "vhh_sequence": "QVQL", "antigen_sequence": "GGGG"},
        {"pdb_id": "AAAA", "vhh_sequence": "XXXX", "antigen_sequence": "YYYY"},
        {"pdb_id": "BBBB", "vhh_sequence": "EVQL", "antigen_sequence": "DDDD"},
    ]
    by_pdb = stage.sequences_by_pdb(supp)
    assert by_pdb["AAAA"]["vhh_sequence"] == "QVQL"
    assert by_pdb["BBBB"]["antigen_sequence"] == "DDDD"


def test_build_feature_rows_joins_and_features() -> None:
    stage = _load("stage_champloo_features.py")
    pairs = [
        {
            "pair_id": "AAAA__AAAA",
            "vhh_pdb": "AAAA",
            "antigen_pdb": "AAAA",
            "label": "1",
            "iptm": "0.9",
        },
        {
            "pair_id": "AAAA__BBBB",
            "vhh_pdb": "AAAA",
            "antigen_pdb": "BBBB",
            "label": "0",
            "iptm": "0.2",
        },
        {
            "pair_id": "AAAA__CCCC",
            "vhh_pdb": "AAAA",
            "antigen_pdb": "CCCC",
            "label": "0",
            "iptm": "0.1",
        },
    ]
    seqs = {
        "AAAA": {"vhh_sequence": "KKKK", "antigen_sequence": "DDDD"},
        "BBBB": {"vhh_sequence": "EVQL", "antigen_sequence": "GGGG"},
    }
    rows = stage.build_feature_rows(pairs, seqs)
    # CCCC has no sequence -> that pair is dropped
    ids = {r["pair_id"] for r in rows}
    assert ids == {"AAAA__AAAA", "AAAA__BBBB"}
    row = next(r for r in rows if r["pair_id"] == "AAAA__AAAA")
    assert row["label"] == "1"
    assert float(row["binder_length"]) == 4.0
    assert "binder_net_charge" in row
