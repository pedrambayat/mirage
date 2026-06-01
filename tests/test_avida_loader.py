from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

from mirage.benchmark._registry import get_loader


def _load_script(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_avida_joins_antigen_sequences() -> None:
    stage = _load_script("stage_avida.py")
    records = [
        {"VHH_sequence": "QVQL", "Ag_label": "IL6", "label": "1"},
        {"VHH_sequence": "EVQL", "Ag_label": "IL6", "label": "0"},
    ]
    antigens = {"IL6": "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP"}
    rows = stage.build_rows(records, antigens)
    assert rows[0]["label"] == "1"
    assert rows[0]["antigen_sequence"].startswith("MNSF")
    assert rows[1]["label"] == "0"


def test_avida_loader_yields_examples(tmp_path: Path) -> None:
    staged = tmp_path / "avida.csv"
    with staged.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["vhh_id", "vhh_sequence", "antigen_label", "antigen_sequence", "label"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "vhh_id": "v1",
                "vhh_sequence": "QVQL",
                "antigen_label": "IL6",
                "antigen_sequence": "MNSF",
                "label": "1",
            }
        )
        writer.writerow(
            {
                "vhh_id": "v2",
                "vhh_sequence": "EVQL",
                "antigen_label": "IL6",
                "antigen_sequence": "MNSF",
                "label": "0",
            }
        )
    loader = get_loader("avida", staged_csv=staged)
    examples = list(loader.load())
    assert len(examples) == 2
    assert {e.label for e in examples} == {"BIND", "NONBIND"}
    assert examples[0].binder_format == "vhh"
    assert examples[0].source == "avida-hil6"
