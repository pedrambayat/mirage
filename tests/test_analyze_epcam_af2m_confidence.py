from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_script() -> Any:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "analyze_epcam_af2m_confidence.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_epcam_af2m_confidence", script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules["analyze_epcam_af2m_confidence"] = module
    spec.loader.exec_module(module)
    return module


def _row(example_id: str, label: str, iptm: float, pae: float) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "value": str(iptm),
        "label": label,
        "__extras": {
            "iptm": iptm,
            "ptm": iptm,
            "ranking_confidence": iptm,
            "iptm_over_ptm": 1.0,
            "plddt_full_mean": iptm,
            "plddt_binder_mean": iptm,
            "plddt_target_mean": iptm,
            "plddt_interface_mean": iptm,
            "pae_interchain_mean": pae,
            "pae_interchain_max": pae,
            "pae_interface_mean": pae,
        },
    }


def test_metric_rows_compare_pos_to_epcam_negative_sets() -> None:
    module = _load_script()
    rows = [
        _row("pos-1", "POS", 0.9, 2.0),
        _row("pos-2", "POS", 0.8, 3.0),
        _row("scr-1", "SCR", 0.3, 8.0),
        _row("off-1", "OFF", 0.2, 9.0),
    ]

    out = module._metric_rows(rows, n_bootstrap=0)
    by_key = {(row["comparison"], row["metric"]): row for row in out}

    assert by_key[("POS_vs_all_negatives", "iptm")]["n_positive"] == 2
    assert by_key[("POS_vs_all_negatives", "iptm")]["n_negative"] == 2
    assert by_key[("POS_vs_all_negatives", "iptm")]["auroc"] == 1.0
    assert by_key[("POS_vs_all_negatives", "iptm")]["ap"] == 1.0
    assert by_key[("POS_vs_all_negatives", "pae_interface_mean")]["auroc"] == 1.0
    assert by_key[("POS_vs_SCR", "iptm")]["n_negative"] == 1
    assert by_key[("POS_vs_OFF", "iptm")]["n_negative"] == 1
