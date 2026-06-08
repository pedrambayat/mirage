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
