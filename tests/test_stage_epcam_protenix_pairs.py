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
