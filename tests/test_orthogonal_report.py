from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any


def _load_script(name: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_safe_replaces_nan_with_none() -> None:
    mod = _load_script("analyze_ms_orthogonal.py")
    safe = mod._json_safe({"precision": math.nan, "recall": 0.0, "ci": [math.nan, 1.0]})
    # Must round-trip through strict JSON (NaN would be rejected).
    text = json.dumps(safe, allow_nan=False)
    back = json.loads(text)
    assert back["precision"] is None
    assert back["recall"] == 0.0
    assert back["ci"] == [None, 1.0]
