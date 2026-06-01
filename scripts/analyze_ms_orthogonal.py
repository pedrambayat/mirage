"""Apply the Champloo-frozen M-S gate to the real-negative orthogonal sets
(AVIDa, labeled-EpCAM) and assemble the cross-regime precision-stability table.

Use::

    uv run python scripts/analyze_ms_orthogonal.py \\
        --model results/published/ms_model_af3.json \\
        --avida-csv ../abdisc-data/avida/avida_staged.csv \\
        --epcam-labels ../abdisc-data/epcam/epcam_killing_labels.csv \\
        --output results/published/mirage_ms_orthogonal.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# ensure built-in loaders self-register (the _registry import below also triggers this)
import mirage.benchmark  # noqa: F401
from mirage.benchmark._registry import get_loader
from mirage.eval.orthogonal import evaluate_frozen_gate, features_for_examples
from mirage.model.ms import MsModel


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats (NaN/inf) with None so the table is valid strict
    JSON. Precision is undefined (None) when the gate predicts no positives."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _evaluate(
    loader_name: str, model: MsModel, *, positive_label: str, **kwargs: Any
) -> dict[str, Any]:
    loader = get_loader(loader_name, **kwargs)
    x, y, _names = features_for_examples(loader.load(), positive_label=positive_label)
    return evaluate_frozen_gate(model, x, y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--avida-csv", type=Path, default=None)
    parser.add_argument("--epcam-labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = MsModel.load(args.model)
    table: dict[str, Any] = {
        "threshold": model.threshold,
        "target_precision": model.target_precision,
        "regimes": {},
    }

    if args.avida_csv is not None:
        table["regimes"]["avida"] = _evaluate(
            "avida", model, positive_label="BIND", staged_csv=args.avida_csv
        )
    if args.epcam_labels is not None:
        table["regimes"]["epcam_killing"] = _evaluate(
            "epcam_killing", model, positive_label="BIND", labels_csv=args.epcam_labels
        )

    safe = _json_safe(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(safe, indent=2))

    def _fmt(v: Any) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    for regime, res in safe["regimes"].items():
        m = res["metrics"]
        print(
            f"{regime}: n={res['n']} precision={_fmt(m['precision'])} "
            f"recall={_fmt(m['recall'])} specificity={_fmt(m['specificity'])}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
