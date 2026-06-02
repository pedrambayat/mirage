"""Apply the frozen SAbDab gate to AVIDa-hIL6 as a held-out same-antigen transfer.

AVIDa is NEVER training data — this is the orthogonal real-negative canary. The
frozen model (BilinearModel for rung 3, MsModel for rung 2) is applied unchanged
through the existing evaluate_frozen_gate harness. AVIDa's unique normalized
sequences must already be present in the embedding cache (embed them first).

Use::

    uv run python scripts/analyze_sabdab_orthogonal.py \\
        --avida-csv data/staged/avida/avida_staged.csv \\
        --embeddings data/staged/sabdab/embeddings.npy \\
        --keys data/staged/sabdab/keys.txt \\
        --model results/published/sabdab_bilinear_model.json --model-type bilinear \\
        --layout concat --output results/published/sabdab_orthogonal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mirage.benchmark._registry import get_loader  # AvidaLoader self-registers as "avida"
from mirage.eval.orthogonal import evaluate_frozen_gate, features_for_examples_embedding
from mirage.features.embeddings import load_embedding_cache
from mirage.model.bilinear import BilinearModel
from mirage.model.ms import MsModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avida-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-type", choices=["bilinear", "ms"], required=True)
    parser.add_argument("--layout", choices=["concat", "hadamard"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-label", default="BIND")
    args = parser.parse_args()

    model = (
        BilinearModel.load(args.model)
        if args.model_type == "bilinear"
        else MsModel.load(args.model)
    )
    cache = load_embedding_cache(args.embeddings, args.keys)
    examples = list(get_loader("avida", staged_csv=args.avida_csv).load())
    x, y = features_for_examples_embedding(
        examples, cache, positive_label=args.positive_label, layout=args.layout
    )
    result = evaluate_frozen_gate(model, x, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["metrics"], indent=2, default=str))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
