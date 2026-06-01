"""Score staged Champloo AF3 structures with the structural-interface scorer.

Reads the structure staging manifest written by ``stage_champloo_structures.py``
and runs the predictor-agnostic ``StructuralInterfaceScorer`` on each
``<predictions-root>/<example_id>/rank1.pdb``. The released AF3 PDBs label the
VHH as chain ``A`` (binder) and the antigen as chain ``B`` (target), so every
example is built with ``binder_chains=("A",)`` / ``target_chains=("B",)`` and
``binder_format="vhh"``.

Output is the standard one-row-per-example score CSV (same header as
``score_pilot_manifest.py``), so the downstream Phase 2 analysis can join
structural features against the released ipTM matrix by ``example_id``.

Use::

    uv run python scripts/score_champloo_structures.py \\
        --manifest data/staged/champloo/champloo_af3_structures_manifest.csv \\
        --predictions-root <abdisc-data>/champloo/af3_structures \\
        --output results/published/champloo_af3_structural_interface.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from mirage.scorers.base import BenchmarkExample
from mirage.scorers.structural_interface import StructuralInterfaceScorer

_HEADER = (
    "example_id",
    "scorer_name",
    "value",
    "label",
    "target_name",
    "source",
    "binder_format",
    "split",
    "extras_json",
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def manifest_to_examples(rows: list[dict[str, str]], *, source: str) -> Iterable[BenchmarkExample]:
    """Build one crystal-independent VHH example per manifest row.

    The structures are AF3-relabelled to chain A (VHH) / chain B (antigen), so
    chain assignment is fixed and does not depend on the original crystal IDs.
    """
    for row in rows:
        label = "COGNATE" if str(row["label"]).strip() == "1" else "SHUFFLED"
        yield BenchmarkExample(
            id=row["example_id"],
            label=label,
            binder_chains=("A",),
            binder_format="vhh",
            target_chains=("B",),
            target_name=row["antigen_pdb"],
            source=source,
            target_pdb_id=row["antigen_pdb"],
            complex_pdb_path=None,
            split=None,
            metadata={
                "vhh_pdb": row["vhh_pdb"],
                "antigen_pdb": row["antigen_pdb"],
                "member_name": row.get("member_name", ""),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=str, default="champloo_af3")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    scorer = StructuralInterfaceScorer(predictions_root=args.predictions_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_missing = 0
    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)
        for example in manifest_to_examples(rows, source=args.source):
            result = scorer.score(example)
            if result.extras.get("missing") == "prediction":
                n_missing += 1
            writer.writerow(
                [
                    example.id,
                    result.scorer_name,
                    result.value,
                    example.label,
                    example.target_name,
                    example.source,
                    example.binder_format,
                    example.split or "",
                    json.dumps(result.extras, sort_keys=True),
                ]
            )
            n += 1

    print(f"Wrote {n} scores ({n_missing} missing structures) to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
