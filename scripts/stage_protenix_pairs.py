"""Pure helpers for staging (binder, antigen) pairs for Protenix structure prediction.

Converts a flat pair CSV (columns: pair_id, binder_seq, antigen_seq, label,
antigen_cluster, fold — the format emitted by stage_sabdab_pairs.py and
stage_champloo_protenix_pairs.py) into BenchmarkExample objects and a
deduplicated unique-sequence manifest used by the pose-prediction step.

``sequence_hash`` is the canonical home for that helper; a later task imports
it from here.

Use::

    uv run python scripts/stage_protenix_pairs.py \\
        --pairs data/staged/champloo/champloo_protenix_pairs.csv \\
        --staged-root data/staged/champloo/protenix

# NOTE: Per-pair Protenix JSON-input emission and --submit are wired in the
# ProtenixPosePredictor task (Task 4); they are NOT implemented here.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer

from mirage.pose_predictors.protenix import sequence_hash  # re-export
from mirage.scorers.base import BenchmarkExample


def examples_from_pairs_csv(path: Path) -> Iterator[BenchmarkExample]:
    """Yield one BenchmarkExample per row of a pairs CSV.

    The CSV must have columns: pair_id, binder_seq, antigen_seq, label,
    antigen_cluster, fold (the schema emitted by stage_sabdab_pairs.py and
    stage_champloo_protenix_pairs.py).
    """
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            yield BenchmarkExample(
                id=row["pair_id"],
                label=row["label"],
                binder_chains=(row["binder_seq"],),
                binder_format="vhh",
                target_chains=(row["antigen_seq"],),
                target_name="",
                source="pairs_csv",
                metadata={
                    "antigen_cluster": int(row["antigen_cluster"]),
                    "fold": int(row["fold"]),
                },
            )


def unique_sequences(path: Path) -> set[str]:
    """Return the set of all distinct binder and antigen sequences across all rows.

    A sequence that appears as both a binder and an antigen, or is reused
    across multiple pair rows, is counted once.
    """
    seqs: set[str] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            seqs.add(row["binder_seq"])
            seqs.add(row["antigen_seq"])
    return seqs


app = typer.Typer(help="Stage unique sequences from a pairs CSV for Protenix prediction.")


@app.command()
def main(
    pairs: Annotated[Path, typer.Option(help="Path to the pairs CSV.")],
    staged_root: Annotated[Path, typer.Option("--staged-root", help="Output directory root.")],
) -> None:
    """Write unique_seqs.txt with one '{seq}\\t{hash}' line per unique sequence.

    # NOTE: Per-pair Protenix JSON-input emission and --submit are wired in the
    # ProtenixPosePredictor task (Task 4); they are NOT implemented here.
    """
    n_rows = 0
    with pairs.open(newline="") as fh:
        n_rows = sum(1 for _ in csv.DictReader(fh))

    uniq = unique_sequences(pairs)
    n_unique = len(uniq)

    staged_root.mkdir(parents=True, exist_ok=True)
    out_path = staged_root / "unique_seqs.txt"
    lines = sorted(f"{seq}\t{sequence_hash(seq)}" for seq in uniq)
    out_path.write_text("\n".join(lines) + "\n")

    typer.echo(f"pairs={n_rows} unique_seqs={n_unique}")


if __name__ == "__main__":
    app()
