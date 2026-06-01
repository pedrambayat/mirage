"""Re-score a staged AF2-M pilot manifest with a registered scorer.

Reconstructs each manifest row's BenchmarkExample directly from `summary.tsv`
via the SAbDab loader's row→candidate→example path, bypassing the loader's
full-dataset ANARCI batch. This is the same pattern the 2026-05-13 morning
session used to publish the first pilot CSVs.

Default scorer is ``rmsd_to_crystal`` (with optional ``--dockq``). Use
``--scorer af2m_confidence`` to run the AF2-M-confidence scorer instead
(no ``--dockq`` flag applies for that scorer), or ``--scorer
structural_interface`` to compute crystal-independent predicted-interface
geometry.
For manifest rows that do not exist in SAbDab ``summary.tsv`` (for example
``*-scramble01`` negative controls), pass ``--from-manifest`` with
``--scorer af2m_confidence`` or ``--scorer structural_interface`` to
reconstruct minimal crystal-independent examples from the staged FASTA paths.

Use::

    uv run python scripts/score_pilot_manifest.py \
        --manifest data/staged/af2m/manifest_20260512_163902.tsv \
        --data-dir /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/abdisc-data/sabdab \
        --output results/published/sabdab_af2m_rmsd_pilot.csv \
        --no-dockq

A second invocation with ``--dockq`` produces the DockQ variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from mirage.benchmark.sabdab import SAbDabLoader
from mirage.scorers.af2m_confidence import AF2MConfidenceScorer
from mirage.scorers.base import AbstractScorer, BenchmarkExample
from mirage.scorers.rmsd_to_crystal import RMSDToCrystalScorer
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

_SYNTHETIC_SUFFIX_RE = re.compile(r"-(?:scramble\d+|wrongtarget\d+)$")


def _manifest_example_ids(manifest_path: Path) -> list[str]:
    with manifest_path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [row["example_id"] for row in reader]


def _manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _summary_rows(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "summary.tsv").open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _parse_example_id(example_id: str) -> tuple[str, str, str]:
    # Format: ``sabdab-{pdb}-{Hchain}-{antigen_chain_first}``. The first chain
    # IDs are single characters (e.g. ``9uo0-3-M``) so a straight split on '-'
    # is unambiguous.
    parts = example_id.split("-")
    if len(parts) != 4 or parts[0] != "sabdab":
        raise ValueError(f"unexpected example id format: {example_id}")
    return parts[1], parts[2], parts[3]


def _build_examples(
    manifest_ids: list[str], loader: SAbDabLoader, rows: list[dict[str, str]]
) -> Iterable[BenchmarkExample]:
    """Yield one BenchmarkExample per manifest id, by matching summary.tsv."""
    index: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        index.setdefault((row["pdb"].lower(), row["Hchain"].strip()), []).append(row)

    for example_id in manifest_ids:
        pdb, hchain, antigen_first = _parse_example_id(example_id)
        matches = index.get((pdb, hchain), [])
        if not matches:
            raise SystemExit(f"no summary row for {example_id} (pdb={pdb} H={hchain})")

        chosen: dict[str, str] | None = None
        for row in matches:
            cand = loader._row_to_candidate(row)
            if cand is None:
                continue
            if cand.target_chain_ids and cand.target_chain_ids[0] == antigen_first:
                chosen = row
                example = loader._make_example(cand)
                break
        if chosen is None:
            raise SystemExit(
                f"no structural candidate matched {example_id} (antigen_first={antigen_first})"
            )
        if example.id != example_id:
            raise SystemExit(
                f"reconstructed id {example.id} does not match manifest id {example_id}"
            )
        yield example


def _read_rmsd_parent_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fh:
        return {row["example_id"]: row for row in csv.DictReader(fh)}


def _parent_id_for_manifest_id(example_id: str) -> str:
    return _SYNTHETIC_SUFFIX_RE.sub("", example_id)


def _label_for_manifest_id(example_id: str, parent_id: str, parent: dict[str, str]) -> str:
    if "-scramble" in example_id:
        return "SCRAMBLE"
    if "-wrongtarget" in example_id:
        return "WRONG_TARGET"
    if parent_id != example_id:
        return "SYNTHETIC_NEGATIVE"
    return parent.get("label", "")


def _parse_colabfold_fasta(path: Path) -> tuple[str, ...]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"FASTA too short: {path}")
    return tuple(lines[1].split(":"))


def _split_manifest_chains(
    chains: tuple[str, ...], binder_format: str, example_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    n_binder = 2 if binder_format == "fab" else 1
    binder = chains[:n_binder]
    target = chains[n_binder:]
    if not binder or not target:
        raise SystemExit(
            f"{example_id}: cannot split FASTA chains for binder_format={binder_format!r}"
        )
    return binder, target


def _build_examples_from_manifest(
    manifest_rows: list[dict[str, str]], parent_rows: dict[str, dict[str, str]]
) -> Iterable[BenchmarkExample]:
    """Yield minimal crystal-independent examples directly from manifest FASTAs."""
    for row in manifest_rows:
        example_id = row["example_id"]
        parent_id = _parent_id_for_manifest_id(example_id)
        parent = parent_rows.get(parent_id)
        if parent is None:
            raise SystemExit(f"no parent RMSD row for {example_id} (parent={parent_id})")

        binder_format = parent.get("binder_format", "")
        chains = _parse_colabfold_fasta(Path(row["fasta_path"]))
        binder_chains, target_chains = _split_manifest_chains(chains, binder_format, example_id)
        yield BenchmarkExample(
            id=example_id,
            label=_label_for_manifest_id(example_id, parent_id, parent),
            binder_chains=binder_chains,
            binder_format=binder_format,
            target_chains=target_chains,
            target_name=parent.get("target_name", ""),
            source="manifest",
            target_pdb_id=None,
            complex_pdb_path=None,
            split=parent.get("split") or None,
            metadata={"manifest_parent_id": parent_id},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--from-manifest",
        action="store_true",
        help="Build minimal examples from manifest FASTAs instead of SAbDab summary.tsv.",
    )
    parser.add_argument(
        "--parent-rmsd-csv",
        type=Path,
        default=Path("results/published/sabdab_af2m_rmsd_n200.csv"),
        help="Parent metadata CSV used by --from-manifest.",
    )
    parser.add_argument(
        "--scorer",
        choices=("rmsd_to_crystal", "af2m_confidence", "structural_interface"),
        default="rmsd_to_crystal",
    )
    dq = parser.add_mutually_exclusive_group()
    dq.add_argument("--dockq", dest="compute_dockq", action="store_true", default=False)
    dq.add_argument("--no-dockq", dest="compute_dockq", action="store_false")
    args = parser.parse_args()

    if args.from_manifest and args.scorer == "rmsd_to_crystal":
        raise SystemExit("--from-manifest is only valid with crystal-independent scorers")

    scorer: AbstractScorer
    if args.scorer == "rmsd_to_crystal":
        scorer = RMSDToCrystalScorer(compute_dockq=args.compute_dockq)
    elif args.scorer == "af2m_confidence":
        scorer = AF2MConfidenceScorer()
    elif args.scorer == "structural_interface":
        scorer = StructuralInterfaceScorer()
    else:  # pragma: no cover — argparse choices guard this
        raise SystemExit(f"unknown scorer: {args.scorer}")

    examples: Iterable[BenchmarkExample]
    if args.from_manifest:
        examples = _build_examples_from_manifest(
            _manifest_rows(args.manifest), _read_rmsd_parent_rows(args.parent_rmsd_csv)
        )
    else:
        loader = SAbDabLoader(data_dir=args.data_dir, use_anarci=False)
        rows = _summary_rows(args.data_dir)
        examples = _build_examples(_manifest_example_ids(args.manifest), loader, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)
        for example in examples:
            result = scorer.score(example)
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

    print(f"Wrote {n} scores to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
