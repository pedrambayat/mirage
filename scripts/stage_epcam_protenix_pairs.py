"""Build the EpCAM real-negative canary pairs CSV for the M-C structure track.

14 designed EpCAM VHHs (CAR-T killing labels) x EpCAM ECD are the positives;
negatives are predict-the-shuffled-pair: each VHH x k wrong antigens drawn from
the SAbDab antigen pool, with the EpCAM antigen cluster excluded (leakage guard).
Emits the standard 6-column pairs schema consumed by stage_protenix_pairs.py and
extract_mc_features.py.

Use::

    uv run python scripts/stage_epcam_protenix_pairs.py \\
        --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \\
        --sabdab-pairs data/staged/sabdab/sabdab_pairs.csv \\
        --output data/staged/epcam/epcam_protenix_pairs.csv \\
        --k 5
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from mirage.benchmark.epcam_killing import EpCAMKillingLoader
from mirage.features.clustering import cluster_antigens
from mirage.features.normalize import normalize_antigen, normalize_binder

_FIELDNAMES = ["pair_id", "binder_seq", "antigen_seq", "label", "antigen_cluster", "fold"]


def epcam_antigen_negative_pool(
    sabdab_antigens: list[str], epcam_antigen: str, *, max_identity: float = 0.9
) -> list[str]:
    """SAbDab antigens that do NOT share a cluster with the EpCAM antigen.

    Clusters the pooled (EpCAM-first) antigen sequences and drops any SAbDab
    antigen whose cluster equals the EpCAM cluster. Multi-chain antigens (``:``)
    are clustered on their concatenated sequence but returned verbatim.
    """
    pooled = [epcam_antigen, *sabdab_antigens]
    clusters = cluster_antigens([a.replace(":", "") for a in pooled], max_identity=max_identity)
    epcam_cluster = clusters[0]
    return [ag for ag, c in zip(sabdab_antigens, clusters[1:], strict=True) if c != epcam_cluster]


def build_epcam_pairs(
    positives: list[tuple[str, str, str, str]],
    negative_pool: list[str],
    *,
    k: int,
    seed: int,
) -> list[dict[str, str]]:
    """Positive rows (VHH x EpCAM) + k shuffled negatives (VHH x pool antigen) each.

    ``positives`` items are ``(vhh_id, binder_seq, epcam_antigen_seq, killing_label)``;
    ``killing_label`` is not written to the CSV (re-derived at analysis time from the
    killing labels) but is accepted so callers pass the full positive record. Negative
    antigens are sampled without replacement per VHH. ``antigen_cluster`` and ``fold``
    are constant ``"0"`` — EpCAM is a frozen-transfer TEST set only, never trained or
    OOF-split, so neither column is consumed downstream.
    """
    if k > len(negative_pool):
        raise ValueError(f"k={k} exceeds negative pool size {len(negative_pool)}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for vhh_id, binder, antigen, _killing in positives:
        rows.append(
            {
                "pair_id": f"epcam-{vhh_id}__epcam",
                "binder_seq": binder,
                "antigen_seq": antigen,
                "label": "1",
                "antigen_cluster": "0",
                "fold": "0",
            }
        )
        idx = rng.choice(len(negative_pool), size=k, replace=False)
        for j, ai in enumerate(idx):
            rows.append(
                {
                    "pair_id": f"epcam-{vhh_id}__neg{j}",
                    "binder_seq": binder,
                    "antigen_seq": negative_pool[int(ai)],
                    "label": "0",
                    "antigen_cluster": "0",
                    "fold": "0",
                }
            )
    return rows


def load_epcam_positives(killing_labels: Path) -> list[tuple[str, str, str, str]]:
    """Load the 14 labeled VHHs as ``(vhh_id, binder, epcam_antigen, killing_label)``.

    Sequences are normalized at the featurization boundary (binder -> ANARCI IMGT
    variable domain; antigen -> signal-peptide/His-tag strip), exactly as the M-C
    feature pipeline does. ``Good`` -> ``functional``, ``Bad`` -> ``nonfunctional``.
    """
    loader = EpCAMKillingLoader(killing_labels)
    out: list[tuple[str, str, str, str]] = []
    for ex in loader.load():
        binder = normalize_binder(ex.binder_chains[0])
        antigen = normalize_antigen(ex.target_chains[0])
        killing = "functional" if ex.label == "BIND" else "nonfunctional"
        out.append((str(ex.metadata["vhh_id"]), binder, antigen, killing))
    return out


def sabdab_antigen_pool(sabdab_pairs: Path) -> list[str]:
    """Unique cognate antigen sequences from the SAbDab pairs CSV (label==1 rows)."""
    with sabdab_pairs.open(newline="") as fh:
        antigens = [r["antigen_seq"] for r in csv.DictReader(fh) if r["label"] == "1"]
    return list(dict.fromkeys(antigens))


app = typer.Typer(add_completion=False, help="Build the EpCAM canary pairs CSV.")


@app.command()
def main(
    killing_labels: Annotated[
        Path, typer.Option("--killing-labels", help="epcam_killing_labels.csv")
    ],
    sabdab_pairs: Annotated[
        Path, typer.Option("--sabdab-pairs", help="SAbDab pairs CSV (antigen pool)")
    ],
    output: Annotated[Path, typer.Option("--output", help="Output pairs CSV")],
    k: Annotated[int, typer.Option(help="Negatives per positive")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed")] = 20260607,
    max_identity: Annotated[float, typer.Option("--max-identity", help="Dedup identity")] = 0.9,
) -> None:
    """Stage EpCAM positives + dedup'd shuffled negatives; write to --output."""
    positives = load_epcam_positives(killing_labels)
    epcam_antigen = positives[0][2]

    # Leakage guard: none of the 14 designed VHHs may appear in SAbDab training.
    with sabdab_pairs.open(newline="") as fh:
        sabdab_binders = {r["binder_seq"] for r in csv.DictReader(fh)}
    overlap = {p[0] for p in positives if p[1] in sabdab_binders}
    if overlap:
        raise SystemExit(f"Leakage: EpCAM VHH(s) {sorted(overlap)} present in SAbDab binders")

    pool = epcam_antigen_negative_pool(
        sabdab_antigen_pool(sabdab_pairs), epcam_antigen, max_identity=max_identity
    )
    if len(pool) < k:
        raise SystemExit(f"Negative pool too small ({len(pool)}) for k={k}")

    rows = build_epcam_pairs(positives, pool, k=k, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == "1")
    typer.echo(
        f"positives={n_pos} negatives={len(rows) - n_pos} total={len(rows)} "
        f"pool={len(pool)} (dedup max_identity={max_identity})"
    )


if __name__ == "__main__":
    app()
