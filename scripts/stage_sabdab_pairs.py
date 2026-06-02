"""Stage SAbDab VHH-antigen pairs for the sequence-only baseline.

Loads cognate positives via the existing SAbDabLoader (VHH-only), normalizes
sequences to mature domains, clusters antigens by sequence identity, assigns
held-out-antigen-cluster folds, and constructs distribution-matched,
cross-cluster, fold-consistent shuffled negatives. Emits a flat pair CSV plus a
unique-sequence manifest for the embedding step.

Use::

    uv run python scripts/stage_sabdab_pairs.py \\
        --data-dir ../abdisc-data/sabdab \\
        --output data/staged/sabdab/sabdab_pairs.csv \\
        --manifest data/staged/sabdab/sabdab_unique_seqs.txt
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Positive:
    pair_id: str
    binder_seq: str
    antigen_seq: str
    antigen_cluster: int
    fold: int


def _antigens_by_cluster(positives: list[Positive], fold: int) -> dict[int, list[str]]:
    """Unique antigen sequences present in ``fold``, grouped by cluster."""
    out: dict[int, list[str]] = {}
    seen: set[tuple[int, str]] = set()
    for p in positives:
        if p.fold != fold:
            continue
        key = (p.antigen_cluster, p.antigen_seq)
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(p.antigen_cluster, []).append(p.antigen_seq)
    return out


def build_pairs(positives: list[Positive], *, k: int, seed: int) -> list[dict[str, str]]:
    """Emit positive rows + k distribution-matched shuffled negatives each.

    For a positive in fold ``f`` with antigen cluster ``c_i``, each negative
    antigen is drawn by (1) sampling a cluster from the fold's positive
    cluster-frequency distribution restricted to clusters != c_i (so the
    negative cluster marginal matches the positive marginal and antigen
    popularity carries no label signal), then (2) a uniform antigen within that
    cluster. Every negative is cross-cluster and fold-consistent by construction.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for fold in sorted({p.fold for p in positives}):
        fold_pos = [p for p in positives if p.fold == fold]
        ag_by_cluster = _antigens_by_cluster(positives, fold)
        cluster_counts = Counter(p.antigen_cluster for p in fold_pos)
        clusters = np.array(sorted(cluster_counts))
        weights = np.array([cluster_counts[int(c)] for c in clusters], dtype=float)
        for p in fold_pos:
            rows.append(
                {
                    "pair_id": p.pair_id,
                    "binder_seq": p.binder_seq,
                    "antigen_seq": p.antigen_seq,
                    "label": "1",
                    "antigen_cluster": str(p.antigen_cluster),
                    "fold": str(fold),
                }
            )
            keep = (clusters != p.antigen_cluster) & np.array(
                [int(c) in ag_by_cluster for c in clusters]
            )
            cand = clusters[keep]
            cand_w = weights[keep]
            if cand.size == 0:
                continue
            probs = cand_w / cand_w.sum()
            for j in range(k):
                cj = int(rng.choice(cand, p=probs))
                pool = ag_by_cluster[cj]
                ag = pool[int(rng.integers(len(pool)))]
                rows.append(
                    {
                        "pair_id": f"{p.pair_id}__neg{j}",
                        "binder_seq": p.binder_seq,
                        "antigen_seq": ag,
                        "label": "0",
                        "antigen_cluster": str(cj),
                        "fold": str(fold),
                    }
                )
    return rows


_FIELDNAMES = ["pair_id", "binder_seq", "antigen_seq", "label", "antigen_cluster", "fold"]


def _build_positives(data_dir: Path, *, n_splits: int, seed: int) -> list[Positive]:
    from mirage.benchmark.sabdab import SAbDabLoader
    from mirage.features.clustering import cluster_antigens
    from mirage.features.normalize import normalize_antigen, normalize_binder
    from mirage.ml.core import assign_folds

    loader = SAbDabLoader(data_dir=data_dir, use_anarci=True)
    raw: list[tuple[str, str, str]] = []  # (pair_id, binder_seq, antigen_seq)
    for ex in loader.load():
        if ex.binder_format != "vhh":
            continue
        binder = normalize_binder(ex.binder_chains[0])
        antigen = ":".join(normalize_antigen(c) for c in ex.target_chains)
        if not binder or not antigen.replace(":", ""):
            continue
        raw.append((ex.id, binder, antigen))

    unique_ag = sorted({a for _, _, a in raw})
    ag_cluster = dict(
        zip(
            unique_ag,
            cluster_antigens([a.replace(":", "") for a in unique_ag]),
            strict=True,
        )
    )
    clusters = np.array([ag_cluster[a] for _, _, a in raw])
    folds = assign_folds(clusters, n_splits=n_splits, seed=seed)
    return [
        Positive(pid, b, a, int(ag_cluster[a]), int(f))
        for (pid, b, a), f in zip(raw, folds, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    positives = _build_positives(args.data_dir, n_splits=args.n_splits, seed=args.seed)
    rows = build_pairs(positives, k=args.k, seed=args.seed)

    # Invariant checks before writing (cross-cluster + fold-consistent).
    by_id = {p.pair_id: p for p in positives}
    for r in rows:
        if r["label"] == "0":
            parent = by_id[r["pair_id"].split("__neg")[0]]
            assert int(r["antigen_cluster"]) != parent.antigen_cluster
            assert int(r["fold"]) == parent.fold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    uniques = sorted({r["binder_seq"] for r in rows} | {r["antigen_seq"] for r in rows})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(uniques) + "\n")

    n_pos = sum(1 for r in rows if r["label"] == "1")
    print(
        f"positives={n_pos} negatives={len(rows) - n_pos} "
        f"unique_clusters={len({p.antigen_cluster for p in positives})} "
        f"unique_seqs={len(uniques)}"
    )
    print(f"Wrote {len(rows)} rows to {args.output} and {len(uniques)} seqs to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
