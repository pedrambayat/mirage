"""M-C cross-regime transfer: a frozen rung gate trained on one regime, applied to
the other, in BOTH directions, with an antigen-overlap dedup guard.

The headline (2a, robust) trains on the larger SAbDab set and tests on the Champloo
diagonal — "do SAbDab's structural rules generalize?". The inverse (2b, caveated)
trains on 91 Champloo positives and tests on the SAbDab reservoir; a failure there can
be underfitting, not absent signal, so 2b is read only in light of 2a. Champloo rows
whose antigen falls in a SAbDab antigen cluster are dropped (leakage guard, both
directions).

Use::

    uv run python scripts/analyze_mc_cross_regime.py \\
        --sabdab-features data/staged/mc/sabdab_features.csv \\
        --champloo-features data/staged/mc/champloo_features.csv \\
        --sabdab-pairs data/staged/sabdab/sabdab_pairs.csv \\
        --champloo-pairs data/staged/protenix/champloo_pairs.csv \\
        --output results/published/mc_cross_regime.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mirage.eval.gate import auroc
from mirage.eval.orthogonal import evaluate_frozen_gate
from mirage.features.clustering import cluster_antigens
from mirage.features.mc_rungs import (
    fit_rung_model,
    folds_array,
    labels_array,
    read_feature_csv,
    rung_matrix,
)

_TRANSFER_RUNG = 3
_CONTRAST_RUNG = 0


def _antigen_by_pair(pairs_path: Path) -> dict[str, str]:
    with pairs_path.open(newline="") as fh:
        return {r["pair_id"]: r["antigen_seq"] for r in csv.DictReader(fh)}


def champloo_antigen_overlap(
    champloo_pairs: Path, sabdab_pairs: Path, *, max_identity: float
) -> set[str]:
    """Champloo pair_ids whose antigen clusters with ANY SAbDab antigen.

    Clusters the pooled antigen sequences at ``max_identity``; a Champloo pair is
    flagged if its antigen shares a cluster with at least one SAbDab antigen.
    """
    cha = _antigen_by_pair(champloo_pairs)
    sab = _antigen_by_pair(sabdab_pairs)
    sab_seqs = list(dict.fromkeys(sab.values()))
    cha_items = list(cha.items())
    pooled = sab_seqs + [ag for _, ag in cha_items]
    clusters = cluster_antigens(pooled, max_identity=max_identity)
    sab_clusters = set(clusters[: len(sab_seqs)])
    cha_clusters = clusters[len(sab_seqs) :]
    return {pid for (pid, _), c in zip(cha_items, cha_clusters, strict=True) if c in sab_clusters}


def _transfer(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    *,
    rung: int,
    l2: float,
    target_precision: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    xtr, names = rung_matrix(train_rows, rung=rung)
    ytr = labels_array(train_rows)
    ftr = folds_array(train_rows)
    model, _ = fit_rung_model(
        xtr, ytr, ftr, feature_names=names, l2=l2, target_precision=target_precision
    )
    xte, _ = rung_matrix(test_rows, rung=rung)
    yte = labels_array(test_rows)
    report = evaluate_frozen_gate(model, xte, yte, n_boot=n_boot, seed=seed)
    report["auroc"] = auroc(model.predict_logit(xte), yte)
    return report


def analyze_cross_regime(
    sabdab_features: Path,
    champloo_features: Path,
    sabdab_pairs: Path,
    champloo_pairs: Path,
    *,
    max_identity: float,
    l2: float,
    target_precision: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    sab_rows = read_feature_csv(sabdab_features)
    cha_rows_all = read_feature_csv(champloo_features)

    overlap = champloo_antigen_overlap(champloo_pairs, sabdab_pairs, max_identity=max_identity)
    cha_rows = [r for r in cha_rows_all if r["pair_id"] not in overlap]

    def both_rungs(train: list[dict[str, str]], test: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "rung3": _transfer(
                train,
                test,
                rung=_TRANSFER_RUNG,
                l2=l2,
                target_precision=target_precision,
                n_boot=n_boot,
                seed=seed,
            ),
            "rung0_contrast": _transfer(
                train,
                test,
                rung=_CONTRAST_RUNG,
                l2=l2,
                target_precision=target_precision,
                n_boot=n_boot,
                seed=seed,
            ),
        }

    return {
        "dedup": {
            "max_identity": max_identity,
            "champloo_total": len(cha_rows_all),
            "champloo_dropped_as_overlap": len(cha_rows_all) - len(cha_rows),
            "champloo_kept": len(cha_rows),
        },
        "sabdab_to_champloo_primary": both_rungs(sab_rows, cha_rows),
        "champloo_to_sabdab_caveated": both_rungs(cha_rows, sab_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sabdab-features", type=Path, required=True)
    parser.add_argument("--champloo-features", type=Path, required=True)
    parser.add_argument("--sabdab-pairs", type=Path, required=True)
    parser.add_argument("--champloo-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-identity", type=float, default=0.9)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    result = analyze_cross_regime(
        args.sabdab_features,
        args.champloo_features,
        args.sabdab_pairs,
        args.champloo_pairs,
        max_identity=args.max_identity,
        l2=args.l2,
        target_precision=args.target_precision,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    d = result["dedup"]
    print(
        f"Dedup: kept {d['champloo_kept']}/{d['champloo_total']} Champloo rows "
        f"(dropped {d['champloo_dropped_as_overlap']} overlapping antigens)"
    )
    print(
        "2a SAbDab->Champloo rung3 AUROC:",
        round(result["sabdab_to_champloo_primary"]["rung3"]["auroc"], 3),
    )
    print(
        "2b Champloo->SAbDab rung3 AUROC:",
        round(result["champloo_to_sabdab_caveated"]["rung3"]["auroc"], 3),
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
