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

import numpy as np

from mirage.features.clustering import cluster_antigens

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
    return [
        ag
        for ag, c in zip(sabdab_antigens, clusters[1:], strict=True)
        if c != epcam_cluster
    ]


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
