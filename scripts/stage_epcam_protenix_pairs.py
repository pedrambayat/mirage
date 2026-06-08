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
