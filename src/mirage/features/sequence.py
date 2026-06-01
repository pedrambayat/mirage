"""Tier-S sequence features: a compact, regularization-friendly physicochemical
descriptor for a (binder, target) sequence pair. Pure Python, no dependencies.

Deliberately compact (~6 per chain) because the Champloo positive cohort is
small (~106); a 40-dim composition vector would overfit. ESM-2 embeddings are
an explicitly deferred Tier-S extension (would add torch) — out of scope here.
"""

from __future__ import annotations

_AROMATIC = frozenset("FWY")
_HYDROPHOBIC = frozenset("AVLIMFWC")
_POLAR = frozenset("STNQ")
_POSITIVE = frozenset("KR")
_NEGATIVE = frozenset("DE")

FEATURE_NAMES: tuple[str, ...] = (
    "binder_length",
    "binder_net_charge",
    "binder_aromatic_frac",
    "binder_hydrophobic_frac",
    "binder_polar_frac",
    "binder_cysteine_frac",
    "target_length",
    "target_net_charge",
    "target_aromatic_frac",
    "target_hydrophobic_frac",
    "target_polar_frac",
    "target_cysteine_frac",
    "length_ratio",
)


def _chain_features(seq: str) -> dict[str, float]:
    seq = seq.strip().upper()
    n = len(seq)
    if n == 0:
        return {
            "length": 0.0,
            "net_charge": 0.0,
            "aromatic_frac": 0.0,
            "hydrophobic_frac": 0.0,
            "polar_frac": 0.0,
            "cysteine_frac": 0.0,
        }
    pos = sum(1 for c in seq if c in _POSITIVE)
    neg = sum(1 for c in seq if c in _NEGATIVE)
    return {
        "length": float(n),
        "net_charge": float(pos - neg),
        "aromatic_frac": sum(1 for c in seq if c in _AROMATIC) / n,
        "hydrophobic_frac": sum(1 for c in seq if c in _HYDROPHOBIC) / n,
        "polar_frac": sum(1 for c in seq if c in _POLAR) / n,
        "cysteine_frac": seq.count("C") / n,
    }


def sequence_features(binder_seq: str, target_seq: str) -> dict[str, float]:
    """Return the Tier-S feature dict for one (binder, target) pair.

    Keys are exactly ``FEATURE_NAMES`` in order.
    """
    b = _chain_features(binder_seq)
    t = _chain_features(target_seq)
    length_ratio = b["length"] / t["length"] if t["length"] > 0 else 0.0
    out = {
        "binder_length": b["length"],
        "binder_net_charge": b["net_charge"],
        "binder_aromatic_frac": b["aromatic_frac"],
        "binder_hydrophobic_frac": b["hydrophobic_frac"],
        "binder_polar_frac": b["polar_frac"],
        "binder_cysteine_frac": b["cysteine_frac"],
        "target_length": t["length"],
        "target_net_charge": t["net_charge"],
        "target_aromatic_frac": t["aromatic_frac"],
        "target_hydrophobic_frac": t["hydrophobic_frac"],
        "target_polar_frac": t["polar_frac"],
        "target_cysteine_frac": t["cysteine_frac"],
        "length_ratio": length_ratio,
    }
    return {name: out[name] for name in FEATURE_NAMES}
