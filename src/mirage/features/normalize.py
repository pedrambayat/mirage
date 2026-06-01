"""Normalize binder/antigen sequences to comparable mature domains before the
Tier-S featurization. ``normalize_binder`` extracts the antibody variable domain
via ANARCI (IMGT) using a mirage-local HMMER; ``normalize_antigen`` strips known
signal peptides and terminal His-tags. Both are idempotent."""

from __future__ import annotations

import re

from mirage.benchmark.targets import SIGNAL_PEPTIDES

_HIS_LEAD = re.compile(r"^[MA]{0,3}H{5,}")
_HIS_TAIL = re.compile(r"H{5,}$")


def _strip_his_tags(seq: str) -> str:
    """Remove an unambiguous terminal His-run (>=5) at either end. Conservative:
    the leading form tolerates a short Met/Ala cloning prefix; no internal cuts."""
    seq = _HIS_LEAD.sub("", seq)
    seq = _HIS_TAIL.sub("", seq)
    return seq


def normalize_antigen(seq: str) -> str:
    """Strip a known precursor signal peptide (if present) then terminal His-tags."""
    seq = seq.strip().upper()
    for signal in SIGNAL_PEPTIDES:
        if seq.startswith(signal):
            seq = seq[len(signal) :]
            break
    return _strip_his_tags(seq)


def normalize_binder(seq: str) -> str:
    """Temporary pass-through; replaced with ANARCI extraction in Task 2."""
    return seq.strip().upper()
