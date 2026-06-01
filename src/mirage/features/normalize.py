"""Normalize binder/antigen sequences to comparable mature domains before the
Tier-S featurization. ``normalize_binder`` extracts the antibody variable domain
via ANARCI (IMGT) using a mirage-local HMMER; ``normalize_antigen`` strips known
signal peptides and terminal His-tags. Both are idempotent."""

from __future__ import annotations

import functools
import os
import re
import shutil
from pathlib import Path

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


def _resolve_hmmer_bin() -> str | None:
    """Locate a HMMER bin dir containing hmmscan: env var, then PATH, then the
    repo-local .tools/hmmer/bin built by scripts/install_hmmer.sh."""
    env = os.environ.get("MIRAGE_HMMER_BIN")
    if env and (Path(env) / "hmmscan").exists():
        return env
    on_path = shutil.which("hmmscan")
    if on_path:
        return str(Path(on_path).parent)
    repo_root = Path(__file__).resolve().parents[3]
    default = repo_root / ".tools" / "hmmer" / "bin"
    if (default / "hmmscan").exists():
        return str(default)
    return None


@functools.cache
def _anarci_domain(seq: str) -> str | None:
    """Return the IMGT variable-domain substring of ``seq``, or None if ANARCI /
    HMMER is unavailable or no antibody domain is found. Memoized over unique
    sequences (AVIDa has few unique VHHs across its 573k rows)."""
    hmmer_bin = _resolve_hmmer_bin()
    if hmmer_bin is None:
        return None
    try:
        from anarci import run_anarci  # type: ignore[import-untyped]

        result = run_anarci([("q", seq)], scheme="imgt", hmmerpath=hmmer_bin)
    except Exception:
        return None
    details = result[2][0]
    if not details:
        return None
    domain = details[0]
    start = int(domain["query_start"])
    end = int(domain["query_end"])
    # query_end is 0-based exclusive (ANARCI convention); use as-is.
    return seq[start:end]


def normalize_binder(seq: str) -> str:
    """Reduce an antibody chain to its IMGT variable domain (drops leader peptide,
    His-tags, framing residues). Falls back to His-strip when ANARCI/HMMER is
    unavailable or no domain is found."""
    seq = seq.strip().upper()
    domain = _anarci_domain(seq)
    if domain is None:
        return _strip_his_tags(seq)
    return _HIS_TAIL.sub("", domain)
