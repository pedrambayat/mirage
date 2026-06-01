from __future__ import annotations

import pytest

from mirage.features import normalize
from mirage.features.normalize import normalize_antigen, normalize_binder

_MATURE = "VPPGEDSKDVAAPHRQ"  # mature-IL6 fragment; not a signal prefix, no His run


def test_normalize_antigen_strips_il6_signal_peptide() -> None:
    raw = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP" + _MATURE + "HHHHHH"
    assert normalize_antigen(raw) == _MATURE


def test_normalize_antigen_strips_terminal_his() -> None:
    assert normalize_antigen(_MATURE + "HHHHHH") == _MATURE


def test_normalize_antigen_strips_leading_his() -> None:
    assert normalize_antigen("MAHHHHHHSSGVSKGEELFTG") == "SSGVSKGEELFTG"


def test_normalize_antigen_idempotent() -> None:
    raw = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP" + _MATURE + "HHHHHH"
    once = normalize_antigen(raw)
    assert normalize_antigen(once) == once


_LEADER = "MKYLLPTAAAGLLLLAAQPAMA"
_VHH = (
    "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKG"
    "RFTISRDNANNTVYLQMNSLKPEDTAVYACAADLLYHPGSWNDYWGQGTQVTVSS"
)
_HAS_HMMER = normalize._resolve_hmmer_bin() is not None
requires_hmmer = pytest.mark.skipif(not _HAS_HMMER, reason="HMMER/hmmscan unavailable")


def test_normalize_binder_falls_back_without_hmmer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the dependency-free path; fallback is His-strip only.
    monkeypatch.setattr(normalize, "_resolve_hmmer_bin", lambda: None)
    normalize._anarci_domain.cache_clear()
    assert normalize_binder("RANDOMSEQ" + "HHHHHH") == "RANDOMSEQ"


@requires_hmmer
def test_normalize_binder_extracts_variable_domain() -> None:
    dom = normalize_binder(_LEADER + _VHH + "HHHHHH")
    assert dom.startswith("QVQL")
    assert dom.endswith("VTVSS")
    assert "MKYLL" not in dom
    assert "HHHHHH" not in dom


@requires_hmmer
def test_normalize_binder_idempotent() -> None:
    once = normalize_binder(_LEADER + _VHH + "HHHHHH")
    assert normalize_binder(once) == once
