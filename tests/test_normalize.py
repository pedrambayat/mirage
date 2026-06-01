from __future__ import annotations

import pytest  # noqa: F401

from mirage.features import normalize  # noqa: F401
from mirage.features.normalize import normalize_antigen, normalize_binder  # noqa: F401

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
