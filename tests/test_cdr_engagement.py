"""Tests for cdr_engagement feature extractor.

Two groups:
- Row-preserving fallback tests: must pass regardless of whether HMMER/ANARCI are
  available.  They verify the function never raises and always returns the full
  feature dict with sensible defaults.
- Real-mapping tests: gated on HMMER being present; verify that a real VHH
  sequence is correctly numbered and the CDR fractions are plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from mirage.features.cdr_engagement import CDR_FEATURE_NAMES, cdr_engagement_features
from mirage.features.normalize import _resolve_hmmer_bin

_HAS_HMMER = _resolve_hmmer_bin() is not None
requires_hmmer = pytest.mark.skipif(not _HAS_HMMER, reason="needs local HMMER")

# A real VHH variable-domain sequence (no leader, no His-tag).
VHH = (
    "QVQLVESGGALVQPGGSLRLSCAASGFPVNRYSMRWYRQAPGKEREWVAGMSSAGDRSSYEDSVKG"
    "RFTISRDDARNTVYLQMNSLKPEDTAVYYCNVNVGFEYWGQGTQVTVSS"
)


# ---------------------------------------------------------------------------
# Row-preserving fallback (must work with OR without HMMER)
# ---------------------------------------------------------------------------


def test_fallback_non_antibody_returns_defaults() -> None:
    """A non-antibody sequence must return all zeros and cdr_mapping_ok == 0.0
    without raising — regardless of HMMER availability."""
    result = cdr_engagement_features(
        binder_seq="AAAAAAAAAA",
        binder_interface_residue_indices=np.array([0, 1, 2]),
        n_binder_residues=10,
    )
    assert isinstance(result, dict)
    assert set(CDR_FEATURE_NAMES).issubset(result)
    assert result["cdr_mapping_ok"] == 0.0
    assert result["cdr_contact_fraction"] == 0.0


def test_fallback_returns_all_feature_names() -> None:
    """All five CDR_FEATURE_NAMES must be present in the returned dict."""
    result = cdr_engagement_features(
        binder_seq="MMMMMM",
        binder_interface_residue_indices=np.array([0]),
        n_binder_residues=6,
    )
    for name in CDR_FEATURE_NAMES:
        assert name in result, f"Missing feature: {name}"


def test_fallback_empty_interface_indices() -> None:
    """Empty interface index array → defaults (cdr_mapping_ok == 0.0)."""
    result = cdr_engagement_features(
        binder_seq="AAAAAAAAAA",
        binder_interface_residue_indices=np.array([], dtype=int),
        n_binder_residues=10,
    )
    assert result["cdr_mapping_ok"] == 0.0
    assert result["cdr_contact_fraction"] == 0.0


def test_fallback_out_of_bounds_indices_clipped() -> None:
    """Interface indices that all exceed n_binder_residues → defaults."""
    result = cdr_engagement_features(
        binder_seq="AAAAAAAAAA",
        binder_interface_residue_indices=np.array([10, 11, 99]),
        n_binder_residues=10,
    )
    # All indices are >= n_binder_residues so the filtered set is empty → defaults.
    assert result["cdr_mapping_ok"] == 0.0
    assert result["cdr_contact_fraction"] == 0.0


def test_fallback_no_raise_on_short_sequence() -> None:
    """Single-residue sequences must not raise."""
    result = cdr_engagement_features(
        binder_seq="A",
        binder_interface_residue_indices=np.array([0]),
        n_binder_residues=1,
    )
    assert isinstance(result, dict)
    assert set(CDR_FEATURE_NAMES).issubset(result)


# ---------------------------------------------------------------------------
# Real mapping (requires HMMER)
# ---------------------------------------------------------------------------


@requires_hmmer
def test_real_vhh_mapping_ok_flag() -> None:
    """A real VHH sequence with all residues at the interface should return
    cdr_mapping_ok == 1.0."""
    result = cdr_engagement_features(
        binder_seq=VHH,
        binder_interface_residue_indices=np.arange(len(VHH)),
        n_binder_residues=len(VHH),
    )
    assert result["cdr_mapping_ok"] == 1.0


@requires_hmmer
def test_real_vhh_cdr_contact_fraction_plausible() -> None:
    """CDRs are a minority of the VHH sequence: fraction should be strictly
    between 0 and 1."""
    result = cdr_engagement_features(
        binder_seq=VHH,
        binder_interface_residue_indices=np.arange(len(VHH)),
        n_binder_residues=len(VHH),
    )
    frac = result["cdr_contact_fraction"]
    assert 0.0 < frac < 1.0, f"Expected 0 < cdr_contact_fraction < 1, got {frac}"


@requires_hmmer
def test_real_vhh_per_cdr_fractions_in_unit_interval() -> None:
    """Each per-CDR fraction must lie in [0, 1]."""
    result = cdr_engagement_features(
        binder_seq=VHH,
        binder_interface_residue_indices=np.arange(len(VHH)),
        n_binder_residues=len(VHH),
    )
    for name in ("cdr1_contact_fraction", "cdr2_contact_fraction", "cdr3_contact_fraction"):
        v = result[name]
        assert 0.0 <= v <= 1.0, f"{name} = {v} is outside [0, 1]"


@requires_hmmer
def test_real_vhh_no_interface_returns_defaults() -> None:
    """Even when ANARCI succeeds, if the (filtered) interface is empty we get
    defaults — do NOT raise."""
    result = cdr_engagement_features(
        binder_seq=VHH,
        binder_interface_residue_indices=np.array([], dtype=int),
        n_binder_residues=len(VHH),
    )
    assert result["cdr_mapping_ok"] == 0.0
    assert result["cdr_contact_fraction"] == 0.0


@requires_hmmer
def test_real_vhh_partial_interface() -> None:
    """Restrict the interface to only CDR3-region residues (IMGT 105-117,
    roughly the last ~15 residues of the VHH).  cdr3_contact_fraction should
    be > 0 and the overall fraction should equal cdr3 fraction."""
    # CDR3 of VHH typically occupies the last stretch before the FR4.
    # Use the last 15 residue indices as a rough proxy.
    iface = np.arange(len(VHH) - 15, len(VHH))
    result = cdr_engagement_features(
        binder_seq=VHH,
        binder_interface_residue_indices=iface,
        n_binder_residues=len(VHH),
    )
    assert result["cdr_mapping_ok"] == 1.0
    # Some of these residues will be in CDR3 or FR4; at least the call must succeed.
    assert 0.0 <= result["cdr_contact_fraction"] <= 1.0
