"""Tests for scripts/extract_mc_features.py — fault-tolerant M-C feature assembler.

TDD: these tests are written BEFORE the implementation. They must fail first,
then pass once the implementation is complete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing from scripts/ without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from extract_mc_features import (  # type: ignore[import-not-found]
    FEATURE_COLUMNS,
    assemble_row,
    build_rows,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Real fixture location
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "protenix"

# Sequences from test_chain_role_resolver.py — chain A = VHH binder, chain B = GFP
SEQ_A = "MQVQLVESGGALVQPGGSLRLSCAASGFPVNRYSMRWYRQAPGKEREWVAGMSSAGDRSSYEDSVKGRFTISRDDARNTVYLQMNSLKPEDTAVYYCNVNVGFEYWGQGTQVTVSSKHHHHHH"  # noqa: E501
SEQ_B = "MAHHHHHHSSGVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"  # noqa: E501


def _minimal_row(
    pair_id: str = "A__B",
    binder_seq: str = "QVQ",
    antigen_seq: str = "EVAL",
    label: str = "1",
    antigen_cluster: str = "3",
    fold: str = "0",
) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "binder_seq": binder_seq,
        "antigen_seq": antigen_seq,
        "label": label,
        "antigen_cluster": antigen_cluster,
        "fold": fold,
    }


# ---------------------------------------------------------------------------
# Test 1: FEATURE_COLUMNS completeness
# ---------------------------------------------------------------------------


def test_feature_columns_completeness() -> None:
    """FEATURE_COLUMNS must contain the passthrough and one key from each feature group."""
    # Passthrough / control columns
    assert "pair_id" in FEATURE_COLUMNS
    assert "prediction_present" in FEATURE_COLUMNS
    # Confidence group
    assert "iptm" in FEATURE_COLUMNS
    assert "interface_pae" in FEATURE_COLUMNS
    # Geometry group
    assert "n_interface_residues_binder" in FEATURE_COLUMNS
    # CDR group
    assert "cdr_contact_fraction" in FEATURE_COLUMNS
    assert "cdr_mapping_ok" in FEATURE_COLUMNS
    # The canonical ordered list
    assert FEATURE_COLUMNS == [
        "pair_id",
        "label",
        "antigen_cluster",
        "fold",
        "prediction_present",
        "iptm",
        "ptm",
        "interface_pae",
        "min_interface_pae",
        "interface_plddt",
        "mean_plddt",
        "n_interface_residues_binder",
        "n_interface_residues_target",
        "buried_sasa_proxy_a2",
        "atom_contacts_5a",
        "shape_complementarity_proxy",
        "atom_clash_fraction_2a",
        "cdr_contact_fraction",
        "cdr1_contact_fraction",
        "cdr2_contact_fraction",
        "cdr3_contact_fraction",
        "cdr_mapping_ok",
    ]


# ---------------------------------------------------------------------------
# Test 2: Missing prediction — nothing on disk
# ---------------------------------------------------------------------------


def test_missing_prediction_returns_blank_features(tmp_path: Path) -> None:
    """When predictions_root has no directory for pair_id, return sentinel row."""
    row = _minimal_row(pair_id="X__Y", binder_seq="QVQ", antigen_seq="EVAL")
    out = assemble_row(row, predictions_root=tmp_path)
    assert out["prediction_present"] == "0"
    # Passthrough columns are filled
    assert out["pair_id"] == "X__Y"
    assert out["label"] == "1"
    assert out["antigen_cluster"] == "3"
    assert out["fold"] == "0"
    # Feature columns are empty strings (not fabricated)
    assert out["iptm"] == ""
    assert out["interface_pae"] == ""
    assert out["n_interface_residues_binder"] == ""
    assert out["cdr_contact_fraction"] == ""
    assert out["cdr_mapping_ok"] == ""
    # All feature columns (non-passthrough, non-prediction_present) must be ""
    passthrough = {"pair_id", "label", "antigen_cluster", "fold", "prediction_present"}
    for col in FEATURE_COLUMNS:
        if col not in passthrough:
            assert out[col] == "", f"Expected '' for {col}, got {out[col]!r}"


# ---------------------------------------------------------------------------
# Test 3: Fault tolerance — build_rows swallows individual failures
# ---------------------------------------------------------------------------


def test_build_rows_fault_tolerant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_rows must not raise when assemble_row raises; the failing pair_id is logged."""
    import extract_mc_features as emf  # type: ignore[import-not-found]

    ok_row = _minimal_row(pair_id="OK__PAIR")
    boom_row = _minimal_row(pair_id="BOOM__PAIR")

    # Make assemble_row raise for "BOOM__PAIR" only
    _real_assemble = emf.assemble_row

    def _patched(row: dict[str, str], predictions_root: Path) -> dict[str, str]:
        if row["pair_id"] == "BOOM__PAIR":
            raise ValueError("synthetic failure")
        return _real_assemble(row, predictions_root)

    monkeypatch.setattr(emf, "assemble_row", _patched)

    failed_log = tmp_path / "failed.txt"
    results = build_rows([ok_row, boom_row], predictions_root=tmp_path, failed_log=failed_log)

    # Run must not have raised
    # OK row must be in results (prediction_present=="0" because nothing on disk, but still returned)
    assert len(results) == 1
    assert results[0]["pair_id"] == "OK__PAIR"

    # BOOM pair_id must be in the failed log
    assert failed_log.exists()
    logged = failed_log.read_text()
    assert "BOOM__PAIR" in logged


# ---------------------------------------------------------------------------
# Test 4: End-to-end integration on the real 3OGO fixture
# ---------------------------------------------------------------------------


def test_end_to_end_real_fixture() -> None:
    """Full assemble_row run on the real Protenix 3OGO fixture.

    Assertions:
    - prediction_present == "1"
    - iptm ≈ 0.9382603 (from the summary JSON)
    - n_interface_residues_binder > 0 (geometry computed correctly with chain A as binder)
    - cdr_mapping_ok == 1.0 (ANARCI resolved the VHH CDRs)
    - interface_pae may be "nan" (trimmed fixture has no cross-chain tokens) — don't assert value
    """
    row = _minimal_row(
        pair_id="3OGO__3OGO",
        binder_seq=SEQ_A,
        antigen_seq=SEQ_B,
        label="1",
        antigen_cluster="0",
        fold="0",
    )
    out = assemble_row(row, predictions_root=FIXTURE_ROOT)

    assert out["prediction_present"] == "1", f"Expected prediction_present=1, got {out}"
    assert float(out["iptm"]) == pytest.approx(0.9382603, abs=1e-4), (
        f"iptm mismatch: {out['iptm']}"
    )
    assert float(out["n_interface_residues_binder"]) > 0, (
        f"Expected >0 binder interface residues, got {out['n_interface_residues_binder']}"
    )
    assert float(out["cdr_mapping_ok"]) == 1.0, (
        f"Expected cdr_mapping_ok=1.0, got {out['cdr_mapping_ok']}"
    )
    # interface_pae is "nan" on the trimmed fixture (all tokens share asym_id=0) — that's OK
    # Just verify it's present and parseable
    assert "interface_pae" in out
    float(out["interface_pae"])  # must be parseable (e.g. "nan")
