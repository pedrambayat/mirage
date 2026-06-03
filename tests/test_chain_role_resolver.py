"""Tests for sequence-based chain-role resolver and CIF support.

TDD: these tests are written BEFORE the implementation. They must fail first,
then pass once the implementation is complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from mirage.scorers._structure import (
    load_structure,
    read_chain_roles_json,
    resolve_chain_roles_by_sequence,
)
from mirage.scorers.base import BenchmarkExample
from mirage.scorers.structural_interface import StructuralInterfaceScorer

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURE_CIF = Path("tests/fixtures/protenix/3OGO__3OGO/3OGO__3OGO_sample_0.cif")

# Sequences extracted from the CIF fixture (chain A = VHH binder, chain B = GFP)
SEQ_A = "MQVQLVESGGALVQPGGSLRLSCAASGFPVNRYSMRWYRQAPGKEREWVAGMSSAGDRSSYEDSVKGRFTISRDDARNTVYLQMNSLKPEDTAVYYCNVNVGFEYWGQGTQVTVSSKHHHHHH"  # noqa: E501
SEQ_B = "MAHHHHHHSSGVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"  # noqa: E501


def _make_ca_residue(resseq: int, ca_xyz: tuple[float, float, float], resname: str) -> Residue:
    """A minimal CA-only residue."""
    res = Residue((" ", resseq, " "), resname, "")
    coord = np.array(ca_xyz, dtype="f")
    res.add(
        Atom(
            name="CA",
            coord=coord,
            bfactor=0.0,
            occupancy=1.0,
            altloc=" ",
            fullname=" CA ",
            serial_number=resseq,
            element="C",
        )
    )
    return res


def _make_ca_chain(
    chain_id: str,
    resnames: list[str],
    start_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    step: float = 4.0,
) -> Chain:
    chain = Chain(chain_id)
    for idx, rn in enumerate(resnames, start=1):
        xyz = (start_xyz[0] + idx * step, start_xyz[1], start_xyz[2])
        chain.add(_make_ca_residue(idx, xyz, rn))
    return chain


def _write_pdb(path: Path, chains: list[Chain]) -> None:
    struct = Structure("synthetic")
    model = Model(0)
    struct.add(model)
    for ch in chains:
        model.add(ch)
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(path))


# Three-letter codes for our short synthetic sequences
# Binder seq "QVW" → GLN VAL TRP
# Target seq "GAP" → GLY ALA PRO
BINDER_RESNAMES = ["GLN", "VAL", "TRP"]
TARGET_RESNAMES = ["GLY", "ALA", "PRO"]
BINDER_SEQ_1L = "QVW"
TARGET_SEQ_1L = "GAP"


# ---------------------------------------------------------------------------
# Test A: load_structure dispatches to MMCIFParser for .cif
# ---------------------------------------------------------------------------


def test_load_structure_cif_returns_correct_chains() -> None:
    """load_structure on a .cif file must return a structure with chains A and B."""
    struct = load_structure(FIXTURE_CIF)
    chain_ids = [ch.id for ch in struct[0]]
    assert "A" in chain_ids
    assert "B" in chain_ids
    # Chain A: 123 residues (VHH), chain B: 249 residues (GFP)
    a_res = [r for r in struct[0]["A"] if r.id[0] == " "]
    b_res = [r for r in struct[0]["B"] if r.id[0] == " "]
    assert len(a_res) == 123
    assert len(b_res) == 249


# ---------------------------------------------------------------------------
# Test B: resolve_chain_roles_by_sequence — reordered synthetic PDB
# ---------------------------------------------------------------------------


def test_resolve_chain_roles_reordered_synthetic(tmp_path: Path) -> None:
    """Core correctness test: chain A holds the ANTIGEN, chain B holds the BINDER.

    The resolver must return binder_ids=["B"], target_ids=["A"] — i.e. it tracks
    sequence, not position.
    """
    pdb_path = tmp_path / "reordered.pdb"
    # A = antigen (TARGET_RESNAMES = GAP), B = binder (BINDER_RESNAMES = QVW)
    _write_pdb(
        pdb_path,
        [
            _make_ca_chain("A", TARGET_RESNAMES, start_xyz=(0.0, 0.0, 0.0)),
            _make_ca_chain("B", BINDER_RESNAMES, start_xyz=(100.0, 0.0, 0.0)),
        ],
    )
    struct = load_structure(pdb_path)
    binder_ids, target_ids = resolve_chain_roles_by_sequence(
        struct,
        binder_seqs=(BINDER_SEQ_1L,),
        target_seqs=(TARGET_SEQ_1L,),
    )
    assert binder_ids == ["B"], f"Expected binder=B, got {binder_ids}"
    assert target_ids == ["A"], f"Expected target=A, got {target_ids}"


# ---------------------------------------------------------------------------
# Test C: resolve_chain_roles_by_sequence — real CIF fixture
# ---------------------------------------------------------------------------


def test_resolve_chain_roles_cif_forward() -> None:
    """With seqA as binder input, resolver maps binder→A, target→B."""
    struct = load_structure(FIXTURE_CIF)
    binder_ids, target_ids = resolve_chain_roles_by_sequence(
        struct,
        binder_seqs=(SEQ_A,),
        target_seqs=(SEQ_B,),
    )
    assert binder_ids == ["A"]
    assert target_ids == ["B"]


def test_resolve_chain_roles_cif_swapped() -> None:
    """With seqB as binder input, resolver maps binder→B, target→A (tracks seq, not position)."""
    struct = load_structure(FIXTURE_CIF)
    binder_ids, target_ids = resolve_chain_roles_by_sequence(
        struct,
        binder_seqs=(SEQ_B,),
        target_seqs=(SEQ_A,),
    )
    assert binder_ids == ["B"]
    assert target_ids == ["A"]


# ---------------------------------------------------------------------------
# Test D: read_chain_roles_json
# ---------------------------------------------------------------------------


def test_read_chain_roles_json_present(tmp_path: Path) -> None:
    p = tmp_path / "chain_roles.json"
    p.write_text(json.dumps({"binder": ["A"], "target": ["B"]}))
    result = read_chain_roles_json(p)
    assert result == (["A"], ["B"])


def test_read_chain_roles_json_missing(tmp_path: Path) -> None:
    result = read_chain_roles_json(tmp_path / "chain_roles.json")
    assert result is None


# ---------------------------------------------------------------------------
# Test E: StructuralInterfaceScorer sequence mode on the real CIF fixture
# ---------------------------------------------------------------------------


def test_structural_interface_sequence_mode_cif() -> None:
    """Sequence-mode scorer on the Protenix CIF fixture.

    The scorer must correctly identify chain A as the binder (123 residues)
    and chain B as the target. It looks for a *sample_0*.cif under the
    predictions_root/example.id directory.
    """
    # Build an example whose binder_chains / target_chains carry the actual seqs
    example = BenchmarkExample(
        id="3OGO__3OGO",
        label="POS",
        binder_chains=(SEQ_A,),
        binder_format="vhh",
        target_chains=(SEQ_B,),
        target_name="GFP",
        source="protenix-fixture",
    )

    scorer = StructuralInterfaceScorer(
        predictions_root=Path("tests/fixtures/protenix"),
        chain_resolution="sequence",
    )
    score = scorer.score(example)

    assert not np.isnan(score.value), f"Score is NaN; extras={score.extras}"
    # Chain A (binder) has 123 residues
    assert float(score.extras["n_residues_binder"]) == pytest.approx(123, abs=2), (
        f"Expected ~123 binder residues, got {score.extras['n_residues_binder']}"
    )
    # Chain B (target) has 249 residues
    assert float(score.extras["n_residues_target"]) == pytest.approx(249, abs=2), (
        f"Expected ~249 target residues, got {score.extras['n_residues_target']}"
    )
