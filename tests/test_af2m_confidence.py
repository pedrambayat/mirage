"""Tests for AF2MConfidenceScorer.

Builds synthetic ColabFold-style outputs (`rank1.pdb` + `scores.json`) and
checks the scorer's aggregates against arithmetic expectations. No
crystal PDB is involved — the confidence scorer is crystal-independent.
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

from mirage.scorers import BenchmarkExample, get_scorer, list_scorers
from mirage.scorers.af2m_confidence import AF2MConfidenceScorer

_BACKBONE = (
    ("N", "N", -1.0, 0.0, 0.0),
    ("CA", "C", 0.0, 0.0, 0.0),
    ("C", "C", 1.0, 0.0, 0.0),
    ("O", "O", 1.5, 0.5, 0.0),
)


def _make_residue(resseq: int, ca_xyz: tuple[float, float, float]) -> Residue:
    res = Residue((" ", resseq, " "), "GLY", "")
    for name, element, dx, dy, dz in _BACKBONE:
        coord = np.array([ca_xyz[0] + dx, ca_xyz[1] + dy, ca_xyz[2] + dz], dtype="f")
        atom = Atom(
            name=name,
            coord=coord,
            bfactor=0.0,
            occupancy=1.0,
            altloc=" ",
            fullname=f" {name:<3}",
            serial_number=0,
            element=element,
        )
        res.add(atom)
    return res


def _make_chain(
    chain_id: str,
    ca_positions: list[tuple[float, float, float]],
    start_resseq: int = 1,
) -> Chain:
    chain = Chain(chain_id)
    for i, pos in enumerate(ca_positions):
        chain.add(_make_residue(start_resseq + i, pos))
    return chain


def _write_structure(path: Path, chains: list[Chain]) -> None:
    structure = Structure("synthetic")
    model = Model(0)
    structure.add(model)
    for chain in chains:
        model.add(chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def _linear(
    n: int, axis: int, start: float = 0.0, step: float = 3.8
) -> list[tuple[float, float, float]]:
    out = []
    for i in range(n):
        xyz = [0.0, 0.0, 0.0]
        xyz[axis] = start + i * step
        out.append((xyz[0], xyz[1], xyz[2]))
    return out


def _example(
    *,
    binder_len: int = 8,
    target_len: int = 10,
) -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-conf-1",
        label="POS",
        binder_chains=("A" * binder_len,),
        binder_format="vhh",
        target_chains=("A" * target_len,),
        target_name="synthetic",
        source="unit-test",
        metadata={"Hchain": "H", "Lchain": "NA", "antigen_chain": "A"},
    )


_RANK1_PDB_NAME = "ex_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
_RANK1_SCORES_NAME = "ex_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"


def _write_prediction(
    pred_dir: Path,
    *,
    binder_positions: list[tuple[float, float, float]],
    target_positions: list[tuple[float, float, float]],
    plddt: list[float] | np.ndarray,
    pae: np.ndarray,
    iptm: float,
    ptm: float,
    max_pae: float | None = None,
    rank1_pdb_name: str = _RANK1_PDB_NAME,
    rank1_scores_name: str = _RANK1_SCORES_NAME,
) -> None:
    """Lay out a synthetic ColabFold-style example directory.

    Writes ``rank1.pdb`` (symlink to a named rank-1 unrelaxed PDB), the
    mirage wrapper's flat ``scores.json`` (summary only), and ColabFold's
    raw rank-1 scores JSON (per-residue ``plddt`` and per-residue-pair
    ``pae``). The scorer derives the raw scores filename from the
    wrapper summary, so the filename pattern matters.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    # Predicted chains are A (binder), B (target), per _predicted_chain_ids.
    _write_structure(
        pred_dir / rank1_pdb_name,
        [_make_chain("A", binder_positions), _make_chain("B", target_positions)],
    )
    # Mirror the production layout: rank1.pdb is a symlink to the named PDB.
    rank1_link = pred_dir / "rank1.pdb"
    if rank1_link.exists() or rank1_link.is_symlink():
        rank1_link.unlink()
    rank1_link.symlink_to(rank1_pdb_name)

    resolved_max_pae = float(pae.max()) if max_pae is None else max_pae
    summary = {
        "mirage_predict_af2m_version": "test",
        "rank1": {
            "iptm": iptm,
            "ptm": ptm,
            "max_pae": resolved_max_pae,
            "mean_plddt": float(np.asarray(plddt).mean()) if len(plddt) else 0.0,
            "pdb": rank1_pdb_name,
        },
    }
    (pred_dir / "scores.json").write_text(json.dumps(summary))
    raw = {
        "iptm": iptm,
        "ptm": ptm,
        "plddt": list(plddt),
        "pae": pae.tolist(),
        "max_pae": resolved_max_pae,
    }
    (pred_dir / rank1_scores_name).write_text(json.dumps(raw))


def test_af2m_confidence_registered() -> None:
    assert "af2m_confidence" in list_scorers()
    scorer = get_scorer("af2m_confidence")
    assert isinstance(scorer, AF2MConfidenceScorer)


def test_af2m_confidence_headline_and_aggregates(tmp_path: Path) -> None:
    example = _example(binder_len=8, target_len=10)
    n = 18  # 8 binder + 10 target
    # All-distinct pLDDT values so means are unambiguous.
    plddt = np.arange(50.0, 50.0 + n, dtype=float)  # [50, 51, ..., 67]
    # PAE: 5 on intra-binder, 7 on intra-target, 12 on inter-chain. This
    # gives controllable inter-chain mean/max.
    pae = np.full((n, n), 5.0)
    pae[8:, 8:] = 7.0  # target-target
    pae[:8, 8:] = 12.0  # binder→target
    pae[8:, :8] = 12.0  # target→binder

    # Binder along x-axis with 3.8 Å spacing; target placed at y=5 so several
    # binder residues are within 8 Å of target heavy atoms.
    binder_pos = _linear(8, axis=0)
    target_pos = _linear(10, axis=0, start=0.0)
    target_pos = [(x, 5.0, 0.0) for (x, _, _) in target_pos]

    _write_prediction(
        tmp_path / "predictions" / example.id,
        binder_positions=binder_pos,
        target_positions=target_pos,
        plddt=plddt,
        pae=pae,
        iptm=0.55,
        ptm=0.70,
    )

    scorer = AF2MConfidenceScorer(predictions_root=tmp_path / "predictions")
    score = scorer.score(example)

    extras = score.extras
    assert score.value == pytest.approx(0.55)
    assert float(extras["iptm"]) == pytest.approx(0.55)
    assert float(extras["ptm"]) == pytest.approx(0.70)
    assert float(extras["ranking_confidence"]) == pytest.approx(0.8 * 0.55 + 0.2 * 0.70)
    assert float(extras["iptm_over_ptm"]) == pytest.approx(0.55 / 0.70)

    # pLDDT aggregates: full mean is 58.5; binder mean is (50..57).mean()=53.5;
    # target mean is (58..67).mean()=62.5.
    assert float(extras["plddt_full_mean"]) == pytest.approx(plddt.mean())
    assert float(extras["plddt_binder_mean"]) == pytest.approx(plddt[:8].mean())
    assert float(extras["plddt_target_mean"]) == pytest.approx(plddt[8:].mean())

    # Inter-chain PAE aggregates: both blocks are filled with 12.
    assert float(extras["pae_interchain_mean"]) == pytest.approx(12.0)
    assert float(extras["pae_interchain_max"]) == pytest.approx(12.0)

    # Counts.
    assert float(extras["n_residues_total"]) == 18
    assert float(extras["n_residues_binder"]) == 8
    assert float(extras["n_residues_target"]) == 10

    # Predicted interface — binder residues within 8 Å of any target heavy
    # atom in the predicted PDB (target sits at y=5; binder at y=0). Heavy
    # atoms include backbone O at offset (1.5, 0.5, 0.0), so the nearest
    # binder→target heavy-atom distance is just under 5 Å for several
    # binder positions. We require at least one interface residue and
    # that the interface-restricted aggregates are computed.
    n_iface = float(extras["n_interface_residues_predicted"])
    assert n_iface >= 1
    plddt_iface = float(extras["plddt_interface_mean"])
    pae_iface = float(extras["pae_interface_mean"])
    # Interface residues are a subset of binder residues, so plddt_interface
    # must lie within the binder-pLDDT range [50, 57].
    assert 50.0 <= plddt_iface <= 57.0
    # PAE to all target residues is 12 from any binder row.
    assert pae_iface == pytest.approx(12.0)


def test_af2m_confidence_missing_prediction(tmp_path: Path) -> None:
    example = _example()
    scorer = AF2MConfidenceScorer(predictions_root=tmp_path / "predictions")
    score = scorer.score(example)
    assert np.isnan(score.value)
    assert score.extras.get("missing") == "prediction"


def test_af2m_confidence_missing_scores_json(tmp_path: Path) -> None:
    example = _example()
    pred_dir = tmp_path / "predictions" / example.id
    pred_dir.mkdir(parents=True)
    _write_structure(
        pred_dir / "rank1.pdb",
        [_make_chain("A", _linear(8, axis=0)), _make_chain("B", _linear(10, axis=0))],
    )
    scorer = AF2MConfidenceScorer(predictions_root=tmp_path / "predictions")
    score = scorer.score(example)
    assert np.isnan(score.value)
    assert score.extras.get("missing") == "scores_json"


def test_af2m_confidence_plddt_length_mismatch(tmp_path: Path) -> None:
    example = _example(binder_len=8, target_len=10)  # expects pLDDT length 18
    _write_prediction(
        tmp_path / "predictions" / example.id,
        binder_positions=_linear(8, axis=0),
        target_positions=_linear(10, axis=0, start=0.0),
        plddt=list(np.full(17, 70.0)),  # WRONG length on purpose
        pae=np.full((17, 17), 5.0),
        iptm=0.5,
        ptm=0.5,
    )
    scorer = AF2MConfidenceScorer(predictions_root=tmp_path / "predictions")
    score = scorer.score(example)
    assert np.isnan(score.value)
    assert "plddt_length_mismatch" in str(score.extras.get("error", ""))


def test_af2m_confidence_no_interface_when_chains_far_apart(tmp_path: Path) -> None:
    example = _example(binder_len=8, target_len=10)
    # Place target 100 Å away — no binder residue is within 8 Å.
    binder_pos = _linear(8, axis=0)
    target_pos = [(x, 100.0, 0.0) for (x, _, _) in _linear(10, axis=0)]
    n = 18
    _write_prediction(
        tmp_path / "predictions" / example.id,
        binder_positions=binder_pos,
        target_positions=target_pos,
        plddt=list(np.full(n, 70.0)),
        pae=np.full((n, n), 10.0),
        iptm=0.2,
        ptm=0.4,
    )
    scorer = AF2MConfidenceScorer(predictions_root=tmp_path / "predictions")
    score = scorer.score(example)
    extras = score.extras
    assert float(extras["n_interface_residues_predicted"]) == 0
    assert np.isnan(float(extras["plddt_interface_mean"]))
    assert np.isnan(float(extras["pae_interface_mean"]))
    # Headline and non-interface aggregates still finite.
    assert float(extras["iptm"]) == pytest.approx(0.2)
    assert float(extras["pae_interchain_mean"]) == pytest.approx(10.0)
