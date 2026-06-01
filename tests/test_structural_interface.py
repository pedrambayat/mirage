from __future__ import annotations

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
from mirage.scorers.structural_interface import StructuralInterfaceScorer


def _make_residue(resseq: int, ca_xyz: tuple[float, float, float], resname: str = "GLY") -> Residue:
    res = Residue((" ", resseq, " "), resname, "")
    for name, dx in (("N", -1.0), ("CA", 0.0), ("C", 1.0), ("O", 1.5)):
        coord = np.array([ca_xyz[0] + dx, ca_xyz[1], ca_xyz[2]], dtype="f")
        res.add(
            Atom(
                name=name,
                coord=coord,
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=f" {name:<3}",
                serial_number=0,
                element="N" if name == "N" else "C",
            )
        )
    return res


def _make_chain(
    chain_id: str,
    ca_positions: list[tuple[float, float, float]],
    resnames: list[str] | None = None,
) -> Chain:
    chain = Chain(chain_id)
    if resnames is None:
        resnames = ["GLY"] * len(ca_positions)
    for idx, pos in enumerate(ca_positions, start=1):
        chain.add(_make_residue(idx, pos, resnames[idx - 1]))
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


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-struct-1",
        label="POS",
        binder_chains=("AAAA",),
        binder_format="vhh",
        target_chains=("AAAA",),
        target_name="synthetic",
        source="unit-test",
    )


def test_structural_interface_registered() -> None:
    assert "structural_interface" in list_scorers()
    assert isinstance(get_scorer("structural_interface"), StructuralInterfaceScorer)


def test_structural_interface_counts_contacts_and_interface_residues(tmp_path: Path) -> None:
    example = _example()
    pred_dir = tmp_path / "predictions" / example.id
    pred_dir.mkdir(parents=True)
    _write_structure(
        pred_dir / "rank1.pdb",
        [
            _make_chain("A", [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], ["LYS", "PHE"]),
            _make_chain("B", [(0.0, 3.0, 0.0), (30.0, 0.0, 0.0)], ["ASP", "TYR"]),
        ],
    )

    scorer = StructuralInterfaceScorer(predictions_root=tmp_path / "predictions")
    score = scorer.score(example)
    extras = score.extras

    assert score.value == pytest.approx(0.5)
    assert float(extras["n_residues_binder"]) == 2.0
    assert float(extras["n_residues_target"]) == 2.0
    assert float(extras["residue_pair_contacts_4a"]) == 1.0
    assert float(extras["residue_pair_contacts_6a"]) == 1.0
    assert float(extras["residue_pair_contacts_8a"]) == 1.0
    assert float(extras["contacts_per_binder_residue_6a"]) == pytest.approx(0.5)
    assert float(extras["residue_pair_contact_fraction_6a"]) == pytest.approx(0.25)
    assert float(extras["n_interface_residues_binder"]) == 1.0
    assert float(extras["n_interface_residues_target"]) == 1.0
    assert float(extras["binder_interface_fraction"]) == pytest.approx(0.5)
    assert float(extras["target_interface_fraction"]) == pytest.approx(0.5)
    assert float(extras["atom_contacts_5a"]) > 0.0
    assert float(extras["atom_close_contacts_3a"]) > 0.0
    assert float(extras["atom_clashes_2a"]) == 0.0
    assert float(extras["buried_area_proxy_5a"]) > 0.0
    assert float(extras["buried_sasa_proxy_a2"]) > 0.0
    assert float(extras["buried_sasa_proxy_binder_a2"]) > 0.0
    assert float(extras["buried_sasa_proxy_target_a2"]) > 0.0
    assert float(extras["buried_sasa_proxy_per_interface_residue"]) > 0.0
    assert 0.0 <= float(extras["buried_sasa_balance"]) <= 1.0
    assert float(extras["mean_binder_atom_exposure_loss"]) > 0.0
    assert float(extras["mean_target_atom_exposure_loss"]) > 0.0
    assert float(extras["atom_packing_pairs_0_1a_gap"]) > 0.0
    assert 0.0 <= float(extras["atom_packing_fraction_0_1a_gap"]) <= 1.0
    assert float(extras["atom_packing_shell_pairs_2a_gap"]) >= float(
        extras["atom_packing_pairs_0_1a_gap"]
    )
    assert 0.0 <= float(extras["binder_atom_packing_coverage_0_1a_gap"]) <= 1.0
    assert 0.0 <= float(extras["target_atom_packing_coverage_0_1a_gap"]) <= 1.0
    assert 0.0 <= float(extras["atom_packing_complementarity_score"]) <= 1.0
    assert float(extras["mean_abs_nearest_surface_gap"]) >= 0.0
    assert float(extras["shape_complementarity_proxy"]) == pytest.approx(0.5)
    assert float(extras["opposite_charge_contact_pairs_6a"]) == 1.0
    assert float(extras["opposite_charge_contact_fraction_6a"]) == pytest.approx(1.0)
    assert float(extras["same_charge_contact_pairs_6a"]) == 0.0
    assert float(extras["mean_interface_residue_distance_8a"]) == pytest.approx(3.0)
    assert float(extras["close_residue_contact_fraction_4a_within_8a"]) == pytest.approx(1.0)
    assert float(extras["min_interchain_heavy_atom_distance"]) < 4.0


def test_structural_interface_missing_prediction(tmp_path: Path) -> None:
    score = StructuralInterfaceScorer(predictions_root=tmp_path / "predictions").score(_example())
    assert np.isnan(score.value)
    assert score.extras["missing"] == "prediction"
