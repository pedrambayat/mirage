"""Tests for RMSDToCrystalScorer.

Builds tiny synthetic predicted/crystal PDB pairs with known geometry so
the scorer's outputs (full-binder CA RMSD, interface CA RMSD, etc.) can
be checked against arithmetic expectations.
"""

from __future__ import annotations

import math
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
from mirage.scorers.rmsd_to_crystal import RMSDToCrystalScorer

_BACKBONE = (
    # (name, element, dx, dy, dz) offsets from the CA position.
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


def _write_structure(path: Path, chains: list[Chain], name: str = "synthetic") -> None:
    structure = Structure(name)
    model = Model(0)
    structure.add(model)
    for chain in chains:
        model.add(chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def _linear_positions(
    n: int, axis: int, start: float = 0.0, step: float = 3.8
) -> list[tuple[float, float, float]]:
    out = []
    for i in range(n):
        xyz = [0.0, 0.0, 0.0]
        xyz[axis] = start + i * step
        out.append((xyz[0], xyz[1], xyz[2]))
    return out


def _shifted(
    positions: list[tuple[float, float, float]], dx: float, dy: float, dz: float
) -> list[tuple[float, float, float]]:
    return [(x + dx, y + dy, z + dz) for (x, y, z) in positions]


def _example(
    tmp_path: Path,
    *,
    binder_format: str = "vhh",
    binder_len: int = 8,
    target_len: int = 10,
    crystal_h_chain: str = "H",
    crystal_l_chain: str | None = None,
    crystal_antigen: str = "A",
) -> BenchmarkExample:
    crystal_path = tmp_path / "crystal.pdb"
    metadata = {
        "crystal_pdb_path": str(crystal_path),
        "Hchain": crystal_h_chain,
        "Lchain": crystal_l_chain or "NA",
        "antigen_chain": crystal_antigen,
    }
    return BenchmarkExample(
        id="ex-1",
        label="POS",
        binder_chains=("A" * binder_len,)
        if binder_format != "fab"
        else ("A" * binder_len, "A" * binder_len),
        binder_format=binder_format,
        target_chains=("A" * target_len,),
        target_name="synthetic",
        source="unit-test",
        complex_pdb_path=crystal_path,
        metadata=metadata,
    )


def test_scorer_is_registered() -> None:
    assert "rmsd_to_crystal" in list_scorers()
    assert isinstance(get_scorer("rmsd_to_crystal", compute_dockq=False), RMSDToCrystalScorer)


def test_missing_prediction_returns_nan(tmp_path: Path) -> None:
    crystal_chains = [
        _make_chain("H", _linear_positions(8, axis=0)),
        _make_chain("A", _linear_positions(10, axis=1)),
    ]
    example = _example(tmp_path)
    _write_structure(example.complex_pdb_path, crystal_chains)

    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "nope", compute_dockq=False)
    score = scorer.score(example)
    assert math.isnan(score.value)
    assert score.extras.get("missing") == "prediction"


def test_missing_crystal_returns_nan(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "ex-1"
    pred_dir.mkdir(parents=True)
    pred_chains = [
        _make_chain("A", _linear_positions(8, axis=0)),
        _make_chain("B", _linear_positions(10, axis=1)),
    ]
    _write_structure(pred_dir / "rank1.pdb", pred_chains)

    example = _example(tmp_path)  # crystal_path file not created
    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "predictions", compute_dockq=False)
    score = scorer.score(example)
    assert math.isnan(score.value)
    assert score.extras.get("missing") == "crystal"


def test_identical_structures_yield_zero_rmsd(tmp_path: Path) -> None:
    """Predicted == crystal (after chain renaming) → all RMSDs ≈ 0."""
    binder_ca = _linear_positions(8, axis=0)
    target_ca = _linear_positions(10, axis=1, start=5.0)

    crystal_chains = [_make_chain("H", binder_ca), _make_chain("A", target_ca)]
    pred_chains = [_make_chain("A", binder_ca), _make_chain("B", target_ca)]

    example = _example(tmp_path)
    _write_structure(example.complex_pdb_path, crystal_chains)
    pred_dir = tmp_path / "predictions" / "ex-1"
    pred_dir.mkdir(parents=True)
    _write_structure(pred_dir / "rank1.pdb", pred_chains)

    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "predictions", compute_dockq=False)
    score = scorer.score(example)
    assert not math.isnan(score.value)
    assert score.value == pytest.approx(0.0, abs=1e-4)
    assert float(score.extras["binder_backbone_rmsd_target_aligned"]) == pytest.approx(
        0.0, abs=1e-4
    )
    assert float(score.extras["target_ca_rmsd"]) == pytest.approx(0.0, abs=1e-4)


def test_pure_binder_translation_after_perfect_target_alignment(tmp_path: Path) -> None:
    """If the target is identical but the binder is shifted by 2 Å in z,
    the target-aligned binder CA RMSD should be exactly 2 Å.
    """
    binder_ca = _linear_positions(8, axis=0)
    target_ca = _linear_positions(10, axis=1, start=5.0)
    shifted_binder = _shifted(binder_ca, 0.0, 0.0, 2.0)

    crystal_chains = [_make_chain("H", binder_ca), _make_chain("A", target_ca)]
    pred_chains = [_make_chain("A", shifted_binder), _make_chain("B", target_ca)]

    example = _example(tmp_path)
    _write_structure(example.complex_pdb_path, crystal_chains)
    pred_dir = tmp_path / "predictions" / "ex-1"
    pred_dir.mkdir(parents=True)
    _write_structure(pred_dir / "rank1.pdb", pred_chains)

    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "predictions", compute_dockq=False)
    score = scorer.score(example)
    assert score.value == pytest.approx(2.0, abs=1e-3)
    assert float(score.extras["target_ca_rmsd"]) == pytest.approx(0.0, abs=1e-4)


def test_interface_rmsd_restricted_to_contact_residues(tmp_path: Path) -> None:
    """Only the first 3 binder residues are within the interface cutoff of the target.
    Translating the entire binder along +y should still yield interface_ca_rmsd == 2 Å
    (same as full-binder, since the entire translation is uniform).
    """
    # Binder along x, residues 0-7
    binder_ca = _linear_positions(8, axis=0)
    # Target chain placed near the first 3 binder residues only
    target_ca = [(0.0, 5.0, 0.0), (3.8, 5.0, 0.0), (7.6, 5.0, 0.0)]
    shifted_binder = _shifted(binder_ca, 0.0, 0.0, 2.0)

    crystal_chains = [_make_chain("H", binder_ca), _make_chain("A", target_ca)]
    pred_chains = [_make_chain("A", shifted_binder), _make_chain("B", target_ca)]

    example = _example(tmp_path, binder_len=8, target_len=3)
    _write_structure(example.complex_pdb_path, crystal_chains)
    pred_dir = tmp_path / "predictions" / "ex-1"
    pred_dir.mkdir(parents=True)
    _write_structure(pred_dir / "rank1.pdb", pred_chains)

    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "predictions", compute_dockq=False)
    score = scorer.score(example)
    n_iface = float(score.extras["n_interface_residues"])
    # Interface residues are those binder CAs within 8 Å of any target heavy atom.
    # Backbone N/C/O extend ±1.5 Å around CA, so the cutoff covers residues whose
    # nearest backbone atom is within 8 Å of the target plane at y=5.
    assert 3 <= n_iface <= 6
    assert float(score.extras["interface_ca_rmsd_target_aligned"]) == pytest.approx(2.0, abs=1e-3)


def test_dockq_chain_map_is_native_to_model_and_filters_to_binder_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two regression contracts in one test:

    1. ``run_on_all_native_interfaces``'s ``chain_map`` arg is keyed by
       *native* chain IDs with *model* chain IDs as values (per
       ``DockQ.py:613``). Reversing the direction silently works only
       when native and model chain letters happen to coincide (the
       pre-fix bug, where 16/20 pilot rows raised ``KeyError`` on chains
       like E/F/H that aren't in the model's A/B/C labelling).
    2. ``dockq_best`` is ``max`` over binder↔target native interfaces
       only; target↔target contacts (e.g. the C-D antigen-antigen
       interface in 9p0m's scFv-vs-G-protein alpha/beta case) must not enter
       the headline number, even when their DockQ is higher than every
       binder↔target interface.
    """
    binder_ca = _linear_positions(8, axis=0)
    target_ca = _linear_positions(10, axis=1, start=5.0)
    crystal_chains = [
        _make_chain("F", binder_ca),  # native binder
        _make_chain("C", target_ca),  # native target 1
        _make_chain("D", target_ca),  # native target 2
    ]
    pred_chains = [
        _make_chain("A", binder_ca),  # model binder
        _make_chain("B", target_ca),  # model target 1
        _make_chain("C", target_ca),  # model target 2
    ]

    # Single-chain binder, two-chain target with non-overlapping native
    # chain letters (F vs ABC) — the exact pattern that triggered the
    # KeyError pre-fix. Construct directly since the _example helper only
    # supports one target chain.
    crystal_path = tmp_path / "crystal.pdb"
    example = BenchmarkExample(
        id="ex-1",
        label="POS",
        binder_chains=("A" * 8,),
        binder_format="vhh",
        target_chains=("A" * 10, "A" * 10),
        target_name="synthetic",
        source="unit-test",
        complex_pdb_path=crystal_path,
        metadata={
            "crystal_pdb_path": str(crystal_path),
            "Hchain": "F",
            "Lchain": "NA",
            "antigen_chain": "C|D",
            "target_chain_ids": ("C", "D"),
        },
    )
    _write_structure(crystal_path, crystal_chains)
    pred_dir = tmp_path / "predictions" / "ex-1"
    pred_dir.mkdir(parents=True)
    _write_structure(pred_dir / "rank1.pdb", pred_chains)

    captured: dict[str, object] = {}

    def fake_load_pdb(path: str, chains: list[str]) -> object:
        del path, chains
        return object()

    def fake_run(
        model: object, native: object, chain_map: dict[str, str]
    ) -> tuple[dict[str, dict[str, float]], float]:
        captured["chain_map"] = dict(chain_map)
        # Fabricate three native interface results: FC and FD (binder-target,
        # which must enter dockq_best) and CD (target-target, which must
        # NOT enter dockq_best). DockQ's own iteration order is
        # itertools.combinations(chain_map.keys(), 2), so the keys here are
        # "".join(pair) in that order.
        results = {
            "FC": {"DockQ": 0.30, "iRMSD": 5.0, "LRMSD": 8.0, "fnat": 0.4, "fnonnat": 0.3},
            "FD": {"DockQ": 0.10, "iRMSD": 9.0, "LRMSD": 15.0, "fnat": 0.2, "fnonnat": 0.5},
            "CD": {"DockQ": 0.90, "iRMSD": 1.0, "LRMSD": 2.0, "fnat": 0.9, "fnonnat": 0.1},
        }
        full_total = sum(r["DockQ"] for r in results.values())
        return results, full_total

    # Patch DockQ's two imported callables. _dockq_metrics imports them
    # lazily from DockQ.DockQ at call time, so patching the module
    # attributes here intercepts them inside the scorer.
    import DockQ.DockQ as dockq_mod  # type: ignore[import-untyped]  # noqa: N813

    monkeypatch.setattr(dockq_mod, "load_PDB", fake_load_pdb)
    monkeypatch.setattr(dockq_mod, "run_on_all_native_interfaces", fake_run)

    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "predictions", compute_dockq=True)
    score = scorer.score(example)

    # Direction: native chain IDs are KEYS, model chain IDs are VALUES.
    assert captured["chain_map"] == {"F": "A", "C": "B", "D": "C"}
    # Headline dockq_best is max over binder-target only:
    # max(0.30 (FC), 0.10 (FD)) = 0.30. The target-target 0.90 (CD) is the
    # highest interface in the full result mapping but must not enter the
    # headline.
    assert float(score.extras["dockq_best"]) == pytest.approx(0.30, abs=1e-9)
    assert "dockq_FC_DockQ" in score.extras
    assert "dockq_FD_DockQ" in score.extras
    assert "dockq_CD_DockQ" not in score.extras


def test_scorer_prefers_target_chain_ids_metadata(tmp_path: Path) -> None:
    """If metadata carries the typed ``target_chain_ids`` list (set by the
    SAbDab loader when only a subset of antigen chains was staged), the
    scorer must use it instead of re-parsing the raw ``antigen_chain``
    field. Otherwise rows like 9u5p (``"A | R"`` raw, only ``A`` staged)
    spuriously fail with ``target_chain_count_mismatch``.
    """
    binder_ca = _linear_positions(8, axis=0)
    target_ca = _linear_positions(10, axis=1, start=5.0)
    crystal_chains = [_make_chain("H", binder_ca), _make_chain("A", target_ca)]
    pred_chains = [_make_chain("A", binder_ca), _make_chain("B", target_ca)]

    example = _example(tmp_path)
    # Simulate the post-fix loader: only chain A was staged, but the raw
    # row had two chains. The typed list is what the scorer should consult.
    example.metadata["antigen_chain"] = "A"  # already-normalized form
    example.metadata["antigen_chain_raw"] = "A | R"
    example.metadata["target_chain_ids"] = ("A",)
    _write_structure(example.complex_pdb_path, crystal_chains)
    pred_dir = tmp_path / "predictions" / "ex-1"
    pred_dir.mkdir(parents=True)
    _write_structure(pred_dir / "rank1.pdb", pred_chains)

    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "predictions", compute_dockq=False)
    score = scorer.score(example)
    assert not math.isnan(score.value)
    assert score.value == pytest.approx(0.0, abs=1e-4)

    # Even if a stale loader left only a multi-chain raw ``antigen_chain``
    # behind, the typed list still takes precedence.
    example.metadata["antigen_chain"] = "A | R"
    score2 = scorer.score(example)
    assert not math.isnan(score2.value)
    assert score2.value == pytest.approx(0.0, abs=1e-4)


def test_chain_count_mismatch_returns_nan(tmp_path: Path) -> None:
    """Fab in metadata (Lchain set) but loader thinks it's a VHH → mismatch."""
    binder_ca = _linear_positions(8, axis=0)
    target_ca = _linear_positions(10, axis=1, start=5.0)
    crystal_chains = [
        _make_chain("H", binder_ca),
        _make_chain("L", binder_ca),
        _make_chain("A", target_ca),
    ]
    pred_chains = [_make_chain("A", binder_ca), _make_chain("B", target_ca)]

    example = _example(tmp_path, binder_format="fab")  # claims fab → 2 binder chains expected
    example.metadata["Lchain"] = "L"
    _write_structure(example.complex_pdb_path, crystal_chains)
    pred_dir = tmp_path / "predictions" / "ex-1"
    pred_dir.mkdir(parents=True)
    _write_structure(pred_dir / "rank1.pdb", pred_chains)  # only 2 chains, not 3

    scorer = RMSDToCrystalScorer(predictions_root=tmp_path / "predictions", compute_dockq=False)
    score = scorer.score(example)
    assert math.isnan(score.value)
    # Either upfront chain-count mismatch or a missing predicted chain — both
    # are acceptable as long as the scorer fails gracefully with an error in extras.
    assert "error" in score.extras
