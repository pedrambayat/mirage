"""Pipeline-agnostic structure/geometry primitives shared by scorers.

These helpers know nothing about any particular scorer's headline metric —
they load PDBs, map the binder/target chain-identification convention, pair
atoms across a predicted/crystal structure, superpose, and find interface
residues. Both :mod:`mirage.scorers.rmsd_to_crystal` and
:mod:`mirage.scorers.af2m_confidence` build on them, so a future scorer
(Protenix, Boltz, ESM, …) can reuse the layer instead of re-importing another
scorer's internals.

Chain-identification convention
-------------------------------
- **Predicted PDB.** ColabFold labels chains ``A``, ``B``, ``C``, … in the
  order they appeared in the input FASTA. The AF2-M wrapper writes
  ``binder_chains`` first then ``target_chains`` (see
  ``mirage.pose_predictors.af2m._example_to_fasta``), so the first
  ``len(example.binder_chains)`` predicted chains are the binder and the
  remaining ``len(example.target_chains)`` are the target.
- **Crystal PDB.** Uses SAbDab's original chain IDs, carried in
  ``example.metadata`` as ``Hchain``, ``Lchain`` (Fab only), and
  ``antigen_chain`` / ``target_chain_ids`` for the antigen.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from mirage.scorers.base import BenchmarkExample

BACKBONE_NAMES = frozenset({"N", "CA", "C", "O"})
INTERFACE_CUTOFF_A = 8.0
PRED_CHAIN_LETTERS = tuple(chr(ord("A") + i) for i in range(26))

AtomSelector = Callable[[list[Residue]], list[Atom]]


def crystal_binder_chain_ids(example: BenchmarkExample) -> list[str]:
    md = example.metadata
    h = str(md.get("Hchain", "")).strip()
    out: list[str] = [h] if h and h != "NA" else []
    if example.binder_format == "fab":
        light = str(md.get("Lchain", "")).strip()
        if light and light != "NA":
            out.append(light)
    return out


def crystal_target_chain_ids(example: BenchmarkExample) -> list[str]:
    # Prefer the loader's typed list of staged chain IDs when present (post-fix
    # SAbDab loader). Falls back to parsing `antigen_chain` for compatibility
    # with loaders that don't emit the typed field. The fallback path can miss
    # subset staging (e.g. RNA/DNA chains dropped at load time but still listed
    # in the raw `antigen_chain` field).
    typed = example.metadata.get("target_chain_ids")
    if typed is not None:
        return [str(c) for c in typed]
    field = str(example.metadata.get("antigen_chain", ""))
    return [c.strip() for c in field.split("|") if c.strip()]


def predicted_chain_ids(example: BenchmarkExample) -> tuple[list[str], list[str]]:
    n_b = len(example.binder_chains)
    n_t = len(example.target_chains)
    if n_b + n_t > len(PRED_CHAIN_LETTERS):
        raise ValueError(f"too many chains for single-letter labelling: {n_b + n_t}")
    return list(PRED_CHAIN_LETTERS[:n_b]), list(PRED_CHAIN_LETTERS[n_b : n_b + n_t])


def load_structure(path: Path) -> Structure:
    parser = PDBParser(QUIET=True)  # type: ignore[no-untyped-call]
    return cast(Structure, parser.get_structure(path.stem, str(path)))  # type: ignore[no-untyped-call]


def chain_residues(structure: Structure, chain_id: str) -> list[Residue]:
    chain = structure[0][chain_id]
    return [res for res in chain if res.id[0] == " "]


def ca_list(residues: list[Residue]) -> list[Atom]:
    return [res["CA"] for res in residues if "CA" in res]


def atoms_by_names(residues: list[Residue], names: frozenset[str]) -> list[Atom]:
    out: list[Atom] = []
    for res in residues:
        for atom in res:
            if atom.get_name() in names:
                out.append(atom)
    return out


def flat_pairs(
    pred_struct: Structure,
    pred_chains: list[str],
    crys_struct: Structure,
    crys_chains: list[str],
    selector: AtomSelector,
) -> tuple[list[Atom], list[Atom]] | None:
    pred_atoms: list[Atom] = []
    crys_atoms: list[Atom] = []
    for p_id, c_id in zip(pred_chains, crys_chains, strict=True):
        p_res = chain_residues(pred_struct, p_id)
        c_res = chain_residues(crys_struct, c_id)
        p_sel = selector(p_res)
        c_sel = selector(c_res)
        if len(p_sel) != len(c_sel):
            return None
        pred_atoms.extend(p_sel)
        crys_atoms.extend(c_sel)
    return pred_atoms, crys_atoms


def heavy_atom_pairs(
    pred: Structure,
    pred_chains: list[str],
    crys: Structure,
    crys_chains: list[str],
) -> tuple[list[Atom], list[Atom]] | None:
    """Pair heavy atoms by (residue_index, atom_name) within each matched chain.

    Crystal residues sometimes have unresolved side-chain atoms; only count atoms
    present in both. Mismatching residue counts is treated as a fatal mismatch.
    """
    pred_atoms: list[Atom] = []
    crys_atoms: list[Atom] = []
    for p_id, c_id in zip(pred_chains, crys_chains, strict=True):
        p_res = chain_residues(pred, p_id)
        c_res = chain_residues(crys, c_id)
        if len(p_res) != len(c_res):
            return None
        for pr, cr in zip(p_res, c_res, strict=True):
            for atom in pr:
                if atom.element == "H":
                    continue
                name = atom.get_name()
                if name in cr:
                    pred_atoms.append(atom)
                    crys_atoms.append(cr[name])
    return pred_atoms, crys_atoms


def rmsd_after_transform(
    moved: list[Atom],
    reference: list[Atom],
    rotation: np.ndarray[Any, Any],
    translation: np.ndarray[Any, Any],
) -> float:
    if not moved:
        return float("nan")
    moved_coords = np.array([a.coord for a in moved]) @ rotation + translation
    ref_coords = np.array([a.coord for a in reference])
    diff = moved_coords - ref_coords
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def interface_residue_indices(
    binder_residues: list[list[Residue]],
    target_residues: list[list[Residue]],
) -> list[tuple[int, int]]:
    """Indices (chain_idx, residue_idx) of binder residues within cutoff of any target atom."""
    target_coords = np.array(
        [
            atom.coord
            for chain in target_residues
            for res in chain
            for atom in res
            if atom.element != "H"
        ]
    )
    if target_coords.size == 0:
        return []
    out: list[tuple[int, int]] = []
    for ci, chain_res in enumerate(binder_residues):
        for ri, res in enumerate(chain_res):
            heavy = np.array([a.coord for a in res if a.element != "H"])
            if heavy.size == 0:
                continue
            diff = heavy[:, None, :] - target_coords[None, :, :]
            d2 = (diff * diff).sum(axis=2)
            if d2.min() <= INTERFACE_CUTOFF_A * INTERFACE_CUTOFF_A:
                out.append((ci, ri))
    return out
