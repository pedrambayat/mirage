"""Predictor-agnostic interface geometry for a predicted complex.

This scorer reads only the predicted rank-1 PDB. It deliberately ignores
AF2-M confidence arrays and any crystal structure, so the same features can be
computed for complexes produced by AF2-M, AF3, Protenix, Boltz, or another
pose generator once their PDBs are staged in the same per-example layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.Residue import Residue

from mirage._paths import default_af2m_predictions_root
from mirage.scorers._registry import register
from mirage.scorers._structure import (
    chain_residues,
    load_structure,
    predicted_chain_ids,
    read_chain_roles_json,
    resolve_chain_roles_by_sequence,
)
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score

CONTACT_CUTOFFS_A = (4.0, 6.0, 8.0)
INTERFACE_CUTOFF_A = 8.0
ATOM_CONTACT_CUTOFF_A = 5.0
ATOM_CLOSE_CONTACT_CUTOFF_A = 3.0
ATOM_CLASH_CUTOFF_A = 2.0
PACKING_SURFACE_GAP_MIN_A = 0.0
PACKING_SURFACE_GAP_MAX_A = 1.0
PACKING_SHELL_GAP_MAX_A = 2.0
SASA_PROBE_RADIUS_A = 1.4
SASA_OCCLUSION_SCALE = 0.35
VDW_RADII_A = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
}

HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"})
AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP", "HIS"})
POLAR_RESIDUES = frozenset({"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS"})
POSITIVE_RESIDUES = frozenset({"LYS", "ARG", "HIS"})
NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})
CHARGED_RESIDUES = POSITIVE_RESIDUES | NEGATIVE_RESIDUES


def _heavy_coords(residue: Residue) -> np.ndarray[Any, Any]:
    coords = [atom.coord for atom in residue if atom.element != "H"]
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _stack_heavy_coords(residues: list[Residue]) -> np.ndarray[Any, Any]:
    arrays = [_heavy_coords(residue) for residue in residues]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return np.empty((0, 3), dtype=float)
    return np.vstack(arrays)


def _atom_vdw_radius(atom: Any) -> float:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if not element:
        element = str(atom.get_name()).strip()[:1].upper()
    return VDW_RADII_A.get(element, 1.70)


def _stack_heavy_coords_and_radii(
    residues: list[Residue],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    coords: list[Any] = []
    radii: list[float] = []
    for residue in residues:
        for atom in residue:
            if atom.element == "H":
                continue
            coords.append(atom.coord)
            radii.append(_atom_vdw_radius(atom))
    if not coords:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float)


def _stack_heavy_coords_with_residue_indices(
    residues: list[Residue],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    coord_arrays: list[np.ndarray[Any, Any]] = []
    index_arrays: list[np.ndarray[Any, Any]] = []
    for idx, residue in enumerate(residues):
        coords = _heavy_coords(residue)
        if not coords.size:
            continue
        coord_arrays.append(coords)
        index_arrays.append(np.full(coords.shape[0], idx, dtype=int))
    if not coord_arrays:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=int)
    return np.vstack(coord_arrays), np.concatenate(index_arrays)


def _distance_stats(
    binder_residues: list[Residue],
    target_residues: list[Residue],
    contact_cutoff: float,
) -> tuple[np.ndarray[Any, Any], float, float, float, float, float, float, int, int]:
    n_binder_res = len(binder_residues)
    n_target_res = len(target_residues)
    residue_distance2 = np.full((n_binder_res, n_target_res), np.inf, dtype=float)
    binder_coords, binder_res_idx = _stack_heavy_coords_with_residue_indices(binder_residues)
    target_coords, target_res_idx = _stack_heavy_coords_with_residue_indices(target_residues)
    if not binder_coords.size or not target_coords.size:
        residue_distances = np.full((n_binder_res, n_target_res), np.nan, dtype=float)
        return residue_distances, float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0, 0, 0

    min_distance = float("inf")
    distance_sum = 0.0
    contact_count = 0.0
    close_contact_count = 0.0
    clash_count = 0.0
    n_pairs = 0.0
    cutoff2 = contact_cutoff * contact_cutoff
    close_cutoff2 = ATOM_CLOSE_CONTACT_CUTOFF_A * ATOM_CLOSE_CONTACT_CUTOFF_A
    clash_cutoff2 = ATOM_CLASH_CUTOFF_A * ATOM_CLASH_CUTOFF_A
    flat_residue_distance2 = residue_distance2.ravel()
    for start in range(0, binder_coords.shape[0], 128):
        chunk = binder_coords[start : start + 128]
        chunk_res_idx = binder_res_idx[start : start + 128]
        diff = chunk[:, None, :] - target_coords[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        min_distance = min(min_distance, float(np.sqrt(d2.min())))
        distance_sum += float(np.sqrt(d2).sum())
        contact_count += float(np.count_nonzero(d2 <= cutoff2))
        close_contact_count += float(np.count_nonzero(d2 <= close_cutoff2))
        clash_count += float(np.count_nonzero(d2 <= clash_cutoff2))
        n_pairs += float(d2.size)
        flat_pair_idx = (chunk_res_idx[:, None] * n_target_res + target_res_idx[None, :]).ravel()
        np.minimum.at(flat_residue_distance2, flat_pair_idx, d2.ravel())

    residue_distances = np.sqrt(residue_distance2)
    residue_distances[~np.isfinite(residue_distances)] = np.nan
    return (
        residue_distances,
        min_distance,
        _safe_div(distance_sum, n_pairs),
        contact_count,
        close_contact_count,
        clash_count,
        n_pairs,
        binder_coords.shape[0],
        target_coords.shape[0],
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _occlusion_scores(
    atom_coords: np.ndarray[Any, Any],
    atom_radii: np.ndarray[Any, Any],
    blocker_coords: np.ndarray[Any, Any],
    blocker_radii: np.ndarray[Any, Any],
    *,
    exclude_self: bool = False,
) -> np.ndarray[Any, Any]:
    if atom_coords.size == 0 or blocker_coords.size == 0:
        return np.zeros(atom_radii.shape[0], dtype=float)

    atom_surface_radii = atom_radii + SASA_PROBE_RADIUS_A
    blocker_surface_radii = blocker_radii + SASA_PROBE_RADIUS_A
    scores = np.zeros(atom_radii.shape[0], dtype=float)
    for start in range(0, atom_coords.shape[0], 128):
        stop = min(start + 128, atom_coords.shape[0])
        chunk = atom_coords[start:stop]
        diff = chunk[:, None, :] - blocker_coords[None, :, :]
        distances = np.sqrt((diff * diff).sum(axis=2))
        if exclude_self:
            rows = np.arange(stop - start)
            distances[rows, start + rows] = np.inf
        occlusion_span = atom_surface_radii[start:stop, None] + blocker_surface_radii[None, :]
        overlap = np.clip(
            (occlusion_span - distances) / atom_surface_radii[start:stop, None],
            0.0,
            None,
        )
        scores[start:stop] = overlap.sum(axis=1)
    return scores


def _add_buried_sasa_proxy_features(
    extras: dict[str, float | str],
    binder_residues: list[Residue],
    target_residues: list[Residue],
    n_binder_interface: float,
    n_target_interface: float,
) -> None:
    binder_coords, binder_radii = _stack_heavy_coords_and_radii(binder_residues)
    target_coords, target_radii = _stack_heavy_coords_and_radii(target_residues)
    if binder_coords.size == 0 or target_coords.size == 0:
        extras["buried_sasa_proxy_a2"] = float("nan")
        extras["buried_sasa_proxy_binder_a2"] = float("nan")
        extras["buried_sasa_proxy_target_a2"] = float("nan")
        extras["buried_sasa_proxy_per_binder_atom"] = float("nan")
        extras["buried_sasa_proxy_per_target_atom"] = float("nan")
        extras["buried_sasa_proxy_per_interface_residue"] = float("nan")
        extras["buried_sasa_balance"] = float("nan")
        extras["mean_binder_atom_exposure_loss"] = float("nan")
        extras["mean_target_atom_exposure_loss"] = float("nan")
        return

    binder_same = _occlusion_scores(
        binder_coords, binder_radii, binder_coords, binder_radii, exclude_self=True
    )
    target_same = _occlusion_scores(
        target_coords, target_radii, target_coords, target_radii, exclude_self=True
    )
    binder_opposite = _occlusion_scores(binder_coords, binder_radii, target_coords, target_radii)
    target_opposite = _occlusion_scores(target_coords, target_radii, binder_coords, binder_radii)

    binder_isolated_exposure = np.exp(-SASA_OCCLUSION_SCALE * binder_same)
    target_isolated_exposure = np.exp(-SASA_OCCLUSION_SCALE * target_same)
    binder_complex_exposure = np.exp(-SASA_OCCLUSION_SCALE * (binder_same + binder_opposite))
    target_complex_exposure = np.exp(-SASA_OCCLUSION_SCALE * (target_same + target_opposite))
    binder_loss = np.clip(binder_isolated_exposure - binder_complex_exposure, 0.0, 1.0)
    target_loss = np.clip(target_isolated_exposure - target_complex_exposure, 0.0, 1.0)
    binder_surface_area = 4.0 * np.pi * (binder_radii + SASA_PROBE_RADIUS_A) ** 2
    target_surface_area = 4.0 * np.pi * (target_radii + SASA_PROBE_RADIUS_A) ** 2
    binder_buried = float((binder_loss * binder_surface_area).sum())
    target_buried = float((target_loss * target_surface_area).sum())
    total_buried = binder_buried + target_buried

    extras["buried_sasa_proxy_a2"] = total_buried
    extras["buried_sasa_proxy_binder_a2"] = binder_buried
    extras["buried_sasa_proxy_target_a2"] = target_buried
    extras["buried_sasa_proxy_per_binder_atom"] = _safe_div(binder_buried, float(binder_radii.size))
    extras["buried_sasa_proxy_per_target_atom"] = _safe_div(target_buried, float(target_radii.size))
    extras["buried_sasa_proxy_per_interface_residue"] = _safe_div(
        total_buried, n_binder_interface + n_target_interface
    )
    extras["buried_sasa_balance"] = _safe_div(
        min(binder_buried, target_buried), max(binder_buried, target_buried)
    )
    extras["mean_binder_atom_exposure_loss"] = float(binder_loss.mean())
    extras["mean_target_atom_exposure_loss"] = float(target_loss.mean())


def _add_atom_packing_features(
    extras: dict[str, float | str],
    binder_residues: list[Residue],
    target_residues: list[Residue],
) -> None:
    binder_coords, binder_radii = _stack_heavy_coords_and_radii(binder_residues)
    target_coords, target_radii = _stack_heavy_coords_and_radii(target_residues)
    if binder_coords.size == 0 or target_coords.size == 0:
        extras["atom_packing_pairs_0_1a_gap"] = float("nan")
        extras["atom_packing_fraction_0_1a_gap"] = float("nan")
        extras["atom_packing_shell_pairs_2a_gap"] = float("nan")
        extras["binder_atom_packing_coverage_0_1a_gap"] = float("nan")
        extras["target_atom_packing_coverage_0_1a_gap"] = float("nan")
        extras["atom_packing_complementarity_score"] = float("nan")
        extras["mean_abs_nearest_surface_gap"] = float("nan")
        return

    binder_packed = np.zeros(binder_radii.shape[0], dtype=bool)
    target_packed = np.zeros(target_radii.shape[0], dtype=bool)
    binder_nearest_abs_gap = np.full(binder_radii.shape[0], np.inf, dtype=float)
    target_nearest_abs_gap = np.full(target_radii.shape[0], np.inf, dtype=float)
    packing_pair_count = 0.0
    shell_pair_count = 0.0

    for start in range(0, binder_coords.shape[0], 128):
        stop = min(start + 128, binder_coords.shape[0])
        chunk = binder_coords[start:stop]
        diff = chunk[:, None, :] - target_coords[None, :, :]
        distances = np.sqrt((diff * diff).sum(axis=2))
        surface_gap = distances - (binder_radii[start:stop, None] + target_radii[None, :])
        abs_gap = np.abs(surface_gap)
        packed = (surface_gap >= PACKING_SURFACE_GAP_MIN_A) & (
            surface_gap <= PACKING_SURFACE_GAP_MAX_A
        )
        shell = (surface_gap >= PACKING_SURFACE_GAP_MIN_A) & (
            surface_gap <= PACKING_SHELL_GAP_MAX_A
        )
        packing_pair_count += float(np.count_nonzero(packed))
        shell_pair_count += float(np.count_nonzero(shell))
        binder_packed[start:stop] = np.any(packed, axis=1)
        target_packed |= np.any(packed, axis=0)
        binder_nearest_abs_gap[start:stop] = abs_gap.min(axis=1)
        np.minimum.at(target_nearest_abs_gap, np.arange(target_radii.shape[0]), abs_gap.min(axis=0))

    binder_coverage = float(binder_packed.mean())
    target_coverage = float(target_packed.mean())
    nearest_abs_gap = np.concatenate([binder_nearest_abs_gap, target_nearest_abs_gap])
    nearest_abs_gap = nearest_abs_gap[np.isfinite(nearest_abs_gap)]

    extras["atom_packing_pairs_0_1a_gap"] = packing_pair_count
    extras["atom_packing_fraction_0_1a_gap"] = _safe_div(packing_pair_count, shell_pair_count)
    extras["atom_packing_shell_pairs_2a_gap"] = shell_pair_count
    extras["binder_atom_packing_coverage_0_1a_gap"] = binder_coverage
    extras["target_atom_packing_coverage_0_1a_gap"] = target_coverage
    extras["atom_packing_complementarity_score"] = float(np.sqrt(binder_coverage * target_coverage))
    extras["mean_abs_nearest_surface_gap"] = (
        float(nearest_abs_gap.mean()) if nearest_abs_gap.size else float("nan")
    )


def _residue_names(residues: list[Residue]) -> np.ndarray[Any, Any]:
    return np.asarray(
        [str(residue.get_resname()).upper() for residue in residues],  # type: ignore[no-untyped-call]
        dtype=object,
    )


def _membership_mask(names: np.ndarray[Any, Any], residues: frozenset[str]) -> np.ndarray[Any, Any]:
    return np.asarray([name in residues for name in names], dtype=bool)


def _pair_count(
    contact_mask: np.ndarray[Any, Any],
    binder_mask: np.ndarray[Any, Any],
    target_mask: np.ndarray[Any, Any],
) -> float:
    if contact_mask.size == 0:
        return 0.0
    pair_mask = binder_mask[:, None] & target_mask[None, :]
    return float(np.count_nonzero(contact_mask & pair_mask))


def _same_charge_count(
    contact_mask: np.ndarray[Any, Any],
    binder_positive: np.ndarray[Any, Any],
    binder_negative: np.ndarray[Any, Any],
    target_positive: np.ndarray[Any, Any],
    target_negative: np.ndarray[Any, Any],
) -> float:
    if contact_mask.size == 0:
        return 0.0
    pair_mask = (binder_positive[:, None] & target_positive[None, :]) | (
        binder_negative[:, None] & target_negative[None, :]
    )
    return float(np.count_nonzero(contact_mask & pair_mask))


def _opposite_charge_count(
    contact_mask: np.ndarray[Any, Any],
    binder_positive: np.ndarray[Any, Any],
    binder_negative: np.ndarray[Any, Any],
    target_positive: np.ndarray[Any, Any],
    target_negative: np.ndarray[Any, Any],
) -> float:
    if contact_mask.size == 0:
        return 0.0
    pair_mask = (binder_positive[:, None] & target_negative[None, :]) | (
        binder_negative[:, None] & target_positive[None, :]
    )
    return float(np.count_nonzero(contact_mask & pair_mask))


def _add_chemistry_features(
    extras: dict[str, float | str],
    residue_distances: np.ndarray[Any, Any],
    binder_residues: list[Residue],
    target_residues: list[Residue],
) -> None:
    contact_mask = residue_distances <= 6.0
    contact_count = float(np.count_nonzero(contact_mask))
    binder_names = _residue_names(binder_residues)
    target_names = _residue_names(target_residues)

    binder_hydrophobic = _membership_mask(binder_names, HYDROPHOBIC_RESIDUES)
    target_hydrophobic = _membership_mask(target_names, HYDROPHOBIC_RESIDUES)
    binder_aromatic = _membership_mask(binder_names, AROMATIC_RESIDUES)
    target_aromatic = _membership_mask(target_names, AROMATIC_RESIDUES)
    binder_polar = _membership_mask(binder_names, POLAR_RESIDUES)
    target_polar = _membership_mask(target_names, POLAR_RESIDUES)
    binder_charged = _membership_mask(binder_names, CHARGED_RESIDUES)
    target_charged = _membership_mask(target_names, CHARGED_RESIDUES)
    binder_positive = _membership_mask(binder_names, POSITIVE_RESIDUES)
    target_positive = _membership_mask(target_names, POSITIVE_RESIDUES)
    binder_negative = _membership_mask(binder_names, NEGATIVE_RESIDUES)
    target_negative = _membership_mask(target_names, NEGATIVE_RESIDUES)

    chemistry_counts = {
        "hydrophobic_contact_pairs_6a": _pair_count(
            contact_mask, binder_hydrophobic, target_hydrophobic
        ),
        "aromatic_contact_pairs_6a": _pair_count(contact_mask, binder_aromatic, target_aromatic),
        "polar_contact_pairs_6a": _pair_count(contact_mask, binder_polar, target_polar),
        "charged_contact_pairs_6a": _pair_count(contact_mask, binder_charged, target_charged),
        "opposite_charge_contact_pairs_6a": _opposite_charge_count(
            contact_mask,
            binder_positive,
            binder_negative,
            target_positive,
            target_negative,
        ),
        "same_charge_contact_pairs_6a": _same_charge_count(
            contact_mask,
            binder_positive,
            binder_negative,
            target_positive,
            target_negative,
        ),
    }
    for name, count in chemistry_counts.items():
        extras[name] = count
        extras[name.replace("_pairs_", "_fraction_")] = _safe_div(count, contact_count)


def _add_interface_distance_features(
    extras: dict[str, float | str], residue_distances: np.ndarray[Any, Any]
) -> None:
    interface_distances = residue_distances[residue_distances <= INTERFACE_CUTOFF_A]
    finite = interface_distances[np.isfinite(interface_distances)]
    if finite.size == 0:
        extras["mean_interface_residue_distance_8a"] = float("nan")
        extras["median_interface_residue_distance_8a"] = float("nan")
        extras["std_interface_residue_distance_8a"] = float("nan")
        extras["close_residue_contact_fraction_4a_within_8a"] = float("nan")
        return
    extras["mean_interface_residue_distance_8a"] = float(finite.mean())
    extras["median_interface_residue_distance_8a"] = float(np.median(finite))
    extras["std_interface_residue_distance_8a"] = float(finite.std())
    extras["close_residue_contact_fraction_4a_within_8a"] = _safe_div(
        float(np.count_nonzero(finite <= 4.0)), float(finite.size)
    )


def _find_structure_for_sequence_mode(pred_dir: Path) -> Path | None:
    """Locate a predicted structure file under ``pred_dir`` for sequence mode.

    Search order (preference): ``rank1.cif``, ``rank1.pdb``, first
    ``*sample_0*.cif`` found recursively.
    """
    for name in ("rank1.cif", "rank1.pdb"):
        candidate = pred_dir / name
        if candidate.is_file():
            return candidate
    # Fall back to any *sample_0*.cif in the directory tree
    for candidate in sorted(pred_dir.rglob("*sample_0*.cif")):
        return candidate
    return None


@register("structural_interface")
class StructuralInterfaceScorer(AbstractScorer):
    """Crystal-independent interface geometry from a predicted rank-1 PDB.

    Parameters
    ----------
    predictions_root:
        Root directory of per-example prediction directories.
    chain_resolution:
        ``"positional"`` (default) — chains are assigned A, B, C, … by input
        order (ColabFold/AF2-M convention).  ``"sequence"`` — chains are mapped
        to binder/target roles by sequence identity against
        ``example.binder_chains`` / ``example.target_chains``; also supports
        mmCIF input and the Protenix nested directory layout.
    """

    name = "structural_interface"

    def __init__(
        self,
        predictions_root: str | Path | None = None,
        chain_resolution: str = "positional",
    ) -> None:
        if predictions_root is None:
            predictions_root = default_af2m_predictions_root()
        self.predictions_root = Path(predictions_root)
        self.chain_resolution = chain_resolution

    def score(self, example: BenchmarkExample) -> Score:
        if self.chain_resolution == "sequence":
            pred_dir = self.predictions_root / example.id
            pred_path = _find_structure_for_sequence_mode(pred_dir)
            if pred_path is None:
                return self.nan_score(example, missing="prediction")
        else:
            pred_path = self.predictions_root / example.id / "rank1.pdb"
            if not pred_path.is_file():
                return self.nan_score(example, missing="prediction")

        try:
            return self._score_real(example, pred_path)
        except Exception as exc:
            return self.nan_score(example, error=f"{type(exc).__name__}: {exc}"[:200])

    def _score_real(self, example: BenchmarkExample, pred_path: Path) -> Score:
        pred = load_structure(pred_path)
        if self.chain_resolution == "sequence":
            roles = read_chain_roles_json(pred_path.parent / "chain_roles.json")
            if roles is not None:
                binder_ids, target_ids = roles
            else:
                binder_ids, target_ids = resolve_chain_roles_by_sequence(
                    pred,
                    binder_seqs=example.binder_chains,
                    target_seqs=example.target_chains,
                )
        else:
            binder_ids, target_ids = predicted_chain_ids(example)
        return self.score_with_chains(example, pred, binder_ids, target_ids)

    def score_with_chains(
        self,
        example: BenchmarkExample,
        pred: Any,
        binder_ids: list[str],
        target_ids: list[str],
    ) -> Score:
        """Compute interface geometry features given an already-loaded structure
        and explicit binder/target chain IDs."""
        binder_residues = [
            residue for chain_id in binder_ids for residue in chain_residues(pred, chain_id)
        ]
        target_residues = [
            residue for chain_id in target_ids for residue in chain_residues(pred, chain_id)
        ]

        (
            residue_distances,
            atom_min_distance,
            atom_mean_distance,
            atom_contact_count,
            atom_close_contact_count,
            atom_clash_count,
            n_atom_pairs,
            n_binder_atoms,
            n_target_atoms,
        ) = _distance_stats(binder_residues, target_residues, ATOM_CONTACT_CUTOFF_A)

        n_binder_res = len(binder_residues)
        n_target_res = len(target_residues)
        n_residue_pairs = n_binder_res * n_target_res

        extras: dict[str, float | str] = {
            "n_residues_binder": float(n_binder_res),
            "n_residues_target": float(n_target_res),
            "n_heavy_atoms_binder": float(n_binder_atoms),
            "n_heavy_atoms_target": float(n_target_atoms),
            "min_interchain_heavy_atom_distance": atom_min_distance,
            "mean_interchain_heavy_atom_distance": atom_mean_distance,
        }

        for cutoff in CONTACT_CUTOFFS_A:
            mask = residue_distances <= cutoff
            contact_count = float(np.count_nonzero(mask))
            suffix = f"{cutoff:g}a"
            extras[f"residue_pair_contacts_{suffix}"] = contact_count
            extras[f"residue_pair_contact_fraction_{suffix}"] = _safe_div(
                contact_count, float(n_residue_pairs)
            )
            extras[f"contacts_per_binder_residue_{suffix}"] = _safe_div(
                contact_count, float(n_binder_res)
            )
            extras[f"contacts_per_target_residue_{suffix}"] = _safe_div(
                contact_count, float(n_target_res)
            )

        interface_mask = residue_distances <= INTERFACE_CUTOFF_A
        binder_interface = (
            np.any(interface_mask, axis=1) if n_target_res else np.zeros(0, dtype=bool)
        )
        target_interface = (
            np.any(interface_mask, axis=0) if n_binder_res else np.zeros(0, dtype=bool)
        )
        n_binder_interface = float(np.count_nonzero(binder_interface))
        n_target_interface = float(np.count_nonzero(target_interface))

        atom_contact_fraction = _safe_div(atom_contact_count, n_atom_pairs)
        atom_close_contact_fraction = _safe_div(atom_close_contact_count, n_atom_pairs)
        atom_clash_fraction = _safe_div(atom_clash_count, n_atom_pairs)

        extras.update(
            {
                "n_interface_residues_binder": n_binder_interface,
                "n_interface_residues_target": n_target_interface,
                "binder_interface_fraction": _safe_div(n_binder_interface, float(n_binder_res)),
                "target_interface_fraction": _safe_div(n_target_interface, float(n_target_res)),
                "atom_contacts_5a": atom_contact_count,
                "atom_contact_fraction_5a": atom_contact_fraction,
                "atom_close_contacts_3a": atom_close_contact_count,
                "atom_close_contact_fraction_3a": atom_close_contact_fraction,
                "atom_clashes_2a": atom_clash_count,
                "atom_clash_fraction_2a": atom_clash_fraction,
                "buried_area_proxy_5a": atom_contact_fraction,
                "shape_complementarity_proxy": _safe_div(
                    float(extras["residue_pair_contacts_6a"]),
                    n_binder_interface + n_target_interface,
                ),
            }
        )
        _add_chemistry_features(extras, residue_distances, binder_residues, target_residues)
        _add_interface_distance_features(extras, residue_distances)
        _add_buried_sasa_proxy_features(
            extras,
            binder_residues,
            target_residues,
            n_binder_interface,
            n_target_interface,
        )
        _add_atom_packing_features(extras, binder_residues, target_residues)

        headline = float(extras["contacts_per_binder_residue_6a"])
        return Score(
            example_id=example.id,
            scorer_name=self.name,
            value=headline,
            extras=extras,
        )
