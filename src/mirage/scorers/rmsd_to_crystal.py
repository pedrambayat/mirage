"""RMSDToCrystalScorer — target-aligned binder RMSD vs crystal pose.

Headline value is the **full-binder, target-aligned CA RMSD**. Extras carry
backbone-RMSD, all-heavy-atom RMSD, interface CA RMSD (binder residues
within 8 Å of any target heavy atom in the crystal), and the DockQ score
plus its components (iRMSD, LRMSD, Fnat).

Predicted- and crystal-chain identification follows the convention documented
in :mod:`mirage.scorers._structure`. Residue pairing relies on the SAbDab
loader feeding ColabFold each chain's CA-by-CA sequence, so a predicted chain
has the same visible residues, in order, as its crystal chain; atoms are paired
1:1 by position. A visible-residue count mismatch bails out with an error in
``extras`` rather than guessing an alignment, and a missing prediction or
crystal yields ``Score(value=nan, …)`` so a whole-pilot run still emits one CSV
row per example.
"""

from __future__ import annotations

from pathlib import Path

from Bio.PDB.Atom import Atom
from Bio.PDB.Superimposer import Superimposer

from mirage._paths import default_af2m_predictions_root
from mirage.scorers._registry import register
from mirage.scorers._structure import (
    BACKBONE_NAMES,
    atoms_by_names,
    ca_list,
    chain_residues,
    crystal_binder_chain_ids,
    crystal_target_chain_ids,
    flat_pairs,
    heavy_atom_pairs,
    interface_residue_indices,
    load_structure,
    predicted_chain_ids,
    rmsd_after_transform,
)
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score


def _dockq_metrics(
    pred_path: Path,
    crystal_path: Path,
    pred_binder: list[str],
    pred_target: list[str],
    crys_binder: list[str],
    crys_target: list[str],
) -> dict[str, float]:
    """Run DockQ and flatten per-interface stats.

    ``dockq_best`` here is the **max over binder↔target native interfaces
    only**. Per-CAPRI bucket thresholds (≥0.23 acceptable, ≥0.49 medium,
    ≥0.80 high) were derived for per-interface DockQ, so for multi-interface
    binders (Fab vs single antigen; scFv vs multi-chain antigen) a max
    keeps the headline number bounded in [0, 1] and directly bucket-
    comparable to literature, at the cost of ignoring the second-best
    interface. Target↔target contacts (e.g. the C-D antigen-antigen
    interface in 9p0m's scFv-vs-G-protein alpha/beta case) and binder↔
    binder contacts never enter the headline. Per-pair metrics are
    emitted only for binder↔target interfaces; non-binding interfaces
    are intentionally dropped from the headline extras.
    """
    import itertools

    from DockQ.DockQ import (  # type: ignore[import-untyped]
        load_PDB,
        run_on_all_native_interfaces,
    )

    model_chains = pred_binder + pred_target
    native_chains = crys_binder + crys_target
    model = load_PDB(str(pred_path), chains=model_chains)
    native = load_PDB(str(crystal_path), chains=native_chains)
    # DockQ's run_on_all_native_interfaces expects ``chain_map`` keyed by
    # *native* chain IDs with *model* chain IDs as values (see DockQ.py:613,
    # ``native_chain_ids = list(chain_map.keys())``). Reversing this only
    # appears to work when native and model share the same chain-letter
    # set (e.g. both are A/B); otherwise DockQ raises ``KeyError`` looking
    # for the model chain on the native structure.
    chain_map = dict(zip(native_chains, model_chains, strict=True))

    results, _total_all = run_on_all_native_interfaces(model, native, chain_map=chain_map)

    binder_set = set(crys_binder)
    target_set = set(crys_target)

    metrics: dict[str, float] = {}
    binder_target_max = 0.0
    # DockQ keys result_mapping by ``"".join(chain_pair)`` over
    # ``itertools.combinations(chain_map.keys(), 2)`` in the order we
    # built chain_map (crys_binder ++ crys_target). Mirror that iteration
    # so we can tell which result came from which native pair.
    for c1, c2 in itertools.combinations(chain_map.keys(), 2):
        key = c1 + c2
        stats = results.get(key)
        if stats is None:
            continue
        is_binder_target = (c1 in binder_set and c2 in target_set) or (
            c1 in target_set and c2 in binder_set
        )
        if not is_binder_target:
            continue
        for stat_name in ("DockQ", "iRMSD", "LRMSD", "fnat", "fnonnat"):
            if stat_name in stats:
                metrics[f"dockq_{key}_{stat_name}"] = float(stats[stat_name])
        binder_target_max = max(binder_target_max, float(stats.get("DockQ", 0.0)))

    metrics["dockq_best"] = binder_target_max
    return metrics


@register("rmsd_to_crystal")
class RMSDToCrystalScorer(AbstractScorer):
    """Target-aligned full-binder CA RMSD between a predicted complex and its crystal.

    See module docstring for chain-identification and residue-pairing conventions.
    """

    name = "rmsd_to_crystal"

    def __init__(
        self,
        predictions_root: str | Path | None = None,
        compute_dockq: bool = True,
    ) -> None:
        if predictions_root is None:
            predictions_root = default_af2m_predictions_root()
        self.predictions_root = Path(predictions_root)
        self.compute_dockq = compute_dockq

    def score(self, example: BenchmarkExample) -> Score:
        pred_path = self.predictions_root / example.id / "rank1.pdb"
        crystal_path = example.complex_pdb_path

        if not pred_path.is_file():
            return self.nan_score(example, missing="prediction")
        if crystal_path is None or not Path(crystal_path).is_file():
            return self.nan_score(example, missing="crystal")

        try:
            return self._score_real(example, pred_path, Path(crystal_path))
        except Exception as exc:  # one bad example shouldn't kill a batch
            return self.nan_score(example, error=f"{type(exc).__name__}: {exc}"[:200])

    def _score_real(self, example: BenchmarkExample, pred_path: Path, crystal_path: Path) -> Score:
        pred_binder_ids, pred_target_ids = predicted_chain_ids(example)
        crys_binder_ids = crystal_binder_chain_ids(example)
        crys_target_ids = crystal_target_chain_ids(example)

        if not crys_binder_ids or not crys_target_ids:
            return self.nan_score(example, error="missing_metadata_chain_ids")
        if len(pred_binder_ids) != len(crys_binder_ids):
            return self.nan_score(example, error="binder_chain_count_mismatch")
        if len(pred_target_ids) != len(crys_target_ids):
            return self.nan_score(example, error="target_chain_count_mismatch")

        pred = load_structure(pred_path)
        crys = load_structure(crystal_path)

        # Pair atoms by visible-residue position within each chain.
        target_ca = flat_pairs(pred, pred_target_ids, crys, crys_target_ids, ca_list)
        if target_ca is None:
            return self.nan_score(example, error="target_residue_count_mismatch")
        binder_ca = flat_pairs(pred, pred_binder_ids, crys, crys_binder_ids, ca_list)
        if binder_ca is None:
            return self.nan_score(example, error="binder_residue_count_mismatch")

        # Superpose predicted target onto crystal target (CA only).
        sup = Superimposer()  # type: ignore[no-untyped-call]
        sup.set_atoms(target_ca[1], target_ca[0])  # type: ignore[no-untyped-call]
        rotran = sup.rotran
        if rotran is None:
            return self.nan_score(example, error="superposition_failed")
        rot, trans = rotran
        if sup.rms is None:
            return self.nan_score(example, error="superposition_failed")
        target_ca_rmsd = float(sup.rms)

        binder_ca_rmsd = rmsd_after_transform(binder_ca[0], binder_ca[1], rot, trans)

        # Backbone RMSD (N/CA/C/O) over the binder, in the same target-aligned frame.
        binder_bb = flat_pairs(
            pred,
            pred_binder_ids,
            crys,
            crys_binder_ids,
            lambda residues: atoms_by_names(residues, BACKBONE_NAMES),
        )
        binder_bb_rmsd = (
            rmsd_after_transform(binder_bb[0], binder_bb[1], rot, trans)
            if binder_bb is not None
            else float("nan")
        )

        # All-heavy-atom RMSD over the binder (names must match in both).
        binder_all = heavy_atom_pairs(pred, pred_binder_ids, crys, crys_binder_ids)
        binder_all_rmsd = (
            rmsd_after_transform(binder_all[0], binder_all[1], rot, trans)
            if binder_all is not None
            else float("nan")
        )

        # Interface CA RMSD — binder residues within cutoff of any target heavy atom in crystal.
        crys_binder_res = [chain_residues(crys, c) for c in crys_binder_ids]
        crys_target_res = [chain_residues(crys, c) for c in crys_target_ids]
        pred_binder_res = [chain_residues(pred, c) for c in pred_binder_ids]
        interface_idx = interface_residue_indices(crys_binder_res, crys_target_res)
        if interface_idx:
            iface_pred: list[Atom] = []
            iface_crys: list[Atom] = []
            for ci, ri in interface_idx:
                if "CA" in pred_binder_res[ci][ri] and "CA" in crys_binder_res[ci][ri]:
                    iface_pred.append(pred_binder_res[ci][ri]["CA"])
                    iface_crys.append(crys_binder_res[ci][ri]["CA"])
            interface_ca_rmsd = rmsd_after_transform(iface_pred, iface_crys, rot, trans)
            n_interface = len(iface_pred)
        else:
            interface_ca_rmsd = float("nan")
            n_interface = 0

        extras: dict[str, float | str] = {
            "binder_ca_rmsd_target_aligned": binder_ca_rmsd,
            "binder_backbone_rmsd_target_aligned": binder_bb_rmsd,
            "binder_all_atom_rmsd_target_aligned": binder_all_rmsd,
            "interface_ca_rmsd_target_aligned": interface_ca_rmsd,
            "target_ca_rmsd": target_ca_rmsd,
            "n_target_ca": float(len(target_ca[0])),
            "n_binder_ca": float(len(binder_ca[0])),
            "n_interface_residues": float(n_interface),
        }

        if self.compute_dockq:
            try:
                extras.update(
                    _dockq_metrics(
                        pred_path,
                        crystal_path,
                        pred_binder_ids,
                        pred_target_ids,
                        crys_binder_ids,
                        crys_target_ids,
                    )
                )
            except Exception as exc:
                extras["dockq_error"] = f"{type(exc).__name__}: {exc}"[:200]

        return Score(
            example_id=example.id,
            scorer_name=self.name,
            value=binder_ca_rmsd,
            extras=extras,
        )
