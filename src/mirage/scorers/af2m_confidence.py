"""AF2MConfidenceScorer — AF2-Multimer's own confidence metrics for a predicted complex.

Headline value is iPTM (the canonical AF-M antibody-antigen discriminator
in the literature). Extras carry pTM, ranking_confidence, iPTM/pTM ratio,
pLDDT aggregates (full, binder-only, target-only, predicted-interface),
and inter-chain PAE aggregates (mean, max, interface-restricted).

Inputs per example (under ``predictions_root/<example_id>/``):

- ``scores.json`` — mirage AF2-M wrapper's flat summary. Used here only
  to locate the rank-1 unrelaxed PDB filename (``rank1.pdb`` is a
  symlink, but the filename pattern carries the model index).
- ``<example_id>_scores_rank_001_alphafold2_multimer_v3_model_<N>_seed_<NNN>.json``
  — ColabFold's raw rank-1 scores file. Contains the per-residue
  ``plddt`` array, the per-residue-pair ``pae`` matrix, and scalar
  ``iptm``/``ptm``/``max_pae``. This is the canonical source for the
  numbers; the wrapper's summary scores.json only has the scalars.
- ``rank1.pdb`` — predicted structure, used only to locate the
  *predicted* binder↔target interface residues for the
  interface-restricted aggregates.

The scorer is **crystal-independent**: it never touches the SAbDab crystal
PDB. This is intentional — the eventual learned discriminator will not
have a crystal at inference time, so any baseline scorer it is compared
against must also work on the prediction alone.

The residue order in ColabFold's ``plddt`` and ``pae`` arrays follows the
input FASTA, which the AF2-M wrapper writes as binder chains first then
target chains (see ``mirage.pose_predictors.af2m._example_to_fasta``).
The same ordering applies to chain letters in ``rank1.pdb``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mirage._paths import default_af2m_predictions_root
from mirage.scorers._registry import register
from mirage.scorers._structure import (
    chain_residues,
    interface_residue_indices,
    load_structure,
    predicted_chain_ids,
)
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score


def _rank1_raw_scores_path(pred_dir: Path, summary_path: Path) -> Path:
    """Return the path to ColabFold's raw rank-1 scores JSON.

    The mirage wrapper's ``scores.json`` carries the rank-1 unrelaxed PDB
    filename under ``["rank1"]["pdb"]``; the matching raw scores file
    swaps ``_unrelaxed_`` for ``_scores_`` and ``.pdb`` for ``.json``.
    Raises :class:`FileNotFoundError` if the summary doesn't carry a
    rank-1 PDB filename.
    """
    summary = json.loads(summary_path.read_text())
    try:
        rank1_pdb_name = str(summary["rank1"]["pdb"])
    except (KeyError, TypeError) as exc:
        raise FileNotFoundError(f"summary scores.json missing rank1.pdb name: {exc}") from exc
    raw_name = rank1_pdb_name.replace("_unrelaxed_", "_scores_").replace(".pdb", ".json")
    return pred_dir / raw_name


@register("af2m_confidence")
class AF2MConfidenceScorer(AbstractScorer):
    """AF2-M's own confidence metrics for a predicted complex.

    Headline ``value`` is iPTM. Extras carry pTM, ranking_confidence,
    iPTM/pTM, pLDDT aggregates (full / binder / target / interface), and
    inter-chain PAE aggregates (mean / max / interface).
    """

    name = "af2m_confidence"

    def __init__(self, predictions_root: str | Path | None = None) -> None:
        if predictions_root is None:
            predictions_root = default_af2m_predictions_root()
        self.predictions_root = Path(predictions_root)

    def score(self, example: BenchmarkExample) -> Score:
        pred_dir = self.predictions_root / example.id
        pred_path = pred_dir / "rank1.pdb"
        summary_path = pred_dir / "scores.json"

        if not pred_path.is_file():
            return self.nan_score(example, missing="prediction")
        if not summary_path.is_file():
            return self.nan_score(example, missing="scores_json")

        try:
            raw_scores_path = _rank1_raw_scores_path(pred_dir, summary_path)
        except FileNotFoundError as exc:
            return self.nan_score(example, missing=str(exc)[:200])
        if not raw_scores_path.is_file():
            return self.nan_score(example, missing=f"raw_scores:{raw_scores_path.name}")

        try:
            return self._score_real(example, pred_path, raw_scores_path)
        except Exception as exc:  # one bad example shouldn't kill a batch
            return self.nan_score(example, error=f"{type(exc).__name__}: {exc}"[:200])

    def _score_real(
        self, example: BenchmarkExample, pred_path: Path, raw_scores_path: Path
    ) -> Score:
        scores = json.loads(raw_scores_path.read_text())
        iptm = float(scores["iptm"])
        ptm = float(scores["ptm"])
        plddt = np.asarray(scores["plddt"], dtype=float)
        pae = np.asarray(scores["pae"], dtype=float)
        max_pae = float(scores["max_pae"])

        ranking_confidence = 0.8 * iptm + 0.2 * ptm
        # iPTM/pTM ratio is SNAP's load-bearing "is the interface specially
        # resolved?" metric. NaN when pTM is zero — protects against a
        # rare degenerate case rather than a real scoring signal.
        iptm_over_ptm = iptm / ptm if ptm > 0 else float("nan")

        pred = load_structure(pred_path)
        pred_binder_ids, pred_target_ids = predicted_chain_ids(example)
        pred_binder_res = [chain_residues(pred, c) for c in pred_binder_ids]
        pred_target_res = [chain_residues(pred, c) for c in pred_target_ids]
        n_binder = sum(len(r) for r in pred_binder_res)
        n_target = sum(len(r) for r in pred_target_res)
        n_total = len(plddt)

        if n_total != n_binder + n_target:
            return self.nan_score(
                example,
                error=(f"plddt_length_mismatch:plddt={n_total},pdb={n_binder + n_target}"),
            )
        if pae.shape != (n_total, n_total):
            return self.nan_score(
                example,
                error=f"pae_shape_mismatch:pae={pae.shape},expected=({n_total},{n_total})",
            )

        plddt_full = float(plddt.mean()) if plddt.size else float("nan")
        plddt_binder = float(plddt[:n_binder].mean()) if n_binder else float("nan")
        plddt_target = float(plddt[n_binder:].mean()) if n_target else float("nan")

        # Predicted interface residues — binder residues within 8 Å of any
        # target heavy atom in the *predicted* structure (same cutoff and
        # geometry as the crystal-side interface used by RMSDToCrystalScorer,
        # but applied to the predicted PDB so the scorer stays crystal-
        # independent). Returns (binder_chain_idx, residue_idx_within_chain).
        interface_idx = interface_residue_indices(pred_binder_res, pred_target_res)
        n_interface = len(interface_idx)

        if interface_idx:
            chain_offsets: list[int] = []
            offset = 0
            for chain_res in pred_binder_res:
                chain_offsets.append(offset)
                offset += len(chain_res)
            flat_iface = np.array([chain_offsets[ci] + ri for ci, ri in interface_idx], dtype=int)
            plddt_interface = float(plddt[flat_iface].mean())
            # PAE from interface binder rows to all target columns. AF-M's
            # PAE is asymmetric (entry [i,j] = expected error in residue
            # j's position when superposed on residue i's frame), so this
            # captures how confident the model is about *target* placement
            # relative to the predicted-interface binder frame.
            pae_interface = float(pae[flat_iface][:, n_binder:].mean())
        else:
            plddt_interface = float("nan")
            pae_interface = float("nan")

        if n_binder and n_target:
            bt = pae[:n_binder, n_binder:]
            tb = pae[n_binder:, :n_binder]
            pae_interchain_mean = float((bt.mean() + tb.mean()) / 2.0)
            pae_interchain_max = float(max(bt.max(), tb.max()))
        else:
            pae_interchain_mean = float("nan")
            pae_interchain_max = float("nan")

        extras: dict[str, float | str] = {
            "iptm": iptm,
            "ptm": ptm,
            "ranking_confidence": ranking_confidence,
            "iptm_over_ptm": iptm_over_ptm,
            "plddt_full_mean": plddt_full,
            "plddt_binder_mean": plddt_binder,
            "plddt_target_mean": plddt_target,
            "plddt_interface_mean": plddt_interface,
            "pae_interchain_mean": pae_interchain_mean,
            "pae_interchain_max": pae_interchain_max,
            "pae_interface_mean": pae_interface,
            "max_pae": max_pae,
            "n_residues_total": float(n_total),
            "n_residues_binder": float(n_binder),
            "n_residues_target": float(n_target),
            "n_interface_residues_predicted": float(n_interface),
        }

        return Score(
            example_id=example.id,
            scorer_name=self.name,
            value=iptm,
            extras=extras,
        )
