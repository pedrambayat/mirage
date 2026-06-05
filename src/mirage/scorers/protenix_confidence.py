"""ProtenixConfidenceScorer — Protenix predictor confidence internals.

Extracts iPTM, pTM, global pLDDT, and token-level interface PAE / pLDDT from
Protenix's two per-sample output JSONs:

- ``*summary_confidence*sample_0*.json`` — scalars + per-chain arrays.
- ``*full_data*sample_0*.json`` — token/atom arrays (PAE, pLDDT, contact_probs).

Interface features are computed over **binder x antigen** token pairs only,
correctly excluding antigen-antigen pairs when the antigen spans multiple chains.
The binder chain(s) are identified by loading the predicted structure and
resolving chain roles via sequence identity (or a ``chain_roles.json`` sidecar).

Output layout per example (under ``predictions_root/<example_id>/``):
  Flat:   ``<id>/<files>``
  Nested: ``<id>/seed_0/predictions/<files>``
Both layouts are discovered via recursive glob.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage._paths import default_protenix_predictions_root
from mirage.scorers._registry import register
from mirage.scorers._structure import (
    load_structure,
    read_chain_roles_json,
    resolve_chain_roles_by_sequence,
)
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score

# Tokens whose max cross-role contact probability meets or exceeds this
# threshold are counted as interface tokens for the interface pLDDT aggregate.
INTERFACE_CONTACT_PROB: float = 0.5


# ---------------------------------------------------------------------------
# Pure reduction helpers (testable without I/O)
# ---------------------------------------------------------------------------


def interface_pae_stats(
    token_pair_pae: np.ndarray[Any, Any],
    binder_token_mask: np.ndarray[Any, Any],
) -> tuple[float, float]:
    """Return (mean_interface_pae, min_interface_pae) over binder x antigen token pairs.

    A pair (i, j) contributes when exactly one of i, j is a binder token
    (``binder_token_mask[i] != binder_token_mask[j]``), i.e. binder→antigen
    and antigen→binder directions are both included (PAE is asymmetric).
    Returns ``(nan, nan)`` when there are no such pairs.

    Parameters
    ----------
    token_pair_pae:
        Shape (N_token, N_token), float Å. Token-level PAE matrix.
    binder_token_mask:
        Shape (N_token,), bool. True for binder tokens, False for antigen tokens.
    """
    mask_b = np.asarray(binder_token_mask, dtype=bool)
    # True for (i, j) where exactly one is a binder token
    interface_mask = mask_b[:, None] != mask_b[None, :]  # (N, N)
    if not interface_mask.any():
        return float("nan"), float("nan")
    pae = np.asarray(token_pair_pae, dtype=float)
    values = pae[interface_mask]
    return float(values.mean()), float(values.min())


def interface_plddt_value(
    atom_plddt: np.ndarray[Any, Any],
    atom_to_token_idx: np.ndarray[Any, Any],
    contact_probs: np.ndarray[Any, Any],
    binder_token_mask: np.ndarray[Any, Any],
) -> float:
    """Return mean per-token pLDDT (0-100 scale) over interface tokens.

    A token is an *interface token* if its maximum contact probability to a
    token of the **opposite role** (binder↔antigen, via ``binder_token_mask``)
    is >= ``INTERFACE_CONTACT_PROB``.  Per-token pLDDT is computed by averaging
    ``atom_plddt`` (0-1 scale) over atoms assigned to that token, then
    multiplying by 100.

    Returns ``nan`` when no interface tokens exist (including when all tokens
    have the same role).

    Parameters
    ----------
    atom_plddt:
        Shape (N_atom,), float 0-1. Per-atom pLDDT from Protenix full-data.
    atom_to_token_idx:
        Shape (N_atom,), int. Maps each atom to its token index.
    contact_probs:
        Shape (N_token, N_token), float 0-1. Predicted contact probabilities.
    binder_token_mask:
        Shape (N_token,), bool. True for binder tokens, False for antigen tokens.
    """
    atom_plddt_arr = np.asarray(atom_plddt, dtype=float)
    atom_to_tok = np.asarray(atom_to_token_idx, dtype=int)
    cp = np.asarray(contact_probs, dtype=float)
    mask_b = np.asarray(binder_token_mask, dtype=bool)

    n_tokens = len(mask_b)

    # --- per-token pLDDT (0-1, then scaled to 0-100) ---
    token_plddt = np.full(n_tokens, float("nan"))
    for t in range(n_tokens):
        atom_mask = atom_to_tok == t
        if atom_mask.any():
            token_plddt[t] = atom_plddt_arr[atom_mask].mean()

    # --- cross-role contact mask (binder↔antigen) ---
    cross_role_mask = mask_b[:, None] != mask_b[None, :]  # (N, N)
    if not cross_role_mask.any():
        return float("nan")

    # max cross-role contact probability for each token
    cp_cross = cp.copy()
    cp_cross[~cross_role_mask] = 0.0  # zero out same-role entries
    max_cross_cp = cp_cross.max(axis=1)  # (N_token,)

    interface_tokens = max_cross_cp >= INTERFACE_CONTACT_PROB
    if not interface_tokens.any():
        return float("nan")

    return float(token_plddt[interface_tokens].mean() * 100.0)


def build_binder_token_mask(
    structure: Any,
    binder_chain_ids: list[str],
    atom_to_token_idx: np.ndarray[Any, Any],
    n_tokens: int,
) -> np.ndarray[Any, Any] | None:
    """Build a boolean mask of length ``n_tokens`` marking binder tokens.

    Iterates atoms in the same order as the Protenix CIF output convention:
    for each chain, for each standard residue (res.id[0] == " "), for each
    atom in that residue.  This matches the order of ``atom_to_token_idx``.

    Returns ``None`` (caller should treat interface features as NaN) when:
    - the number of atoms iterated does not match ``len(atom_to_token_idx)``, or
    - ``max(atom_to_token_idx) >= n_tokens`` (inconsistent sizes).

    Parameters
    ----------
    structure:
        Biopython ``Structure`` object (model 0 is used).
    binder_chain_ids:
        Chain IDs (in the predicted structure) that belong to the binder.
    atom_to_token_idx:
        Array mapping each atom (in CIF order) to its token index.
    n_tokens:
        Total number of tokens (length of token-level arrays).
    """
    a2t = np.asarray(atom_to_token_idx, dtype=int)

    # Basic size sanity: if declared n_tokens is smaller than the highest token
    # index in the mapping, the arrays are inconsistent.
    if a2t.size > 0 and int(a2t.max()) >= n_tokens:
        return None

    # Build per-atom chain-id list following the verified CIF atom order.
    atom_chain_ids: list[str] = []
    for chain in structure[0]:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for _atom in residue:
                atom_chain_ids.append(chain.id)

    if len(atom_chain_ids) != len(a2t):
        return None

    binder_set = set(binder_chain_ids)
    # A token is a binder token if ANY atom that maps to it is from a binder chain.
    token_is_binder = np.zeros(n_tokens, dtype=bool)
    for k, tok_idx in enumerate(a2t):
        if atom_chain_ids[k] in binder_set:
            token_is_binder[tok_idx] = True

    return token_is_binder


# ---------------------------------------------------------------------------
# Structure discovery helper (shared with geometry scorer logic)
# ---------------------------------------------------------------------------


def _find_structure(pred_dir: Path) -> Path | None:
    """Locate a predicted structure file under ``pred_dir``.

    Search order: ``rank1.cif``, ``rank1.pdb``, first ``*sample_0*.cif``
    found recursively (covers nested ``seed_0/predictions/`` layout and flat
    fixture layout).
    """
    for name in ("rank1.cif", "rank1.pdb"):
        candidate = pred_dir / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(pred_dir.rglob("*sample_0*.cif")):
        return candidate
    return None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@register("protenix_confidence")
class ProtenixConfidenceScorer(AbstractScorer):
    """Protenix's own confidence metrics for a predicted complex.

    Headline ``value`` is iPTM. Extras carry pTM, global pLDDT, and
    token-level interface PAE / pLDDT aggregates derived from the full-data
    JSON.  Interface features are restricted to binder x antigen token pairs
    so multi-chain antigens do not pollute the interface statistics.

    File discovery uses recursive globs so it handles both the flat fixture
    layout (``predictions_root/<id>/<files>``) and the nested campaign layout
    (``predictions_root/<id>/seed_0/predictions/<files>``).
    """

    name = "protenix_confidence"

    def __init__(self, predictions_root: str | Path | None = None) -> None:
        if predictions_root is None:
            predictions_root = default_protenix_predictions_root()
        self.predictions_root = Path(predictions_root)

    def score(self, example: BenchmarkExample) -> Score:
        try:
            return self._score_impl(example)
        except Exception as exc:
            return self.nan_score(example, error=f"{type(exc).__name__}: {exc}"[:200])

    def _score_impl(self, example: BenchmarkExample) -> Score:
        pred_dir = self.predictions_root / example.id

        # Discover summary file
        summary_matches = list(pred_dir.glob("**/*summary_confidence*sample_0*.json"))
        if not summary_matches:
            return self.nan_score(example, missing="prediction")
        summary_path = summary_matches[0]

        summary = json.loads(summary_path.read_text())
        iptm = float(summary["iptm"])
        ptm = float(summary["ptm"])
        mean_plddt = float(summary["plddt"])  # 0-100 scale (global)

        # Discover full-data file (optional)
        interface_pae: float = float("nan")
        min_interface_pae: float = float("nan")
        interface_plddt: float = float("nan")

        full_matches = list(pred_dir.glob("**/*full_data*sample_0*.json"))
        if full_matches:
            full_data = json.loads(full_matches[0].read_text())

            token_pair_pae = np.asarray(full_data["token_pair_pae"], dtype=float)
            atom_plddt = np.asarray(full_data["atom_plddt"], dtype=float)
            atom_to_token = np.asarray(full_data["atom_to_token_idx"], dtype=int)
            contact_probs = np.asarray(full_data["contact_probs"], dtype=float)

            n_tokens = token_pair_pae.shape[0]

            # Build binder token mask from structure + chain role resolution
            btmask: np.ndarray[Any, Any] | None = None
            structure_path = _find_structure(pred_dir)
            if structure_path is not None:
                struct = load_structure(structure_path)
                roles = read_chain_roles_json(structure_path.parent / "chain_roles.json")
                if roles is not None:
                    binder_ids, _target_ids = roles
                else:
                    binder_ids, _target_ids = resolve_chain_roles_by_sequence(
                        struct,
                        binder_seqs=example.binder_chains,
                        target_seqs=example.target_chains,
                    )
                btmask = build_binder_token_mask(struct, binder_ids, atom_to_token, n_tokens)

            if btmask is not None:
                interface_pae, min_interface_pae = interface_pae_stats(token_pair_pae, btmask)
                interface_plddt = interface_plddt_value(
                    atom_plddt, atom_to_token, contact_probs, btmask
                )
            # else: structure missing / size mismatch → leave interface features as NaN

        extras: dict[str, float | str] = {
            "iptm": iptm,
            "ptm": ptm,
            "mean_plddt": mean_plddt,
            "interface_pae": interface_pae,
            "min_interface_pae": min_interface_pae,
            "interface_plddt": interface_plddt,
        }

        return Score(
            example_id=example.id,
            scorer_name=self.name,
            value=iptm,
            extras=extras,
        )
