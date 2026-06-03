"""ProtenixConfidenceScorer — Protenix predictor confidence internals.

Extracts iPTM, pTM, global pLDDT, and token-level interface PAE / pLDDT from
Protenix's two per-sample output JSONs:

- ``*summary_confidence*sample_0*.json`` — scalars + per-chain arrays.
- ``*full_data*sample_0*.json`` — token/atom arrays (PAE, pLDDT, contact_probs).

This is a **pure-JSON extractor**: it never loads the structure file and does
not depend on any chain resolver. The headline ``value`` is iPTM.

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
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score

# Tokens whose max cross-chain contact probability meets or exceeds this
# threshold are counted as interface tokens for the interface pLDDT aggregate.
INTERFACE_CONTACT_PROB: float = 0.5


# ---------------------------------------------------------------------------
# Pure reduction helpers (testable without I/O)
# ---------------------------------------------------------------------------


def interface_pae_stats(
    token_pair_pae: np.ndarray[Any, Any],
    token_asym_id: np.ndarray[Any, Any],
) -> tuple[float, float]:
    """Return (mean_interface_pae, min_interface_pae) over all cross-chain token pairs.

    A pair (i, j) is cross-chain when ``token_asym_id[i] != token_asym_id[j]``.
    Both (i→j) and (j→i) entries are included (PAE is asymmetric).
    Returns ``(nan, nan)`` when there are no cross-chain pairs.

    Parameters
    ----------
    token_pair_pae:
        Shape (N_token, N_token), float Å. Token-level PAE matrix.
    token_asym_id:
        Shape (N_token,), int. Chain index per token (0 = first chain, etc.).
    """
    asym = np.asarray(token_asym_id)
    # Boolean mask: True for all (i, j) where chains differ
    mask = asym[:, None] != asym[None, :]  # (N, N)
    if not mask.any():
        return float("nan"), float("nan")
    pae = np.asarray(token_pair_pae, dtype=float)
    cross_values = pae[mask]
    return float(cross_values.mean()), float(cross_values.min())


def interface_plddt_value(
    atom_plddt: np.ndarray[Any, Any],
    atom_to_token_idx: np.ndarray[Any, Any],
    contact_probs: np.ndarray[Any, Any],
    token_asym_id: np.ndarray[Any, Any],
) -> float:
    """Return mean per-token pLDDT (0-100 scale) over interface tokens.

    A token is an *interface token* if its maximum cross-chain contact
    probability (to any token of a different chain) is >=
    ``INTERFACE_CONTACT_PROB``.  Per-token pLDDT is computed by averaging
    ``atom_plddt`` (0-1 scale) over atoms assigned to that token, then
    multiplying by 100.

    Returns ``nan`` when no interface tokens exist (including when all
    tokens belong to the same chain).

    Parameters
    ----------
    atom_plddt:
        Shape (N_atom,), float 0-1. Per-atom pLDDT from Protenix full-data.
    atom_to_token_idx:
        Shape (N_atom,), int. Maps each atom to its token index.
    contact_probs:
        Shape (N_token, N_token), float 0-1. Predicted contact probabilities.
    token_asym_id:
        Shape (N_token,), int. Chain index per token.
    """
    atom_plddt_arr = np.asarray(atom_plddt, dtype=float)
    atom_to_tok = np.asarray(atom_to_token_idx, dtype=int)
    cp = np.asarray(contact_probs, dtype=float)
    asym = np.asarray(token_asym_id, dtype=int)

    n_tokens = len(asym)

    # --- per-token pLDDT (0-1, then scaled to 0-100) ---
    token_plddt = np.full(n_tokens, float("nan"))
    for t in range(n_tokens):
        atom_mask = atom_to_tok == t
        if atom_mask.any():
            token_plddt[t] = atom_plddt_arr[atom_mask].mean()

    # --- cross-chain contact mask ---
    cross_chain_mask = asym[:, None] != asym[None, :]  # (N, N)
    if not cross_chain_mask.any():
        return float("nan")

    # max cross-chain contact probability for each token
    cp_cross = cp.copy()
    cp_cross[~cross_chain_mask] = 0.0  # zero out same-chain entries
    max_cross_cp = cp_cross.max(axis=1)  # (N_token,)

    interface_tokens = max_cross_cp >= INTERFACE_CONTACT_PROB
    if not interface_tokens.any():
        return float("nan")

    return float(token_plddt[interface_tokens].mean() * 100.0)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@register("protenix_confidence")
class ProtenixConfidenceScorer(AbstractScorer):
    """Protenix's own confidence metrics for a predicted complex.

    Headline ``value`` is iPTM. Extras carry pTM, global pLDDT, and
    token-level interface PAE / pLDDT aggregates derived from the full-data
    JSON.

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
            token_asym_id = np.asarray(full_data["token_asym_id"], dtype=int)
            atom_plddt = np.asarray(full_data["atom_plddt"], dtype=float)
            atom_to_token = np.asarray(full_data["atom_to_token_idx"], dtype=int)
            contact_probs = np.asarray(full_data["contact_probs"], dtype=float)

            interface_pae, min_interface_pae = interface_pae_stats(token_pair_pae, token_asym_id)
            interface_plddt = interface_plddt_value(
                atom_plddt, atom_to_token, contact_probs, token_asym_id
            )

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
