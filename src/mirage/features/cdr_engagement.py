"""CDR-engagement interface features for predicted antibody-antigen complexes.

Given the predicted binder chain's sequence and the set of binder residue indices
that form the interface, this module reports what fraction of those interface
residues fall in the CDR loops (by IMGT numbering via ANARCI).

**Row-preserving contract:** on any failure (HMMER unavailable, ANARCI import
error, no domain found, empty filtered interface) the function returns ``_defaults()``
— every feature is 0.0 — and sets ``cdr_mapping_ok = 0.0``. It **never raises**.
This ensures that a downstream row assembler can always obtain a full feature
vector without dropping the pair.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mirage.features.normalize import _resolve_hmmer_bin

CDR_FEATURE_NAMES: tuple[str, ...] = (
    "cdr_contact_fraction",  # fraction of binder interface residues in ANY CDR
    "cdr1_contact_fraction",
    "cdr2_contact_fraction",
    "cdr3_contact_fraction",
    "cdr_mapping_ok",  # 1.0 if ANARCI mapped the chain, else 0.0
)

# IMGT CDR ranges (inclusive).
_CDR1_LO, _CDR1_HI = 27, 38
_CDR2_LO, _CDR2_HI = 56, 65
_CDR3_LO, _CDR3_HI = 105, 117


def _defaults() -> dict[str, float]:
    return {name: 0.0 for name in CDR_FEATURE_NAMES}


def _imgt_cdr_masks(
    binder_seq: str,
    n_residues: int,
) -> dict[str, npt.NDArray[np.bool_]] | None:
    """Build per-residue CDR boolean masks using ANARCI IMGT numbering.

    Returns a dict with keys ``"any"``, ``"cdr1"``, ``"cdr2"``, ``"cdr3"``
    (each a boolean array of length *n_residues*), or ``None`` on any failure.
    Residues that lie outside the numbered variable domain (e.g. tags, linkers)
    are False.
    """
    hmmer_bin = _resolve_hmmer_bin()
    if hmmer_bin is None:
        return None

    try:
        from anarci import run_anarci  # type: ignore[import-untyped]

        result = run_anarci([("q", binder_seq)], scheme="imgt", hmmerpath=hmmer_bin)
    except Exception:
        return None

    # result[1][0] is the list of numbered domains for sequence 0.
    domains = result[1][0]
    if not domains:
        return None

    domain_numbering, dom_start, _dom_end = domains[0]

    # Walk the numbered alignment, mapping each non-gap entry to its query
    # residue index and IMGT position.
    mask_cdr1 = np.zeros(n_residues, dtype=bool)
    mask_cdr2 = np.zeros(n_residues, dtype=bool)
    mask_cdr3 = np.zeros(n_residues, dtype=bool)

    qi = dom_start  # 0-based index into binder_seq
    for (imgt_pos, _ins), aa in domain_numbering:
        if aa == "-":
            # Gap in the query — do not advance qi.
            continue
        if 0 <= qi < n_residues:
            if _CDR1_LO <= imgt_pos <= _CDR1_HI:
                mask_cdr1[qi] = True
            elif _CDR2_LO <= imgt_pos <= _CDR2_HI:
                mask_cdr2[qi] = True
            elif _CDR3_LO <= imgt_pos <= _CDR3_HI:
                mask_cdr3[qi] = True
        qi += 1

    mask_any = mask_cdr1 | mask_cdr2 | mask_cdr3
    return {"any": mask_any, "cdr1": mask_cdr1, "cdr2": mask_cdr2, "cdr3": mask_cdr3}


def cdr_engagement_features(
    binder_seq: str,
    binder_interface_residue_indices: npt.NDArray[Any],
    n_binder_residues: int,
) -> dict[str, float]:
    """Compute CDR-engagement fractions for the binder interface.

    Parameters
    ----------
    binder_seq:
        Full sequence of the predicted binder chain (1-letter AA codes).
    binder_interface_residue_indices:
        0-based indices into the predicted binder chain identifying which residues
        form the interface.  Indices ``>= n_binder_residues`` are silently ignored.
    n_binder_residues:
        Length of the binder chain (= ``len(binder_seq)`` in normal usage;
        provided explicitly so callers that already have the integer avoid a
        redundant ``len()`` call and for clarity in the row-assembler contract).

    Returns
    -------
    dict[str, float]
        Keys are exactly ``CDR_FEATURE_NAMES``.  On any failure returns all-zero
        defaults with ``cdr_mapping_ok = 0.0`` and **never raises**.
    """
    masks = _imgt_cdr_masks(binder_seq, n_binder_residues)
    if masks is None:
        return _defaults()

    # Restrict interface to valid indices.
    iface = binder_interface_residue_indices[binder_interface_residue_indices < n_binder_residues]
    if iface.size == 0:
        return _defaults()

    return {
        "cdr_contact_fraction": float(masks["any"][iface].mean()),
        "cdr1_contact_fraction": float(masks["cdr1"][iface].mean()),
        "cdr2_contact_fraction": float(masks["cdr2"][iface].mean()),
        "cdr3_contact_fraction": float(masks["cdr3"][iface].mean()),
        "cdr_mapping_ok": 1.0,
    }
