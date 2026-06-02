"""Read the ESM embedding cache and build per-rung paired feature matrices.

numpy-only — mirage never imports torch. The cache is produced by the separate
``scripts/embed_sequences.py`` and keyed on the raw (normalized) sequence
string, so reader and writer agree without hashing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_embedding_cache(npy_path: Path, keys_path: Path) -> dict[str, np.ndarray[Any, Any]]:
    """Map each cached sequence to its embedding row."""
    keys = [k for k in keys_path.read_text().splitlines() if k]
    arr = np.load(npy_path)
    if arr.shape[0] != len(keys):
        raise ValueError(f"cache mismatch: {arr.shape[0]} embeddings vs {len(keys)} keys")
    return {k: arr[i] for i, k in enumerate(keys)}


def paired_matrix(
    pairs: Sequence[tuple[str, str]],
    cache: dict[str, np.ndarray[Any, Any]],
    *,
    layout: str,
) -> np.ndarray[Any, Any]:
    """Build a feature matrix for (binder_seq, antigen_seq) pairs.

    ``layout="concat"`` -> ``[e_ab | e_ag]`` (2d-wide; used by the additive
    logistic and split back into halves by the bilinear model).
    ``layout="hadamard"`` -> ``e_ab * e_ag`` (d-wide; diagonal bilinear).
    Raises KeyError if a sequence is absent from the cache (embed it first)."""
    rows: list[np.ndarray[Any, Any]] = []
    for binder, antigen in pairs:
        e_b = cache[binder]
        e_g = cache[antigen]
        if layout == "concat":
            rows.append(np.concatenate([e_b, e_g]))
        elif layout == "hadamard":
            rows.append(e_b * e_g)
        else:
            raise ValueError(f"unknown layout: {layout!r}")
    return np.asarray(rows, dtype=float)
