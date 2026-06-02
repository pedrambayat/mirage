from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mirage.features.embeddings import load_embedding_cache, paired_matrix


def _write_cache(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    keys = ["AAAA", "CCCC", "DDDD"]
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    npy = tmp_path / "emb.npy"
    kf = tmp_path / "keys.txt"
    np.save(npy, arr)
    kf.write_text("\n".join(keys) + "\n")
    return npy, kf, {k: arr[i] for i, k in enumerate(keys)}


def test_load_roundtrip(tmp_path: Path):
    npy, kf, expected = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    assert set(cache) == set(expected)
    assert np.allclose(cache["CCCC"], [3.0, 4.0])


def test_paired_matrix_concat(tmp_path: Path):
    npy, kf, _ = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    x = paired_matrix([("AAAA", "CCCC")], cache, layout="concat")
    assert x.shape == (1, 4)
    assert np.allclose(x[0], [1.0, 2.0, 3.0, 4.0])


def test_paired_matrix_hadamard(tmp_path: Path):
    npy, kf, _ = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    x = paired_matrix([("AAAA", "CCCC")], cache, layout="hadamard")
    assert np.allclose(x[0], [1.0 * 3.0, 2.0 * 4.0])


def test_missing_key_raises(tmp_path: Path):
    npy, kf, _ = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    with pytest.raises(KeyError):
        paired_matrix([("AAAA", "ZZZZ")], cache, layout="concat")
