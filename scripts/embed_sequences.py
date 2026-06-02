"""Embed unique sequences with ESM-2 650M and cache mean-pooled vectors.

Runs in a SEPARATE env (torch + fair-esm), NOT the mirage uv env. Reads a
manifest (one normalized sequence per line; multi-chain antigens are
':'-joined), mean-pools the final-layer per-residue representations per chain
(chunking sequences longer than the model context into non-overlapping windows,
length-weighted), averages chains length-weighted, and writes ``embeddings.npy``
(N x 1280, float32) plus ``keys.txt`` (the manifest lines, in row order). The
reader keys on the raw sequence string, so no hashing is involved.

Use (inside the esm env)::

    python scripts/embed_sequences.py \\
        --manifest data/staged/sabdab/sabdab_unique_seqs.txt \\
        --out-embeddings data/staged/sabdab/embeddings.npy \\
        --out-keys data/staged/sabdab/keys.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_MAX_LEN = 1022  # ESM-2 context (1024 minus BOS/EOS)


def iter_windows(n: int, max_len: int = _MAX_LEN) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) windows covering [0, n)."""
    if n <= 0:
        return [(0, 0)]
    return [(s, min(s + max_len, n)) for s in range(0, n, max_len)]


def _embed_one(seq, model, alphabet, batch_converter, device):
    import torch

    vecs: list[np.ndarray] = []
    lengths: list[int] = []
    for chain in seq.split(":"):
        if not chain:
            continue
        for start, end in iter_windows(len(chain)):
            window = chain[start:end]
            _, _, toks = batch_converter([("q", window)])
            toks = toks.to(device)
            with torch.no_grad():
                out = model(toks, repr_layers=[33], return_contacts=False)
            rep = out["representations"][33][0, 1 : len(window) + 1].mean(0)
            vecs.append(rep.float().cpu().numpy())
            lengths.append(len(window))
    weights = np.array(lengths, dtype=float)
    return np.average(np.stack(vecs), axis=0, weights=weights).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-embeddings", type=Path, required=True)
    parser.add_argument("--out-keys", type=Path, required=True)
    args = parser.parse_args()

    import esm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    seqs = [s for s in args.manifest.read_text().splitlines() if s]
    embeddings = np.zeros((len(seqs), 1280), dtype=np.float32)
    for i, seq in enumerate(seqs):
        embeddings[i] = _embed_one(seq, model, alphabet, batch_converter, device)
        if (i + 1) % 200 == 0:
            print(f"embedded {i + 1}/{len(seqs)}", flush=True)

    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.out_keys.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_embeddings, embeddings)
    args.out_keys.write_text("\n".join(seqs) + "\n")
    print(f"Wrote {embeddings.shape} to {args.out_embeddings} and keys to {args.out_keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
