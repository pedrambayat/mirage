"""Per-residue ESM-2 650M embeddings for the SAbDab unique sequences.

Runs in the `esm` env (torch). Unlike scripts/embed_sequences.py (which
mean-pools), this stores the FULL per-residue final-layer representations,
concatenating per-window outputs for chains longer than the 1022 context and
per-chain blocks for ':'-joined multi-chain antigens. Output is a ragged cache:
``perres.npz`` (one float16 array ``"<i>"`` of shape [L_i, 1280] per sequence)
plus ``perres_keys.txt`` (sequences in index order).

Use (esm env):
    python scripts/embed_perresidue.py \\
      --manifest data/staged/sabdab/sabdab_unique_seqs.txt \\
      --out-npz data/staged/sabdab/perres.npz \\
      --out-keys data/staged/sabdab/perres_keys.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_MAX_LEN = 1022


def iter_windows(n: int, max_len: int = _MAX_LEN) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) windows covering [0, n)."""
    if n <= 0:
        return [(0, 0)]
    return [(s, min(s + max_len, n)) for s in range(0, n, max_len)]


def _per_residue_one(seq, model, batch_converter, device):
    import torch

    blocks: list[np.ndarray] = []
    for chain in seq.split(":"):
        if not chain:
            continue
        for start, end in iter_windows(len(chain)):
            window = chain[start:end]
            _, _, toks = batch_converter([("q", window)])
            with torch.no_grad():
                out = model(toks.to(device), repr_layers=[33], return_contacts=False)
            rep = out["representations"][33][0, 1 : len(window) + 1]  # drop BOS/EOS
            blocks.append(rep.float().cpu().numpy().astype(np.float16))
    return np.concatenate(blocks, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-keys", type=Path, required=True)
    args = parser.parse_args()

    import esm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    seqs = [s for s in args.manifest.read_text().splitlines() if s]
    arrays: dict[str, np.ndarray] = {}
    for i, seq in enumerate(seqs):
        arrays[str(i)] = _per_residue_one(seq, model, batch_converter, device)
        if (i + 1) % 100 == 0:
            print(f"embedded {i + 1}/{len(seqs)}", flush=True)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.out_keys.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **arrays)
    args.out_keys.write_text("\n".join(seqs) + "\n")
    print(f"Wrote {len(arrays)} ragged per-residue arrays to {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
