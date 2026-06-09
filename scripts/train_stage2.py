"""Stage-2 OOF trainer: cross-attention + pooled-MLP over frozen ESM-2 650M
embeddings, under the held-out-antigen-cluster split. Runs in the `esm` env.

Torch is imported lazily so the pure helpers (load_perres, oof_folds) import
without torch for mirage-env unit tests. Writes per-rung OOF logits to a CSV;
metrics are computed by scripts/analyze_stage2.py in the mirage env.

Use (esm env):
    python scripts/train_stage2.py \\
      --pairs data/staged/sabdab/sabdab_pairs.csv \\
      --perres data/staged/sabdab/perres.npz --perres-keys data/staged/sabdab/perres_keys.txt \\
      --pooled data/staged/sabdab/embeddings.npy --pooled-keys data/staged/sabdab/keys.txt \\
      --out data/staged/sabdab/stage2_oof_scores.csv --seed 20260601
Smoke (esm env, CPU):
    python scripts/train_stage2.py --smoke
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_EPOCHS = 40
_BATCH = 32
_LR = 1e-3
_WD = 1e-2


def load_perres(npz_path: Path, keys_path: Path) -> dict[str, np.ndarray]:
    keys = [k for k in Path(keys_path).read_text().splitlines() if k]
    data = np.load(npz_path)
    return {k: data[str(i)] for i, k in enumerate(keys)}


def oof_folds(folds: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (test_mask, train_mask) per unique fold value."""
    for f in np.unique(folds):
        test = folds == f
        yield test, ~test


def _make_models(seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class XAttnGate(nn.Module):
        def __init__(self, d_in: int = 1280, d: int = 128, heads: int = 4) -> None:
            super().__init__()
            self.proj_b = nn.Linear(d_in, d)
            self.proj_a = nn.Linear(d_in, d)
            self.ln_b = nn.LayerNorm(d)
            self.ln_a = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
            )

        def forward(self, binder, antigen, bmask, amask):
            qb = self.ln_b(self.proj_b(binder))
            ka = self.ln_a(self.proj_a(antigen))
            out, _ = self.attn(qb, ka, ka, key_padding_mask=~amask)
            m = bmask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
            return self.head(pooled).squeeze(-1)

    class PooledMLP(nn.Module):
        def __init__(self, d_in: int = 1280) -> None:
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(3 * d_in, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
            )

        def forward(self, feat):
            return self.head(feat).squeeze(-1)

    return XAttnGate(), PooledMLP()


def _pad_batch(arrs, device):
    import torch

    lengths = [a.shape[0] for a in arrs]
    lmax = max(lengths)
    out = torch.zeros(len(arrs), lmax, arrs[0].shape[1], dtype=torch.float32, device=device)
    mask = torch.zeros(len(arrs), lmax, dtype=torch.bool, device=device)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = torch.from_numpy(a.astype(np.float32)).to(device)
        mask[i, : a.shape[0]] = True
    return out, mask


def _train_xattn(model, binders, antigens, y, idx_train, idx_test, device, seed):
    import torch
    from torch import nn

    rng = np.random.default_rng(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=_LR, weight_decay=_WD)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    for _ in range(_EPOCHS):
        model.train()
        order = rng.permutation(idx_train)
        for s in range(0, len(order), _BATCH):
            bi = order[s : s + _BATCH]
            b, bm = _pad_batch([binders[i] for i in bi], device)
            a, am = _pad_batch([antigens[i] for i in bi], device)
            opt.zero_grad()
            logits = model(b, a, bm, am)
            loss = lossf(logits, yt[bi])
            loss.backward()
            opt.step()
    model.eval()
    scores = np.full(y.shape, np.nan, dtype=float)
    with torch.no_grad():
        for s in range(0, len(idx_test), _BATCH):
            bi = idx_test[s : s + _BATCH]
            b, bm = _pad_batch([binders[i] for i in bi], device)
            a, am = _pad_batch([antigens[i] for i in bi], device)
            scores[bi] = model(b, a, bm, am).cpu().numpy()
    return scores


def _train_mlp(model, feats, y, idx_train, idx_test, device, seed):
    import torch
    from torch import nn

    rng = np.random.default_rng(seed)
    mean = feats[idx_train].mean(0)
    std = feats[idx_train].std(0)
    std[std == 0] = 1.0
    fz = torch.from_numpy(((feats - mean) / std).astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=_LR, weight_decay=_WD)
    lossf = nn.BCEWithLogitsLoss()
    tr = np.asarray(idx_train)
    for _ in range(_EPOCHS):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), _BATCH):
            bi = order[s : s + _BATCH]
            opt.zero_grad()
            loss = lossf(model(fz[bi]), yt[bi])
            loss.backward()
            opt.step()
    model.eval()
    scores = np.full(y.shape, np.nan, dtype=float)
    with torch.no_grad():
        scores[idx_test] = model(fz[idx_test]).cpu().numpy()
    return scores


def _read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def run(args: argparse.Namespace) -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = _read_pairs(args.pairs)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    folds = np.array([int(r["fold"]) for r in rows], dtype=int)

    perres = load_perres(args.perres, args.perres_keys)
    binders = [perres[r["binder_seq"]] for r in rows]
    antigens = [perres[r["antigen_seq"]] for r in rows]

    pooled_keys = [k for k in args.pooled_keys.read_text().splitlines() if k]
    pooled_arr = np.load(args.pooled)
    pooled = {k: pooled_arr[i] for i, k in enumerate(pooled_keys)}
    bp = np.stack([pooled[r["binder_seq"]] for r in rows])
    ap = np.stack([pooled[r["antigen_seq"]] for r in rows])
    feats = np.concatenate([bp, ap, bp * ap], axis=1)

    sx = np.full(y.shape, np.nan, dtype=float)
    sm = np.full(y.shape, np.nan, dtype=float)
    for test, train in oof_folds(folds):
        idx_train = np.flatnonzero(train)
        idx_test = np.flatnonzero(test)
        xm, mm = _make_models(args.seed)
        sx[idx_test] = _train_xattn(
            xm, binders, antigens, y, idx_train, idx_test, device, args.seed
        )[idx_test]
        sm[idx_test] = _train_mlp(mm, feats, y, idx_train, idx_test, device, args.seed)[idx_test]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "label", "fold", "score_xattn", "score_mlp"])
        for i, r in enumerate(rows):
            w.writerow([r["pair_id"], int(y[i]), int(folds[i]), float(sx[i]), float(sm[i])])
    print(f"Wrote {len(rows)} OOF rows to {args.out}")
    return 0


def smoke() -> int:
    """Both models must overfit a tiny random set (sanity: training works)."""
    import torch  # noqa: F401

    rng = np.random.default_rng(0)
    n, d = 64, 1280
    y = rng.integers(0, 2, n)
    binders = [rng.normal(size=(rng.integers(8, 16), d)).astype(np.float16) for _ in range(n)]
    antigens = [rng.normal(size=(rng.integers(20, 40), d)).astype(np.float16) for _ in range(n)]
    bp = np.stack([b.mean(0) for b in binders]).astype(float)
    ap = np.stack([a.mean(0) for a in antigens]).astype(float)
    feats = np.concatenate([bp, ap, bp * ap], axis=1)
    idx = np.arange(n)
    xm, mm = _make_models(0)
    sx = _train_xattn(xm, binders, antigens, y, idx, idx, "cpu", 0)
    sm = _train_mlp(mm, feats, y, idx, idx, "cpu", 0)

    def _auroc(s, yy):
        pos, neg = s[yy == 1], s[yy == 0]
        return float((pos[:, None] > neg[None, :]).mean())

    ax, am = _auroc(sx, y), _auroc(sm, y)
    print(f"smoke train AUROC: xattn={ax:.3f} mlp={am:.3f}")
    assert ax > 0.9 and am > 0.9, "a model failed to overfit — training is broken"
    print("SMOKE OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--perres", type=Path)
    parser.add_argument("--perres-keys", type=Path)
    parser.add_argument("--pooled", type=Path)
    parser.add_argument("--pooled-keys", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
