# SAbDab Stage-2 Cross-Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a model over *per-residue* ESM-2 650M embeddings (cross-attention) beats the 0.496 pooled-bilinear floor under the same held-out-antigen-cluster OOF split, alongside a pooled-MLP ablation. Report AUROC + CIs; decide after.

**Architecture:** Frozen per-residue ESM-2 embeddings (new cache) → a torch OOF trainer (esm env) for two models, cross-attention + pooled-MLP, writing OOF logits to a CSV → a numpy metrics script (mirage env) that reuses `eval/gate.py`. Torch lives only in `scripts/` run in the `esm` env; `src/mirage/` stays numpy-only.

**Tech Stack:** Python 3.11; the `esm` conda env (torch 2.12 + fair-esm) for the embedding + training scripts; the mirage uv env (numpy) for metrics + tests. Reuses `eval/gate.py`, the pooled `embeddings.npy`, `sabdab_pairs.csv`, `sabdab_baseline.json`.

**Spec:** `docs/superpowers/specs/2026-06-02-sabdab-stage2-cross-attention-design.md`

**Conventions:** TDD where CI can run it (the pure helpers + the numpy metrics). Torch scripts import torch **lazily** inside functions so their pure helpers import without torch; the torch models are validated by a `--smoke` mode run with the `esm` python (CPU), not CI. `scripts/` is not mypy-scoped. **Commits authored by Pedram — never add a Claude/Anthropic trailer.** Branch `sabdab-sequence-baseline`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/embed_perresidue.py` | NEW — per-residue ESM-2 650M cache (esm env, torch lazy) |
| `scripts/train_stage2.py` | NEW — pure helpers (`load_perres`, `oof_folds`) + torch models (`XAttnGate`, `PooledMLP`) + OOF trainer + `--smoke` (esm env) |
| `scripts/slurm/train_stage2.slurm` | NEW — SLURM wrapper for the two GPU jobs |
| `scripts/analyze_stage2.py` | NEW — OOF scores → gate metrics + CIs (mirage env, numpy) |
| `tests/test_stage2_helpers.py` | NEW — `load_perres`, `oof_folds` (importlib, mirage env) |
| `tests/test_analyze_stage2.py` | NEW — `analyze_scores` path (mirage env) |
| `results/published/sabdab_stage2.json` | NEW — results |
| `results/published/sabdab_baseline_summary.md` | EDIT — add Stage-2 subsection (after runs) |

---

## Task 1: Per-residue embedding script

**Files:** Create `scripts/embed_perresidue.py`, `tests/test_perres_windows.py`.

- [ ] **Step 1: Write the failing test `tests/test_perres_windows.py`**

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "embed_perresidue", REPO / "scripts" / "embed_perresidue.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embed_perresidue"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_iter_windows_non_overlapping():
    mod = _load()
    assert mod.iter_windows(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert mod.iter_windows(3, 1022) == [(0, 3)]
```

- [ ] **Step 2: Run `uv run pytest tests/test_perres_windows.py -v`** — expect FAIL (module missing).

- [ ] **Step 3: Write `scripts/embed_perresidue.py`**

```python
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


def _per_residue_one(seq, model, batch_converter, device):  # noqa: ANN001
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
```

- [ ] **Step 4: Verify** — `uv run pytest tests/test_perres_windows.py -v && uv run ruff check scripts/embed_perresidue.py tests/test_perres_windows.py && uv run ruff format scripts/embed_perresidue.py tests/test_perres_windows.py`. Expect 2 pass, ruff clean. Do NOT run main (needs torch+GPU).

- [ ] **Step 5: Commit**

```bash
git add scripts/embed_perresidue.py tests/test_perres_windows.py
git commit -m "Add per-residue ESM-2 650M embedding script (stage 2)"
```

---

## Task 2: OOF trainer (pure helpers + torch models + smoke)

**Files:** Create `scripts/train_stage2.py`, `tests/test_stage2_helpers.py`.

- [ ] **Step 1: Write the failing test `tests/test_stage2_helpers.py`**

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_stage2", REPO / "scripts" / "train_stage2.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_stage2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_load_perres_roundtrip(tmp_path):
    mod = _load()
    a = np.arange(6, dtype=np.float16).reshape(3, 2)
    b = np.arange(4, dtype=np.float16).reshape(2, 2)
    np.savez(tmp_path / "p.npz", **{"0": a, "1": b})
    (tmp_path / "k.txt").write_text("AAA\nCC\n")
    cache = mod.load_perres(tmp_path / "p.npz", tmp_path / "k.txt")
    assert set(cache) == {"AAA", "CC"}
    assert cache["AAA"].shape == (3, 2) and np.allclose(cache["CC"], b)


def test_oof_folds_disjoint_and_exhaustive():
    mod = _load()
    folds = np.array([0, 0, 1, 2, 2, 1])
    seen = np.zeros(6, dtype=bool)
    n = 0
    for test, train in mod.oof_folds(folds):
        assert not (test & train).any()        # disjoint
        assert (test | train).all()             # exhaustive
        seen |= test
        n += 1
    assert n == 3 and seen.all()                # each row held out exactly once
```

- [ ] **Step 2: Run `uv run pytest tests/test_stage2_helpers.py -v`** — expect FAIL (module missing).

- [ ] **Step 3: Write `scripts/train_stage2.py`**

```python
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


def _make_models(seed: int):  # noqa: ANN201
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

        def forward(self, binder, antigen, bmask, amask):  # noqa: ANN001
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

        def forward(self, feat):  # noqa: ANN001
            return self.head(feat).squeeze(-1)

    return XAttnGate(), PooledMLP()


def _pad_batch(arrs, device):  # noqa: ANN001
    import torch

    lengths = [a.shape[0] for a in arrs]
    lmax = max(lengths)
    out = torch.zeros(len(arrs), lmax, arrs[0].shape[1], dtype=torch.float32, device=device)
    mask = torch.zeros(len(arrs), lmax, dtype=torch.bool, device=device)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = torch.from_numpy(a.astype(np.float32)).to(device)
        mask[i, : a.shape[0]] = True
    return out, mask


def _train_xattn(model, binders, antigens, y, idx_train, idx_test, device, seed):  # noqa: ANN001
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


def _train_mlp(model, feats, y, idx_train, idx_test, device, seed):  # noqa: ANN001
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
    import torch

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
```

- [ ] **Step 4: Verify pure helpers (mirage env)** — `uv run pytest tests/test_stage2_helpers.py -v && uv run ruff check scripts/train_stage2.py tests/test_stage2_helpers.py && uv run ruff format scripts/train_stage2.py tests/test_stage2_helpers.py`. Expect 2 pass, ruff clean.

- [ ] **Step 5: Run the torch smoke test (esm env, CPU)** — `~/miniconda3/envs/esm/bin/python scripts/train_stage2.py --smoke`. Expected: prints `smoke train AUROC: xattn=… mlp=…` both > 0.9 and `SMOKE OK` (proves both models + the training loop learn when signal exists). If it fails, the models/training are broken — fix before proceeding.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_stage2.py tests/test_stage2_helpers.py
git commit -m "Add stage-2 OOF trainer: cross-attention + pooled-MLP (esm env)"
```

---

## Task 3: Metrics from OOF scores (mirage env)

**Files:** Create `scripts/analyze_stage2.py`, `tests/test_analyze_stage2.py`.

- [ ] **Step 1: Write the failing test `tests/test_analyze_stage2.py`**

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from mirage.eval.gate import auroc

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_stage2", REPO / "scripts" / "analyze_stage2.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_stage2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_analyze_scores_matches_gate_auroc():
    mod = _load()
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = (scores + rng.normal(scale=0.5, size=200) > 0).astype(int)
    out = mod.analyze_scores(scores, labels, target_precision=0.9, seed=1)
    assert abs(out["auroc"] - auroc(scores, labels)) < 1e-9
    assert "metrics" in out and "auroc_ci" in out
    assert out["auroc_ci"][0] <= out["auroc"] <= out["auroc_ci"][1]
```

- [ ] **Step 2: Run `uv run pytest tests/test_analyze_stage2.py -v`** — expect FAIL (module missing).

- [ ] **Step 3: Write `scripts/analyze_stage2.py`**

```python
"""Compute gate metrics for the stage-2 OOF scores and compare to the floor.

Reads the OOF scores CSV produced by scripts/train_stage2.py (in the esm env)
and computes, per rung, AUROC + a fixed-precision operating point + PPV sweep +
a bootstrap CI on AUROC, reusing mirage's numpy eval/gate.py. Carries the stage-1
bilinear floor (rung3 AUROC) for the head-to-head.

Use:
    uv run python scripts/analyze_stage2.py \\
      --scores data/staged/sabdab/stage2_oof_scores.csv \\
      --baseline results/published/sabdab_baseline.json \\
      --output results/published/sabdab_stage2.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.gate import auroc, bootstrap_ci, choose_threshold_for_precision, summary_dict


def analyze_scores(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    target_precision: float,
    seed: int,
) -> dict[str, Any]:
    finite = np.isfinite(scores)
    s, y = scores[finite], labels[finite]
    thr = choose_threshold_for_precision(s, y, target_precision=target_precision)
    out = summary_dict(s, y, threshold=thr)
    out["auroc"] = auroc(s, y)
    out["auroc_ci"] = bootstrap_ci(auroc, s, y, n_boot=1000, seed=seed)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    with args.scores.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows], dtype=int)

    result: dict[str, Any] = {"n": int(y.size), "n_positive": int((y == 1).sum())}
    for rung in ("score_xattn", "score_mlp"):
        s = np.array([float(r[rung]) for r in rows], dtype=float)
        result[rung] = analyze_scores(s, y, target_precision=args.target_precision, seed=args.seed)

    floor = json.loads(args.baseline.read_text())
    result["bilinear_floor_auroc"] = floor["rung3_bilinear"]["auroc"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for rung in ("score_xattn", "score_mlp"):
        m = result[rung]
        print(f"{rung}: AUROC={m['auroc']:.3f} CI={tuple(round(v, 3) for v in m['auroc_ci'])}")
    print(f"bilinear floor AUROC={result['bilinear_floor_auroc']:.3f}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify** — `uv run pytest tests/test_analyze_stage2.py -v && uv run ruff check scripts/analyze_stage2.py tests/test_analyze_stage2.py && uv run ruff format scripts/analyze_stage2.py tests/test_analyze_stage2.py`. Expect 1 pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_stage2.py tests/test_analyze_stage2.py
git commit -m "Add stage-2 OOF-scores metrics analysis (mirage env)"
```

---

## Task 4: SLURM wrapper

**Files:** Create `scripts/slurm/train_stage2.slurm`.

- [ ] **Step 1: Write `scripts/slurm/train_stage2.slurm`**

```bash
#!/bin/bash
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=stage2
#SBATCH --output=slurm-stage2-%j.out
set -euo pipefail
ESM_PY="$HOME/miniconda3/envs/esm/bin/python"
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
"$ESM_PY" scripts/train_stage2.py \
  --pairs data/staged/sabdab/sabdab_pairs.csv \
  --perres data/staged/sabdab/perres.npz --perres-keys data/staged/sabdab/perres_keys.txt \
  --pooled data/staged/sabdab/embeddings.npy --pooled-keys data/staged/sabdab/keys.txt \
  --out data/staged/sabdab/stage2_oof_scores.csv --seed 20260601
```

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/slurm/train_stage2.slurm
git add scripts/slurm/train_stage2.slurm
git commit -m "Add stage-2 training SLURM wrapper"
```

---

## Task 5: Run on PARCC + write the result (controller)

These steps run the GPU jobs and fill the summary with real numbers — executed by the controller, not a code subagent.

- [ ] **Step 1: Per-residue embeddings** — pre-cache is already present (ESM-2 weights). Submit:
  `sbatch --wrap="$HOME/miniconda3/envs/esm/bin/python scripts/embed_perresidue.py --manifest data/staged/sabdab/sabdab_unique_seqs.txt --out-npz data/staged/sabdab/perres.npz --out-keys data/staged/sabdab/perres_keys.txt"` with the dgx-b200 SBATCH flags. Verify `perres.npz` covers all 844 seqs.

- [ ] **Step 2: Train OOF** — `sbatch scripts/slurm/train_stage2.slurm`. Verify `stage2_oof_scores.csv` has 2,688 rows with finite `score_xattn` / `score_mlp`.

- [ ] **Step 3: Metrics** — `uv run python scripts/analyze_stage2.py --scores data/staged/sabdab/stage2_oof_scores.csv --baseline results/published/sabdab_baseline.json --output results/published/sabdab_stage2.json`.

- [ ] **Step 4: Summary** — add a "## Stage 2 — per-residue cross-attention" subsection to `results/published/sabdab_baseline_summary.md`: a table of cross-attention + pooled-MLP AUROC [CI] vs the 0.496 bilinear floor, and the read (cleared the floor → pursue sequence-only; ≈ floor → confirmed, move to M-C). Commit `results/published/sabdab_stage2.json` + the summary edit.

---

## Verification (end-to-end)

- `uv run pytest -p no:warnings -q`, `uv run ruff check`, `uv run mypy src/mirage` — all green (new code is in `scripts/`; pure helpers tested via importlib; torch never imported in the mirage env).
- The `--smoke` run shows both models overfit a tiny set (train AUROC > 0.9) — so a chance OOF result on real data is a genuine finding, not a dead model.
- `sabdab_stage2.json` reports cross-attention + pooled-MLP OOF AUROC with CIs and the bilinear floor reference.
- Decision: if either rung's AUROC clears the 0.496 floor with a CI clearly above it, sequence-only is worth pursuing; otherwise the floor is confirmed and the next step is M-C.

## Self-review notes (resolved)

- **Spec coverage:** per-residue cache (T1), cross-attention + pooled-MLP OOF trainer with smoke (T2), numpy metrics reusing gate.py (T3), SLURM (T4), runs + summary (T5). All spec sections map to a task.
- **Type/name consistency:** `load_perres(npz, keys)`, `oof_folds(folds)`, `analyze_scores(scores, labels, *, target_precision, seed)`, OOF CSV columns `pair_id,label,fold,score_xattn,score_mlp`, and the `_train_xattn`/`_train_mlp` signatures are consistent across T2/T3/T5.
- **Deviation from spec, noted:** training uses a fixed 40 epochs (no early-stop validation slice) — simpler, deterministic, and adequate given dropout + weight decay for a bounded shot. If overfitting is severe in the OOF result, early stopping is the first knob to add.
- **Placeholders:** none — all steps carry complete code. The torch models are validated by the `--smoke` run (T2 Step 5), the only non-CI check.
