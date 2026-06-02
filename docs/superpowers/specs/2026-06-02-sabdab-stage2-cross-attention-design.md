# SAbDab stage-2 sequence-only: cross-attention over per-residue embeddings — design

**Date:** 2026-06-02
**Status:** approved (design); implementation pending
**Branch:** `sabdab-sequence-baseline` (continues the baseline work)

## Problem

The stage-1 ladder established a rigorous sequence-only floor: under a
leakage-controlled held-out-antigen-cluster split, every rung — including a
low-rank bilinear on mean-pooled ESM-2 650M embeddings — sits at **AUROC ≈ 0.50**
(the frozen bilinear: 0.496). The random-split contrast confirmed this is a
genuine absence of signal, not a held-out-antigen artifact.

But every stage-1 rung used **mean-pooled** embeddings. Pooling collapses the
per-residue structure, destroying any CDR-epitope **contact** information — the
one thing a binding model arguably needs. The committed "next rung" is a model
that operates on **per-residue** embeddings and can learn which binder residues
attend to which antigen residues. This is the bounded stage-2 shot: build it,
evaluate it on the *same* split, and report whether it clears the floor.

## Goal

Train two models over **frozen** per-residue ESM-2 650M embeddings of the SAbDab
VHH–antigen pairs and evaluate them **out-of-fold under the same
held-out-antigen-cluster split** as stage 1, reporting AUROC + bootstrap CIs
against the 0.496 bilinear floor:

1. **Cross-attention head** — binder residues cross-attend to antigen residues
   (the model that can use contact structure).
2. **Pooled-MLP ablation** — mean-pool both chains, feed `[B | A | B⊙A]` to an
   MLP. Isolates "attention captured contacts" from "just a deeper nonlinear
   head with an interaction term."

**Decision rule:** none pre-registered — run it, report AUROC + CIs for both
rungs vs the floor, and decide afterward whether sequence-only is worth pursuing
or the floor is confirmed and we move to M-C.

**Non-goals.** No PLM fine-tuning (unfreezing ESM-2 would overfit 448 positives
and is not "bounded"). No new dataset, split, or negative scheme — reuse
`data/staged/sabdab/sabdab_pairs.csv` exactly. No hyperparameter sweep — one or
two seeds, fixed architecture.

## Locked decisions

| Axis | Decision |
|------|----------|
| Models | Cross-attention head **and** pooled-MLP ablation (both rungs). |
| Embeddings | **Frozen** per-residue ESM-2 650M (`esm2_t33_650M_UR50D`). No fine-tuning. |
| Split / eval | Same held-out-antigen-cluster 5-fold OOF (the `fold` column), same gate metrics + bootstrap CIs, compared to the 0.496 bilinear. |
| Torch boundary | All torch lives in `scripts/` run in the `esm` conda env. The `mirage` package + its tests stay numpy-only. Torch produces **OOF scores**; numpy computes the **metrics** with the existing `eval/gate.py`. |
| Decision rule | Just report AUROC + CIs; decide after. |

## Design

### 1. Per-residue embeddings — `scripts/embed_perresidue.py` (esm env, GPU)

Standalone (torch imported lazily). Reads the existing
`data/staged/sabdab/sabdab_unique_seqs.txt` manifest (844 seqs), runs ESM-2 650M,
and stores the **per-residue** final-layer representations (not pooled). For
chains > 1022 aa it windows as in `embed_sequences.py` but **concatenates** the
per-window per-residue outputs to reconstruct the full `[L, 1280]` array.
Multi-chain antigens (`":"`-joined) concatenate per-chain residue blocks.

Output: a ragged cache `data/staged/sabdab/perres.npz` (one float16 array
`"<i>"` of shape `[L_i, 1280]` per sequence) + `perres_keys.txt` (the sequence
strings in index order). ~460 MB, gitignored.

Reuses the pure window helper `iter_windows` (copied locally — the script stays
standalone, no mirage import in the esm env).

### 2. OOF training — `scripts/train_stage2.py` (esm env, torch)

Standalone, torch imported lazily so its pure helpers are importable for tests.

**Pure helpers (unit-tested via importlib in the mirage env):**
- `load_perres(npz_path, keys_path) -> dict[str, np.ndarray]` — ragged cache reader.
- `oof_folds(folds: np.ndarray) -> Iterator[tuple[test_mask, train_mask]]` — yields
  the held-out / train boolean masks per unique fold value.

**Models (torch; smoke-tested in the esm env, not CI):**
- **Cross-attention** (`XAttnGate`): `Linear(1280→128)` projections for binder and
  antigen (separate), then `nn.MultiheadAttention(embed_dim=128, num_heads=4,
  batch_first=True)` with query = projected binder residues, key/value =
  projected antigen residues, `key_padding_mask` for antigen padding;
  masked-mean-pool the attended binder representation over binder residues →
  `Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)` → logit. A `LayerNorm`
  after each projection handles embedding scale.
- **Pooled-MLP** (`PooledMLP`): input `[B_pool | A_pool | B_pool⊙A_pool]`
  (3×1280, from the **existing** pooled cache `embeddings.npy`, standardized on
  train), `Linear(3840→128) → ReLU → Dropout(0.3) → Linear(128→1)` → logit.

**Training:** per fold, train on the other 4 folds with `BCEWithLogitsLoss`,
`Adam(lr=1e-3, weight_decay=1e-2)`, a fixed 40 epochs with early stop on a 10%
train-validation slice (patience 5), minibatches with antigen-length padding +
masks. `torch.manual_seed(seed)`. Predict the held-out fold; collect OOF logits.

Output: `data/staged/sabdab/stage2_oof_scores.csv` with columns
`pair_id, label, fold, score_xattn, score_mlp`. Run via
`scripts/slurm/train_stage2.slurm` (dgx-b200).

### 3. Metrics — `scripts/analyze_stage2.py` (mirage env, numpy)

Reads `stage2_oof_scores.csv`; for each of `score_xattn` and `score_mlp` computes
`auroc`, `choose_threshold_for_precision` + `summary_dict` (gate metrics + PPV
sweep) and a bootstrap CI on AUROC (reusing `eval/gate.py`). Reads the stage-1
`results/published/sabdab_baseline.json` to carry the 0.496 bilinear reference.
Writes `results/published/sabdab_stage2.json` and prints the comparison.

### 4. Summary — `results/published/sabdab_baseline_summary.md`

Add a "Stage 2 — per-residue cross-attention" subsection: cross-attention +
pooled-MLP AUROCs (with CIs) vs the bilinear floor, and the read (cleared the
floor → pursue; ≈ floor → confirmed, move to M-C).

## Wiring summary

| Path | Change |
|------|--------|
| `scripts/embed_perresidue.py` | NEW — per-residue ESM-2 cache (esm env) |
| `scripts/train_stage2.py` | NEW — XAttn + PooledMLP OOF trainer (esm env, torch) |
| `scripts/slurm/train_stage2.slurm` | NEW — SLURM wrapper |
| `scripts/analyze_stage2.py` | NEW — OOF-scores → gate metrics (mirage env) |
| `results/published/sabdab_stage2.json` | NEW — results |
| `results/published/sabdab_baseline_summary.md` | EDIT — add Stage-2 subsection |
| `tests/test_stage2_helpers.py` | NEW — `load_perres`, `oof_folds` via importlib |
| `tests/test_analyze_stage2.py` | NEW — OOF→metrics path |

Reused: `eval/gate.py` (all metrics), the existing pooled `embeddings.npy` +
`keys.txt` (pooled-MLP), `sabdab_pairs.csv` (pairs + folds), `sabdab_baseline.json`
(floor reference), the `esm` conda env, the cached ESM-2 weights.

## Testing & CI

- `tests/test_stage2_helpers.py` (mirage env): load a tiny synthetic ragged
  `perres.npz` → `load_perres` round-trips shapes/values; `oof_folds` yields
  disjoint, exhaustive train/test masks per fold. Loaded via importlib (torch is
  lazy, so import succeeds without torch).
- `tests/test_analyze_stage2.py` (mirage env): a synthetic `stage2_oof_scores.csv`
  → `analyze_stage2` produces a result dict whose `auroc` matches
  `eval.gate.auroc` on the same scores; structure (per-rung metrics + CIs + floor
  ref) is correct.
- **Torch smoke test** (esm env, documented, not CI): `XAttnGate` and `PooledMLP`
  run a forward pass on random padded tensors and overfit a tiny planted-attention
  dataset to AUROC > 0.9 (sanity that the models can learn when signal exists).
- Full mirage battery stays green: `uv run ruff check`, `ruff format --check`,
  `mypy src/mirage`, `pytest` (the new scripts are in `scripts/`, not mypy-scoped;
  the torch-lazy imports keep test collection torch-free).

## Verification (end-to-end)

1. `sbatch` per-residue embed → `perres.npz` + `perres_keys.txt` covering all 844 seqs.
2. `sbatch` train_stage2 → `stage2_oof_scores.csv` with finite scores for both rungs over all 2,688 rows.
3. `analyze_stage2.py` → `sabdab_stage2.json` with AUROC + CIs for both rungs and the floor reference; prints the comparison.
4. Smoke test confirms both models learn a planted signal (so a chance result on real data is a real finding, not a dead model — the same guard stage 1 used).

## Risks / assumptions

- **Overfitting** (448 positives, 164k-param projections): mitigated by frozen
  embeddings, small `d`, dropout, weight decay, early stopping. The OOF split is
  the honest check.
- **A chance result is an acceptable, expected outcome** — it confirms the floor
  and the smoke test proves the models aren't dead. Report plainly.
- **Per-residue storage** (~460 MB ragged float16): gitignored; only the 844
  SAbDab uniques (AVIDa not needed for the in-distribution stage-2 comparison).
- **Torch-free invariant**: `embed_perresidue.py` and `train_stage2.py` are the
  only torch-touching code; they live in `scripts/` and run in the `esm` env, so
  `src/mirage/` stays numpy-only.
- **Commits authored by Pedram, no Claude/Anthropic trailer.**
