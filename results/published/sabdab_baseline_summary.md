# mirage — SAbDab sequence-only binding baseline (M-S floor)

> **STATUS (2026-06-02): in-distribution ladder COMPLETE.** All numbers below are
> real, produced by the committed pipeline on mature-domain-normalized SAbDab
> sequences (ANARCI IMGT variable-domain extraction for VHH binders;
> signal-peptide + His-tag stripping for antigens). ESM-2 650M embeddings are
> mean-pooled (chunked length-weighted for antigens > 1022 aa). No numbers are
> fabricated. The AVIDa held-out transfer (orthogonal guardrail) is computed
> separately; see that section.

**Caveat up front:** M-S is the *pre-structure baseline*. It does not consume
predictor confidence (ipTM / PAE / pLDDT) and so does **not** answer mirage's
headline question — that is M-C (Phase B). This baseline establishes the
strongest *sequence-only* reference floor we can build, replacing the weak
Champloo strawman (91 cognate positives, AUROC 0.346) with a well-powered,
leakage-controlled SAbDab measurement.

## Dataset & protocol

- **Source:** SAbDab protein–antigen complexes, **VHH-only** (`Lchain == NA`),
  via the existing `SAbDabLoader` (resolution ≤ 3.0 Å, protein antigen ≥ 30 aa,
  ANARCI-resolvable heavy chain, k-mer-Jaccard binder dedup at ~90% identity).
- **Positives:** **448** cognate (VHH, antigen) pairs across **241** antigen
  sequence-identity clusters (one unique VHH per positive after dedup).
- **Negatives:** **2,240** shuffled (VHH × wrong antigen) pairs — k = 5 per
  positive, drawn **cross-cluster**, **fold-consistent**, and
  **distribution-matched** to the per-fold positive antigen-cluster marginal
  (so antigen popularity carries no label signal and the model is forced onto
  the interaction). Total **2,688** rows.
- **Split:** grouped 5-fold by **antigen cluster** (whole clusters held out
  together — spike/HA/GPCR variants cannot straddle a fold). In-distribution
  numbers are **out-of-fold** (OOF) under this leakage control.
- **Operating point:** target precision 0.90 (none of the rungs reach it, so
  each row reports its best-precision fallback operating point — the gate
  collapses to predict-almost-everything-negative, as expected at chance).

## In-distribution head-to-head (OOF, held-out antigen clusters) — n=2,688, 448 positives

| rung | model | AUROC | recall | specificity | precision | threshold |
|---|---|---|---|---|---|---|
| 0 | Tier-S (13 bulk-composition feats), additive | **0.520** | 0.038 | 0.975 | 0.236 | −1.421 |
| 1 | ESM-2 concat `[e_ab \| e_ag]`, additive | **0.491** | 0.004 | 0.998 | 0.286 | 3.482 |
| 2 | ESM-2 diagonal bilinear `e_ab ⊙ e_ag` | **0.495** | 0.058 | 0.953 | 0.197 | 11.400 |
| 3 | ESM-2 low-rank bilinear `(P_a e_ab)·(P_g e_ag)` (rank 32) | **0.496** | 0.013 | 0.996 | 0.375 | 15.398 |

PPV at deployed prevalence (frozen rung 3): 1:100 → 0.029, 1:1,000 → 0.0030,
1:10,000 → 0.0003. (PPV ≈ prevalence at every rarity ⇒ the gate adds essentially
nothing — consistent with AUROC ≈ chance.)

## Read

- **Every rung sits at chance (AUROC 0.49–0.52) under the held-out-antigen-cluster
  split** — including ESM-2 650M embeddings *with* an explicit interaction term.
- **The interaction rungs (2, 3) do not beat the additive rungs (0, 1).** The
  bilinear models *can* express antigen-specific compatibility (and provably learn
  a planted interaction in unit tests), yet on real held-out antigens they recover
  no signal beyond chance. So the collapse is not the "additive-is-chance" artifact
  of the Champloo floor — it is a genuine **generalization** failure: pooled
  sequence features carry **no transferable antigen-specific binding signal** when
  the test antigens are entirely unseen clusters.
- This is the **expected, defensible, well-powered floor.** It is a real
  replacement for the 0.346 Champloo strawman: 5× the positives, 241 antigen
  clusters, proper cluster holdout, and the strongest sequence-only model we set
  out to build. A sequence-only gate is non-discriminative for held-out antigens —
  which is precisely the motivation for the predictor-conditional structure track
  (M-C, Phase B).

> **Note on "strongest":** within sequence-only, stage 2 (cross-attention over
> per-residue embeddings, PLM fine-tuning) could in principle extract more, and is
> the deferred next rung. But the additive→diagonal→low-rank ladder already shows
> that adding interaction capacity buys *nothing* on held-out antigens here, so the
> burden of proof is on a much richer model to beat chance — a strong prior that the
> floor is real, not a capacity artifact.

## Orthogonal validation — AVIDa-hIL6 (held-out same-antigen, NOT training)

AVIDa-hIL6 is the orthogonal **real-negative** canary: ~574k VHH–IL6 pairs with
true binder/non-binder labels against the *correct* antigen. It is **never**
training data. The SAbDab-frozen rung-3 gate (`sabdab_bilinear_model.json`) is
applied unchanged through the existing `evaluate_frozen_gate` harness
(`scripts/analyze_sabdab_orthogonal.py`), scoring AVIDa's mature-domain-normalized
sequences from the cached ESM-2 embeddings.

_Transfer numbers pending — computed in a separate run (38.6k unique AVIDa VHHs
require an ANARCI normalization pass + ESM embedding). This row will be filled from
`results/published/sabdab_orthogonal.json`. Expected behavior, given the
in-distribution chance result: a degenerate near-all-negative transfer (the gate
carries no signal to export), mirroring the Phase A M-S AVIDa canary._

## How to reproduce

All inputs are in `../abdisc-data/sabdab/`. Binder normalization needs the local
HMMER for ANARCI (`bash scripts/install_hmmer.sh`, installs to `.tools/hmmer/`).

```bash
# 1. Stage positives + leakage-safe negatives + antigen-cluster folds
uv run python scripts/stage_sabdab_pairs.py \
  --data-dir ../abdisc-data/sabdab \
  --output data/staged/sabdab/sabdab_pairs.csv \
  --manifest data/staged/sabdab/sabdab_unique_seqs.txt

# 2. Embed the 844 unique sequences with ESM-2 650M (separate `esm` env, GPU)
sbatch scripts/slurm/embed_esm.slurm   # -> data/staged/sabdab/embeddings.npy + keys.txt

# 3. Train the four-rung ladder, OOF under antigen-cluster folds, freeze the best
uv run python scripts/analyze_sabdab_baseline.py \
  --pairs data/staged/sabdab/sabdab_pairs.csv \
  --embeddings data/staged/sabdab/embeddings.npy \
  --keys data/staged/sabdab/keys.txt \
  --output results/published/sabdab_baseline.json \
  --model-out results/published/sabdab_bilinear_model.json
```

Defaults: `--l2 1.0`, `--rank 32`, `--lr 0.05`, `--n-iter 2000`,
`--bilinear-l2 1e-2`, `--target-precision 0.9`, `--seed 20260601`. The bilinear
trainer uses gradient-norm clipping (`max_grad_norm=1.0`) for numerical stability
on the high-dimensional embeddings.
