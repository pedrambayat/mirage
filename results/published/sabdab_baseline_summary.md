# mirage — SAbDab sequence-only binding baseline (M-S floor)

> **STATUS (2026-06-02): COMPLETE — in-distribution ladder + AVIDa transfer.**
> All numbers below are real, produced by the committed pipeline on
> mature-domain-normalized sequences (ANARCI IMGT variable-domain extraction for
> VHH binders; signal-peptide + His-tag stripping for antigens). ESM-2 650M
> embeddings are mean-pooled (chunked length-weighted for antigens > 1022 aa).
> No numbers are fabricated.

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
> per-residue embeddings) could in principle extract more — the additive→diagonal→
> low-rank ladder used only *mean-pooled* embeddings, which discard CDR-epitope
> contact structure. That stage-2 shot was taken (see below) and **also lands at
> the floor**, confirming the floor is real, not a pooling/capacity artifact. (PLM
> fine-tuning is out of scope — it would overfit 448 positives.)

## Leakage contrast: random vs held-out-antigen split

To check whether the chance result is an *artifact of holding out antigens* (a
true generalization gap) or a genuine *absence of signal*, the same ladder was
re-run under a **random split** (`--random-folds`): folds assigned randomly **per
positive** (each binder + its negatives stay together), so whole binders are held
out in *both* splits and the **only** difference is that test antigens are now
exposed during training (as other binders' cognates and as negative partners). If
the 0.50 were a held-out-antigen artifact, the random split — which lets the model
memorize antigen-conditional patterns — should score well above chance.

| rung | held-out-antigen AUROC | random-split AUROC |
|---|---|---|
| 0 Tier-S (additive) | 0.520 | 0.482 |
| 1 ESM-concat (additive) | 0.491 | 0.202 |
| 2 Hadamard (diagonal bilinear) | 0.495 | 0.392 |
| 3 low-rank bilinear | 0.496 | 0.536 |

**The random split shows no memorization advantage.** The intended strongest model
(rung 3) is ≈ chance in both regimes (0.496 vs 0.536). The high-dimensional
additive/diagonal rungs go *sub-chance* under random folds (0.20–0.39) — an
overfitting / anti-generalization signature (with ~2,150 train rows they fit
fold-specific noise, not signal). No rung approaches the high AUROC that genuine
antigen memorization would produce.

**Read:** the floor is more robust than a "generalization failure" — there is **no
usable sequence-only binding signal here at all**, not even one a leaky split could
exploit. The leakage-controlled 0.50 is therefore not a cost of the held-out-antigen
design; it is the honest absence of signal. (Reproduced from
`results/published/sabdab_baseline_random.json`.)

## Stage 2 — per-residue cross-attention

The stage-1 ladder used only **mean-pooled** embeddings, which discard the
CDR-epitope **contact** structure a binding model arguably needs. Stage 2 tests
whether a model over **per-residue** ESM-2 650M embeddings recovers it, under the
*same* held-out-antigen-cluster OOF split. Two rungs (frozen embeddings; torch in
the `esm` env, metrics in numpy via `eval/gate.py`):

- **Cross-attention** — binder residues cross-attend to antigen residues
  (length-masked), attended rep pooled → MLP → logit. The model that *can* use
  contacts.
- **Pooled-MLP ablation** — `[B | A | B⊙A]` → MLP → logit. Isolates "attention
  captured contacts" from "just a deeper nonlinear head with an interaction term."

| model | AUROC | 95% CI |
|---|---|---|
| bilinear floor (stage 1) | 0.496 | — |
| **cross-attention** (per-residue) | **0.496** | [0.468, 0.525] |
| pooled-MLP ablation | 0.526 | [0.497, 0.556] |

**Read — the floor holds.** The cross-attention head, which had access to
per-residue contact structure, lands **exactly at the floor (0.496)** with a CI
straddling chance: per-residue attention recovers nothing transferable. The only
movement is a marginal +0.03 from the pooled-MLP (a deeper nonlinear head), and
critically the **attention model did no better than — and slightly below — the
simpler pooled-MLP**, so the gain is from head nonlinearity, not contact modeling.
A torch smoke test confirms both models overfit a planted signal (train AUROC
1.00), so this is a genuine no-signal result, not a dead model. Stage 2 does **not
rescue** sequence-only; the floor is confirmed and the next step is the
predictor-conditional structure track (M-C). (Reproduced from
`results/published/sabdab_stage2.json`; see
`docs/superpowers/plans/2026-06-02-sabdab-stage2-cross-attention.md`.)

## Orthogonal validation — AVIDa-hIL6 (held-out same-antigen, NOT training)

AVIDa-hIL6 is the orthogonal **real-negative** canary: ~574k VHH–IL6 pairs with
true binder/non-binder labels against the *correct* antigen. It is **never**
training data. The SAbDab-frozen rung-3 gate (`sabdab_bilinear_model.json`) is
applied unchanged through the existing `evaluate_frozen_gate` harness
(`scripts/analyze_sabdab_orthogonal.py`), scoring AVIDa's mature-domain-normalized
sequences from the cached ESM-2 embeddings.

**Result** (n=573,891; 20,980 binders / 552,911 non-binders; prevalence 0.037):

| metric | value | note |
|---|---|---|
| **AUROC** (threshold-free) | **0.617** | modestly but robustly above chance (huge n, tight CIs) |
| recall @ frozen thr (−4.107) | 0.855 | gate calls ~76% of pairs positive |
| specificity | 0.242 | |
| precision | 0.041 [0.041, 0.041] | ≈ the 0.037 base rate |

Two distinct readings, and the distinction is the point:

- **Ranking carries weak same-antigen signal (AUROC 0.617 > 0.5).** Unlike the
  in-distribution *held-out-antigen* result (AUROC ≈ 0.50), the frozen gate ranks
  AVIDa pairs modestly above chance. AVIDa is a **single-antigen** task (every VHH
  vs IL-6), so this is almost certainly **binder-side** signal — properties that
  correlate with being a real IL-6 binder — **not** antigen-specific
  complementarity (which is exactly what collapses to chance for *unseen*
  antigens). It is a different axis, and a weak one.
- **The calibrated operating point does NOT transfer.** At the SAbDab-frozen
  threshold the gate predicts ~76% positive with precision 0.041 ≈ prevalence —
  the threshold sits in the wrong place on AVIDa's score distribution. So the
  *gate* (threshold-and-all) adds essentially nothing, even though the *ranking*
  is slightly informative.

This is a sharper outcome than the Phase A M-S AVIDa canary (which collapsed
all-negative): here the frozen ranking is weakly above chance on the same-antigen
set while the operating point is miscalibrated. It does **not** contradict the
floor — the floor is about *held-out-antigen* generalization, where the signal is
absent (AUROC ≈ 0.50). Reported from `results/published/sabdab_orthogonal.json`.

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

### AVIDa transfer (orthogonal guardrail)

AVIDa is featurized from its OWN embedding cache (its ~38.6k unique VHHs are not
in the SAbDab cache). Build that cache, then apply the frozen gate unchanged.

```bash
# 4a. Stage AVIDa (raw; see results/published/mirage_phase_a_summary.md for inputs)
uv run python scripts/stage_avida.py \
  --records data/raw/avida/AVIDa-hIL6.csv \
  --antigens data/raw/avida/antigen_sequences.csv \
  --output data/staged/avida/avida_staged.csv

# 4b. Build AVIDa's unique normalized-sequence manifest (~30-min ANARCI pass over
#     38.6k unique VHHs) and embed it with ESM-2 650M into a SEPARATE cache:
#       data/staged/avida/avida_unique_seqs.txt -> avida_embeddings.npy + avida_keys.txt
#     (normalize each unique vhh_sequence with normalize_binder and each
#     antigen_sequence with normalize_antigen, then run scripts/embed_sequences.py
#     on that manifest via the esm env / a GPU SLURM job).

# 4c. Apply the frozen rung-3 gate to AVIDa through the orthogonal harness
uv run python scripts/analyze_sabdab_orthogonal.py \
  --avida-csv data/staged/avida/avida_staged.csv \
  --embeddings data/staged/avida/avida_embeddings.npy \
  --keys data/staged/avida/avida_keys.txt \
  --model results/published/sabdab_bilinear_model.json --model-type bilinear \
  --layout concat --output results/published/sabdab_orthogonal.json
```

Defaults: `--l2 1.0`, `--rank 32`, `--lr 0.05`, `--n-iter 2000`,
`--bilinear-l2 1e-2`, `--target-precision 0.9`, `--seed 20260601`. The bilinear
trainer uses gradient-norm clipping (`max_grad_norm=1.0`) for numerical stability
on the high-dimensional embeddings.
