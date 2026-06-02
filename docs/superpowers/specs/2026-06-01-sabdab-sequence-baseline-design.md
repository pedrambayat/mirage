# SAbDab sequence-only binding baseline (mirage M-S floor) — design

**Date:** 2026-06-01
**Status:** approved (design); implementation pending
**Branch:** `sabdab-sequence-baseline`

## Problem

The Phase A M-S sequence-only gate scores **AUROC 0.346** in-distribution
(`results/published/mirage_phase_a_summary.md`). That number is a weak strawman
for two compounding reasons:

1. **Additive features.** The gate is the numpy L2-logistic in `ml/core.py` over
   the 13 **bulk-composition** Tier-S features in `features/sequence.py`
   (per-chain length, net charge, aromatic/hydrophobic/polar/cysteine fractions,
   plus length ratio). For a fixed VHH, its cognate pair and its shuffled pairs
   share **identical binder composition** and differ only in *which* antigen is
   on the other side. Under negatives that are distribution-matched on the
   antigen marginal, a purely additive model
   `score = w_a·e_ab + w_g·e_ag + b` has `E[score|pos] = E[score|neg]` and is
   provably ~chance. The interaction term is **mandatory**.
2. **Tiny training set.** Champloo supplies only **91 unique cognate
   positives** (the released ipTM matrices are keyed at `pdb_id` granularity).

SAbDab is the obvious remedy: ~3,021 VHH rows (`Lchain == NA`), 6,147 PDB
structures already on disk under `abdisc-data/sabdab/sabdab_dataset/`, and
1,978 unique antigen names. Sequences are recoverable from the chothia PDBs by
the **already-built** `SAbDabLoader` (`src/mirage/benchmark/sabdab.py`) — no
co-crystal structure parsing is required for a sequence-only baseline.

## Goal

Build the **strongest sequence-only VHH–antigen binding discriminator** trainable
on SAbDab cognate pairs, evaluated under a **leakage-controlled held-out-antigen
split**, as a rigorous floor for the M-S gate. Two outcomes are useful:

- **(a)** Clear the 0.346 strawman with a model that actually uses the
  interaction term.
- **(b)** Measure whether sequence-only generalization *collapses* when whole
  antigen clusters are held out. A collapse here is not a failure — it is the
  result that motivates the predictor-conditional structure track (M-C,
  Phase B). Report it plainly with the M-S-is-pre-structure caveat, mirroring
  the existing Phase A summary.

**Non-goals.** This is a baseline/floor, **not** M-C: it does not consume
predictor confidence (ipTM/PAE/pLDDT). No Fab/scFv this round. No
cross-attention or PLM fine-tuning this round (deferred to a later spec). mirage
stays **torch-free** — ESM embeddings are produced in a separate GPU env and
cached, exactly as the GPU pose predictors shell out.

## Locked decisions (brainstorm, 2026-06-01)

| Axis | Decision |
|------|----------|
| Format | **VHH-only** (`Lchain == NA`, ~3,021 rows). No Fab/scFv. |
| Embeddings | **ESM-2 650M** (`esm2_t33_650M_UR50D`, 1280-dim), local GPU, mean-pooled, cached. |
| Model depth | Full **stage-1 ladder**: additive → diagonal bilinear (Hadamard) → low-rank bilinear two-tower. Deep nets deferred. |
| Splits | Held-out **antigen clusters** + antibody dedup. Epitope-level deferred. |
| Negatives | Shuffled (VHH × wrong antigen), **distribution-matched** to the positive antigen marginal, **same-cluster excluded**, **fold-consistent**. |
| Guardrail | AVIDa-hIL6 stays a **held-out same-antigen** frozen-gate transfer, NOT training. |

## Design

### 1. Staging — `scripts/stage_sabdab_pairs.py` (PARCC, local HMMER)

Produces a flat pair table for the analysis step. Steps:

1. **Positives.** Run `SAbDabLoader(use_anarci=True)`. It already filters
   resolution ≤ 3.0 Å, protein antigen, antigen chain-sum ≥ 30 aa, an
   ANARCI-resolvable heavy chain, and dedups binders by k-mer Jaccard at ~90%
   identity. Keep only `example.binder_format == "vhh"` (loader sets this when
   `Lchain == NA`).
2. **Normalize at the featurization boundary.** `normalize_binder` (ANARCI IMGT)
   on the VHH; `normalize_antigen` (signal-peptide + His strip) on each antigen
   chain. SAbDab antigens are crystal-derived (mature), so antigen normalization
   is mostly His-tag cleanup; applying it keeps SAbDab featurized like Champloo
   and AVIDa. Keep antigen chains **separate** for length-weighted pooling later.
3. **Antigen clustering** (`features/clustering.py`). Cluster the unique
   normalized antigen sequences by sequence identity. Prefer `mmseqs
   easy-cluster` when the `mmseqs` binary is on PATH; otherwise a deterministic
   greedy k-mer-Jaccard clusterer reusing the loader's `_kmer_set` / `_jaccard`
   and `_identity_to_jaccard_threshold` helpers (default 90% identity). Assign
   each positive an integer `antigen_cluster`.
4. **Folds.** `assign_folds(antigen_cluster_ids, n_splits=5, seed)` (existing in
   `ml/core.py`) so every member of an antigen cluster lands in the same fold —
   spike/HA/GPCR variants cannot straddle the split.
5. **Negatives** (algorithm below). k=5 per positive, deterministic in `--seed`.
6. **Emit** `data/staged/sabdab/sabdab_pairs.csv` with columns
   `pair_id, binder_seq, antigen_seq, label, antigen_cluster, fold` and a
   sidecar `sabdab_unique_seqs.txt` (deduplicated normalized binder + antigen
   sequences) for the embedding step.

#### Negative-sampling algorithm (the leakage control)

For each cognate positive `(VHH_i, Ag_i)` in fold `f` with antigen cluster
`c_i`, draw `k=5` negatives `(VHH_i, Ag_j)` subject to:

- **Cross-cluster:** `cluster(Ag_j) ≠ c_i` — prevents calling a near-identical
  antigen a non-binder (false-negative guard).
- **Fold-consistent:** `fold(Ag_j) == f` — each fold stays self-contained so OOF
  is honest.
- **Distribution-matched:** `Ag_j` is sampled so the **antigen-cluster marginal
  of the negatives within fold `f` matches the antigen-cluster marginal of the
  positives within fold `f`** (excluding `c_i` per draw). Concretely: build the
  positive cluster-frequency distribution for the fold, then sample negative
  clusters from that distribution (renormalized to exclude `c_i`), then pick a
  uniform antigen within the chosen cluster. This neutralizes
  antigen-popularity shortcuts (e.g. "spike is rarely the right partner"), so the
  antigen marginal carries no label signal and the model is forced onto the
  interaction.

The staging step asserts these three invariants on its own output before
writing.

### 2. Embeddings — separate env, cached (mirage stays torch-free)

`scripts/embed_sequences.py` runs in a dedicated `esm` env (torch + `fair-esm`
or `transformers`), **not** the mirage uv env, submitted via
`scripts/slurm/embed_esm.slurm` (`--partition=dgx-b200`, `--qos=dgx`,
`--account=dbgoodma-goodman-laboratory`). It:

- Reads the unique-sequence manifest, runs **ESM-2 650M**, and **mean-pools** the
  final-layer per-residue representations to a single 1280-dim vector per
  sequence.
- **Long antigens (> 1022 residues, e.g. spike):** chunk into ≤1022-residue
  non-overlapping windows, mean-pool each window, then take the
  **length-weighted mean** of the window vectors. Documented behavior, not silent
  truncation.
- **Multi-chain antigens:** length-weighted mean of per-chain pooled vectors.
- Writes `data/staged/sabdab/embeddings.npy` (N×1280, float32) and `keys.txt`
  mapping `sha1(normalized_seq)` → row index. Binders and antigens share one
  cache, keyed by sequence hash, so re-runs are idempotent.

`src/mirage/features/embeddings.py` (numpy-only) provides
`load_embedding_cache(npy_path, keys_path) -> dict[str, np.ndarray]` and
`paired_matrix(pairs, cache, layout)` which builds the per-rung feature matrix in
one of two layouts: `"concat"` → `[e_ab | e_ag]` (2560-dim; consumed by the
additive logistic in rung 1 and split back into halves by the bilinear model in
rung 3) and `"hadamard"` → `e_ab ⊙ e_ag` (1280-dim; rung 2). Sequences are
hashed with the same `sha1(normalized_seq)` rule as the embed step.

### 3. Model ladder

Let `e_ab, e_ag ∈ ℝ^1280` be the (per-pair, standardized) pooled embeddings.

- **Rung 0 — additive Tier-S.** The existing 13 features → `train_ms` grouped by
  antigen cluster. Reproduces the chance floor on SAbDab (sanity: the additive
  claim holds on the new, larger data).
- **Rung 1 — additive ESM-concat.** `X = [std(e_ab) | std(e_ag)]` (2560-dim) →
  `train_ms`. Still additive; expected ~chance under distribution-matched
  negatives. This is the control that shows concatenation is not enough even
  with rich features.
- **Rung 2 — diagonal bilinear (Hadamard).** `X = std(e_ab) ⊙ std(e_ag)`
  (1280-dim) → the **existing** `train_ms`/logistic, because
  `w·(e_ab ⊙ e_ag) = e_abᵀ diag(w) e_ag` is a convex, diagonal bilinear form.
  **Zero new training code.** Freezes as a plain `MsModel`.
- **Rung 3 — low-rank bilinear two-tower (headline).**
  `score = (P_a e_ab)·(P_g e_ag) + b`, projections `P_a, P_g ∈ ℝ^{r×1280}`,
  rank `r ≈ 32`. New numpy GD trainer `src/mirage/ml/bilinear.py` (logistic loss
  + L2 on `P_a, P_g`, deterministic seed, grouped OOF via `assign_folds`). This
  is the strongest model expressible without a deep net. Freezes as a new
  `BilinearModel` (`src/mirage/model/bilinear.py`) exposing
  `predict_logit(x=[e_a | e_g]) -> logits`, so the existing
  `evaluate_frozen_gate(model, x, y)` works unchanged.

Bilinear training is non-convex; mitigate with L2, multiple seeds, and reported
CIs. Rung 2 (convex, diagonal) is the documented fallback if rung 3 is unstable.

### 4. Evaluation — `scripts/analyze_sabdab_baseline.py`

- Reuse `eval/gate.py` wholesale: tie-aware `auroc`,
  `choose_threshold_for_precision`, `summary_dict` (PPV-prevalence sweep),
  stratified `bootstrap_ci`. Operating point at **target precision 0.9** and the
  default prevalence grid, consistent with the Phase A gate; AUROC (ranking,
  prevalence-robust) is the primary head-to-head number, with the k=5 training
  ratio not biasing it.
- **In-distribution:** OOF scores under the antigen-cluster grouped folds for
  every rung. One head-to-head table — AUROC, recall@target-precision, the PPV
  sweep, and bootstrap CIs — across rungs 0–3.
- **Freeze** the best rung's full-fit artifact to `results/published/`
  (`sabdab_ms_model.json` for an `MsModel` rung, or
  `sabdab_bilinear_model.json` for rung 3).
- **AVIDa transfer (guardrail).** Apply the frozen model to AVIDa-hIL6 through
  the existing `eval/orthogonal.py::evaluate_frozen_gate`, adding a sibling
  `features_for_examples_embedding(...)` that looks up cached embeddings and
  builds the paired matrix (the current Tier-S `features_for_examples` is left
  untouched). Held-out same-antigen real-negative canary; **NOT** training.
- **Writeup:** `results/published/sabdab_baseline_summary.md`. No fabricated
  numbers; report any collapse plainly with the pre-structure caveat.

## Wiring summary

| Path | Change |
|------|--------|
| `scripts/stage_sabdab_pairs.py` | NEW — load, normalize, cluster, fold, negatives |
| `src/mirage/features/clustering.py` | NEW — antigen clustering (mmseqs or greedy k-mer) |
| `scripts/embed_sequences.py` | NEW — ESM-2 650M mean-pool (separate env) |
| `scripts/slurm/embed_esm.slurm` | NEW — SLURM wrapper |
| `src/mirage/features/embeddings.py` | NEW — cache loader + paired-matrix builders |
| `src/mirage/ml/bilinear.py` | NEW — numpy low-rank bilinear GD trainer + OOF |
| `src/mirage/model/bilinear.py` | NEW — frozen `BilinearModel` |
| `scripts/analyze_sabdab_baseline.py` | NEW — train ladder, OOF table, freeze best |
| `src/mirage/eval/orthogonal.py` | EDIT — add `features_for_examples_embedding` |
| `results/published/sabdab_baseline_summary.md` | NEW — results writeup |

### Reused (do not reimplement)

- `benchmark/sabdab.py::SAbDabLoader` — positives, binder dedup, ANARCI gate.
- `features/normalize.py` — `normalize_binder` / `normalize_antigen`.
- `ml/core.py` — `fit_logistic_regression`, `standardizer`, `apply_standardizer`,
  `assign_folds`, `oof_logistic_scores`.
- `model/ms.py::train_ms` / `MsModel` — rungs 0–2 freeze as `MsModel`.
- `eval/gate.py` — all metrics.
- `eval/orthogonal.py::evaluate_frozen_gate` — AVIDa transfer (model/x/y interface).

## Testing & CI

- `tests/test_clustering.py` — greedy clusterer is deterministic and merges
  near-identical sequences; mmseqs path skips gracefully when the binary is
  absent.
- `tests/test_sabdab_negatives.py` — on a synthetic positive table, every
  negative is cross-cluster, fold-consistent, and the negative cluster marginal
  matches the positive marginal within tolerance; output is deterministic in
  seed.
- `tests/test_embeddings_cache.py` — round-trip a tiny synthetic cache; hashing
  and the three paired-matrix layouts are correct.
- `tests/test_bilinear.py` — on synthetic data where the label is the sign of a
  planted bilinear interaction (and additive models are at chance), the trainer
  reaches AUROC ≫ 0.5 while an additive logistic stays ≈ 0.5; OOF is
  deterministic; `BilinearModel` save/load/predict_logit round-trips.
- Full battery before commit: `uv run ruff check`, `uv run ruff format --check`,
  `uv run mypy src/mirage`, `uv run pytest`. ANARCI/HMMER and mmseqs tests skip
  gracefully where those tools are absent, keeping CI green.

## Verification (end-to-end)

1. Staging on PARCC produces `sabdab_pairs.csv` with the invariants asserted.
2. SLURM embed job produces `embeddings.npy` + `keys.txt` covering every unique
   sequence in the manifest.
3. `analyze_sabdab_baseline.py` reports rung 0 ≈ chance, rung 1 ≈ chance, and
   rungs 2/3 strictly above — the interaction term earns its keep — and writes
   the head-to-head table + frozen artifact.
4. The AVIDa transfer row is produced via the frozen model through the existing
   orthogonal harness.
5. Re-running staging → embed → analyze reproduces the committed
   `sabdab_baseline_summary.md` numbers from `abdisc-data/sabdab/` + the cache.

## Risks / assumptions

- **Held-out-antigen collapse is an acceptable, informative outcome** — report
  plainly with the pre-structure caveat; do not present it as a mirage win/loss.
- **ESM context limit (1022)** — chunked length-weighted mean-pool for long
  antigens; documented, not silent truncation.
- **Negative false-negatives** — cross-cluster exclusion prevents labeling a
  near-identical antigen a non-binder; distribution-matching prevents
  antigen-popularity leakage.
- **Bilinear non-convexity** — multiple seeds + L2 + reported CIs; the convex
  diagonal rung is the fallback.
- **Separate embedding env** — `embed_sequences.py` and its SLURM wrapper are the
  only torch-touching code; they live in `scripts/`, not the importable package,
  preserving the torch-free invariant for `src/mirage/`.
- **Work on branch `sabdab-sequence-baseline`**; commits authored by Pedram, no
  Claude/Anthropic trailer (workspace convention).
