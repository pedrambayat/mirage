# M-C structure-track — Protenix prediction campaign record

> **STATUS (2026-06-04): COMPLETE.** 3,234 / 3,234 predicted complexes; feature CSVs
> extracted with 0 failures. This is the input substrate for Phase B2 (rung-ladder
> modeling vs the M-S 0.496 floor).

## What was produced

- **Predictor:** Protenix 2.0.0 (open AF3-class), one diffusion sample (top-ranked
  `sample_0`) per pair, MSA reused from a pre-computed ColabFold cache (one MSA per
  of 967 unique sequences). Blackwell lock-in flags: `--trimul_kernel torch
  --triatt_kernel torch --enable_fusion false --need_atom_confidence true
  --use_template false` with `LAYERNORM_TYPE=native` + cu128 torch.
- **Pairs:** 3,234 = SAbDab 2,688 (448 cognate VHH positives / 241 antigen clusters
  + 2,240 distribution-matched cross-cluster negatives, k=5) + Champloo 546 (91
  cognate + 455 negatives). Multi-chain antigens are split into separate
  `proteinChain`s (5.8% of SAbDab positives); interface confidence is computed
  binder-vs-antigen.
- **Feature CSVs** (`data/staged/mc/{sabdab,champloo}_features.csv`, gitignored):
  per pair — `iptm, ptm, interface_pae, min_interface_pae, interface_plddt,
  mean_plddt` (confidence), 6 interface-geometry descriptors, 5 CDR-engagement
  features, passthrough `label/antigen_cluster/fold`. **0 extraction failures.**

## Compute (SLURM, polite footprint)

Size-split hybrid: ≤800-token pairs (96%) on `b200-mig45` (`--qos=mig`, `%8`, 8
MIG slices); >800-token tail (130 pairs) on `dgx-b200` (`--qos=dgx`, `%4`). The
torch triangle kernels are slow for large N on a quarter-B200 MIG slice (a
1684-token pair: stuck >13 min on mig45 vs ~8 min on a full B200), hence the split.
**MIG tasks require explicit `--cpus-per-task=8 --mem≈56G`** — the default 1 CPU /
8 GB OOM-kills Protenix MSA featurization (N_msa 10–13k). Each prediction keeps only
`sample_0` (~7 MB); samples 1–4 are deleted in post-process to bound the shared
project quota.

## Discriminative sanity (SAbDab, cognate vs shuffled medians)

| feature | cognate (pos) | shuffled (neg) |
|---|---|---|
| ipTM | 0.490 | 0.281 |
| interface PAE (Å) | 19.4 | 23.4 |

ipTM alone already separates (the expected abdisc wall) — so the Phase B2 headline
is the **paired Δ(geometry vs ipTM)**, i.e. whether interface geometry/CDR features
add signal on top of confidence, on the same held-out-antigen-cluster split that put
the sequence-only M-S floor at **AUROC 0.496**.

## Reproduce

Predictions: `scripts/stage_protenix_pairs.py` (+ `stage_champloo_protenix_pairs.py`)
→ `precompute_protenix_msa.py` (per-unique-sequence MSA) → `predict_protenix.slurm` +
`run_protenix_chunk.py` (batched, per-job fault-tolerant). Features:
`extract_mc_features.py --predictions-root data/raw/predictions/protenix/all`.

## Phase B2 (modeling) — COMPLETE (2026-06-04)

The rung-ladder discriminator + the apples-to-apples test vs the M-S 0.496 floor are
done. **Headline: a clean negative.** On the same 2,688 SAbDab rows / 241 antigen
clusters / same held-out-antigen-cluster folds as M-S:

- Rung 0 (ipTM alone): **OOF AUROC 0.690** [0.658, 0.722] — clears the 0.496 sequence
  floor *and* the ~0.50 AF2-M-confidence wall. Protenix confidence is genuinely
  informative.
- Rungs 1/2/3 (+confidence internals / +geometry / +CDR): all **0.691**. Paired
  **ΔAUROC(R2−R0) = +0.001 [−0.019, +0.022]** and **ΔAUROC(R3−R0) = +0.001
  [−0.019, +0.022]** — every CI includes 0. Interface geometry and CDR engagement add
  **no** discriminative signal on top of ipTM.
- Top standardized coefficients are all confidence terms (`interface_pae` −0.70, `iptm`
  +0.69, `min_interface_pae` +0.66); geometry/CDR weights are an order of magnitude
  smaller. Random-split AUROC ≈ OOF (~0.68–0.70) → genuine signal, not a held-out
  artifact.
- **Cross-regime (Protenix on both):** dedup dropped 455/546 Champloo rows (SAbDab
  antigen clusters cover most Champloo antigens at ≥0.9 id), 91 leakage-free kept
  (14 pos). 2a SAbDab→Champloo rung3 0.758 (rung0 0.763); 2b Champloo→SAbDab rung3
  0.678 (rung0 0.691). Same pattern both directions: geometry/CDR add nothing over
  ipTM; ipTM transfers at ~0.69–0.76.

Full writeup: `results/published/mc_indist_summary.md`. Artifacts: `mc_indist.json`,
`mc_sabdab_model.json` (frozen Rung-3 gate), `mc_cross_regime.json`. Reproduce:
`scripts/analyze_mc_indist.py` and `scripts/analyze_mc_cross_regime.py` (see the
summary's "How to reproduce"). The optional AF3 companion (Champloo AF3 rung 0→2 +
AF3-vs-Protenix ipTM) remains a non-blocking follow-up.

## EpCAM real-negative canary — COMPLETE (2026-06-08)

First test in the real-negative tier: 14 real designed EpCAM VHHs (CAR-T killing labels)
+ 70 shuffled negatives, Protenix-predicted (84/84, 0 failures), scored by the **frozen
SAbDab M-C gate** applied unchanged. Same predictor config as B1. Validation gate passed
(positive median ipTM 0.620 vs shuffled 0.327).

- **Primary (designed-binders-vs-shuffled, frozen transfer):** rung-0 (ipTM) AUROC
  **0.761**, rung-3 **0.777**, paired **ΔAUROC(R3−R0) = +0.015 [−0.035, +0.072]** (CI
  includes 0). ipTM generalizes to the designed-binder regime (above the 0.690
  in-distribution level); geometry/CDR add nothing — the B2 negative reproduces.
- **Calibration (headline):** the frozen SAbDab rung-3 gate at its P=0.90 threshold
  (0.331), applied unchanged, holds **precision 0.833** (5 TP / 1 FP), recall 0.357,
  specificity 0.986 — vs the M-S sequence gate's recall-0 collapse on AVIDa. Cross-regime
  precision stability holds for the structure gate.
- **Secondary (exploratory, N=14, killing≠binding):** mirage score vs functional killing
  descriptive AUROC **0.771** — encouraging hint, to test at power on AVIDa.

Full writeup: `results/published/mc_epcam_canary_summary.md`. Artifact:
`mc_epcam_canary.json`. Spec/plan: `docs/superpowers/{specs,plans}/2026-06-07-mirage-mc-epcam-canary*`.
Next: the powered real-negative test (AVIDa-hIL6, real assay negatives).
