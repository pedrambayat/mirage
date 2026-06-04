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
