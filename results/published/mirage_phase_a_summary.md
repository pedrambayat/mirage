# mirage Phase A — sequence-only gate (M-S) results

> **STATUS (2026-06-01): in-distribution DONE; orthogonal PENDING.** The
> in-distribution numbers below are real (AF3 Champloo, produced by the committed
> pipeline). The orthogonal rows are still `[fill]` because AVIDa-hIL6 and the
> labeled-EpCAM killing set are not yet staged (see **How to produce these
> numbers**). No numbers are fabricated.

**Caveat up front:** M-S is the *pre-structure baseline*. It does not consume
predictor confidence and so does **not** answer mirage's headline question (where
ipTM is insufficient) — that is M-C (Phase B). M-S establishes the
data/feature/validation infrastructure and a sequence-only reference floor.

## In-distribution (Champloo AF3, held-out-antigen OOF) — n=8228, 91 cognate positives

Target operating point was P=0.90. **Neither model reaches P=0.90**, so each row
reports its highest-precision (fallback) operating point.

| model | AUROC | recall | specificity | precision | threshold |
|---|---|---|---|---|---|
| M-S (Tier-S, OOF) | **0.362** | 1.000 | 0.001 | 0.011 | −5.213 |
| raw ipTM (floor) | **0.754** | 0.044 | 1.000 | 0.667 | 0.910 |

- M-S OOF AUROC **0.362 (< 0.5)**: the bulk-composition Tier-S features carry **no
  usable cognate-vs-shuffled signal** under the leakage-controlled held-out-antigen
  split — its "operating point" collapses to predict-almost-everything-positive
  (recall 1.0 / precision ≈ the 1.1% base rate). This is the *expected, defensible*
  weak-baseline floor: for a given VHH, its cognate and shuffled pairs share
  identical VHH composition and differ only in *which* antigen, so marginal
  composition cannot encode antigen-specific matching.
- raw ipTM AUROC **0.754**: AF3 ipTM is an informative ranker on in-distribution
  Champloo, but still caps at precision 0.667 / recall 0.044 at its best operating
  point (high-specificity, low-recall).

PPV at deployed prevalence (M-S): 1:100 → 0.010, 1:1,000 → 0.0010, 1:10,000 → 0.0001.
(PPV ≈ prevalence at every rarity ⇒ the gate adds essentially nothing — consistent
with AUROC ≈ chance.)

## Feature attribution (top 3 by |standardized coef|)
1. `length_ratio` (coef +0.031) 2. `target_length` (+0.029) 3. `binder_length` (−0.003)

All standardized coefficients are tiny (|coef| < 0.04). The weak signal that exists
is length-driven, not interaction-driven — consistent with there being no real
binding signal in bulk composition (and consistent with the sub-chance AUROC).

## Orthogonal validation (Champloo-frozen threshold) — PENDING
The frozen `results/published/ms_model_af3.json` is committed and ready; this table
fills once AVIDa + labeled-EpCAM are staged and `analyze_ms_orthogonal.py` is run.

| regime | n (pos/neg) | precision [CI] | recall [CI] | specificity [CI] |
|---|---|---|---|---|
| AVIDa-hIL6 | [fill] | [fill] | [fill] | [fill] |
| EpCAM killing | 14 (8/6) | [fill] | [fill] | [fill] |

## Read
- In-distribution: M-S (sequence-only) is **non-discriminative** (AUROC 0.36);
  raw AF3 ipTM is a moderate ranker (AUROC 0.75) but low-recall at high precision.
- Does the frozen threshold hold its precision off Champloo? **[pending orthogonal run]**
- EpCAM canary (designed binders): **[pending — labels not yet authored]**

**Framing:** a frozen-threshold *collapse* on the orthogonal sets is an
*expected* outcome for a sequence-only pre-structure baseline, **not** a mirage
failure. M-S trains on synthetic shuffled-pair negatives and never sees
predictor confidence; the orthogonal sets have real negatives against the
*correct* antigen — a genuinely harder, different decision. The headline
cross-regime question belongs to M-C (Phase B); M-S only establishes the
data/feature/validation infrastructure and a sequence-only reference floor.
Report collapse plainly with that caveat; do not present it as evidence for or
against mirage's scientific claim. (The in-distribution AUROC 0.36 already shows
M-S has no sequence-only signal even *before* the regime shift.)

---

## How to produce these numbers

**In-distribution (done — reproducible):** all inputs are present in
`../abdisc-data/champloo/`.
```bash
uv run python scripts/stage_champloo_pairs.py \
  --metadata ../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \
  --matrix ../abdisc-data/champloo/iptm_confidence_scores/iptm_confidence_scores/af3_matrix_clean.csv \
  --predictor af3 --output data/staged/champloo/champloo_pairs_af3.csv
uv run python scripts/stage_champloo_features.py \
  --pairs data/staged/champloo/champloo_pairs_af3.csv \
  --supp ../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \
  --output data/staged/champloo/champloo_features_af3.csv
uv run python scripts/analyze_ms_indist.py \
  --features data/staged/champloo/champloo_features_af3.csv \
  --model-out results/published/ms_model_af3.json \
  --output results/published/mirage_ms_indist_af3.json
```

**Orthogonal (pending — needs inputs):** stage AVIDa (download raw
`COGNANO/AVIDa-hIL6` → `scripts/stage_avida.py`) and author
`../abdisc-data/epcam/epcam_killing_labels.csv` (Good: 10,25,26,34,57,61,74,86;
Bad: 14,15,16,18,21,73), then:
```bash
uv run python scripts/analyze_ms_orthogonal.py \
  --model results/published/ms_model_af3.json \
  --avida-csv ../abdisc-data/avida/avida_staged.csv \
  --epcam-labels ../abdisc-data/epcam/epcam_killing_labels.csv \
  --output results/published/mirage_ms_orthogonal.json
```
Then transcribe the orthogonal table + answer the remaining **Read** bullets.
