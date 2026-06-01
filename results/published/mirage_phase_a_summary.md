# mirage Phase A — sequence-only gate (M-S) results

> **STATUS (2026-06-01): Phase A COMPLETE — in-distribution baseline + both
> orthogonal real-negative sets (AVIDa-hIL6, EpCAM-killing) done.** All numbers below are real,
> produced by the committed pipeline on **mature-domain-normalized** sequences
> (ANARCI IMGT variable-domain extraction for binders; signal-peptide + His-tag
> stripping for antigens — see `docs/superpowers/specs/2026-06-01-sequence-normalization-design.md`).
> No numbers are fabricated.

**Caveat up front:** M-S is the *pre-structure baseline*. It does not consume
predictor confidence and so does **not** answer mirage's headline question (where
ipTM is insufficient) — that is M-C (Phase B). M-S establishes the
data/feature/validation infrastructure and a sequence-only reference floor.

**Normalization note:** binder/antigen sequences are reduced to comparable mature
domains before featurization, so Champloo (train) and AVIDa (orthogonal) are
featurized like-for-like. This removed a preprocessing artifact (AVIDa VHHs
carried a ~22-residue secretion leader Champloo's PDB-derived VHHs lack). The
in-distribution AUROC barely moved (0.362 → **0.346**), confirming normalization
removed a *nuisance axis* without adding binding signal — and the AVIDa collapse
(below) **persists under clean featurization**, so it reflects M-S's lack of
transferable signal, not the leader-peptide artifact.

## In-distribution (Champloo AF3, held-out-antigen OOF) — n=8228, 91 cognate positives

Target operating point was P=0.90. **Neither model reaches P=0.90**, so each row
reports its highest-precision (fallback) operating point.

| model | AUROC | recall | specificity | precision | threshold |
|---|---|---|---|---|---|
| M-S (Tier-S, OOF) | **0.346** | 1.000 | 0.004 | 0.011 | −4.943 |
| raw ipTM (floor) | **0.754** | 0.044 | 1.000 | 0.667 | 0.910 |

- M-S OOF AUROC **0.346 (< 0.5)**: the bulk-composition Tier-S features carry **no
  usable cognate-vs-shuffled signal** under the leakage-controlled held-out-antigen
  split — its "operating point" collapses to predict-almost-everything-positive
  (recall 1.0 / precision ≈ the 1.1% base rate). This is the *expected, defensible*
  weak-baseline floor: for a given VHH, its cognate and shuffled pairs share
  identical VHH composition and differ only in *which* antigen, so marginal
  composition cannot encode antigen-specific matching.
- raw ipTM AUROC **0.754**: AF3 ipTM is an informative ranker on in-distribution
  Champloo, but still caps at precision 0.667 / recall 0.044 at its best operating
  point (high-specificity, low-recall). (Unchanged by normalization — ipTM is a
  predictor confidence score, not a sequence feature.)

PPV at deployed prevalence (M-S): 1:100 → 0.010, 1:1,000 → 0.0010, 1:10,000 → 0.0001.
(PPV ≈ prevalence at every rarity ⇒ the gate adds essentially nothing — consistent
with AUROC ≈ chance.)

## Feature attribution (top 3 by |standardized coef|)
1. `length_ratio` (coef +0.019) 2. `target_length` (+0.019) 3. `binder_polar_frac` (+0.002)

All standardized coefficients are tiny (|coef| < 0.02). The weak signal that exists
is length-driven, not interaction-driven — consistent with there being no real
binding signal in bulk composition (and consistent with the sub-chance AUROC).

## Orthogonal validation (Champloo-frozen threshold = −4.472)

The frozen `results/published/ms_model_af3.json` is applied unchanged to held-out
real-negative sets.

| regime | n (pos/neg) | precision [CI] | recall [CI] | specificity [CI] |
|---|---|---|---|---|
| AVIDa-hIL6 | 573,891 (20,980 / 552,911) | n/a (0 predicted positive) | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| EpCAM killing | 14 (8/6) | n/a (0 predicted positive) | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |

- **AVIDa-hIL6:** the Champloo-frozen gate predicts **NONBIND for all 573,891
  pairs** — zero positive predictions, so recall 0.000 and precision is undefined
  (`null`). Specificity is trivially 1.000 (everything is called negative). This is
  a *degenerate all-negative* transfer, the expected collapse for a sequence-only
  pre-structure baseline with no in-distribution signal (AUROC 0.346). Crucially,
  it holds **after** mature-domain normalization, so it is not an artifact of
  AVIDa's secretion-leader contamination.
- **EpCAM-killing (designed-binder canary):** same all-negative transfer — the
  frozen gate predicts NONBIND for all 14 designed VHHs (8 effective / 6
  ineffective CAR-T killers vs AsPC1), recall 0.000, precision undefined. A
  sequence-only gate trained on Champloo cognate-vs-shuffled carries nothing that
  separates functional from non-functional designed EpCAM binders. Tiny n —
  report with the wide-CI caveat; the point estimate is the expected collapse.

## Read
- In-distribution: M-S (sequence-only) is **non-discriminative** (AUROC 0.35);
  raw AF3 ipTM is a moderate ranker (AUROC 0.75) but low-recall at high precision.
- Does the frozen threshold hold its precision off Champloo? **No — it collapses
  to an all-negative classifier on AVIDa's real negatives (recall 0, precision
  undefined).** Expected for a no-signal sequence-only floor; reported plainly.
- EpCAM canary (designed binders): **also collapses all-negative** (recall 0 on
  14 designed VHHs) — M-S doesn't separate functional from non-functional binders.

**Framing:** a frozen-threshold *collapse* on the orthogonal sets is an
*expected* outcome for a sequence-only pre-structure baseline, **not** a mirage
failure. M-S trains on synthetic shuffled-pair negatives and never sees
predictor confidence; the orthogonal sets have real negatives against the
*correct* antigen — a genuinely harder, different decision. The headline
cross-regime question belongs to M-C (Phase B); M-S only establishes the
data/feature/validation infrastructure and a sequence-only reference floor.
Report collapse plainly with that caveat; do not present it as evidence for or
against mirage's scientific claim. (The in-distribution AUROC 0.35 already shows
M-S has no sequence-only signal even *before* the regime shift.)

---

## How to produce these numbers

All sequences are normalized to mature domains at the featurization boundary.
Binder normalization needs a local HMMER for ANARCI — build it once with
`bash scripts/install_hmmer.sh` (installs to `.tools/hmmer/`, gitignored).

**In-distribution (reproducible):** all inputs are present in `../abdisc-data/champloo/`.
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

**AVIDa orthogonal (reproducible):** raw `COGNANO/AVIDa-hIL6` is in `data/raw/avida/`.
Staging is raw (instant); the harness normalizes the 38.6k unique VHHs at scoring
time via ANARCI (~30 min).
```bash
uv run python scripts/stage_avida.py \
  --records data/raw/avida/AVIDa-hIL6.csv \
  --antigens data/raw/avida/antigen_sequences.csv \
  --output data/staged/avida/avida_staged.csv
uv run python scripts/analyze_ms_orthogonal.py \
  --model results/published/ms_model_af3.json \
  --avida-csv data/staged/avida/avida_staged.csv \
  --output results/published/mirage_ms_orthogonal.json
```

**EpCAM orthogonal (done):** `../abdisc-data/epcam/epcam_killing_labels.csv`
(`vhh_id,vhh_sequence,label`; Good: 10,25,26,34,57,61,74,86; Bad: 14,15,16,18,21,73 —
sequences pulled from `snap/epcam/vhh_epcam_don.csv`) is staged. Add
`--epcam-labels ../abdisc-data/epcam/epcam_killing_labels.csv` to the AVIDa orthogonal
command above to reproduce its row.
