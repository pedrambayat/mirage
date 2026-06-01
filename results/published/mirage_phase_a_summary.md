# mirage Phase A — sequence-only gate (M-S) results

> **STATUS: SCAFFOLD — PENDING DATA (2026-06-01).** The `[fill]` values below are
> intentionally unfilled: the in-distribution and orthogonal *real-data runs* have
> not been executed because their source inputs are not present in this repo (see
> **How to produce these numbers** at the bottom). All Phase A *code* (staging,
> training, gate metrics, attribution, orthogonal harness) is implemented, tested,
> and integration-verified on synthetic input. Do **not** fill these brackets by
> hand — run the pipeline and transcribe from the emitted JSON. No numbers here
> are fabricated.

**Caveat up front:** M-S is the *pre-structure baseline*. It does not consume
predictor confidence and so does **not** answer mirage's headline question (where
ipTM is insufficient) — that is M-C (Phase B). M-S establishes the
data/feature/validation infrastructure and a sequence-only reference.

## In-distribution (Champloo, held-out-antigen OOF), operating point P=0.90
| model | recall | specificity | precision |
|---|---|---|---|
| M-S (Tier-S) | [fill] | [fill] | [fill] |
| raw ipTM (floor) | [fill] | [fill] | [fill] |

PPV at deployed prevalence (M-S): 1:100 → [fill], 1:1,000 → [fill], 1:10,000 → [fill].

## Feature attribution (top 3 by |standardized coef|)
1. [fill] 2. [fill] 3. [fill]
Sanity: signal is not dominated by trivial length/charge mismatch alone (else
expect EpCAM collapse below).

## Orthogonal validation (Champloo-frozen threshold)
| regime | n (pos/neg) | precision [CI] | recall [CI] | specificity [CI] |
|---|---|---|---|---|
| AVIDa-hIL6 | [fill] | [fill] | [fill] | [fill] |
| EpCAM killing | 14 (8/6) | [fill] | [fill] | [fill] |

## Read
- Does the frozen threshold hold its precision off Champloo? [yes/no + numbers]
- EpCAM canary (designed binders): [held / collapsed]

**Framing:** a frozen-threshold *collapse* on the orthogonal sets is an
*expected* outcome for a sequence-only pre-structure baseline, **not** a mirage
failure. M-S trains on synthetic shuffled-pair negatives and never sees
predictor confidence; the orthogonal sets have real negatives against the
*correct* antigen — a genuinely harder, different decision. The headline
cross-regime question belongs to M-C (Phase B); M-S only establishes the
data/feature/validation infrastructure and a sequence-only reference floor.
Report collapse plainly with that caveat; do not present it as evidence for or
against mirage's scientific claim.

---

## How to produce these numbers (blocked — required inputs)

The pipeline code is complete; only the inputs are missing. To fill this summary:

1. **Stage the Champloo feature table** (needs the labeled pair manifest
   `data/staged/champloo/champloo_pairs_af3.csv`, which is an upstream staging
   artifact **not currently present** — only the supp table is in
   `../abdisc-data/champloo/`):
   ```bash
   uv run python scripts/stage_champloo_features.py \
     --pairs data/staged/champloo/champloo_pairs_af3.csv \
     --supp ../abdisc-data/champloo/Supplementary_Table_1_*.csv \
     --output data/staged/champloo/champloo_features_af3.csv
   ```
2. **In-distribution analysis** → fills the first two tables + attribution:
   ```bash
   uv run python scripts/analyze_ms_indist.py \
     --features data/staged/champloo/champloo_features_af3.csv \
     --model-out results/published/ms_model_af3.json \
     --output results/published/mirage_ms_indist_af3.json
   ```
3. **Orthogonal validation** → fills the orthogonal table. Requires staged AVIDa
   (`../abdisc-data/avida/` — download raw `COGNANO/AVIDa-hIL6` CSVs, then
   `scripts/stage_avida.py`) and the owner-authored
   `../abdisc-data/epcam/epcam_killing_labels.csv` (Good: 10,25,26,34,57,61,74,86;
   Bad: 14,15,16,18,21,73):
   ```bash
   uv run python scripts/analyze_ms_orthogonal.py \
     --model results/published/ms_model_af3.json \
     --avida-csv ../abdisc-data/avida/avida_staged.csv \
     --epcam-labels ../abdisc-data/epcam/epcam_killing_labels.csv \
     --output results/published/mirage_ms_orthogonal.json
   ```
4. Transcribe the metrics from `mirage_ms_indist_af3.json` and
   `mirage_ms_orthogonal.json` into the tables above and answer the **Read** bullets.
