# mirage — M-C EpCAM real-negative canary

> **STATUS (2026-06-08): COMPLETE.** First test in the real-negative tier. All numbers
> below are real, produced by the committed pipeline: Protenix-predicted complexes
> (84 pairs, 0 failures) scored by the **frozen SAbDab M-C gate**, applied unchanged.
> No numbers are fabricated.

## What this is

The first test of mirage in the **designed-binder deployment regime**: 14 real
designed EpCAM VHHs (collaborator CAR-T killing assay vs AsPC1 EpCAM+) against EpCAM,
plus shuffled negatives, scored by the gate frozen on SAbDab in M-C Phase B2. It asks
three questions the in-distribution / cross-regime SAbDab↔Champloo work could not:

1. Does **ipTM** still separate real *designed* binders from shuffled pairings — does
   the B2 in-distribution signal (AUROC 0.690) generalize to designed positives?
2. Does **interface geometry / CDR engagement** add anything on top of ipTM *here*?
3. Does the **frozen SAbDab operating threshold** hold under cross-regime transfer, or
   collapse the way the M-S sequence gate did on AVIDa (recall 0)?

This is explicitly a **probe** (N = 14 positives), not a powered result — it scouts the
regime the powered AVIDa follow-up will measure properly.

## Dataset & protocol

- **Positives (14):** labeled EpCAM VHHs × EpCAM ECD (UniProt P16422 res 24–265).
  Functional (8): 10, 25, 26, 34, 57, 61, 74, 86. Non-functional (6): 14, 15, 16, 18,
  21, 73. Label = functional CAR-T killing (one step downstream of binding).
- **Negatives (70):** predict-the-shuffled-pair — each VHH × k=5 wrong antigens drawn
  distribution-matched from the SAbDab antigen pool (396 after deduping EpCAM's antigen
  cluster at ≥0.90 id). No EpCAM VHH appears in the SAbDab training set (leakage guard
  passed).
- **Predictor:** Protenix (open AF3-class), identical config to the B1 campaign
  (templates off, Blackwell torch kernels, `need_atom_confidence`). **84 / 84 predicted,
  0 feature-extraction failures.**
- **Model:** the SAbDab M-C gate, **frozen and applied unchanged** (never trained on
  EpCAM). Rung-3 reproduces `mc_sabdab_model.json`; rung-0 (ipTM) is the contrast.

## Install-validation gate (cognate ≫ shuffled)

| set | n | median ipTM | range |
|---|---|---|---|
| positives (designed EpCAM binders) | 14 | **0.620** | [0.220, 0.929] |
| negatives (shuffled VHH × wrong-Ag) | 70 | **0.327** | [0.165, 0.857] |

Protenix reproduces the cognate ≫ shuffled separation on real designed binders. Gate passed.

## Read 1 — primary: designed-binders-vs-shuffled (frozen transfer, threshold-free)

n = 84 (14 positive / 70 negative).

| rung | feature set | AUROC |
|---|---|---|
| 0 | ipTM only | **0.761** |
| 3 | + confidence internals + geometry + CDR | **0.777** |

| contrast | ΔAUROC | 95% CI | interpretation |
|---|---|---|---|
| rung 3 − rung 0 | +0.015 | [−0.035, +0.072] | CI includes 0 |

- **ipTM generalizes to the designed-binder regime** — AUROC 0.761, *above* the B2
  in-distribution level (0.690) and far above the M-S sequence floor (0.496) and the
  AF2-M ~0.50 wall. The confidence signal is not a SAbDab/Champloo artifact; it holds on
  real designed VHHs against a real tumor target.
- **Interface geometry + CDR add nothing on top of ipTM** — the paired ΔAUROC CI
  includes 0, reproducing the clean B2 negative in a new, more decision-relevant regime.

## Read 2 — calibration: does the frozen SAbDab threshold hold? (the headline)

The frozen rung-3 gate at its **SAbDab-chosen P=0.90 threshold (0.331)**, applied to
EpCAM **unchanged**:

| gate | threshold | TP | FP | FN | TN | precision | recall | specificity |
|---|---|---|---|---|---|---|---|---|
| rung 3 (full) | 0.331 | 5 | 1 | 9 | 69 | **0.833** | 0.357 | 0.986 |
| rung 0 (ipTM) | 0.012 | 3 | 1 | 11 | 69 | 0.750 | 0.214 | 0.986 |

**This is the most important result of the canary.** The frozen SAbDab gate transfers to
the EpCAM designed-binder regime with **precision ≈ 0.83** — close to its 0.90 target —
and specificity 0.99. This is qualitatively different from the M-S sequence gate, which
when frozen on Champloo and applied to AVIDa's real negatives collapsed to **recall 0 /
undefined precision**. **Cross-regime precision stability — mirage's intended headline
metric — holds for the predictor-conditional structure gate**, at least into this
designed-binder regime. (Recall is low at this conservative high-precision operating
point, as designed for an FP-costly gate; N = 14 makes the recall CI wide.)

## Read 3 — secondary, exploratory: functional vs non-functional killing (N=14)

> **Caveat (load-bearing):** N = 14, and the label is *functional killing*, one step
> downstream of binding — a "non-functional" VHH may still bind. Descriptive only; **no
> inferential claim**. This is a hypothesis-generator for the powered AVIDa follow-up.

Frozen rung-3 mirage score (logit), 8 functional vs 6 non-functional:

- descriptive AUROC = **0.771**
- functional scores:    [0.87, 0.53, 0.51, 0.44, −0.31, −1.35, −1.76, −1.87]
- non-functional scores: [0.47, −0.01, −1.47, −2.11, −2.43, −2.45]

The binding-correctness score trends higher for functional than non-functional designs,
with clear exceptions (one functional design scores low; one non-functional scores
high). Encouraging but not conclusive at this N — exactly what the AVIDa step is for.

## Read

- **ipTM generalizes to real designed binders** (AUROC 0.761), extending the B2 finding
  beyond crystal-derived positives.
- **Structure-beyond-confidence still adds nothing** (paired ΔAUROC CI includes 0) — the
  B2 negative reproduces in the deployment regime.
- **The frozen gate keeps its precision across regimes** (0.83 vs the 0.90 target),
  where the sequence-only gate did not — the first positive evidence for mirage's
  cross-regime precision-stability headline on a real-negative-adjacent set.
- **The mirage score tracks functional killing** descriptively (AUROC 0.77, N=14) — a
  decision-relevant hint to test at power on AVIDa (real assay negatives).

This is a probe, and it points the same way the rest of the project does: the
predictor's confidence carries the binding signal and transfers robustly; richer
structure has not yet earned its keep; the honest powered test is the real-negative
tier (AVIDa next).

## How to reproduce

```bash
# 1. Stage pairs (14 positives + k=5 shuffled negatives, dedup'd pool)
uv run python scripts/stage_epcam_protenix_pairs.py \
    --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \
    --sabdab-pairs data/staged/sabdab/sabdab_pairs.csv \
    --output data/staged/epcam/epcam_protenix_pairs.csv --k 5

# 2. (Protenix campaign: MSA precompute -> SLURM array -> features; see campaign record)

# 3. Frozen-transfer analysis (the three reads)
uv run python scripts/analyze_mc_epcam.py \
    --sabdab-features data/staged/mc/sabdab_features.csv \
    --epcam-features data/staged/mc/epcam_features.csv \
    --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \
    --output results/published/mc_epcam_canary.json
```

Artifacts: `results/published/mc_epcam_canary.json`. Spec:
`docs/superpowers/specs/2026-06-07-mirage-mc-epcam-canary-design.md`. Plan:
`docs/superpowers/plans/2026-06-07-mirage-mc-epcam-canary.md`.
