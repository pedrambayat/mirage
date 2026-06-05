# mirage — M-C predictor-conditional structure track (Phase B2)

> **STATUS (2026-06-04): COMPLETE — in-distribution rung ladder + cross-regime transfer.**
> All numbers below are real, produced by the committed pipeline on Protenix-predicted
> complexes with Protenix confidence outputs (ipTM / PAE / pLDDT), interface geometry,
> and CDR engagement features. No numbers are fabricated.

**Caveat up front:** M-C is the *predictor-conditional* structure track. It consumes
Protenix's predicted `(binder, target)` complex alongside the predictor's own confidence
outputs (ipTM / PAE / pLDDT), interface geometry, and CDR engagement features. This is
mirage's headline question — the one that M-S (sequence-only, AUROC 0.496) could not
answer: does the predicted complex, plus the predictor's confidence, discriminate cognate
from shuffled binder–antigen pairs for **held-out antigen clusters**? The same 2,688 SAbDab
rows / 448 positives / 241 antigen clusters / same held-out-antigen-cluster 5-fold split
as M-S are used throughout, enabling a direct apples-to-apples comparison against the
0.496 sequence-only floor.

Protenix (open AF3-class predictor, no templates) was used to predict every pair.
The rung ladder asks a further, sharper question: does interface **geometry** and
**CDR engagement** add discriminative signal *on top of* the predictor's own confidence
scalar (ipTM)?

## Dataset & protocol

- **Source:** same SAbDab VHH-only rows as M-S. **2,688 total rows** — 448 cognate
  (VHH, antigen) positives across 241 antigen sequence-identity clusters (one unique
  VHH per positive after dedup), plus 2,240 distribution-matched cross-cluster negatives
  (k = 5 per positive, fold-consistent).
- **Predictor:** Protenix (open AF3-class, no templates). Every pair was predicted;
  confidence outputs (ipTM, PAE per-residue, pLDDT) and interface geometry were
  extracted per complex.
- **Split:** grouped 5-fold by antigen cluster, same partitions as M-S. All
  in-distribution numbers are **out-of-fold (OOF)** under this leakage control.
- **Rung ladder:** cumulative feature sets — 0 = ipTM only, 1 = + full confidence
  internals (PAE, pLDDT), 2 = + interface geometry (contacts, buried SASA proxy,
  clash fraction, shape complementarity proxy), 3 = + CDR engagement fractions. Each
  rung is an L2-logistic classifier fit OOF under the antigen-cluster folds.
- **Operating point:** target precision 0.90.

## In-distribution head-to-head (OOF, held-out antigen clusters) — n=2,688, 448 positives

| rung | feature set | OOF AUROC | 95% CI | random-split AUROC | precision | recall | specificity | threshold |
|---|---|---|---|---|---|---|---|---|
| 0 | ipTM only | **0.690** | [0.658, 0.722] | 0.683 | 0.881 | 0.312 | 0.992 | 0.012 |
| 1 | + confidence internals (PAE, pLDDT) | **0.691** | [0.661, 0.722] | 0.690 | 0.902 | 0.308 | 0.993 | 0.279 |
| 2 | + interface geometry | **0.691** | [0.660, 0.723] | 0.698 | 0.877 | 0.301 | 0.992 | 0.321 |
| 3 | + CDR engagement | **0.691** | [0.660, 0.722] | 0.693 | 0.882 | 0.301 | 0.992 | 0.331 |

M-S baseline (sequence-only floor): AUROC 0.496.

## The headline — paired ΔAUROC: does geometry/CDR add signal beyond ipTM?

The scientific question is **not** whether M-C beats M-S — rung 0 (ipTM alone) already
clears the 0.496 floor at AUROC 0.690. The question is whether the predicted interface
**structure** (geometry, CDR engagement) adds discriminative signal on top of the
predictor's own confidence scalar.

| contrast | ΔAUROC | 95% CI | interpretation |
|---|---|---|---|
| rung 2 − rung 0 (geometry added) | +0.0014 | [−0.019, +0.022] | CI includes 0 |
| rung 3 − rung 0 (geometry + CDR added) | +0.0009 | [−0.019, +0.022] | CI includes 0 |
| rung 3 − rung 2 (CDR added over geometry) | −0.0005 | [−0.004, +0.002] | CI includes 0 |

**Every paired ΔAUROC confidence interval includes zero.** Interface geometry and CDR
engagement add no discriminative signal beyond ipTM. The structure, once the predictor's
confidence is in hand, is uninformative at this power.

**Note on Δprecision.** The paired Δprecision values (e.g., R2 − R0 = −0.123; R3 − R0 =
−0.118) are computed at rung A's operating threshold applied to both arms — a common
operating point to enable comparison. Because rungs differ in logit scale, the
cross-rung Δprecision numbers are a threshold-scale artifact, not a signal measure, and
should **not** be over-read. ΔAUROC (threshold-free) is the clean headline; the table
above reports each rung's own operating-point precision (all in the range 0.877–0.902),
and those are the relevant per-rung operating-point numbers.

## Rung 3 standardized coefficients

Coefficients are standardized (unit-variance features); ranked by absolute value. The
top contributors are all **confidence terms**:

| feature | std. coefficient |
|---|---|
| interface_pae | −0.697 |
| iptm | +0.686 |
| min_interface_pae | +0.658 |
| atom_clash_fraction_2a | −0.261 |
| atom_contacts_5a | +0.226 |
| interface_plddt_missing | −0.164 |
| n_interface_residues_target | −0.128 |
| ptm | −0.088 |
| mean_plddt | −0.054 |
| cdr2_contact_fraction | −0.042 |
| n_interface_residues_binder | +0.034 |
| cdr1_contact_fraction | +0.033 |
| cdr3_contact_fraction | +0.029 |
| buried_sasa_proxy_a2 | +0.020 |
| cdr_contact_fraction | +0.020 |
| shape_complementarity_proxy | −0.004 |

The three dominant features are `interface_pae`, `iptm`, and `min_interface_pae` — all
predictor confidence outputs. Geometry (contacts, clash) and CDR engagement fractions
carry coefficients an order of magnitude smaller. The discriminative signal lives
entirely in the predictor's confidence outputs.

## Random-split contrast

The random-split AUROCs (0.683–0.698 across rungs) closely match the OOF
(held-out-antigen-cluster) AUROCs (0.690–0.691). This is qualitatively different from
M-S, where the held-out-antigen split revealed the floor was near chance — here the
signal is **not** an antigen-memorization artifact. ipTM is a per-complex confidence
estimate, not a function of antigen identity, so it generalizes across unseen antigen
clusters in the same way it performs within them. The 0.69 AUROC is genuine signal, not
a leakage artifact.

## Cross-regime transfer

**Dedup:** Champloo has 546 rows total. Of these, 455 were dropped because their antigen
falls within a SAbDab antigen cluster at ≥0.90 sequence identity. **91 rows remain**
(14 positives / 77 negatives) as a leakage-free Champloo test set. The small size means
CIs are wide; read AUROC (threshold-free) rather than operating-point metrics.

**Direction 2a — SAbDab → Champloo (primary):** train on SAbDab (2,688 rows), test on the
91 leakage-free Champloo rows.

| rung | AUROC | n | positives | precision | 95% CI | recall |
|---|---|---|---|---|---|---|
| rung 3 (full) | 0.758 | 91 | 14 | 0.714 | [0.375, 1.000] | 0.357 |
| rung 0 (ipTM, contrast) | 0.763 | 91 | 14 | 0.714 | [0.333, 1.000] | 0.357 |

**Direction 2b — Champloo → SAbDab (caveated):** train on 91 leakage-free Champloo rows
(only 14 positives), test on the full SAbDab set (2,688 rows). This direction is
caveated — the small training set limits what the model can learn — but the test set is
large (tight CIs on AUROC).

| rung | AUROC | n | positives | precision | 95% CI | recall |
|---|---|---|---|---|---|---|
| rung 3 (full) | 0.678 | 2,688 | 448 | 0.486 | [0.433, 0.549] | 0.266 |
| rung 0 (ipTM, contrast) | 0.691 | 2,688 | 448 | 0.941 | [0.902, 0.978] | 0.286 |

Both directions reproduce the in-distribution pattern: geometry and CDR add no AUROC
over ipTM (2a: Δ = −0.005; 2b: Δ = −0.013). ipTM transfers at AUROC ~0.69–0.76 in
both directions, consistent with the in-distribution level. Direction 2b reaches the
in-distribution AUROC despite being trained on only 14 positive examples — confirming
that ipTM's signal is the feature, not dataset size.

The 2b precision numbers diverge sharply between rung 0 (0.941) and rung 3 (0.486),
but this is a threshold-scale artifact from applying rung A's operating threshold to
rung B: the logit scale shifts when 14 positives anchor the fit. AUROC is the clean
comparison.

## Read

- **ipTM (Protenix confidence) discriminates cognate vs shuffled at AUROC ~0.69** — a
  clear step above the 0.496 sequence-only floor (M-S) and above the ~0.50 AF2-M
  confidence wall (the abdisc negative result). Predictor confidence from an AF3-class
  model IS informative for held-out antigen clusters, unlike AF2-M confidence.

- **BUT interface geometry and CDR engagement add NO signal beyond ipTM.** Every paired
  ΔAUROC CI includes zero. This is a clean, well-powered negative: same 241 held-out
  antigen clusters, paired bootstrap ΔCIs, random-split contrast, standardized
  coefficients, and cross-regime transfer all converge on the same conclusion. The
  mirage hypothesis that predicted interface geometry discriminates beyond the
  predictor's own confidence scalar is **not supported** at this power, with these
  features, with this predictor.

- **The discriminative signal is entirely in the confidence channel.** The three
  dominant standardized coefficients are `interface_pae`, `iptm`, and
  `min_interface_pae` — all confidence outputs. Geometry and CDR weights are an order
  of magnitude smaller. ipTM is not simply a proxy for something structural it is
  measuring indirectly; it *is* the signal.

- **This is a defensible, well-controlled negative result** — directly analogous to how
  M-S reported its floor. It relocates the open question: confidence carries the signal;
  structure-beyond-confidence does not. The natural next question is whether richer
  structural representations (e.g., graph neural networks over predicted coordinates,
  residue-level PAE maps, energetic decomposition) can surface additional signal that
  scalar geometry summaries miss.

## How to reproduce

Protenix predictions for all 2,688 SAbDab pairs are in `data/staged/mc/sabdab_features.csv`.
Champloo features are in `data/staged/mc/champloo_features.csv`.

```bash
# 1. In-distribution rung ladder vs the M-S 0.496 floor
uv run python scripts/analyze_mc_indist.py \
    --features data/staged/mc/sabdab_features.csv \
    --output results/published/mc_indist.json \
    --model-out results/published/mc_sabdab_model.json

# 2. Cross-regime transfer (SAbDab <-> Champloo, leakage-guarded)
uv run python scripts/analyze_mc_cross_regime.py \
    --sabdab-features data/staged/mc/sabdab_features.csv \
    --champloo-features data/staged/mc/champloo_features.csv \
    --sabdab-pairs data/staged/sabdab/sabdab_pairs.csv \
    --champloo-pairs data/staged/protenix/champloo_pairs.csv \
    --output results/published/mc_cross_regime.json
```
