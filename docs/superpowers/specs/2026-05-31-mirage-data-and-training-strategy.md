# mirage — Data & Training Strategy  (2026-05-31)

**Status:** addendum to and refinement of the locked design spec
`2026-05-28-mirage-design.md`. That spec defined *what* mirage is (a
binding-correctness discriminator) and left the primary success metric and the
data strategy open. This document closes the data/ML side: it fixes the
scientific framing, the metric *shape*, the dataset registry, and the
train/test/validate strategy. Decisions here were reached in the 2026-05-31
brainstorming session and reflect feedback from Dan.

---

## 1. Scientific motivation (framing)

### The core problem
Structure predictors (AlphaFold3 and successors) render a stereochemically
perfect antibody–antigen complex for **essentially any pair** — cognate or not.
Hand AF3 a nanobody and an antigen that have never met and it still produces a
clean, plausible-looking interface. The model is excellent at *docking geometry*
and largely blind to *whether binding actually happens*. This is the gap the
Smorodina paper named: **structural plausibility without binding specificity.**

### Why it matters
This is not academic. In-silico binder design — including the lab's downstream
driver, CAR-T binders against tumor-restricted antigens (CHL1, MSLN; CD19
control) — rests on screening *predicted* complexes to decide what to build and
test. If the predictor confidently renders convincing interfaces for
non-binders, the screen is contaminated with **mirages**: beautiful structures
of binding events that do not exist, burning scarce wet-lab effort on false
leads. The field needs to ask, of a predicted complex, "is this a real binding
event, or a plausible-but-incorrect rendering?"

### The honest complication (why this is hard, and why the project pivoted)
The obvious answer is "use the predictor's own confidence (ipTM)." The
inconvenient finding from the abdisc/Champloo work that motivates this line is
that **ipTM is already fairly good** at separating cognate from shuffled pairs
(AUROC ~0.754 / AP ~0.197 locally) and nothing cheap we tried beats it. The
earlier framing — a predictor-agnostic discriminator with positive =
RMSD-to-crystal pose — hit a wall: RMSD measures *pose accuracy*, not *binding
existence* (the wrong target), and agnostic geometry lost to ipTM. That negative
result is why mirage exists in its current form, and why mirage is deliberately
**predictor-conditional** (it consumes ipTM/PAE/pLDDT rather than discarding
them).

### The scientific question we are planning
> **Can we judge whether a predicted antibody–antigen complex reflects a true
> binding event, beyond what the predictor's own confidence already tells us —
> and specifically, *where and when* does that confidence become insufficient?**

Made concrete and falsifiable:

- **Operationalized as a gate.** Not an abstract score — a binary
  binder/non-binder filter, **FP-costly** (better to miss a binder than waste a
  wet-lab slot on a non-binder). Measured in sensitivity/specificity at a
  high-precision operating point.
- **The real test is generalization, not in-sample accuracy.** Dan's skepticism
  — "feature signal varies with dataset, the positive cohort is small, I don't
  buy these results" — *is* the scientific question in disguise. A gate that only
  works on the dataset it was tuned on is itself a mirage of a discriminator. So
  the headline is **not** "beat ipTM on Champloo"; it is whether a
  Champloo-*frozen* gate holds its precision on **orthogonal true positives it
  never saw** (SAbDab, AVIDa), across **binder formats** (VHH → Fab/scFv), and on
  **designed binders** (EpCAM).
- **Real experimental negatives earn first-class status.** Two held-out sets
  carry them — **AVIDa-hIL6** (binding assay, natural VHHs) and **EpCAM** (CAR-T
  killing, designed VHHs, N=14) — letting us probe the regime where predictor
  confidence is most likely to fail, which constructed shuffled negatives cannot.
  EpCAM is the smaller but more deployment-faithful of the two.

### One-sentence version
Structure predictors hallucinate convincing antibody–antigen interfaces whether
or not the pair binds; **mirage builds, and rigorously stress-tests across
datasets and formats, a gate that distinguishes true binding from a confident
structural mirage — and maps exactly where the predictor's own confidence stops
being enough.** That last clause is the genuinely new science: v1 is **baseline
characterization of where ipTM is and isn't sufficient**, with the
orthogonal/real-negative datasets as the instrument for finding the "isn't."

---

## 2. Metric shape (resolved at altitude)

The primary metric is fixed in *shape*; precise thresholds are tuned during
implementation, not pre-locked.

- **Use:** binary gate. **Cost:** FP-costly → fix a **high-precision operating
  point**, report the **recall/sensitivity** achieved there, plus
  **specificity** (1 − false-pass rate). This is exactly Dan's
  sensitivity/specificity language.
- **Floor (necessary, not sufficient):** match or beat raw ipTM on
  cognate-vs-shuffled. Clearing it counts as "anything works"; it is **not** the
  headline (locking it would re-create the wall that retired abdisc).
- **Headline:** **cross-regime precision stability** — freeze the threshold that
  achieves the target precision on a calibration split, then measure realized
  precision/sensitivity on held-out regimes (held-out antigen, held-out VHH,
  Fab/scFv formats, EpCAM designed binders, and the orthogonal datasets).
  Report with **bootstrap confidence intervals** so small cohorts are handled
  honestly. Also report the prevalence-independent false-pass rate and a
  **precision-vs-prevalence sweep extending to ~1:10,000**, reported as the
  positive predictive value (PPV) the lab should expect when pulling hits from a
  real in-silico screen. Precision at Champloo's ~1:105 ratio badly overstates
  deployed PPV; the sweep is what the wet-lab decision actually consumes.

---

## 3. Dataset registry (v1)

| Dataset | Role | True positives (binding evidence) | Negatives | Predicted complex? | Confidence (ipTM/PAE/pLDDT)? | Format | Key caveats |
|---|---|---|---|---|---|---|---|
| **Champloo / Smorodina** | **Primary** (train + in-dist test) | 106 cognate VHH–Ag, co-crystal | ~11,130 **constructed** shuffled non-cognate (106×106 off-diagonal) | Yes — AF3 staged (8,223 files) | Yes — ipTM staged | VHH | Negatives synthetic, not assayed; small positive cohort (106); heavy class imbalance (~1:105) |
| **SAbDab** | **Orthogonal TP reservoir** + format axis | ~2,392 nanobody-vs-protein rows → **~1k nonredundant VHH**; +thousands Fab/scFv; all co-crystal | none native | No — crystal only | No (requires AF3 run) | VHH + Fab/scFv | Predictor-conditional use **requires running AF3**; needs redundancy clustering |
| **EpCAM (SNAP + collaborator)** | **Orthogonal test — designed-binder deployment regime, real labels** | **8 functional ("Good") VHHs** (CAR-T killing assay vs AsPC1 EpCAM⁺): 10, 25, 26, 34, 57, 61, 74, 86; plus the broader 43 designed POS | **6 non-functional ("Bad") VHHs (REAL):** 14, 15, 16, 18, 21, 73; plus 86 SCR + 24 OFF (designed, unassayed) | not yet (verify SNAP for AF2-M/mBER structures) | not yet | VHH (designed) | **N=14 labeled — tiny → held-out test only, wide CIs.** Label = **functional CAR-T killing**, one step downstream of binding (a "Bad" VHH may be a non-binder *or* a non-productive binder). Most deployment-relevant set we have. Lives in SNAP data dir; VHH IDs need mapping to sequences |
| **AVIDa-hIL6** | **Orthogonal test only** (first-class) | many VHH binders to IL-6 family (assay label) | many **real assay-based non-binders** | No — sequence only | No | VHH | Sequence-only (no structures); HF config broken → use raw CSVs. Its negatives are **real** — the one set escaping the synthetic-negative limitation |
| **Germinal** | Parked design case study | 24 BLI-positive designed | no clean public negatives | — | — | designed | Not benchmark-ready; parked until per-design negatives recovered |

**Positive label = experimental evidence of binding** (co-crystallization for
Champloo/SAbDab; CAR-T killing for EpCAM; binding assay for AVIDa). **v1
negatives are constructed shuffled non-cognate** for the training set (Champloo),
but **two datasets carry real experimental negatives** and are reserved as
held-out tests: **AVIDa** (binding assay, natural VHHs) and **EpCAM** (CAR-T
killing, designed VHHs — the deployment regime). Together they are the
orthogonal probe for the regime ipTM is expected to miss, and they **partially
reach the experimental-non-binder tier the 2026-05-28 spec had deferred.**
EpCAM's label is *functional* (killing), one step downstream of binding — a
noisier but maximally decision-relevant negative. *(Open PI item, deferred to
Dan: exact experimental-binding evidence accepted per dataset.)*

---

## 4. Train / test / validate strategy

### ML setup
- **Task:** binary classification — true binding vs structurally-plausible-but-
  false — used as an FP-costly gate.
- **Model class:** simple and **regularized** (logistic regression /
  gradient-boosted trees), **not** a large net. With ~106 training positives a
  heavy model overfits and feeds exactly the "I don't buy it" critique. Matches
  the Phase-1/2 precedent.

### Feature tiers (what makes the staged plan work)
- **Tier-S — sequence-level:** computable on *every* dataset, including
  structure-less AVIDa and crystal-only SAbDab (VHH/antigen sequence properties;
  optionally ESM-2 perplexity/embeddings).
- **Tier-C — predictor-conditional:** ipTM / PAE / pLDDT + interface geometry
  from the predicted complex. Champloo today; AF3-lifted SAbDab subset later.

### Two model variants
- **M-S (sequence-only) — *pre-structure baseline only*.** Trainable and
  orthogonally validatable *today*; it proves the data/feature/validation
  infrastructure and gives a sequence-from-binding reference. **It does not
  answer the headline question.** mirage's stated motivation is the failure of
  *structural-predictor confidence* (hallucinated interfaces despite confident
  ipTM); a sequence-only model answers the older "can we predict binding from
  sequence alone?" question instead. M-S must be framed to Dan strictly as the
  baseline, not as evidence for or against the scientific claim.
- **M-C (full predictor-conditional) — *the model that fulfills the
  motivation*.** Only M-C, which consumes ipTM/PAE/pLDDT + interface geometry,
  speaks to where predictor confidence is and isn't sufficient. Champloo now;
  orthogonal validation once the AF3-lift lands. The project's headline result
  comes from M-C, not M-S.

### Splits & roles
- **Train:** Champloo (106 cognate positives + shuffled non-cognate negatives).
- **In-distribution test:** Champloo **grouped** held-out — **held-out-VHH** and
  **held-out-antigen** splits (leakage controls; already specified in the
  Phase-1 plan). Random-pair split as a sanity check only. **Feature-attribution
  check (SHAP):** because the simple/regularized models invite it, pull SHAP
  values on the in-distribution test to confirm the model is not keying on
  trivial sequence mismatches from the shuffle construction (e.g. CDR-charge ↔
  antigen-charge gross complementarity) rather than binding-relevant signal. A
  model that learns "this VHH doesn't *look like* it belongs near this antigen
  class" will pass cognate-vs-shuffled yet collapse on EpCAM designed binders,
  whose sequences are highly optimized and just barely fail to bind — **EpCAM is
  the designated canary for this failure mode.**
- **Orthogonal validation (the generalization test):**
  - **SAbDab ~1k nonredundant VHH positives** → sensitivity of a Champloo-frozen
    gate (does it keep real binders it never saw?). Fab/scFv subsets exercise the
    format-generalization axis.
  - **AVIDa-hIL6** → held-out test with **real assay negatives** → genuine
    sensitivity *and* specificity on an independent antigen system.
  - **EpCAM (N=14: 8 functional / 6 non-functional)** → small but
    maximally deployment-relevant held-out test: **designed VHHs with real CAR-T
    killing labels** on a target the lab actually pursues. Confirmatory only,
    reported with wide bootstrap CIs; never trained on. This is also where the
    SHAP canary becomes a *quantitative* labeled check rather than a qualitative
    worry.
- **Floor baseline, everywhere available:** raw ipTM.

### Staged execution
- **Phase A (now, no GPU):** commit the registry artifact → assemble Tier-S
  features → train M-S on Champloo → orthogonally validate on SAbDab positives +
  AVIDa. Produces the first sensitivity/specificity-with-CIs and cross-dataset
  generalization numbers Dan asked to see.
- **Phase B (parallel):** queue AF3 on a nonredundant SAbDab VHH subset → Tier-C
  features → train M-C → full predictor-conditional orthogonal validation.

### Feasibility risk (flagged, not hidden)
Champloo's AF3 structures came **pre-computed from Zenodo** — we have not run AF3
ourselves. AF3 weights are access-gated, so Phase B's "run AF3 on SAbDab" has an
access/compute dependency to confirm before committing. **Fallback:** a runnable
predictor (Boltz-2 / Protenix / ColabFold AF2-M), accepting a different
confidence base and documenting the predictor swap. **Critical constraint on the
fallback:** ipTM/PAE/pLDDT are calibrated *per predictor*, so M-C must **never**
be trained on AF3 (Champloo) and validated on a fallback predictor (SAbDab) —
the confidence features would be on incomparable scales. If the fallback is
triggered, the Champloo **training** set must be **re-run through the same
fallback predictor** so train and test share one uniform confidence base. **Phase
A does not depend on this** — it unblocks immediately.

### Methodological guards (from the 2026-05-31 critical review)
Carry these into the implementation plan as explicit checks, not afterthoughts:

1. **M-S is a baseline, not the answer.** Only M-C (predictor-conditional)
   addresses the scientific motivation; present M-S accordingly.
2. **Watch what the model rejects.** SHAP on the in-distribution test; EpCAM
   designed binders are the canary for trivial-mismatch shortcutting.
3. **Report PPV across prevalence to ~1:10,000**, not just precision at
   Champloo's ~1:105.
4. **One predictor across train and test.** If AF3 is unavailable, re-run
   Champloo through the fallback predictor; never mix confidence bases.

---

## 5. How this answers Dan

- *"Feature signal varies with dataset"* → freeze the model on Champloo, measure
  on SAbDab/AVIDa. If sensitivity holds, signal generalizes; if it collapses, the
  critique is confirmed with numbers. Either outcome is an honest, citable
  result.
- *"Positive cohort is small"* → SAbDab contributes ~1k orthogonal positives
  (≫106); all metrics reported with bootstrap CIs.
- *"Validate with an orthogonal true-positive dataset"* → SAbDab (TP reservoir),
  AVIDa (orthogonal, real negatives), and EpCAM (designed-binder deployment
  regime, real CAR-T killing labels).
- *"Generate a registry of datasets with caveats and think through test/train/
  validate"* → Section 3 + Section 4; the registry becomes a living artifact in
  the repo.

---

## 6. Open items (carried to PI / parallel tracks)
- (PI) Exact experimental-binding evidence accepted per dataset — including
  whether EpCAM's **functional CAR-T killing** label is accepted as the
  binding-evidence proxy, or down-weighted relative to direct binding assays.
- (Data staging) Map the labeled EpCAM VHH IDs (Good: 10, 25, 26, 34, 57, 61,
  74, 86 / Bad: 14, 15, 16, 18, 21, 73) to sequences. **EpCAM predicted complexes
  do not exist and must be generated** (predictor track, alongside the SAbDab
  lift) — so EpCAM tests **M-S in Phase A**, and **M-C only after generation**.
- (PI) Ratification of the working name "mirage."
- (Parallel) AF3 (or fallback predictor) run feasibility on PARCC for the SAbDab
  lift; mBER / additional orthogonal datasets from Pierce.
- (Implementation) Precise high-precision operating point and prevalence
  assumptions for the headline metric.
