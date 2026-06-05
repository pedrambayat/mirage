# mirage M-C — predictor-conditional structure track — Design Spec (2026-06-02)

> **Status:** design locked 2026-06-02. Successor to the concluded M-S
> sequence-only floor. Addendum to the 2026-05-28 mirage design spec and the
> 2026-05-31 data & training strategy. Implementation plan to follow.

## 1. Context & lineage

M-S (the sequence-only, no-GPU gate) is **done and concluded**: on SAbDab VHH-only
(448 cognate positives / 241 antigen clusters), under a held-out-antigen-cluster
5-fold OOF split with distribution-matched cross-cluster negatives (k=5), a
four-rung ladder (additive Tier-S → ESM-2 concat → diagonal bilinear → low-rank
bilinear) lands at **AUROC ≈ 0.50** (frozen bilinear **0.496**). A random-split
contrast confirmed this is a genuine absence of transferable antigen-specific
signal, not a held-out artifact. Full writeup:
`results/published/sabdab_baseline_summary.md`.

M-S is, by construction, the **pre-structure baseline** — it does not consume the
predictor's confidence and so does not answer mirage's headline question. **M-C is
that model**: predictor-conditional, consuming a *predicted* `(binder, target)`
complex + the predictor's own confidence + interface geometry.

**M-C must beat 0.496 on the *same* held-out-antigen-cluster OOF split, over the
*same* SAbDab cognate set (448 VHH / 241 clusters) + distribution-matched
negatives**, for an apples-to-apples claim.

## 2. Scientific question

Structure predictors render a stereochemically plausible antibody–antigen complex
for essentially any pair — cognate or not (the Smorodina gap: structural
plausibility without binding specificity). Given a *predicted* complex and the
predictor's own confidence, can a lean discriminator judge whether the complex
reflects a **real binding event** — and in particular, does the **structure of the
predicted interface** add discriminative signal *beyond* the predictor's headline
confidence (ipTM)? The abdisc-era negative result — AF2-M confidence ≈ 0.50 AUROC
for binder/non-binder — is the wall M-C is built to test against with a modern
AF3-class predictor and explicit interface-geometry features.

## 3. Decisions locked (this session)

1. **Predictor: Protenix** (open AF3-class), installed into its own shell-out env.
   AF3 proper is not runnable on PARCC (gated weights; Champloo's AF3 came from the
   paper). Protenix emits full confidence (ipTM / PAE / pLDDT) + structure and runs
   on B200.
2. **Scope: single predictor (Protenix) spans both datasets** — Champloo as the
   in-distribution training source and SAbDab as the held-out reservoir. This is the
   most faithful realization of the original "Champloo trains, SAbDab validates"
   design and automatically satisfies the "one predictor across train/test" guard.
   Bonus: Champloo already carries AF3 structures + ipTM, so the Protenix run yields
   a direct AF3-vs-Protenix comparison on the same 91 cells.
3. **Negatives: predict-the-shuffled-pair, full apples-to-apples campaign.** Each
   non-cognate (VHH × wrong-antigen) pair gets its own predicted complex. SAbDab =
   all 2,688 M-S rows (k=5). Champloo = 91 positives + 455 matched k=5 negatives
   (546 rows). No decoy docking.
4. **Model: lean, regularized L2-logistic ladder** with a curated handful of
   interpretable descriptors per rung; coefficients + SHAP. Dan-defensible at 448
   positives.
5. **Real-negative *structural* orthogonal tier (AVIDa / EpCAM via Protenix):
   deferred** to a later phase. v1 = Champloo + SAbDab only. AVIDa stays held-out
   regardless.

## 4. Predictor & staging (A)

- **`ProtenixPosePredictor`** mirrors the existing `AbstractPosePredictor` /
  `AF2MPosePredictor` contract: `stage(examples) → manifest`, SLURM array `submit`,
  per-example output `<root>/<id>/rank1.pdb` + `confidence.json`. Account
  `dbgoodma-goodman-laboratory`, partition `b200-mig45` / `dgx-b200`, `--gres=gpu:1`.
- **Leakage guard — MSA-only, templates DISABLED.** Protenix must not receive
  structural templates (would let it copy the SAbDab crystal). This is the
  structural analog of the af2m no-templates lock-in and is a hard invariant.
- **MSA caching per unique sequence** (~800 unique sequences across both sets —
  448 SAbDab VHH + ~241 SAbDab antigens + 91 Champloo binders + 91 antigens),
  reused across all pairings. Only per-pair inference scales with pair count
  (~3,200 inferences total).
- **Chain convention:** binder chain(s) first, then antigen — matches the af2m
  FASTA convention so `predicted_chain_ids(example)` and `StructuralInterfaceScorer`
  keep working unchanged.
- **mirage stays torch-free.** GPU work shells out; mirage reads only cached PDBs +
  confidence JSON + a staged feature CSV. Predictions live under
  `data/raw/predictions/protenix/` (gitignored).
- **Install-validation gate:** Protenix must reproduce the qualitative
  cognate ≫ shuffled ipTM gap on the 91 Champloo diagonal before the full campaign
  proceeds. (No Champloo Protenix-ipTM matrix exists to match cell-for-cell;
  optionally compare Protenix diagonal structures to the staged AF3 ones.)

## 5. Datasets & negatives (B)

- **Negative construction = predict-the-shuffled-pair.** The predictor renders a
  "mirage" for non-cognate pairs; its confidence + interface geometry on that
  rendering is the discriminative signal. Matches M-S's negative semantics exactly.
- **SAbDab (held-out reservoir, headline):** all **2,688** rows = 448 cognate
  positives + 2,240 distribution-matched cross-cluster negatives (k=5) — the *exact*
  rows behind the 0.496 floor, reusing `stage_sabdab_pairs` output unchanged so the
  M-C AUROC is computed on the same rows and folds.
- **Champloo (in-distribution train source):** 91 cognate-diagonal positives +
  455 matched k=5 cross-cluster negatives (546 rows total), staged with the same
  negative-matching logic.
- **Single predictor across both** (Protenix) → never mix predictors within a
  train→test pipeline.

## 6. Features — the rung ladder (C)

Each rung adds one isolated signal (mirrors the M-S additive→bilinear ladder) so we
can read exactly where discrimination appears:

| Rung | Adds | Source |
|---|---|---|
| **0** | ipTM-alone | Protenix confidence — the abdisc wall / floor *within* M-C |
| **1** | + confidence internals: pTM, interface-PAE (binder↔antigen block), interface pLDDT, mean pLDDT | new `protenix_confidence` extractor |
| **2** | + interface geometry (curated handful: interface-residue count, buried-SASA proxy, atom-contacts, shape-complementarity proxy, clash fraction) | **reuse `StructuralInterfaceScorer`** (predictor-agnostic), select a defensible subset of its ~50 columns |
| **3** | + CDR-engagement: fraction of interface contacts via CDR vs framework residues | **new** extractor — ANARCI IMGT CDR annotation mapped onto the predicted binder chain |

- Every rung is compared against the **M-S 0.496** anchor on the same OOF split.
- **No RMSD-to-crystal / DockQ-to-crystal feature.** Negatives have no cognate
  crystal of the non-cognate complex, and that label was retired with abdisc.
  Geometry is computed from the predicted complex alone.
- Curated subset keeps the head ~6–10 features per rung — the Dan-defensible
  discipline for 448 positives.
- **Rung 3 robustness — row-preserving CDR fallback.** Predictors can render
  distorted CDR loops when a VHH is forced against a non-cognate antigen, and ANARCI
  may fail to confidently map the CDR3 onto the predicted binder chain. When that
  happens the row is **not dropped** (dropping would desync the rung tables and break
  the apples-to-apples comparison against Rung 0); instead the CDR-engagement
  features take a **default value (0.0)** and a per-row `cdr_mapping_ok` flag plus an
  overall **CDR-mapping failure rate** are recorded and reported. A high failure rate
  is itself a finding (a structural signal that non-cognate poses degrade the CDR),
  not a silent gap. All rungs are evaluated on the **same row set** so deltas between
  rungs are paired.

## 7. Model & gate (D)

- **Lean L2-logistic ladder** (extends the M-S logistic). One curated head per rung;
  report **coefficients + SHAP** for interpretability and the EpCAM-style
  trivial-shortcut canary discipline.
- **Gate framing carried from M-S:** FP-costly → high-precision operating point,
  sensitivity/specificity, **bootstrap CIs** (reuse `eval/gate.py`), PPV-vs-prevalence
  sweep toward deployment rarity.
- **Threshold-scale fix (from M-S review):** pick the operating threshold on the
  *full-fit* model's own logits (the shipped `predict_logit` scale), never on OOF
  scores. OOF is for the honest in-distribution report only.

## 8. Evaluation design (E)

1. **SAbDab-internal OOF** (held-out-antigen-cluster 5-fold, the *same* folds as
   M-S), two nested questions:
   - **Floor check (necessary, not the headline):** does each rung beat **0.496**?
     Rung 0 (ipTM-alone) is *expected* to clear it comfortably — predictor confidence
     already separates cognate from shuffled — so beating 0.496 alone proves little.
   - **The real headline — does geometry beat confidence?** The primary scientific
     claim is **Rung 2/3 > Rung 0**: interface geometry (and CDR engagement) adds
     discriminative signal *on top of* ipTM. This is measured as a **paired
     rung-delta**: a single bootstrap that resamples rows once and computes
     ΔAUROC and Δprecision-at-operating-point *between rungs on the same resampled
     rows* (CI on the difference, not two overlapping marginal CIs). The result
     counts only if the ΔAUROC CI excludes 0. (`eval/gate.py` gains a paired-delta
     bootstrap helper.)
2. **Cross-regime transfer — run BOTH directions** (enabled by having one predictor
   on both sets). Train a frozen gate on one regime, apply to the other via the
   `FrozenGate` / `evaluate_frozen_gate` harness — mirage's "cross-regime precision
   stability" headline. **Guard (both directions):** dedup Champloo antigens against
   SAbDab antigen clusters so the transfer is not leakage.
   - **2a. SAbDab → Champloo (the robust direction):** train on SAbDab (~450
     positives), test on the Champloo diagonal. The larger, more diverse training set
     yields a better-fit gate; this asks whether SAbDab's structural rules generalize
     to Champloo. Treat as the *primary* transfer result.
   - **2b. Champloo → SAbDab (the original direction):** train on Champloo (546 rows),
     test on the SAbDab reservoir. **Inverted-transfer caveat:** training on 91
     positives and testing on a large diverse reservoir means a *failure here may be
     a false negative from underfitting*, not evidence the signal is absent — so 2b is
     read only in light of 2a, never on its own.
3. **Random-split leakage contrast** (M-S `--random-folds` analog): confirms any
   held-out signal is not a memorization artifact.
4. **Free AF3 companion:** the staged Champloo **AF3** matrix runs the *same*
   rung-0→2 ladder. AF3 has structures + pLDDT-in-B-factor + scalar ipTM but **no
   PAE JSON locally**, so its rung-1 is partial. Answers the design-spec question —
   *with the strong predictor, does interface geometry add beyond ipTM?* — and gives
   a direct AF3-vs-Protenix ipTM comparison on the 91 cells. Self-consistent within
   AF3; never pooled with Protenix.

## 9. Reuse / new / out-of-scope (F)

- **Reuse wholesale:** `features/clustering.py`, `scripts/stage_sabdab_pairs.py`
  (splits + distribution-matched negatives), `eval/gate.py`, `eval/orthogonal.py`
  (`FrozenGate`), `scorers/structural_interface.py` (`StructuralInterfaceScorer`),
  the antigen-cluster folds, ANARCI normalization (`features/normalize.py`).
- **New:** `pose_predictors/protenix.py` (`ProtenixPosePredictor`) + a
  `scripts/slurm/predict_protenix.slurm` wrapper; a `protenix_confidence` extractor;
  a CDR-engagement feature extractor; `scripts/stage_protenix_pairs.py` (Champloo +
  SAbDab manifests); `scripts/analyze_mc_indist.py` and
  `scripts/analyze_mc_cross_regime.py`; the rung-ladder trainer extending the M-S
  logistic.
- **Out of scope (v1):** AVIDa / EpCAM real-negative *structural* tests (deferred);
  non-VHH formats; decoy docking; any GPU package in the mirage uv env; RMSD/DockQ
  -to-crystal as a feature or label.

## 10. Success criteria & risks (G)

- **Success (the headline):** **Rung 2/3 > Rung 0 (ipTM-alone)** with a paired
  ΔAUROC bootstrap CI that excludes 0 on the SAbDab OOF split — interface geometry
  adds discriminative signal *on top of* the predictor's own confidence. Beating the
  M-S **0.496** floor is a *necessary precondition*, not the headline: Rung 0 is
  expected to clear 0.496 on its own, so the floor-beat alone proves little. A clean
  *negative* (geometry does not beat ipTM) is also a publishable, Dan-defensible
  result given the rigor (same folds, paired deltas, random-split contrast, bootstrap
  CIs).
- **Risks:**
  1. **Protenix install/runtime on PARCC** — the first feasibility gate; validated by
     the cognate ≫ shuffled ipTM check on the 91 diagonal before the full campaign.
  2. **ipTM-alone may already clear 0.496** (the abdisc wall reappearing) — acceptable;
     it relocates the bar to "geometry beats ipTM," which is the real M-C hypothesis.
  3. **448 positives × a curated head** — defended by the random-split contrast,
     SHAP, and L2 regularization, exactly as M-S defended its floor.
  4. **CDR-mapping failure on distorted non-cognate poses** (Rung 3) — handled by the
     row-preserving 0.0 default + `cdr_mapping_ok` flag + reported failure rate (§6),
     so it never silently drops rows or breaks the paired rung deltas.
  5. **Inverted-transfer false negative** — the Champloo→SAbDab direction (2b) can
     fail from underfitting on 91 positives rather than absent signal; mitigated by
     running the robust SAbDab→Champloo direction (2a) as the primary transfer read.

## 11. Invariants

- mirage package stays **torch-free**; all GPU work shells out, results cached.
- **Same held-out-antigen-cluster split** as M-S for apples-to-apples vs 0.496.
- **AVIDa stays held-out** (never trained), even when its structural test lands later.
- **No templates** in Protenix (crystal-leakage guard).
- Commits **Pedram-authored, no Claude/Anthropic trailer**.
- Build via **subagent-driven execution**, like the M-S work.
- **Sequencing dependency:** M-C's random-split contrast (§8.3) and the new
  paired-delta bootstrap (§8.1) extend `eval/gate.py` code that currently lives on the
  open M-S **PR #2** (leakage contrast + `--random-folds`). `main` is still partial
  through commit `e277371`. M-C should branch off PR #2 (or land it first) rather than
  off the partial `main`, so it inherits that scaffolding.

## 12. References

- `results/published/sabdab_baseline_summary.md` — the M-S floor (0.496) this beats.
- `docs/superpowers/specs/2026-05-28-mirage-design.md` — locked mirage design.
- `docs/superpowers/specs/2026-05-31-mirage-data-and-training-strategy.md` — staged
  M-S/M-C plan, gate framing, Dan's review-driven guards.
- `docs/datasets/dataset-registry.md` — dataset roles, label provenance, caveats.
- `src/mirage/scorers/structural_interface.py`, `src/mirage/pose_predictors/af2m.py`,
  `src/mirage/eval/{gate,orthogonal}.py`, `scripts/stage_sabdab_pairs.py` — reused
  scaffolding.
