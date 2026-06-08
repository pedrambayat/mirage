# mirage M-C — EpCAM real-negative canary — Design Spec (2026-06-07)

> **Status:** design locked 2026-06-07. First test in the **real-negative tier**
> (deferred in the 2026-06-02 M-C structure-track spec). Successor to the concluded
> M-C Phase B2 (rung ladder; clean negative — interface geometry/CDR add no signal
> on top of ipTM). Addendum to the 2026-05-28 mirage design and the 2026-06-02 M-C
> spec. Implementation plan to follow.

## 1. Context & lineage

The work so far has established, all on the **same held-out-antigen-cluster split**:

- **M-S (sequence-only):** AUROC ≈ 0.50 — chance for held-out antigens (the floor).
- **M-C Phase B2 (predictor-conditional):** ipTM alone = **AUROC 0.690** (clears the
  floor and the old ~0.50 AF2-M wall), but interface geometry + CDR engagement add
  **nothing** on top of ipTM — every paired ΔAUROC CI includes 0. A clean,
  well-powered **negative**: the discriminative signal lives entirely in the
  predictor's confidence channel.

**Every result to date rests on *constructed shuffled* non-cognate negatives.** That
is the easy regime. The claim mirage exists to make — flag a predicted complex that
*looks* like binding but isn't — is only really tested against **real** non-binders,
and against **real designed binders** in the deployment setting (not crystal-derived
positives). The 2026-06-02 spec named this the deferred **real-negative tier**. This
canary is its first, cheapest step.

## 2. Scientific question

In the CAR-T **designed-binder deployment regime** — real VHHs proposed by an
in-silico pipeline, scored against a single tumor target (EpCAM) — does the
predictor-conditional gate still behave as it did in-distribution?

1. Does **ipTM** still separate real designed binders from shuffled pairings (does
   the 0.690 in-distribution signal generalize to designed positives)?
2. Does **interface geometry / CDR engagement** add anything on top of ipTM *here*
   (re-asking the B2 question in a new regime)?
3. Does the **frozen SAbDab gate's operating threshold** hold under cross-regime
   transfer, or collapse the way the M-S Champloo gate did on AVIDa (recall 0)?

This is explicitly a **probe**, not a powered result (N = 14 positives). It is
informative whichever way it falls, and it scouts the regime that the powered AVIDa
follow-up (step 2, out of scope here) will measure properly.

## 3. Decisions locked (this session)

1. **Scope: EpCAM only.** AVIDa-hIL6 is the powered step 2, a separate spec.
2. **Approach A — apples-to-apples replay.** EpCAM negatives are constructed exactly
   as M-C did (predict-the-shuffled-pair), so the result is directly comparable to
   the 0.690 in-distribution number. The 14 real designed positives are the only
   genuinely new ingredient.
3. **Frozen transfer, no fit on EpCAM.** EpCAM is **never trained on** (too small;
   held-out forever). The SAbDab M-C gate is applied unchanged via the existing
   `FrozenGate` harness — a pure cross-regime evaluation. **Note:** only the rung-3
   gate was persisted (`mc_sabdab_model.json`); the rung-0 contrast model is **re-fit
   full-fit on the committed SAbDab feature CSV** via the same `fit_rung_model` used
   in B2 (deterministic; the rung-3 re-fit reproduces `mc_sabdab_model.json`
   bit-for-bit). "Frozen" means fit on SAbDab and applied unchanged to EpCAM, never
   trained on EpCAM — not that a persisted artifact exists for every rung.
4. **Primary readout = designed-binders-vs-shuffled** (clean, comparable). The
   functional-vs-non-functional killing read is **secondary and exploratory**
   (killing is one step downstream of binding; a "non-functional" VHH may still
   bind). Both are reported; only the primary carries an inferential claim.
5. **Predictor: Protenix**, identical config to the B1 campaign. No new GPU code.

## 4. Datasets & negative construction (A)

- **Positives (14):** the labeled EpCAM VHHs × EpCAM ECD.
  - Functional (8): IDs 10, 25, 26, 34, 57, 61, 74, 86.
  - Non-functional (6): IDs 14, 15, 16, 18, 21, 73.
  - Label provenance: collaborator CAR-T killing assay vs AsPC1 EpCAM+. Sequences
    come from the existing `epcam_killing` loader; target = `targets.EPCAM_ECD`
    (UniProt P16422 residues 24–265, 4MZV chain A).
- **Negatives (~70):** predict-the-shuffled-pair, M-C-identical — each of the 14 VHHs
  × **k = 5** wrong antigens drawn distribution-matched from the SAbDab antigen pool.
  → **84 total predicted complexes** (14 positive / 70 negative).
- **Leakage guards (hard):**
  - Dedup EpCAM (P16422) against SAbDab antigen clusters at **≥ 0.90** identity (same
    guard as the cross-regime Champloo dedup); abort/flag if EpCAM collides with a
    SAbDab training cluster.
  - Confirm none of the 14 designed VHHs appear in the SAbDab training set.
  - Verify the k = 5 drawn wrong antigens are not EpCAM-related.

## 5. Prediction & features (B) — reuse, nothing new on the GPU side

- **`ProtenixPosePredictor`** + `predict_protenix.slurm` + `run_protenix_chunk.py`,
  identical config to B1: `LAYERNORM_TYPE=native`, cu128 torch, ColabFold MSA
  (`MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com` + `--msa_server_mode colabfold`),
  `--trimul_kernel torch --triatt_kernel torch --enable_fusion false`,
  `--need_atom_confidence true`, **`--use_template false`** (crystal-leakage guard).
- **MSA precompute** per unique sequence (~14 VHH + a handful of antigens), reused
  across pairings. 84 inferences → a couple hours on `b200-mig45`; trivial vs the
  3,234-pair campaign. MIG tasks get `--cpus-per-task=8 --mem≈56G`.
- **Features:** reuse `extract_mc_features` → ipTM / pTM / interface-PAE / pLDDT
  (confidence) + 6 interface-geometry descriptors + 5 CDR-engagement features;
  assemble rungs with `mc_rungs` (`interface_plddt_missing` flag handling unchanged).
- **Sequence normalization** at the featurization boundary as in B2 (`normalize.py`:
  binders → ANARCI IMGT variable domain; antigen → signal-peptide/His-tag strip).

## 6. Evaluation — three reads (C)

1. **Primary (threshold-free).** Re-fit the SAbDab rung-0 and rung-3 models (full-fit
   on the committed `sabdab_features.csv`; rung-3 reproduces `mc_sabdab_model.json`),
   then apply their `predict_logit` to the 84 EpCAM feature rows; compute AUROC for
   designed-binders-vs-shuffled and the **paired** `paired_delta_bootstrap`
   Δ(R3 − R0). Rung-0 is monotone in ipTM, so its AUROC ≈ raw-ipTM AUROC on EpCAM.
   *Reads: does ipTM generalize to designed positives, and does geometry/CDR add
   anything here?*
2. **Calibration.** The frozen rung-3 gate at its **SAbDab P = 0.90 threshold**
   applied unchanged via `evaluate_frozen_gate` → confusion / precision / recall /
   specificity on the 84, with bootstrap CIs. *Reads: does the frozen threshold hold,
   or collapse like the M-S Champloo gate did on AVIDa?* — feeds the cross-regime
   precision-stability headline.
3. **Secondary (exploratory).** Among the 14, the frozen rung-3 mirage score for the
   8 functional vs 6 non-functional — descriptive separation (distributions +
   a descriptive AUROC) flagged **N = 14 / killing ≠ binding / hypothesis-generating
   for the AVIDa follow-up**. No inferential claim, no frozen-threshold verdict.

Outputs: `results/published/mc_epcam_canary.json` + a `mc_epcam_canary_summary.md`
writeup mirroring the B2 summary's structure and caveat discipline.

## 7. Success criteria & risks (D)

- **Success = a clean, interpretable read** (both outcomes publish, as with the B2
  floor):
  - rung-0 AUROC holding ~0.69+ → ipTM generalizes to the designed-binder regime
    (expands the claim beyond crystal positives).
  - Δ(R3 − R0) CI again including 0 → reinforces the B2 negative in a new regime.
  - The calibration read (threshold holds vs collapses) is itself a key result for
    the precision-stability headline, independent of the AUROC.
- **Risks / caveats:**
  1. **N = 14 positives** → wide CIs on every EpCAM-specific number. Mitigation: this
     is framed as a probe; the powered version is AVIDa (step 2).
  2. **killing ≠ binding** confound on read 3 → kept strictly secondary/exploratory.
  3. **Shuffle negatives still synthetic** — but the positives are now *real designed
     binders*, which is the upgrade this canary delivers over B2.
  4. **Leakage** (EpCAM vs SAbDab clusters; the 14 VHHs in SAbDab) → hard dedup guards
     in §4; abort on collision.
  5. **ANARCI mapping** on designed VHHs → row-preserving default + `cdr_mapping_ok`
     diagnostic, exactly as B2.

## 8. Reuse / new / out-of-scope (E)

- **Reuse wholesale:** `ProtenixPosePredictor` + SLURM runner + `run_protenix_chunk`,
  `precompute_protenix_msa.py`, `extract_mc_features.py`, `features/mc_rungs.py`,
  `eval/gate.py` (`paired_delta_bootstrap`, `choose_threshold_for_precision`),
  `eval/orthogonal.py` (`FrozenGate` / `evaluate_frozen_gate`),
  `mc_sabdab_model.json` (the frozen gate), the `epcam_killing` loader,
  `targets.EPCAM_ECD`, `features/normalize.py`, the antigen-cluster dedup helper.
- **New (small):** `scripts/stage_epcam_protenix_pairs.py` (14 positives + dedup'd
  distribution-matched negatives, k=5) and `scripts/analyze_mc_epcam.py` (the three
  reads → JSON + summary md). A reused/extended dedup guard if not already a callable.
- **Out of scope:** refitting on EpCAM; AVIDa (step 2); non-VHH formats; richer
  structural representations (GNN / residue-level PAE maps); decoy docking; any GPU
  package in the mirage uv env.

## 9. Invariants

- mirage package stays **torch-free**; all GPU work shells out, results cached.
- **EpCAM is never trained on** — held-out forever; frozen-transfer evaluation only.
- **No templates** in Protenix (crystal-leakage guard); identical predictor config to
  the B1 campaign so EpCAM features are commensurable with the SAbDab gate.
- **Same feature extraction + rung assembly** as B2.
- Commits **Pedram-authored, no Claude/Anthropic trailer**.
- Build via **subagent-driven execution**, like the M-S / M-C work.
- **PI note:** the primary success metric is still formally open; this canary is
  consistent with the working "cross-regime precision stability of an FP-costly gate"
  assumption and is cheap enough to run ahead of formal ratification.

## 10. References

- `results/published/mc_indist_summary.md` — the B2 rung-ladder negative this extends.
- `results/published/mc_campaign_record.md` — the B1 Protenix campaign + config.
- `docs/superpowers/specs/2026-06-02-mirage-mc-structure-track-design.md` — names the
  deferred real-negative tier this canary opens.
- `docs/superpowers/specs/2026-05-28-mirage-design.md` — locked mirage design.
- `docs/datasets/dataset-registry.md` — EpCAM label provenance + caveats.
- `src/mirage/benchmark/epcam_killing.py`, `src/mirage/eval/{gate,orthogonal}.py`,
  `src/mirage/features/mc_rungs.py`, `results/published/mc_sabdab_model.json` — reused
  scaffolding.
