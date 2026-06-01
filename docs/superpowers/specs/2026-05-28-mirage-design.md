# mirage (working name) — Design Spec  (2026-05-28)

**Scientific question.** Structure predictors (AlphaFold3 and successors) render a stereochemically
plausible antibody–antigen complex for essentially any pair — cognate or not. mirage asks: given a
predicted complex, can a scorer judge whether it reflects a *real binding event* — whether the
antibody and antigen actually bind — beyond what the predictor's own confidence (ipTM) already
tells you? This is the Smorodina gap: structural plausibility without binding specificity. The
project is named for the failure mode it detects — a predictor renders a convincing *mirage* of
binding whether or not the pair binds. mirage is a binding-correctness discriminator that assigns a
*mirage score* estimating how likely a predicted interface is a plausible-but-incorrect rendering (a
mirage) rather than a true binding event. [cite: Smorodina paper note — "Structural Plausibility
Without Binding Specificity"; AF3 AP 0.187, Chai-1 0.067, Boltz-2 0.026, baseline ~0.011]

**Motivation.** The downstream driver is CAR-T binder design against tumor-restricted splice-variant
antigens in the Goodman Lab (CHL1, MSLN; CD19 control), where in-silico screens propose VHH binders
and we must know which predicted complexes are genuine recognition before spending wet-lab effort.
But the failure mode — confident, plausible structures for non-binders — is general to
antibody–antigen prediction, so mirage is scoped to the general problem with CAR-T as the
motivating application, not the boundary.

**Operational positive label.** A positive is a cognate antibody–antigen pair with *experimental
evidence of binding*. In v1 that evidence is co-crystallization (Smorodina/Champloo cognate diagonal
pairs and SAbDab complexes) plus the SNAP-team-validated EpCAM POS designs — a **proxy for binding,
not a direct binding-vs-non-binding assay label**. Negatives in v1 are **constructed shuffled
non-cognate pairings, not experimentally verified non-binders** — a known limitation (and the same
care the Champloo paper itself takes with this construction) that motivates the deferred
experimental-non-binder tier. The label is binding *existence* — NOT RMSD-to-crystal pose accuracy
(the retired abdisc label, which does not measure what binder design needs) and NOT a wet-lab screen
readout (we are not predicting a screen; assay-status and multiplexing are out of scope).
[to confirm: Dan — exact experimental-binding evidence accepted per dataset]

**Target distribution (v1 datasets).**
- *Smorodina/Champloo (primary):* VHH–antigen cognate diagonal positives + constructed shuffled
  non-cognate negatives, with AF3 structures + ipTM already staged from the Champloo reproduction —
  defines the core correct-vs-plausible-incorrect task.
- *EpCAM VHH (positive control):* SNAP-validated real-binder VHHs, anchoring the positive class in
  designed-binder evidence beyond co-crystallization.
- *SAbDab (multi-format reservoir):* broad antibody–antigen complexes spanning Fab/scFv/VHH,
  feeding the format-generalization evaluation axis.

**Negatives (tiered).** v1 = constructed shuffled non-cognate (Smorodina construction). Deferred,
named tiers: experimental non-binders; failed designs. The deferred tiers are where ipTM is most
likely to be insufficient and are the intended proving ground for mirage's headline metric.

**Binder-format scope.** v1 trains/evaluates on VHH. Fab, scFv, minibinders, peptides are a
committed *evaluation axis* — the project explicitly tests format generalization (the existing
`binder_format` field on benchmark examples and the format-agnostic `StructuralInterfaceScorer`
already span these), not a vague someday.

**Predictor & input interface.** AF3 in practice — tractable and the only predictor with a usable
specificity base (AP 0.197 vs Boltz-2 0.026 / Chai-1 0.067 in the Champloo reproduction). The
predictor is swappable, but mirage is **predictor-conditional, not pose-pipeline-agnostic**: the
mirage score consumes the predicted complex *and* the predictor's confidence outputs (ipTM / PAE /
pLDDT) as features. This is a deliberate reversal of abdisc's implicit agnostic-by-design framing,
which Champloo Phase 2 showed loses to raw ipTM (predictor-agnostic AF3 geometry did not beat ipTM;
AF3 outputs essentially never clash). We own that the signal lives partly in the predictor's own
confidence and build on it rather than discarding it.

**Improvement metric [OPEN — resolve in-session, not pre-locked].** Beating raw ipTM on
cognate-vs-shuffled (AP/AUROC, Smorodina-style) is the **floor a v1 result must clear to count as
anything — not the headline success criterion.** Locking "beat ipTM on the same task" as the
primary goal would re-create the exact wall that retired abdisc (Champloo Phase 1+2: nothing cheap
beats AF3 ipTM, AP 0.197 / AUROC 0.754). The headline metric should instead capture *where and when*
predictor confidence is insufficient — e.g. calibration error, top-k precision in regimes where ipTM
is known to fail, and performance on the negative tiers ipTM misses (deferred experimental-non-binder
/ failed-design tiers). v1 data alone cannot fully measure this; **v1 is therefore explicitly
baseline characterization of where ipTM is and isn't sufficient.** The primary metric is decided in
the spec-authoring session, not here.

**In scope / out of scope.** *In:* binding-correctness scoring of predicted antibody–antigen
complexes; VHH v1 with multi-format evaluation; AF3 + its confidence as input; characterizing where
ipTM is/isn't sufficient. *Out:* screen-readout prediction; CAR-T-specific-only scope;
RMSD-to-crystal as primary label; predictor-agnostic-by-design ambition; "beat ipTM on
cognate-vs-shuffled" as the headline goal; a learned net as a goal in itself;
experimental-non-binder and failed-design negative tiers (deferred); non-VHH formats beyond the
evaluation axis.

**First experiment (post-spec): formalize + publish the Smorodina reconciliation.** Phase 1+2 already
largely reproduced Smorodina; the v1 first task is to **write up the methodology deltas** between our
run and the paper, confirm the headline numbers reconcile within tolerance (local AF3 ipTM AP 0.197 /
AUROC 0.754 vs paper AF3 AP 0.187; Chai-1 0.067, Boltz-2 0.026, baseline ~0.011), and publish a
single citable reconciliation artifact (e.g. `mirage_smorodina_reconciliation.md` + table). This
defines "where ipTM is sufficient." The first *new-science* experiment is whatever the metric
decision implies — the first regime where we have reason to believe ipTM fails (a deferred negative
tier).

**What abdisc carries forward.** *Code:* scorer framework (`src/abdisc/scorers/base.py`),
`StructuralInterfaceScorer` (`src/abdisc/scorers/structural_interface.py`, format-agnostic), Champloo
staging / Zenodo range-extraction pipeline (`scripts/stage_champloo_pairs.py`,
`scripts/stage_champloo_structures.py`), controls infra, and the SAbDab + EpCAM loaders
(`src/abdisc/benchmark/sabdab.py`, `src/abdisc/benchmark/loaders.py`). *Findings (as evidence):*
nothing cheap beats AF3 ipTM at cognate-vs-shuffled; AF3 outputs essentially never clash;
clash-feature sign flips AF2-M→AF3 across datasets; SAbDab pose-quality baseline (AUROC ~0.94,
N=200). *Discipline:* wiki/log/progress-note workflow, no-co-author commits, uv stack, CI.

**abdisc retirement.** abdisc was framed as a learned, pose-pipeline-agnostic discriminator that
beats predictor confidence, positive = RMSD-to-crystal. Champloo Phase 1+2 retired that framing:
raw AF3 ipTM already separates cognate from shuffled (AP 0.197 / AUROC 0.754) and nothing cheap
beats it — including predictor-agnostic geometry — while RMSD-to-crystal was shown not to measure
binding existence, the thing binder design needs. abdisc is reclassified as preparatory baseline
characterization for mirage: it established the baselines, data pipelines, and the negative result
that motivates measuring binding correctness directly.

**Open items requiring PI input before v1 implementation.** (a) exact experimental-binding evidence
accepted per dataset; (b) ratification of the working name "mirage"; (c) the primary success metric
(see Improvement metric). To be raised with Dan at the next meeting [date to fill].
