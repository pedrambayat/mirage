# abdisc — Project Design Doc

**Date:** 2026-05-10
**Author:** Pedram Bayat
**Status:** Brainstorm-validated; awaiting spec review and implementation plan
**Successor to:** the SNAP project (`/Users/pedrambayat/.../SNAP/snap`) — this doc supersedes `notes/execution-plan-multi-tier.md` for forward planning. SNAP remains as the data and code source for benchmark v1.

> **2026-05-11 amendment — scope broadening.** The framing in this spec is narrower than the project's current scope:
>
> - abdisc is now positioned as a *general* antibody-antigen binder discrimination tool. The splice-variant / CAR-T / CHL1 / MSLN framing below is no longer load-bearing for the headline contribution — those targets remain useful concrete examples and motivating cases, but they are not the project's identity.
> - The design track has been downgraded from a co-equal track to a *secondary investigation*. mBER, BindCraft, and any other specific generator are not committed dependencies. `src/abdisc/design/` is retained for forward optionality, but week 1–6 work focuses entirely on the benchmark and scorer comparison.
>
> Sections 2 (Headline contribution), 3 (Scope), and 7 (Design campaign track) should be read with this amendment in mind. The detailed implementation choices in sections 4–6 (repo structure, benchmark format, scorer wrappers) remain accurate.

> **2026-05-12 amendment — pose-quality unification (post-PI meeting).** Sharpens the framing further:
>
> - "Is this binder real?" is now **operationalized** as "is the predicted pose close to a crystal pose?" The two questions are treated as the same — a real binder is one whose predicted complex matches the crystal complex (within tolerance). This collapses what had been two separate framings (binder discrimination vs pose quality) into one.
> - **Ground truth source:** SAbDab. For each crystal-validated complex, run a pose predictor (AF2M / AF3 / Protenix / Boltz / …) and compute RMSD between the predicted and crystal poses. This is the supervision signal for the learned discriminator. Fast and quantitative.
> - **Positive control:** EpCAM (from SNAP). Take known binders, predict poses, verify the trained discriminator scores them as real.
> - **The discriminator is pose-pipeline-agnostic.** The same model must score Protenix outputs and AF3 outputs equivalently. The input representation must not encode predictor-specific signals.
> - **Dataset priority inverts** from earlier sections: SAbDab becomes the *primary* training source (it carries the crystal labels), EpCAM becomes the validation positive control. Earlier sections imply the reverse — that should be read with this amendment in mind.
> - **Two diagnostic analyses** are tracked alongside the headline discriminator (not the headline themselves): inter-predictor pose concordance, and hotspot agreement (see `…/abdisc-wiki/wiki/Methods/Hotspot Agreement.md`).
> - **Live sources for the current framing:** `…/abdisc-wiki/wiki/Current State.md` and `…/Research/abdisc/03 - Meetings/2026-05-12 - Dan meeting.md`. Trust those over earlier sections of this spec when the framings disagree.

---

## 1. Context

SNAP set out to design CAR-T binders against tumor-restricted splice variants
(CHL1 exon-out, MSLN stump). The benchmark sub-project established that
AlphaFold-Multimer biophysical confidence scores (iPTM, BSA, CDR engagement,
inter-chain PAE) have ~0.5 AUROC for distinguishing true binders from
controlled negatives — wrong-target Abs and CDR-scrambled variants. Direct
quote from `benchmark/RESULTS.md`: *"BSA, CDR engagement, and iPTM have 0%
specificity."* This finding invalidated the original plan to filter generated
designs through AF2-M scoring, and it is the empirical wedge for the new
project.

The design pipeline (mBER) works end-to-end and produces candidate VHH
binders, but without a working in-silico filter the candidates cannot be
prioritized for wet-lab pursuit. The bottleneck is the discriminator, not the
generator. This project builds that discriminator and the controlled
benchmark needed to validate it.

## 2. Headline contribution

> Existing structure-prediction confidence scores (AF2-M, Protenix, Boltz,
> IntelliFold) cannot reliably distinguish true antibody-antigen binders from
> controlled hard negatives — CDR-scrambled variants, off-target antibodies,
> and unverified generative designs. We release **`abdisc-bench`**, a
> controlled benchmark with crystal-validated positives and four classes of
> negatives (including in-silico designs from a generative pipeline). We
> benchmark N≥6 off-the-shelf scorers and propose **`abdisc-net`**, an
> antibody-antigen discriminator [architecture chosen at week 6 by what the
> baselines reveal] trained on the benchmark.

**Two-track paper shape:**
- **Floor (week 6):** benchmark + leaderboard table + analysis = workshop
  paper minimum. Publishable as "all current methods fail" if no model
  works.
- **Ceiling (week 12-14):** add `abdisc-net` + apply it to in-silico design
  campaigns on CHL1 / MSLN. If it beats baselines on hard negatives, the
  paper is method + benchmark + applied story.

**Venue targets:**
1. NeurIPS 2026 AI-for-Science / GenBio / MLSB workshops (primary; deadline
   typically late August / early September).
2. ICLR 2027 MLDD or TML4Bio workshops (fallback).
3. bioRxiv preprint (independent of acceptance).

## 3. Scope

**In scope:**
- A controlled benchmark dataset with four classes (positives, off-target,
  CDR-scrambled, in-silico designs).
- Wrappers for ≥4 off-the-shelf scorers (AF2-M, Protenix, ESM-2 perplexity,
  ipSAE, plus Boltz / IntelliFold / Rosetta-physics if cheap).
- One new learned scorer (`abdisc-net`), architecture chosen mid-summer by
  the baseline gate.
- Background design campaigns (mBER + BindCraft) on 5 targets, ingested into
  the benchmark as the in-silico-design class.
- A workshop paper draft.

**Explicitly out of scope (= not in v1, not in the paper):**
- Wet-lab validation / handoff of designed binders.
- Multi-tier evaluation stack as originally drafted in
  `notes/execution-plan-multi-tier.md` (Tiers 3-4 specifically — Rosetta
  energetics is a single baseline here, negative-design docking is deferred).
- A full design-pipeline benchmark in the style of the obsolete
  `benchmark/phase4_pipeline_eval.py`.
- Web leaderboard or hosted dataset infrastructure beyond a Markdown table
  and a versioned manifest.
- Fine-tuning Protenix or any other large pretrained model — `abdisc-net`
  is built fresh, optionally on top of frozen Protenix features.

## 4. Repository structure

**Location:** `~/abdisc/` outside Google Drive (SNAP's Drive location has
fought sync conflicts with conda envs, large PDBs, and git internals). Push
to a private GitHub repo. SNAP referenced only as a data source; specific
files copied in, then the cord is cut.

```
abdisc/
├── pyproject.toml
├── environment.yml            # conda env, mirrors mber where possible
├── README.md
├── docs/
│   ├── benchmark-card.md      # HuggingFace-style dataset card
│   ├── scorer-api.md
│   └── design-campaigns.md
├── src/abdisc/
│   ├── benchmark/
│   │   ├── manifest.py        # canonical dataset
│   │   ├── loaders.py         # SAbDab / PDB / mmCIF
│   │   ├── negatives/{scramble,off_target,designs}.py
│   │   └── splits.py
│   ├── scorers/
│   │   ├── base.py            # AbstractScorer interface
│   │   ├── af2m.py · protenix.py · boltz.py · intellifold.py
│   │   ├── esm2_perplexity.py · ipsae.py
│   │   └── neural/            # abdisc-net implementation, populated week 7+
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── leaderboard.py     # generates Table 1
│   │   └── plots.py
│   ├── design/
│   │   ├── mber_runner.py     # SLURM batch launcher
│   │   ├── targets/{chl1,msln,…}.yaml
│   │   └── ingest.py          # mBER outputs → benchmark/negatives/designs
│   └── cli.py                 # `abdisc bench score`, `abdisc design run`, …
├── data/
│   ├── manifest/              # tracked CSVs
│   └── raw/                   # gitignored: PDBs, cached predictions
├── results/                   # gitignored except results/published/
├── scripts/slurm/
├── tests/
└── paper/                     # LaTeX draft, regenerates figures from results/
```

**Five non-obvious design choices:**

1. **`AbstractScorer` is the central abstraction.** Every scorer takes a
   `BenchmarkExample` and returns a `Score`. That single interface is what
   makes "swap in N models" cheap — adding a 7th scorer should be half a
   day.
2. **Manifest-driven.** One CSV is the source of truth for the benchmark.
   Everything else (negatives, splits, leaderboard) derives from it.
3. **Designs are first-class data.** `design/ingest.py` is a one-way pipe
   from mBER outputs into the benchmark's in-silico-design class. The two
   tracks talk through this file and only this file.
4. **CLI for everything reproducible.** `abdisc bench score --scorer
   protenix --split test` rather than ad-hoc scripts.
5. **`paper/` lives in the repo.** Figures regenerate from
   `results/published/`. Avoids SNAP's "where did this number come from"
   problem.

**Explicitly NOT doing:** DVC / Git LFS (manifest is small, raw PDBs come
from public sources on demand); web UI / hosted leaderboard; premature
training-loop framework.

## 5. The benchmark artifact (`abdisc-bench`)

### 5.1 Classes

| Class | Target N | Source | Label |
|---|---:|---|---|
| **POS** crystal-validated binders | 60–100 | SAbDab Ab-Ag complexes (≤3.0 Å, paratope on CDR) | 1 |
| **OFF** wrong-target negatives | 60–100 | Cross-pair sampling: POS Abs against unrelated antigens | 0 |
| **SCR** CDR-scrambled negatives | 60–100 | Per POS, scramble CDRs preserving framework | 0 |
| **DES** in-silico designs | 50–200 | mBER (and BindCraft) campaigns on 5 targets | unknown — **eval-only** |

Total ~250–500 examples. Workshop-paper sized.

### 5.2 Construction rules

- **POS:** SAbDab filtered on resolution (≤3.0 Å), single-chain VHH or Fab,
  identifiable CDRs via ANARCI, antigen ≥30 residues. Sequence-identity
  de-duplication at >90%. Crystal contacts pre-computed via the ported
  `analyze_contacts`.
- **OFF:** for each POS antibody, sample K=2 antigens outside the
  antibody's target species/family. Excluded if known cross-reactivity.
  Deterministic seed.
- **SCR:** per POS antibody, generate 2–3 scrambles using two strategies —
  (i) **shuffle** CDR residues within each loop (preserves AA composition);
  (ii) **resample** from SAbDab CDR-residue frequency distribution
  (doesn't). Both kept and reported separately.
- **DES:** outputs from mBER (primary) and BindCraft (secondary) on 5
  targets — CHL1 exon-out, MSLN stump, plus 3 SAbDab targets where the
  true positive exists for comparison. Each design tagged with target,
  generator, hyperparameters, generation timestamp. **DES is eval-only,
  never in training**, to avoid the discriminator memorizing generator
  idiosyncrasies.

### 5.3 Splits

- **Train / val / test by antigen identity** (not by complex). All
  complexes sharing an antigen go to the same split. Forces generalization
  to unseen antigens.
- **Held-out OOD eval set:** the SNAP biology — CHL1 exon-in vs exon-out,
  MSLN stump vs full-length. Tiny (4–8 pairs) but the splice-variant story
  motivates the paper biologically.

### 5.4 Format

- `data/manifest/abdisc_v1.csv` — one row per example
  (`id, class, antibody_seq, antigen_seq, antigen_pdb_id,
  complex_pdb_id, source, split, …`).
- Predicted structures cached at
  `data/raw/predictions/{scorer}/{example_id}.pdb`.
- `data/manifest/abdisc_v1.lock.json` — canonical hash so future runs
  verify "this is exactly v1."

### 5.5 Leaderboard (Table 1 of the paper)

| Scorer | AUROC POS↔OFF | AUROC POS↔SCR-shuffle | AUROC POS↔SCR-resample | DES rate-as-positive | Wall-clock / example (single GPU or CPU as noted) |

The DES column ("what fraction of designs does this scorer call positive?")
has no ground truth — disagreement *between* scorers on the DES set is
itself a signal worth analyzing.

### 5.6 Validation criteria for "the benchmark is good"

- POS pairs reproduce known interfaces (BSA, CDR engagement) — port and run
  the SNAP `analyze_contacts` test.
- SCR negatives change ≥80% of CDR residues, preserve framework Levenshtein
  ≤5.
- OFF pairs have no shared antigen substring ≥10 residues.
- Train / val / test splits each contain POS + OFF + SCR. DES is a separate eval-only set (per §6.4) and is not split-partitioned.
- All cached predictions reproducible by re-running with the recorded seed.

## 6. Scorer exploration

### 6.1 Phase 4a — baselines (priority order)

| # | Scorer | Why included | Cost / example | Risk |
|---|---|---|---|---|
| 1 | `af2m_iptm` | The score the paper says fails | ~5 min GPU | low — port from SNAP |
| 2 | `esm2_perplexity` | Cheapest baseline; sequence-only sanity check | ~1 sec CPU | low |
| 3 | `ipsae` | Better PAE-based score; vendored in SNAP | seconds (post-AF2-M) | low |
| 4 | `protenix` | Current SOTA structure model | ~10 min GPU | medium — SLURM launcher |
| 5 | `boltz2` | Diversity in the structure-model class | ~5 min GPU | medium |
| 6 | `intellifold` | Orthogonal AF3 reproduction | ~10 min GPU | low-medium |
| 7 | `physics_baseline` | Rosetta InterfaceAnalyzer dG/dSASA | ~minutes CPU | medium — Rosetta install |

Items 1–4 are non-negotiable. Items 5–7 added if cheap.

### 6.2 Decision gate at end of Phase 4a (~week 6)

Three possible worlds based on Table 1:

- **World A:** all baselines fail uniformly (~0.5 AUROC). Paper writes
  itself as "no current method works, here's the benchmark and the
  negative result." `abdisc-net` is *bonus*, not load-bearing.
- **World B:** one baseline (likely Protenix) clearly dominates.
  `abdisc-net` must beat it specifically on POS↔SCR.
- **World C:** mixed — different scorers win on different negative classes.
  `abdisc-net` targets the gaps. Most interesting world.

### 6.3 Phase 4b — `abdisc-net` (architecture chosen week 6)

The menu, in order of prior on what's worth trying:

1. **Joint-embedding interface model (JEPA-style).** Encode binder and
   target separately into a shared embedding space; train with InfoNCE
   contrastive loss using POS as positives, SCR + OFF as hard negatives.
   No structure prediction in-loop. Cheap to train, cheap to score.
   **Default if World A or C.**
2. **Pair-residue transformer over Protenix features.** Take Protenix's
   pair representation, learn a small head that scores interface quality.
   Reuses an SOTA model's geometry knowledge. **Default if World B.**
3. **SE(3)-equivariant interface scorer.** EquiformerV2-style head over
   predicted interface atoms. Heaviest engineering — only with surplus
   time/appetite.
4. **Trajectory-aware scorer (the dynamics hook).** Generate K=5–10
   binding-process snapshots via short MD relaxation OR Boltz-flow
   interpolation, score trajectory plausibility, not just endpoint.
   Highest novelty / risk. Only viable if data definition is solved by
   week 8.

This design records the menu and decision criteria, **not** the choice. The
choice happens at the gate.

### 6.4 Training data for `abdisc-net`

- Train on POS vs (SCR-shuffle ∪ SCR-resample ∪ OFF), held out by antigen
  identity per §5.3.
- DES is eval-only.
- Augmentation: subsample CDR loops, paratope masking, antigen surface
  jitter.

### 6.5 What "winning" means

- **Primary:** AUROC POS↔SCR > best baseline by ≥0.05 on held-out test
  split.
- **Secondary:** AUROC POS↔OFF doesn't drop below the best baseline.
- **Stretch:** SNAP biology — correctly orders CHL1 exon-out > exon-in for
  known binders, MSLN stump > full-length.

If primary fails by week 12 → fall back to World A's "negative result"
framing. Paper still ships.

## 7. Design campaign track

**Targets (5):**
- CHL1 exon-out (glioma neoepitope)
- MSLN stump (post-cleavage neo-N-term)
- 3× SAbDab targets with known binders (e.g. PD-L1, HER2, VEGF) — re-design
  where the true positive exists, enabling the question "do scorers rate
  designs more like the known positive or like SCR negatives?"

**Generators:** mBER (primary) + BindCraft (secondary, off-the-shelf).
BoltzDesign deferred unless cheap.

**Cadence:** SLURM batch every Monday (set-and-forget). ~10–30 designs/week
per *target* (summed across active generators), staggered so not all 5
targets run every week. By week 10: ~150–300 designs in DES total.

**Single integration point:** `abdisc design ingest <run_dir>` pulls
completed designs into `data/manifest/abdisc_v1.csv` as DES rows. Tracks
generator + hyperparameters per design.

**Effort budget:** ~30% of summer time. Mostly SLURM launches and ingest
runs; not active development.

## 8. Timeline (14 weeks, 2026-05-11 → 2026-08-16)

| Week | Date | Benchmark / scorer | Design |
|---|---|---|---|
| 1 | May 11 | Repo scaffold, `AbstractScorer` interface, port `analyze_contacts` | — |
| 2 | May 18 | Benchmark v0: POS+OFF+SCR (no DES); pytest harness | First mBER campaigns (CHL1, MSLN) |
| 3 | May 25 | Baselines #1-3 (af2m, esm2, ipsae); first Table 1 | Cron-style weekly batch live |
| 4 | Jun 1 | Protenix wrapper + SLURM launcher (#4) | Add 3 SAbDab targets |
| 5 | Jun 8 | Boltz / IntelliFold (#5-6) | First DES rows ingested |
| 6 | Jun 15 | Physics baseline (#7); **architecture decision gate** | — |
| 7 | Jun 22 | Begin `abdisc-net` (chosen architecture), training scaffold | — |
| 8 | Jun 29 | Train v1, debug, first eval | BindCraft as 2nd generator |
| 9 | Jul 6 | Iterate model, ablations | — |
| 10 | Jul 13 | Run model on DES + SNAP biology eval set | DES ~150 examples |
| 11 | Jul 20 | Lock results; first paper figures | — |
| 12 | Jul 27 | Writing — intro, methods, results | — |
| 13 | Aug 3 | Writing — discussion, related work, figures | — |
| 14 | Aug 10 | Internal review, polish, submit (~Aug 14) | — |

Week 14 deliberately under-loaded for slippage. If everything goes well, it
becomes time for ablations or wet-lab handoff prep.

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Protenix install / SLURM friction eats week 4-5 | high | medium | Start dry-run in week 2, not week 4. If blocked, drop to AF2-M + ESM-2 + ipSAE for v1. |
| `mber-open` env breaks on PARCC mid-summer | medium | high | Pin a known-good env snapshot week 1; rebuild fresh to verify. |
| DES class arrives slow, isn't in benchmark v1 | medium | low | Benchmark v1 ships without DES if needed; DES is a v1.1 addition. |
| `abdisc-net` doesn't beat baselines | medium | medium | World-A fallback framing is publishable. |
| NeurIPS workshop deadline slips | low-medium | medium | Fallback: ICLR 2027 MLDD (Jan deadline). bioRxiv unconditional. |
| Scope creep into D (specificity / wet lab) | high | medium-high | D is *explicitly out of scope*. Designs accumulate; wet-lab is a follow-up. |
| Stop running design campaigns mid-summer | medium | low | DES is nice-to-have; benchmark + scorer is the paper. |
| Reviewer says "this is just a benchmark" | medium | low | Pre-empt with `abdisc-net` framing; lean on controlled-negative novelty. |

## 10. Open questions / explicit non-decisions

These are deliberately deferred, not forgotten:

- **`abdisc-net` architecture** — chosen at week 6 by the baseline gate.
- **BoltzDesign as a third generator** — only if integration is cheap.
- **Whether to publish the benchmark dataset on HuggingFace Hub** — depends
  on licensing of underlying SAbDab + PDB structures; check before
  paper-submission week.
- **Author list / collaborators** — not addressed here.
- **Whether `binder-discrimination/` from SNAP gets ported or archived** —
  defer to the implementation plan; likely archived because it's EpCAM-
  specific and uses AF2-M features, but a few utility functions may be
  worth porting.
- **Wet-lab handoff plan if the project succeeds** — out of scope for the
  paper, but if `abdisc-net` works, follow-up is to nominate top-K designs
  per target for synthesis. Track that as a separate project.

## 11. Success criteria (paper-shippable definition of done)

Concrete bar at submission week (week 14):

1. `abdisc-bench` v1.0 manifest is frozen, hashed, validated per §5.6.
2. Table 1 contains ≥4 baseline scorers with metrics on all four classes.
3. `abdisc-net` (one architecture) trained and evaluated; either beats best
   baseline on POS↔SCR by ≥0.05 AUROC, **or** the World-A "negative result"
   narrative is fully written and defended.
4. Splice-variant OOD eval (CHL1 / MSLN) reported, even if results are
   negative.
5. Paper draft compiles, all figures regenerate from `results/published/`,
   reproducibility statement included.
6. Repo public on GitHub with `pyproject.toml`, `environment.yml`, `README`,
   and a `Reproducing the paper` section.

## 12. References

- `notes/technical-report-2026-04-09.md` — the SNAP project overview that
  this design supersedes for forward planning.
- `benchmark/RESULTS.md` — empirical foundation (AF2-M ~0.5 AUROC).
- `notes/execution-plan-multi-tier.md` — the prior multi-tier plan;
  Tier 1-2 partly absorbed into Phase 4a baselines, Tier 3-4 deferred.
- `evaluation/EVALUATION.md` — `analyze_contacts` reference; ported into
  abdisc benchmark validation.
- `binder-discrimination/EXPERIMENT_REVIEW.md` — historical EpCAM
  classifier experiment; documents the "RF on AF2-M features" failure mode
  this project's approach is designed to avoid.
