# mirage M-C Phase B2 — rung-ladder discriminator — Design Spec (2026-06-04)

> **Status:** design locked 2026-06-04. Implements the modeling half of the M-C
> structure track. Phase B1 (the 3,234-pair Protenix prediction campaign + feature
> extraction) is **COMPLETE** — see `results/published/mc_campaign_record.md`. This
> spec is the modeling addendum to the locked
> `docs/superpowers/specs/2026-06-02-mirage-mc-structure-track-design.md`;
> implementation plan to follow.

## 1. Context & what B1 left us

B1 predicted all 3,234 `(binder, antigen)` complexes with Protenix (open AF3-class,
no templates) and extracted per-pair feature CSVs (0 failures, every row
`prediction_present=1`):

- `data/staged/mc/sabdab_features.csv` — **2,688 rows** = 448 cognate VHH positives
  / 241 antigen clusters + 2,240 distribution-matched cross-cluster negatives (k=5).
  These are the **exact rows + folds** behind the M-S **AUROC 0.496** floor.
- `data/staged/mc/champloo_features.csv` — **546 rows** = 91 cognate + 455 matched
  negatives (k=5).

Both CSVs carry the M-S passthrough columns `pair_id, label, antigen_cluster, fold,
prediction_present`, plus confidence (`iptm, ptm, interface_pae, min_interface_pae,
interface_plddt, mean_plddt`), 6 interface-geometry descriptors, and 5 CDR-engagement
columns. The `fold` column **is** the M-S held-out-antigen-cluster 5-fold assignment.

Observed B1 signal (SAbDab cognate vs shuffled medians): ipTM 0.490 vs 0.281,
interface-PAE 19.4 vs 23.4 Å — so Rung 0 (ipTM-alone) is expected to clear 0.496.

### Data facts that shaped this design (verified on the CSVs)

- **`interface_plddt` is 74.0% NaN on SAbDab (71.4% on Champloo); every other feature
  column is dense.** The NaN arises because most pairs — especially negatives — have
  no high-confidence cross-chain contacts to average pLDDT over. The missingness is
  itself plausibly discriminative (no confident interface ≈ non-binding).
- **`cdr_mapping_ok` is 99.6% (SAbDab) / 100% (Champloo).** ANARCI maps the binder
  *sequence* regardless of pose, so the §6 row-preserving fallback of the B-design is
  essentially never triggered; the per-pose CDR *contact* fractions still vary (the
  real signal). A ~constant column is useless as a model feature but its failure rate
  is a reportable diagnostic.
- **`eval/attribution.py` already provides `standardized_contributions`** — the
  numpy SHAP stand-in. No new SHAP dependency.

## 2. Scientific question (carried from the B design, §2)

Given a *predicted* complex and the predictor's own confidence, does the **structure
of the predicted interface** add discriminative signal *beyond* the predictor's
headline confidence (ipTM)? Beating the M-S 0.496 floor is a *necessary precondition*,
not the headline — Rung 0 is expected to clear it. **The headline is the paired
Δ(Rung 2/3 vs Rung 0): does interface geometry (+ CDR engagement) beat ipTM-alone on
the same held-out-antigen-cluster OOF split?**

## 3. Decisions locked (this session)

1. **`interface_plddt`: drop the continuous column, keep a missingness flag.** Rung 1
   gains a derived `interface_plddt_missing` ∈ {0,1}; the continuous value (mostly NaN,
   so mostly a constant impute) is not used. `mean_plddt` remains the dense pLDDT term.
2. **Geometry: all 6 descriptors in Rung 2**, with standardized coefficients reported
   to expose collinearity — L2 + the attribution readout handle the redundant
   "interface-size" proxies. Cumulative head stays ≤ 16 features (28:1 vs 448 positives).
3. **Rungs are cumulative/nested** (Rung k = Rung k−1 columns + the new block), so the
   paired delta isolates the added block.
4. **Same folds as M-S, read directly from the CSV `fold` column** — `train_ms` gains
   an optional `folds` override so no fold is re-derived (zero apples-to-apples risk).
5. **Frozen transfer model = Rung 3** (geometry + CDR), the full structure-aware gate,
   with Rung 0 transfer reported as the confidence-only contrast. Not a dynamic
   best-OOF pick — fixed for interpretability.
6. **Self-contained random-split contrast** — implemented via `assign_folds(per-row
   ids)`; B2 does **not** depend on or merge the open PR #2 (which carries unrelated
   stage-2 cross-attention work). The new paired-delta helper is built fresh.
7. **AF3 companion = optional final task, non-blocking.** Built only after the Protenix
   headline lands.

## 4. Feature schema & the rung ladder

Built at load time from `sabdab_features.csv` / `champloo_features.csv`:

| Rung | Adds | Cumulative features |
|---|---|---|
| **0** | `iptm` | 1 |
| **1** | `ptm, interface_pae, min_interface_pae, mean_plddt, interface_plddt_missing` | 6 |
| **2** | `n_interface_residues_binder, n_interface_residues_target, buried_sasa_proxy_a2, atom_contacts_5a, shape_complementarity_proxy, atom_clash_fraction_2a` | 12 |
| **3** | `cdr_contact_fraction, cdr1_contact_fraction, cdr2_contact_fraction, cdr3_contact_fraction` | 16 |

- `interface_plddt_missing` is derived: `1.0` if the raw `interface_plddt` cell is
  empty / NaN, else `0.0`. The raw continuous `interface_plddt` is not a feature.
- `cdr_mapping_ok` is **not** a feature (near-constant); its failure rate is a
  diagnostic in the in-distribution report.
- **No RMSD/DockQ-to-crystal feature** (retired with abdisc; negatives have no cognate
  crystal). Geometry is computed from the predicted complex alone (already done in B1).

## 5. Model & gate

- **Reuse `model/ms.py` `train_ms` essentially unchanged.** Per rung:
  `train_ms(x, y, feature_names=<cumulative cols>, l2=1.0, target_precision=0.90,
  seed=<fixed>, folds=<CSV fold column>)` → `(MsModel, oof_scores)`.
- **One small backward-compatible change to `train_ms`:** add an optional
  `folds: np.ndarray | None = None` parameter that, when provided, is used directly
  for the OOF split instead of `assign_folds(groups, …)`. When `None`, behaviour is
  unchanged. This lets B2 consume the exact M-S `fold` column.
- **L2 = 1.0** (matches M-S). Threshold chosen on the full-fit model's own logits at
  `target_precision=0.90` (the M-S review fix — never on OOF scores).
- `MsModel` already satisfies the `FrozenGate` protocol (`threshold` + `predict_logit`),
  so cross-regime transfer reuses `eval/orthogonal.evaluate_frozen_gate(model, x, y)`
  unchanged — it already accepts a feature matrix directly.

## 6. Paired-delta bootstrap — new helper in `eval/gate.py`

```python
def paired_delta_bootstrap(
    scores_a, scores_b, labels, *, statistic, n_boot=1000, seed=0, alpha=0.05
) -> tuple[float, float, float]:  # (delta_point, ci_lo, ci_hi)
```

- Resample row indices **once** per replicate, **stratified by class** (positives and
  negatives resampled separately, mirroring the existing `bootstrap_ci`), and apply the
  **same** indices to both score vectors and the labels.
- Compute `statistic(scores_a[idx], y[idx]) − statistic(scores_b[idx], y[idx])` per
  replicate; the CI is the `[alpha/2, 1−alpha/2]` quantiles of the **differences**
  (not two overlapping marginal CIs).
- Runs on **precomputed frozen OOF score vectors** — no per-bootstrap refit.
- Two statistics are used:
  - **AUROC** → ΔAUROC (the headline).
  - **precision at each rung's own frozen operating threshold** → Δprecision
    (each rung's threshold is fixed from its frozen full-fit model; precision is
    recomputed at that fixed threshold inside each bootstrap sample).
- A delta "counts" iff its CI excludes 0.

## 7. In-distribution analysis — `scripts/analyze_mc_indist.py`

Input: `data/staged/mc/sabdab_features.csv`. Output: `results/published/mc_indist.json`
+ a printed summary table; freezes the Rung-3 model to
`results/published/mc_sabdab_model.json`.

Per rung (0–3):
- Build the cumulative feature matrix; `train_ms` with `folds=<CSV fold col>`,
  `groups=<antigen_cluster>` (groups retained for provenance though `folds` overrides).
- Marginal **AUROC + stratified bootstrap CI** (`eval/gate.auroc`, `bootstrap_ci`).
- Gate metrics at the 0.90-precision operating point + **PPV-vs-prevalence sweep**.
- **Standardized coefficients** (`eval/attribution.standardized_contributions`) — the
  trivial-shortcut canary.

Cross-rung:
- **Paired deltas** R2−R0, R3−R0, R3−R2 (ΔAUROC + Δprecision with CIs) — the headline.

Contrasts & diagnostics:
- **Random-split contrast:** re-run all rungs with `folds = assign_folds(per-row ids,
  …)` (ordinary K-fold); a held-out-vs-random AUROC gap localizes any
  antigen-generalization effect. Self-contained (no PR #2).
- **CDR-mapping failure rate** reported from the `cdr_mapping_ok` column.

## 8. Cross-regime transfer — `scripts/analyze_mc_cross_regime.py`

Output: `results/published/mc_cross_regime.json`.

- **Dedup guard (both directions):** cluster Champloo antigens against SAbDab antigens
  by sequence identity (`features/clustering.py`); drop Champloo rows whose antigen
  falls in a SAbDab antigen cluster, so transfer is not leakage. The deduped Champloo
  set is used for both directions.
- **2a SAbDab → Champloo (primary, robust):** freeze the Rung-3 `MsModel` on the full
  SAbDab set, apply to deduped Champloo via `evaluate_frozen_gate(model, x, y)`.
  AUROC + gate metrics + bootstrap CIs. Report **Rung 0** transfer as the
  confidence-only contrast. This is the "cross-regime precision stability" headline.
- **2b Champloo → SAbDab (caveated):** freeze Rung 3 on the 546 Champloo rows, apply to
  the SAbDab reservoir. **Read only in light of 2a** — a failure here can be an
  underfitting false-negative on 91 positives, not absent signal.

## 9. AF3 companion — `scripts/analyze_mc_af3.py` (OPTIONAL, non-blocking, last)

Built only after the Protenix headline lands. Extract `iptm`/`ptm`/pLDDT/geometry
(no local PAE → Rung 1 partial: `interface_pae`/`min_interface_pae` unavailable) from
the staged Champloo AF3 structures (`../abdisc-data/champloo/af3_structures` + the ipTM
matrices). Run Rung 0→2 self-consistently; produce an AF3-vs-Protenix ipTM comparison
on the 91 Champloo cells. Self-consistent within AF3 — never pooled with Protenix.

## 10. Reuse / new / out-of-scope

- **Reuse wholesale:** `model/ms.py` `train_ms` (+ the tiny `folds` override),
  `ml/core` (`assign_folds`, `oof_logistic_scores`, `fit_logistic_regression`,
  standardizers), `eval/gate.py` (`auroc`, `bootstrap_ci`, threshold/operating-point
  helpers, PPV sweep), `eval/orthogonal.py` (`evaluate_frozen_gate`, `FrozenGate`),
  `eval/attribution.py` (`standardized_contributions`), `features/clustering.py`
  (cross-dataset antigen dedup), the `MsModel` JSON artifact contract.
- **New:** `eval/gate.paired_delta_bootstrap`; a nested rung feature-matrix builder
  (small module or in-script helper) with the `interface_plddt_missing` derivation;
  `scripts/analyze_mc_indist.py`; `scripts/analyze_mc_cross_regime.py`;
  `scripts/analyze_mc_af3.py` (optional); the `train_ms` `folds` parameter.
- **Out of scope (v1):** AVIDa/EpCAM real-negative *structural* tests (deferred; AVIDa
  stays held-out); non-VHH formats; decoy docking; any GPU package in the mirage uv
  env; RMSD/DockQ-to-crystal as feature or label; re-prediction or re-extraction of
  features (B1's CSVs are the substrate; `interface_plddt` threshold relaxation was
  considered and declined in favor of the missingness flag).

## 11. Success criteria & risks

- **Success (headline):** Rung 2/3 > Rung 0 with a paired ΔAUROC bootstrap CI that
  **excludes 0** on the SAbDab OOF split. A clean *negative* (geometry does not beat
  ipTM) is equally publishable given the rigor (same folds, paired deltas, random-split
  contrast, bootstrap CIs, coefficient readout).
- **Necessary precondition:** each rung beats the M-S 0.496 floor (Rung 0 expected to).
- **Risks:**
  1. ipTM-alone may already saturate the separable signal → ΔAUROC ≈ 0; acceptable,
     it's the real result.
  2. Collinear geometry proxies → unstable coefficients; mitigated by L2 + the
     standardized-coefficient readout (interpretation, not significance, is affected;
     AUROC is unaffected).
  3. 448 positives × 16 features → defended by L2, the random-split contrast, and
     coefficients, exactly as M-S defended its floor.
  4. Cross-regime 2b underfitting false-negative → mitigated by reading 2a first.

## 12. Tests (TDD-adjacent, land with the code)

- `paired_delta_bootstrap`: planted A-strictly-dominates-B → ΔAUROC CI excludes 0 and
  is positive; identical score vectors → CI contains 0.
- Nested feature-matrix builder selects the correct cumulative columns per rung.
- `interface_plddt_missing` derivation (NaN/empty → 1.0, finite → 0.0).
- `train_ms` `folds` override reproduces the OOF scores of an explicit
  `oof_logistic_scores(... folds ...)` call and leaves the `folds=None` path unchanged.

## 13. Invariants

- mirage package stays **torch-free**; pure-numpy; no GPU deps added.
- **Same held-out-antigen-cluster split** as M-S (the CSV `fold` column) for the 0.496
  comparison.
- **AVIDa stays held-out.**
- Commits **Pedram-authored, no Claude/Anthropic trailer**.
- Build via **subagent-driven execution**, on `mc-structure-track` (no PR #2 merge).

## 14. References

- `docs/superpowers/specs/2026-06-02-mirage-mc-structure-track-design.md` — the locked
  M-C structure-track design this implements.
- `results/published/mc_campaign_record.md` — what B1 produced (the substrate).
- `results/published/sabdab_baseline_summary.md` — the M-S 0.496 floor this measures
  against.
- `src/mirage/model/ms.py`, `src/mirage/ml/core.py`, `src/mirage/eval/{gate,orthogonal,
  attribution}.py`, `src/mirage/features/clustering.py`, `scripts/extract_mc_features.py`
  — reused scaffolding.
