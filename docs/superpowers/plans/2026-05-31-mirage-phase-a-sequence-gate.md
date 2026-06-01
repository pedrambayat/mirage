# mirage Phase A — Sequence-Only Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-GPU sequence-only mirage gate (M-S): train an L2-logistic binder/non-binder classifier on Champloo Tier-S sequence features, evaluate it as an FP-costly gate (sensitivity/specificity at a fixed-precision operating point, PPV-vs-prevalence sweep, bootstrap CIs), and validate a Champloo-*frozen* threshold on the real-negative orthogonal sets (AVIDa, labeled-EpCAM).

**Architecture:** Pure-Python/numpy library code under `src/mirage/` (no sklearn/torch/shap), analysis pipelines under `scripts/`, importlib-loaded unit tests under `tests/` — matching the existing mirage/Champloo-classifier conventions. Reusable numeric primitives (logistic regression, grouped folds, AUROC/AP) are lifted out of the existing `scripts/analyze_champloo_classifier.py` into `mirage.ml.core` so training, gate metrics, and orthogonal apply all share one implementation. The trained model freezes to a JSON artifact (standardization + coefficients + operating threshold) that the orthogonal harness loads and applies unchanged.

**Tech Stack:** Python 3.11, numpy, pandas, uv. Dev: pytest, ruff, mypy (strict). No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-05-31-mirage-data-and-training-strategy.md`. This plan implements **Phase A only**. M-S is explicitly the *pre-structure baseline*; the predictor-conditional M-C and the AF3/structure track (Phase B) are out of scope here.

**Reuse note:** `scripts/analyze_champloo_classifier.py` already implements numpy logistic regression, `assign_folds`, `oof_logistic_scores`, `_auroc`, `_average_precision`, and the three grouped splits. Task 2 moves those into a library module and re-exports them from the script so existing tests stay green; do not duplicate them.

---

## File Structure

**New library modules (`src/mirage/`):**
- `ml/__init__.py`, `ml/core.py` — logistic regression, standardization, grouped folds, AUROC/AP (moved from the analysis script; the single source of truth).
- `features/__init__.py`, `features/sequence.py` — Tier-S per-pair sequence features (compact physicochemical set, numpy-free pure Python returning floats).
- `eval/gate.py` — gate metrics: precision/recall/specificity at a threshold, `recall_at_precision`, `choose_threshold_for_precision`, `ppv_at_prevalence`, `ppv_prevalence_sweep`, `bootstrap_ci`.
- `eval/attribution.py` — logistic standardized-coefficient attribution (the numpy stand-in for the SHAP guard).
- `eval/orthogonal.py` — dataset-agnostic frozen-gate evaluation over `(features, label)` arrays.
- `model/__init__.py`, `model/ms.py` — `MsModel` frozen artifact (save/load JSON, `predict_logit`) + `train_ms` (grouped OOF scores + final fit + threshold).
- `benchmark/avida.py` — AVIDa-hIL6 sequence-label loader.
- `benchmark/epcam_killing.py` — labeled-EpCAM (CAR-T killing) loader.

**New scripts (`scripts/`):**
- `stage_champloo_features.py` — join supplementary-table sequences onto staged pairs and emit a Tier-S feature CSV.
- `stage_avida.py` — normalize raw AVIDa CSVs into one staged CSV.
- `analyze_ms_indist.py` — train M-S, in-distribution gate metrics vs the ipTM floor, PPV sweep, attribution.
- `analyze_ms_orthogonal.py` — load frozen M-S, score AVIDa + labeled-EpCAM, frozen-threshold cross-regime stability table.

**New docs:**
- `docs/datasets/dataset-registry.md` — the committed dataset registry (Dan's ask).
- `results/published/mirage_phase_a_summary.md` — Phase A results write-up (produced at the end).

**Modified:**
- `scripts/analyze_champloo_classifier.py` — re-export moved primitives from `mirage.ml.core`.
- `src/mirage/benchmark/__init__.py` — side-effect imports for the two new loaders.

**Tests:** one `tests/test_<module>.py` per new module/script, following the importlib idiom in `tests/test_champloo_classifier.py`.

---

## Task -1: Repository bootstrap (run once, before Task 0)

The mirage code repo is a populated working tree but **not yet a git repo**, so
the per-task commits below would otherwise fail. The two companion vaults
(`mirage-notes/`, `mirage-wiki/`) are likewise un-initialized. Each is its **own**
git repo — never cross-commit between them.

- [ ] **Step 1: Initialize the code repo with a clean baseline commit**

Working dir: the mirage code repo
(`/vast/projects/.../binder-discrimination/mirage`). Commit the existing tree
*first* so Task 0's scrub lands as a readable diff rather than being buried in a
giant initial commit. `.gitignore` already excludes `.venv/`, caches,
`data/raw/`, `data/staged/`, and `results/*` (except `results/published/`).

```bash
git init
git branch -M main
git add -A
git commit -m "Initial mirage repo (migrated from abdisc)"
```

- [ ] **Step 2: Wire the GitHub remote and push the baseline**

The remote exists and SSH auth is confirmed. The GitHub repo must be **empty**
(no README/license/.gitignore); if it already has a commit, run
`git pull --rebase origin main` before the push.

```bash
git remote add origin git@github.com:pedrambayat/mirage.git
git push -u origin main
```

After this, push after each task's commit (`git push`) so the remote tracks
progress. (If `git push` is ever rejected for non-fast-forward, stop and surface
it — do not force-push.)

- [ ] **Step 3: Initialize the companion vaults (local-only for now)**

`mirage-notes/` and `mirage-wiki/` (siblings of the code repo) currently hold
only `.gitkeep` placeholders. Initialize each as its own repo, commit the
scaffold so later progress/wiki commits have a baseline, wire its remote, and
push. Both GitHub repos must be **empty** (no README/license); if either already
has a commit, `git -C <dir> pull --rebase origin main` before its push.

```bash
git -C ../mirage-notes init
git -C ../mirage-notes branch -M main
git -C ../mirage-notes add -A
git -C ../mirage-notes commit -m "Initialize mirage-notes vault scaffold"
git -C ../mirage-notes remote add origin git@github.com:pedrambayat/mirage-notes.git
git -C ../mirage-notes push -u origin main

git -C ../mirage-wiki init
git -C ../mirage-wiki branch -M main
git -C ../mirage-wiki add -A
git -C ../mirage-wiki commit -m "Initialize mirage-wiki vault scaffold"
git -C ../mirage-wiki remote add origin git@github.com:pedrambayat/mirage-wiki.git
git -C ../mirage-wiki push -u origin main
```

- [ ] **Step 4: Sanity check**

```bash
git -C . status                 # clean tree, on main, tracking origin/main
git -C ../mirage-notes log --oneline -1
git -C ../mirage-wiki  log --oneline -1
```

All commit messages across all three repos: **no `Co-Authored-By: Claude`
trailer** (overrides default Claude Code behavior).

---

## Task 0: Scrub stray `abdisc` references (repo hygiene)

The `mirage` repo was bootstrapped by copying `abdisc`; the package was renamed
to `src/mirage/` but ~20 stray `abdisc` references remain in docstrings,
comments, one functional metadata key, and the top-level docs. mirage must not
reference abdisc. This task removes them **before** new code lands so nothing
new inherits the old name.

**Do NOT rename:** `abdisc-data` / `../abdisc-data/...` — that is the real,
shared on-disk dataset directory name (see CLAUDE.md), not part of this repo.

- [ ] **Step 1: Cosmetic references (docstrings + comments)**

Rewrite `abdisc` → `mirage` in the prose of these files (verify each reads
correctly afterward — these are comments, not identifiers):
- `src/mirage/_paths.py` (module docstring + `repo_root` docstring)
- `src/mirage/cli.py` (CLI docstrings)
- `src/mirage/scorers/af2m_confidence.py` (docstrings referencing "abdisc AF2-M wrapper")
- `src/mirage/pose_predictors/__init__.py` (docstring)
- `scripts/slurm/run_af2m_chunk.py`, `scripts/score_champloo_structures.py`,
  `scripts/stage_champloo_structures.py`, `scripts/analyze_af2m_confidence_vs_rmsd.py`
  (docstrings/comments only — leave `abdisc-data` path examples intact)

- [x] **Step 2: Functional metadata key (artifact-schema change) — DONE 2026-06-01**

Already applied during plan review: the AF2-M provenance key value was renamed
`"abdisc_predict_af2m_version"` → `"mirage_predict_af2m_version"` in
`scripts/slurm/run_af2m_chunk.py` (`MIRAGE_VERSION_KEY`) and
`tests/test_af2m_confidence.py`. Verified no reader checks the literal (it is a
write-only provenance stamp) and no committed/staged `scores.json` carried the
old key, so the artifact-schema change is safe; AF2-M staging is Phase B and
re-run from scratch anyway.

- [ ] **Step 3: Top-level docs**

`CLAUDE.md`, `README.md`, and `AGENTS.md` still open with abdisc framing (e.g.
"abdisc trains a pose-pipeline-agnostic discriminator"). Replace `abdisc` →
`mirage` there. **CLAUDE.md needs a genuine content pass, not just a token
swap** — its project-overview paragraph describes the abdisc framing and should
match the mirage spec. Keep this scoped to the name/framing; do not rewrite
architecture sections that are still accurate.

- [ ] **Step 4: Verify nothing but `abdisc-data` remains, then run the suite**

```bash
grep -rn 'abdisc' --include='*.py' --include='*.md' --include='*.toml' src scripts tests *.md \
  | grep -v 'abdisc-data'   # expected: no output
uv run ruff check && uv run mypy src/mirage && uv run pytest
```
Expected: the grep prints nothing; the existing suite stays green (the metadata-key
rename is covered by the updated `test_af2m_confidence.py`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Scrub stray abdisc references from migrated mirage repo"
```

---

## Task 1: Dataset registry document

**Files:**
- Create: `docs/datasets/dataset-registry.md`

No code/test — this is the committed artifact Dan requested. Content is transcribed from Section 3 of the spec.

- [ ] **Step 1: Write the registry doc**

Create `docs/datasets/dataset-registry.md`:

```markdown
# mirage dataset registry

Living registry of every dataset, its role, label provenance, and caveats.
Source of truth for the data side of `docs/superpowers/specs/2026-05-31-mirage-data-and-training-strategy.md`.

| Dataset | Role | True positives (evidence) | Negatives | Predicted complex? | Confidence? | Format | Caveats |
|---|---|---|---|---|---|---|---|
| Champloo / Smorodina | Primary (train + in-dist test) | 106 cognate VHH–Ag, co-crystal | ~11,130 constructed shuffled non-cognate | Yes — AF3 staged | Yes — ipTM | VHH | Synthetic negatives; small positives (106); ~1:105 imbalance |
| SAbDab | Orthogonal TP reservoir + format axis | ~1k nonredundant VHH (+Fab/scFv), co-crystal | none native | No (crystal only) | No | VHH + Fab/scFv | Predictor-conditional use needs AF3 runs; cluster for redundancy |
| EpCAM (SNAP + collaborator) | Orthogonal test — designed-binder deployment regime, real labels | 8 functional VHHs (CAR-T killing vs AsPC1): 10,25,26,34,57,61,74,86 | 6 non-functional (real): 14,15,16,18,21,73; +86 SCR/24 OFF (unassayed) | No — must be generated | No | VHH (designed) | N=14 → test only, wide CIs; label = functional killing (downstream of binding) |
| AVIDa-hIL6 | Orthogonal test only | many VHH binders to IL-6 family (assay) | many real assay non-binders | No | No | VHH | Sequence-only; HF config broken → use raw CSVs |
| Germinal | Parked design case study | 24 BLI-positive designed | no clean public negatives | — | — | designed | Not benchmark-ready |

**Train vs test:** Champloo trains; SAbDab/AVIDa/EpCAM are held-out validation. Only AVIDa and EpCAM carry real negatives. EpCAM is functional-killing-labeled and tiny (test-only).
```

- [ ] **Step 2: Commit**

```bash
git add docs/datasets/dataset-registry.md
git commit -m "Add committed mirage dataset registry"
```

---

## Task 2: Move numeric core into `mirage.ml.core`

**Files:**
- Create: `src/mirage/ml/__init__.py`, `src/mirage/ml/core.py`
- Modify: `scripts/analyze_champloo_classifier.py`
- Test: `tests/test_ml_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ml_core.py`:

```python
from __future__ import annotations

import numpy as np

from mirage.ml import core


def test_auroc_perfect_and_inverted() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert core.auroc(scores, labels) == 1.0
    assert core.auroc(-scores, labels) == 0.0


def test_average_precision_matches_known_value() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    labels = np.asarray([1, 0, 1, 0])
    ap = core.average_precision(scores, labels)
    assert abs(ap - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_assign_folds_no_group_leakage_and_deterministic() -> None:
    groups = np.asarray([f"g{i // 4}" for i in range(40)])
    folds = core.assign_folds(groups, n_splits=5, seed=1)
    for g in np.unique(groups):
        assert np.unique(folds[groups == g]).size == 1
    again = core.assign_folds(groups, n_splits=5, seed=1)
    assert np.array_equal(folds, again)


def test_oof_logistic_recovers_separable_signal() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, 0.5, 30), rng.normal(3, 0.5, 30)])[:, None]
    y = np.concatenate([np.zeros(30), np.ones(30)])
    folds = np.tile(np.arange(5), 12)
    scores = core.oof_logistic_scores(x, y, folds, l2=1.0)
    assert np.isfinite(scores).all()
    assert core.auroc(scores, y.astype(int)) > 0.95


def test_fit_predict_roundtrip() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-2, 0.5, 40), rng.normal(2, 0.5, 40)])[:, None]
    y = np.concatenate([np.zeros(40), np.ones(40)])
    mean, std = core.standardizer(x)
    intercept, coef = core.fit_logistic_regression(core.apply_standardizer(x, mean, std), y, l2=1.0)
    logits = intercept + core.apply_standardizer(x, mean, std) @ coef
    assert core.auroc(logits, y.astype(int)) > 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ml_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mirage.ml'`

- [ ] **Step 3: Create the module**

Create `src/mirage/ml/__init__.py`:

```python
"""Reusable numeric primitives shared by training, eval, and analysis."""
```

Create `src/mirage/ml/core.py` (logic copied verbatim from `scripts/analyze_champloo_classifier.py`, renamed to public names and with explicit standardizer helpers):

```python
"""Numpy-only logistic regression, grouped folds, and ranking metrics.

Single source of truth: the Champloo analysis script and all mirage Phase A
code import these rather than re-implementing them.
"""

from __future__ import annotations

import math

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.size * neg.size))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    if scores.size == 0 or n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(float)
    precision = np.cumsum(labels_sorted) / np.arange(1, labels_sorted.size + 1)
    return float((precision * labels_sorted).sum() / n_pos)


def standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) with zero-variance columns forced to std=1."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> tuple[float, np.ndarray]:
    beta = np.zeros(x.shape[1] + 1, dtype=float)
    design = np.column_stack([np.ones(x.shape[0], dtype=float), x])
    penalty = np.diag(np.concatenate([[0.0], np.full(x.shape[1], l2, dtype=float)]))
    prev_loss = math.inf
    for _ in range(max_iter):
        logits = np.clip(design @ beta, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(pred * (1.0 - pred), 1e-9)
        gradient = design.T @ (pred - y) + penalty @ beta
        hessian = (design.T * weights) @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        loss = float(
            -np.sum(y * np.log(pred + 1e-12) + (1.0 - y) * np.log(1.0 - pred + 1e-12))
            + 0.5 * float(beta @ penalty @ beta)
        )
        if abs(prev_loss - loss) < tolerance or float(np.linalg.norm(step)) < tolerance:
            break
        prev_loss = loss
    return float(beta[0]), beta[1:]


def assign_folds(groups: np.ndarray, *, n_splits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    shuffled = rng.permutation(unique.size)
    fold_of_group = {g: int(shuffled[i] % n_splits) for i, g in enumerate(unique)}
    return np.asarray([fold_of_group[g] for g in groups], dtype=int)


def oof_logistic_scores(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    l2: float,
) -> np.ndarray:
    out = np.full(y.shape, math.nan, dtype=float)
    for fold in np.unique(folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        y_train = y[train_mask]
        if y_train.size < 2 or np.unique(y_train).size < 2:
            continue
        mean, std = standardizer(x[train_mask])
        x_train = apply_standardizer(x[train_mask], mean, std)
        x_test = apply_standardizer(x[test_mask], mean, std)
        intercept, coef = fit_logistic_regression(x_train, y_train, l2=l2)
        out[test_mask] = intercept + x_test @ coef
    return out
```

- [ ] **Step 4: Re-export from the analysis script (keep existing tests green)**

In `scripts/analyze_champloo_classifier.py`, replace the local definitions of `_auroc`, `_average_precision`, `assign_folds`, `oof_logistic_scores`, `_fit_logistic_regression`, `_standardize` with imports. Add near the top (after `import numpy as np`):

```python
from mirage.ml.core import (
    apply_standardizer,
    assign_folds,
    auroc as _auroc,
    average_precision as _average_precision,
    fit_logistic_regression as _fit_logistic_regression,
    oof_logistic_scores,
    standardizer,
)
```

Delete the now-duplicated function bodies (`_auroc`, `_average_precision`, `assign_folds`, `oof_logistic_scores`, `_fit_logistic_regression`, `_standardize`). The `_standardize(x_train, x_apply)` two-arg helper, if still referenced, is replaced by `standardizer` + `apply_standardizer`; update its one call site in `oof_logistic_scores` (now imported) — no other call sites remain.

- [ ] **Step 5: Run both test files**

Run: `uv run pytest tests/test_ml_core.py tests/test_champloo_classifier.py -v`
Expected: PASS (existing Champloo tests still pass via re-exports).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/ml tests/test_ml_core.py scripts/analyze_champloo_classifier.py
git commit -m "Lift numpy ML core out of Champloo script into mirage.ml.core"
```

---

## Task 3: Tier-S sequence features

**Files:**
- Create: `src/mirage/features/__init__.py`, `src/mirage/features/sequence.py`
- Test: `tests/test_sequence_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sequence_features.py`:

```python
from __future__ import annotations

from mirage.features.sequence import FEATURE_NAMES, sequence_features


def test_feature_names_match_dict_keys() -> None:
    feats = sequence_features("ACDEFGHIK", "KKKKDDDD")
    assert list(feats.keys()) == list(FEATURE_NAMES)


def test_lengths_are_reported() -> None:
    feats = sequence_features("AAAA", "GGGGGG")
    assert feats["binder_length"] == 4.0
    assert feats["target_length"] == 6.0


def test_net_charge_sign() -> None:
    # all-lysine binder is strongly positive; all-aspartate is negative
    assert sequence_features("KKKK", "A")["binder_net_charge"] > 0
    assert sequence_features("DDDD", "A")["binder_net_charge"] < 0


def test_fractions_in_unit_interval() -> None:
    feats = sequence_features("FWYAVLIMCDEKR", "ACDEFGHIKLMNPQRSTVWY")
    for key, value in feats.items():
        if key.endswith("_frac"):
            assert 0.0 <= value <= 1.0


def test_empty_sequence_is_safe() -> None:
    feats = sequence_features("", "")
    assert feats["binder_length"] == 0.0
    assert feats["binder_aromatic_frac"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sequence_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mirage.features'`

- [ ] **Step 3: Implement**

Create `src/mirage/features/__init__.py`:

```python
"""Feature extractors. Tier-S = sequence-only (usable on every dataset)."""
```

Create `src/mirage/features/sequence.py`:

```python
"""Tier-S sequence features: a compact, regularization-friendly physicochemical
descriptor for a (binder, target) sequence pair. Pure Python, no dependencies.

Deliberately compact (~6 per chain) because the Champloo positive cohort is
small (~106); a 40-dim composition vector would overfit. ESM-2 embeddings are
an explicitly deferred Tier-S extension (would add torch) — out of scope here.
"""

from __future__ import annotations

_AROMATIC = frozenset("FWY")
_HYDROPHOBIC = frozenset("AVLIMFWC")
_POLAR = frozenset("STNQ")
_POSITIVE = frozenset("KR")
_NEGATIVE = frozenset("DE")

FEATURE_NAMES: tuple[str, ...] = (
    "binder_length",
    "binder_net_charge",
    "binder_aromatic_frac",
    "binder_hydrophobic_frac",
    "binder_polar_frac",
    "binder_cysteine_frac",
    "target_length",
    "target_net_charge",
    "target_aromatic_frac",
    "target_hydrophobic_frac",
    "target_polar_frac",
    "target_cysteine_frac",
    "length_ratio",
)


def _chain_features(seq: str) -> dict[str, float]:
    seq = seq.strip().upper()
    n = len(seq)
    if n == 0:
        return {
            "length": 0.0,
            "net_charge": 0.0,
            "aromatic_frac": 0.0,
            "hydrophobic_frac": 0.0,
            "polar_frac": 0.0,
            "cysteine_frac": 0.0,
        }
    pos = sum(1 for c in seq if c in _POSITIVE)
    neg = sum(1 for c in seq if c in _NEGATIVE)
    return {
        "length": float(n),
        "net_charge": float(pos - neg),
        "aromatic_frac": sum(1 for c in seq if c in _AROMATIC) / n,
        "hydrophobic_frac": sum(1 for c in seq if c in _HYDROPHOBIC) / n,
        "polar_frac": sum(1 for c in seq if c in _POLAR) / n,
        "cysteine_frac": seq.count("C") / n,
    }


def sequence_features(binder_seq: str, target_seq: str) -> dict[str, float]:
    """Return the Tier-S feature dict for one (binder, target) pair.

    Keys are exactly ``FEATURE_NAMES`` in order.
    """
    b = _chain_features(binder_seq)
    t = _chain_features(target_seq)
    length_ratio = b["length"] / t["length"] if t["length"] > 0 else 0.0
    out = {
        "binder_length": b["length"],
        "binder_net_charge": b["net_charge"],
        "binder_aromatic_frac": b["aromatic_frac"],
        "binder_hydrophobic_frac": b["hydrophobic_frac"],
        "binder_polar_frac": b["polar_frac"],
        "binder_cysteine_frac": b["cysteine_frac"],
        "target_length": t["length"],
        "target_net_charge": t["net_charge"],
        "target_aromatic_frac": t["aromatic_frac"],
        "target_hydrophobic_frac": t["hydrophobic_frac"],
        "target_polar_frac": t["polar_frac"],
        "target_cysteine_frac": t["cysteine_frac"],
        "length_ratio": length_ratio,
    }
    return {name: out[name] for name in FEATURE_NAMES}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sequence_features.py -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/features tests/test_sequence_features.py
git commit -m "Add Tier-S sequence feature extractor"
```

---

## Task 4: Gate metrics

**Files:**
- Create: `src/mirage/eval/gate.py`
- Test: `tests/test_gate_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gate_metrics.py`:

```python
from __future__ import annotations

import numpy as np

from mirage.eval import gate


def test_confusion_at_threshold() -> None:
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    m = gate.metrics_at_threshold(scores, labels, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0
    assert m["fpr"] == 0.0


def test_choose_threshold_for_precision_hits_target() -> None:
    # one negative scores above one positive; precision 1.0 only above that neg
    scores = np.asarray([0.2, 0.5, 0.55, 0.7, 0.9])
    labels = np.asarray([0, 1, 0, 1, 1])
    thr = gate.choose_threshold_for_precision(scores, labels, target_precision=1.0)
    m = gate.metrics_at_threshold(scores, labels, threshold=thr)
    assert m["precision"] >= 1.0 - 1e-9


def test_recall_at_precision_perfect_separation() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert gate.recall_at_precision(scores, labels, target_precision=0.9) == 1.0


def test_ppv_at_prevalence_bayes() -> None:
    # recall=0.9, specificity=0.9, prevalence=0.5 -> PPV=0.9
    assert abs(gate.ppv_at_prevalence(recall=0.9, specificity=0.9, prevalence=0.5) - 0.9) < 1e-9
    # at prevalence 1e-4 the same gate is nearly worthless
    low = gate.ppv_at_prevalence(recall=0.9, specificity=0.9, prevalence=1e-4)
    assert low < 0.01


def test_ppv_sweep_is_monotone_decreasing_as_rarer() -> None:
    sweep = gate.ppv_prevalence_sweep(recall=0.9, specificity=0.99, prevalences=(0.5, 0.1, 1e-3, 1e-4))
    values = [row["ppv"] for row in sweep]
    assert values == sorted(values, reverse=True)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    lo, hi = gate.bootstrap_ci(
        lambda s, y: gate.metrics_at_threshold(s, y, threshold=1.5)["recall"],
        scores,
        labels,
        n_boot=200,
        seed=1,
    )
    point = gate.metrics_at_threshold(scores, labels, threshold=1.5)["recall"]
    assert lo <= point <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gate_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mirage.eval.gate'`

- [ ] **Step 3: Implement**

Create `src/mirage/eval/gate.py`:

```python
"""Gate metrics for an FP-costly binder/non-binder gate.

The mirage gate fixes a high-precision operating point and reports the
sensitivity/specificity achieved there, plus the deployed PPV across a
prevalence sweep (Champloo's ~1:105 ratio badly overstates real-screen PPV).
All functions are numpy-only and take 1-D score/label arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, *, threshold: float
) -> dict[str, float]:
    """Predict positive when score >= threshold; return gate metrics."""
    pred = scores >= threshold
    y = labels.astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    fpr = fp / (fp + tn) if (fp + tn) else math.nan
    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
    }


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(scores[np.isfinite(scores)])
    # one threshold just above each distinct score, plus a floor below the min
    bumped = np.nextafter(uniq, np.inf)
    return np.concatenate([[np.nextafter(uniq.min(), -np.inf)], bumped]) if uniq.size else np.array([0.0])


def choose_threshold_for_precision(
    scores: np.ndarray, labels: np.ndarray, *, target_precision: float
) -> float:
    """Lowest threshold whose precision >= target (maximizing recall at target).

    If no threshold reaches the target, return the threshold with the highest
    precision observed (so the caller still gets a defined operating point).
    """
    best_thr = math.inf
    best_recall = -1.0
    fallback_thr = 0.0
    fallback_prec = -1.0
    for thr in _candidate_thresholds(scores):
        m = metrics_at_threshold(scores, labels, threshold=thr)
        prec = m["precision"]
        if math.isnan(prec):
            continue
        if prec > fallback_prec:
            fallback_prec, fallback_thr = prec, thr
        if prec >= target_precision and m["recall"] > best_recall:
            best_recall, best_thr = m["recall"], thr
    return float(best_thr if math.isfinite(best_thr) else fallback_thr)


def recall_at_precision(
    scores: np.ndarray, labels: np.ndarray, *, target_precision: float
) -> float:
    thr = choose_threshold_for_precision(scores, labels, target_precision=target_precision)
    return metrics_at_threshold(scores, labels, threshold=thr)["recall"]


def ppv_at_prevalence(*, recall: float, specificity: float, prevalence: float) -> float:
    """Bayesian PPV translation: sensitivity & specificity are prevalence-free,
    so deployed PPV is recomputed at the target prevalence."""
    tpr = recall * prevalence
    fpr_mass = (1.0 - specificity) * (1.0 - prevalence)
    denom = tpr + fpr_mass
    return float(tpr / denom) if denom > 0 else math.nan


def ppv_prevalence_sweep(
    *, recall: float, specificity: float, prevalences: Sequence[float]
) -> list[dict[str, float]]:
    return [
        {"prevalence": float(p), "ppv": ppv_at_prevalence(recall=recall, specificity=specificity, prevalence=p)}
        for p in prevalences
    ]


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Stratified bootstrap CI: resample positives and negatives separately so
    small cohorts keep both classes present in every replicate."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    vals: list[float] = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        bn = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([bp, bn])
        v = statistic(scores[idx], labels[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (math.nan, math.nan)
    arr = np.asarray(vals)
    return (float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2)))


# default prevalence grid: from balanced down to a realistic in-silico screen rate
DEFAULT_PREVALENCES: tuple[float, ...] = (0.5, 0.1, 0.01, 1e-3, 1e-4)


def summary_dict(scores: np.ndarray, labels: np.ndarray, *, threshold: float) -> dict[str, Any]:
    """Convenience bundle: metrics at threshold + PPV sweep at that operating point."""
    m = metrics_at_threshold(scores, labels, threshold=threshold)
    sweep = ppv_prevalence_sweep(
        recall=m["recall"], specificity=m["specificity"], prevalences=DEFAULT_PREVALENCES
    )
    return {"metrics": m, "ppv_sweep": sweep}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gate_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/eval/gate.py tests/test_gate_metrics.py
git commit -m "Add FP-costly gate metrics (sens/spec, recall@precision, PPV sweep, bootstrap CI)"
```

---

## Task 5: Feature attribution (SHAP guard, numpy)

**Files:**
- Create: `src/mirage/eval/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_attribution.py`:

```python
from __future__ import annotations

import numpy as np

from mirage.eval.attribution import standardized_contributions


def test_dominant_feature_ranks_first() -> None:
    # feature 0 carries all signal; feature 1 is noise
    rng = np.random.default_rng(0)
    n = 200
    x0 = np.concatenate([rng.normal(-2, 0.3, n), rng.normal(2, 0.3, n)])
    x1 = rng.normal(0, 1, 2 * n)
    x = np.column_stack([x0, x1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    ranked = standardized_contributions(x, y, feature_names=["signal", "noise"], l2=1.0)
    assert ranked[0]["feature"] == "signal"
    assert abs(ranked[0]["abs_contribution"]) > abs(ranked[1]["abs_contribution"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/mirage/eval/attribution.py`:

```python
"""Feature attribution for the linear gate.

For an L2-logistic model on standardized features, the standardized coefficient
is the per-feature contribution to the log-odds — the linear-model analogue of
a global SHAP value. This is the numpy stand-in for the spec's SHAP guard
(detecting whether the gate keys on trivial sequence mismatches); `shap` is not
a project dependency.
"""

from __future__ import annotations

import numpy as np

from mirage.ml.core import apply_standardizer, fit_logistic_regression, standardizer


def standardized_contributions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str],
    l2: float,
) -> list[dict[str, float | str]]:
    """Fit on standardized features; return features ranked by |coefficient|."""
    mean, std = standardizer(x)
    xs = apply_standardizer(x, mean, std)
    _intercept, coef = fit_logistic_regression(xs, y, l2=l2)
    rows: list[dict[str, float | str]] = [
        {"feature": name, "coefficient": float(c), "abs_contribution": float(abs(c))}
        for name, c in zip(feature_names, coef, strict=True)
    ]
    rows.sort(key=lambda r: float(r["abs_contribution"]), reverse=True)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/eval/attribution.py tests/test_attribution.py
git commit -m "Add standardized-coefficient feature attribution (numpy SHAP stand-in)"
```

---

## Task 6: Stage Champloo Tier-S feature table

**Files:**
- Create: `scripts/stage_champloo_features.py`
- Test: `tests/test_stage_champloo_features.py`

The staged pair table (`data/staged/champloo/champloo_pairs_af3.csv`) has no
sequences. This script joins the supplementary table's `vhh_sequence` /
`antigen_sequence` (keyed by `pdb_id`) onto each pair via `vhh_pdb` /
`antigen_pdb`, computes Tier-S features, and writes a feature CSV carrying
`pair_id, vhh_pdb, antigen_pdb, label, iptm` + the Tier-S columns.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_champloo_features.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sequences_by_pdb_picks_first() -> None:
    stage = _load("stage_champloo_features.py")
    supp = [
        {"pdb_id": "AAAA", "vhh_sequence": "QVQL", "antigen_sequence": "GGGG"},
        {"pdb_id": "AAAA", "vhh_sequence": "XXXX", "antigen_sequence": "YYYY"},
        {"pdb_id": "BBBB", "vhh_sequence": "EVQL", "antigen_sequence": "DDDD"},
    ]
    by_pdb = stage.sequences_by_pdb(supp)
    assert by_pdb["AAAA"]["vhh_sequence"] == "QVQL"
    assert by_pdb["BBBB"]["antigen_sequence"] == "DDDD"


def test_build_feature_rows_joins_and_features() -> None:
    stage = _load("stage_champloo_features.py")
    pairs = [
        {"pair_id": "AAAA__AAAA", "vhh_pdb": "AAAA", "antigen_pdb": "AAAA", "label": "1", "iptm": "0.9"},
        {"pair_id": "AAAA__BBBB", "vhh_pdb": "AAAA", "antigen_pdb": "BBBB", "label": "0", "iptm": "0.2"},
        {"pair_id": "AAAA__CCCC", "vhh_pdb": "AAAA", "antigen_pdb": "CCCC", "label": "0", "iptm": "0.1"},
    ]
    seqs = {
        "AAAA": {"vhh_sequence": "KKKK", "antigen_sequence": "DDDD"},
        "BBBB": {"vhh_sequence": "EVQL", "antigen_sequence": "GGGG"},
    }
    rows = stage.build_feature_rows(pairs, seqs)
    # CCCC has no sequence -> that pair is dropped
    ids = {r["pair_id"] for r in rows}
    assert ids == {"AAAA__AAAA", "AAAA__BBBB"}
    row = next(r for r in rows if r["pair_id"] == "AAAA__AAAA")
    assert row["label"] == "1"
    assert float(row["binder_length"]) == 4.0
    assert "binder_net_charge" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stage_champloo_features.py -v`
Expected: FAIL (script does not exist)

- [ ] **Step 3: Implement**

Create `scripts/stage_champloo_features.py`:

```python
"""Join Champloo supplementary-table sequences onto staged pairs and emit a
Tier-S feature CSV for the mirage sequence-only gate (M-S).

Use::

    uv run python scripts/stage_champloo_features.py \\
        --pairs data/staged/champloo/champloo_pairs_af3.csv \\
        --supp ../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \\
        --output data/staged/champloo/champloo_features_af3.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mirage.features.sequence import FEATURE_NAMES, sequence_features

_BASE_COLUMNS = ("pair_id", "vhh_pdb", "antigen_pdb", "label", "iptm")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def sequences_by_pdb(supp_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map pdb_id -> {vhh_sequence, antigen_sequence}; first row per pdb wins."""
    out: dict[str, dict[str, str]] = {}
    for row in supp_rows:
        pdb = row["pdb_id"]
        if pdb not in out:
            out[pdb] = {
                "vhh_sequence": row.get("vhh_sequence", ""),
                "antigen_sequence": row.get("antigen_sequence", ""),
            }
    return out


def build_feature_rows(
    pairs: list[dict[str, str]], seqs: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in pairs:
        vhh = seqs.get(p["vhh_pdb"])
        ant = seqs.get(p["antigen_pdb"])
        if vhh is None or ant is None:
            continue
        binder_seq = vhh["vhh_sequence"]
        target_seq = ant["antigen_sequence"]
        if not binder_seq or not target_seq:
            continue
        feats = sequence_features(binder_seq, target_seq)
        row: dict[str, str] = {col: p.get(col, "") for col in _BASE_COLUMNS}
        for name in FEATURE_NAMES:
            row[name] = repr(feats[name])
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*_BASE_COLUMNS, *FEATURE_NAMES]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--supp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = read_csv(args.pairs)
    supp = read_csv(args.supp)
    rows = build_feature_rows(pairs, sequences_by_pdb(supp))
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} feature rows to {args.output} (from {len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stage_champloo_features.py -v`
Expected: PASS

- [ ] **Step 5: Generate the real feature table (manual data step)**

Run:
```bash
uv run python scripts/stage_champloo_features.py \
  --pairs data/staged/champloo/champloo_pairs_af3.csv \
  --supp ../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \
  --output data/staged/champloo/champloo_features_af3.csv
```
Expected: prints a non-zero "Wrote N feature rows" count. (`data/staged/` is gitignored — the CSV is a local artifact, not committed.)

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add scripts/stage_champloo_features.py tests/test_stage_champloo_features.py
git commit -m "Stage Champloo Tier-S feature table (join supp-table sequences)"
```

---

## Task 7: M-S model — train, freeze, threshold

**Files:**
- Create: `src/mirage/model/__init__.py`, `src/mirage/model/ms.py`
- Test: `tests/test_ms_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ms_model.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mirage.model.ms import MsModel, train_ms


def _separable_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.4, 60), rng.normal(2, 0.4, 60)])[:, None]
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(int)
    return x, y, ["f0"]


def test_train_ms_produces_usable_model() -> None:
    x, y, names = _separable_data()
    model, oof = train_ms(x, y, feature_names=names, l2=1.0, target_precision=0.9, seed=1)
    assert isinstance(model, MsModel)
    assert np.isfinite(oof).all()
    # frozen model separates the classes
    logits = model.predict_logit(x)
    assert ((logits >= model.threshold).astype(int) == y).mean() > 0.9


def test_threshold_is_on_the_frozen_model_scale() -> None:
    # The shipped threshold must hit the target precision on the frozen model's
    # OWN logits (predict_logit) — not on the OOF scores. This is the scale-
    # consistency guarantee the orthogonal harness depends on.
    from mirage.eval.gate import metrics_at_threshold

    x, y, names = _separable_data()
    model, _ = train_ms(x, y, feature_names=names, l2=1.0, target_precision=0.9, seed=1)
    logits = model.predict_logit(x)
    m = metrics_at_threshold(logits, y, threshold=model.threshold)
    assert m["precision"] >= 0.9 - 1e-9


def test_ms_model_save_load_roundtrip(tmp_path: Path) -> None:
    x, y, names = _separable_data()
    model, _ = train_ms(x, y, feature_names=names, l2=1.0, target_precision=0.9, seed=1)
    path = tmp_path / "ms.json"
    model.save(path)
    loaded = MsModel.load(path)
    assert loaded.feature_names == model.feature_names
    assert abs(loaded.threshold - model.threshold) < 1e-12
    assert np.allclose(loaded.predict_logit(x), model.predict_logit(x))
    # artifact is human-readable JSON
    json.loads(path.read_text())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ms_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mirage.model'`

- [ ] **Step 3: Implement**

Create `src/mirage/model/__init__.py`:

```python
"""Trained mirage models. M-S = sequence-only frozen gate."""
```

Create `src/mirage/model/ms.py`:

```python
"""M-S: the sequence-only mirage gate.

`train_ms` fits an L2-logistic model on Tier-S features, computes leakage-aware
out-of-fold scores for honest in-distribution metrics, fits a final model on all
rows, and picks the operating threshold at a target precision. The frozen
`MsModel` (standardization + coefficients + threshold) serializes to JSON and is
applied unchanged to orthogonal datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mirage.eval.gate import choose_threshold_for_precision
from mirage.ml.core import (
    apply_standardizer,
    assign_folds,
    fit_logistic_regression,
    oof_logistic_scores,
    standardizer,
)


@dataclass(frozen=True)
class MsModel:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    intercept: float
    coef: list[float]
    threshold: float
    target_precision: float

    def predict_logit(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        xs = apply_standardizer(x, mean, std)
        return self.intercept + xs @ np.asarray(self.coef)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> MsModel:
        data = json.loads(path.read_text())
        return cls(**data)


def train_ms(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str],
    l2: float,
    target_precision: float,
    seed: int,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
) -> tuple[MsModel, np.ndarray]:
    """Return (frozen model, out-of-fold scores).

    `groups` drives the leakage-controlled OOF split (e.g. antigen PDB). If
    None, an ordinary K-fold over rows is used.

    The returned OOF scores are for honest held-out *reporting only*. The frozen
    model's operating threshold is chosen on the **full-fit model's own logits**
    so it is on the same scale the frozen model emits (`predict_logit`) — the
    orthogonal harness applies that threshold unchanged and must reproduce the
    target operating point. (Picking the threshold on OOF scores and applying it
    to the full-fit model would be a scale mismatch: each OOF fold has its own
    standardizer + intercept.)
    """
    y = y.astype(float)
    if groups is None:
        groups = np.arange(x.shape[0]).astype(str)

    # OOF scores: purely for the honest in-distribution report downstream.
    folds = assign_folds(groups, n_splits=n_splits, seed=seed)
    oof = oof_logistic_scores(x, y, folds, l2=l2)

    # Final full-data fit — this is what freezes into the artifact.
    mean, std = standardizer(x)
    xs = apply_standardizer(x, mean, std)
    intercept, coef = fit_logistic_regression(xs, y, l2=l2)

    # Threshold on the full-fit model's own logits (same scale as predict_logit).
    full_logits = intercept + xs @ coef
    finite = np.isfinite(full_logits)
    threshold = choose_threshold_for_precision(
        full_logits[finite], y[finite].astype(int), target_precision=target_precision
    )

    model = MsModel(
        feature_names=list(feature_names),
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
        intercept=float(intercept),
        coef=[float(v) for v in coef],
        threshold=float(threshold),
        target_precision=float(target_precision),
    )
    return model, oof
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ms_model.py -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/model tests/test_ms_model.py
git commit -m "Add M-S sequence-only gate: train, freeze to JSON, threshold at target precision"
```

---

## Task 8: In-distribution gate analysis script

**Files:**
- Create: `scripts/analyze_ms_indist.py`
- Test: `tests/test_analyze_ms_indist.py`

Trains M-S on the Champloo feature CSV with the held-out-antigen grouped split,
writes the frozen model artifact, and reports gate metrics (M-S vs the raw-ipTM
floor) at the operating point, the PPV-prevalence sweep, and the feature
attribution ranking.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_ms_indist.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def _load(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_feature_matrix_parses_columns() -> None:
    analyze = _load("analyze_ms_indist.py")
    rows = [
        {"pair_id": "a", "label": "1", "iptm": "0.9", "antigen_pdb": "AAAA",
         "binder_length": "120.0", "binder_net_charge": "2.0"},
        {"pair_id": "b", "label": "0", "iptm": "0.1", "antigen_pdb": "BBBB",
         "binder_length": "118.0", "binder_net_charge": "-1.0"},
    ]
    x, y, iptm, groups, names = analyze.load_feature_matrix(
        rows, feature_names=("binder_length", "binder_net_charge")
    )
    assert x.shape == (2, 2)
    assert list(y) == [1, 0]
    assert np.allclose(iptm, [0.9, 0.1])
    assert list(groups) == ["AAAA", "BBBB"]
    assert names == ("binder_length", "binder_net_charge")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyze_ms_indist.py -v`
Expected: FAIL (script does not exist)

- [ ] **Step 3: Implement**

Create `scripts/analyze_ms_indist.py`:

```python
"""Train M-S on Champloo Tier-S features and report in-distribution gate metrics.

Compares the sequence-only gate (M-S, out-of-fold under the held-out-antigen
split) against the raw-ipTM floor at a fixed-precision operating point, writes
the frozen M-S artifact, the PPV-prevalence sweep, and the feature-attribution
ranking (the SHAP guard).

Use::

    uv run python scripts/analyze_ms_indist.py \\
        --features data/staged/champloo/champloo_features_af3.csv \\
        --model-out results/published/ms_model_af3.json \\
        --output results/published/mirage_ms_indist_af3.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from mirage.eval.attribution import standardized_contributions
from mirage.eval.gate import DEFAULT_PREVALENCES, choose_threshold_for_precision, summary_dict
from mirage.features.sequence import FEATURE_NAMES
from mirage.model.ms import train_ms


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_feature_matrix(
    rows: list[dict[str, str]], *, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    x = np.array([[_to_float(r[name]) for name in feature_names] for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    iptm = np.array([_to_float(r["iptm"]) for r in rows], dtype=float)
    groups = np.array([r["antigen_pdb"] for r in rows])
    return x, y, iptm, groups, feature_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    rows = read_csv(args.features)
    x, y, iptm, groups, names = load_feature_matrix(rows, feature_names=FEATURE_NAMES)
    finite_rows = np.isfinite(x).all(axis=1)
    x, y, iptm, groups = x[finite_rows], y[finite_rows], iptm[finite_rows], groups[finite_rows]

    model, oof = train_ms(
        x, y, feature_names=list(names), l2=args.l2,
        target_precision=args.target_precision, seed=args.seed, groups=groups,
    )
    model.save(args.model_out)

    # Honest in-distribution report: the OOF scores live on a different scale
    # than the frozen full-fit model, so the operating point is re-derived on
    # the OOF scores. (model.threshold is the *shipped* full-fit threshold used
    # unchanged by the orthogonal harness — do NOT apply it to OOF scores.)
    finite_oof = np.isfinite(oof)
    oof_thr = choose_threshold_for_precision(
        oof[finite_oof], y[finite_oof], target_precision=args.target_precision
    )
    ms_summary = summary_dict(oof[finite_oof], y[finite_oof], threshold=oof_thr)

    iptm_finite = np.isfinite(iptm)
    iptm_thr = choose_threshold_for_precision(
        iptm[iptm_finite], y[iptm_finite], target_precision=args.target_precision
    )
    iptm_summary = summary_dict(iptm[iptm_finite], y[iptm_finite], threshold=iptm_thr)

    attribution = standardized_contributions(x, y.astype(float), feature_names=list(names), l2=args.l2)

    result = {
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "target_precision": args.target_precision,
        "prevalences": list(DEFAULT_PREVALENCES),
        "oof_threshold": float(oof_thr),
        "shipped_threshold": float(model.threshold),
        "ms": ms_summary,
        "iptm_floor": iptm_summary,
        "attribution": attribution,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"ms_recall": ms_summary["metrics"]["recall"],
                      "iptm_recall": iptm_summary["metrics"]["recall"]}, indent=2))
    print(f"Wrote {args.output} and model {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_analyze_ms_indist.py -v`
Expected: PASS

- [ ] **Step 5: Run the real analysis (manual data step)**

Run:
```bash
uv run python scripts/analyze_ms_indist.py \
  --features data/staged/champloo/champloo_features_af3.csv \
  --model-out results/published/ms_model_af3.json \
  --output results/published/mirage_ms_indist_af3.json
```
Expected: prints `ms_recall` and `iptm_recall`; writes the JSON + model artifact.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add scripts/analyze_ms_indist.py tests/test_analyze_ms_indist.py results/published/ms_model_af3.json results/published/mirage_ms_indist_af3.json
git commit -m "Add M-S in-distribution gate analysis (vs ipTM floor, PPV sweep, attribution)"
```

---

## Task 9: AVIDa-hIL6 loader + staging

**Files:**
- Create: `src/mirage/benchmark/avida.py`, `scripts/stage_avida.py`
- Modify: `src/mirage/benchmark/__init__.py`
- Test: `tests/test_avida_loader.py`

**Prerequisite (manual):** download the raw AVIDa-hIL6 files (`AVIDa-hIL6.csv`,
`antigen_sequences.csv`) from `COGNANO/AVIDa-hIL6` (the HF dataset viewer is
misconfigured — use the raw files) into `../abdisc-data/avida/raw/`. The loader
reads a single staged CSV produced by `stage_avida.py`; it does not hit the
network.

- [ ] **Step 1: Write the failing test**

Create `tests/test_avida_loader.py`:

```python
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

from mirage.benchmark._registry import get_loader


def _load_script(script_name: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_avida_joins_antigen_sequences() -> None:
    stage = _load_script("stage_avida.py")
    records = [
        {"VHH_sequence": "QVQL", "Ag_label": "IL6", "label": "1"},
        {"VHH_sequence": "EVQL", "Ag_label": "IL6", "label": "0"},
    ]
    antigens = {"IL6": "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP"}
    rows = stage.build_rows(records, antigens)
    assert rows[0]["label"] == "1"
    assert rows[0]["antigen_sequence"].startswith("MNSF")
    assert rows[1]["label"] == "0"


def test_avida_loader_yields_examples(tmp_path: Path) -> None:
    staged = tmp_path / "avida.csv"
    with staged.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["vhh_id", "vhh_sequence", "antigen_label", "antigen_sequence", "label"]
        )
        writer.writeheader()
        writer.writerow({"vhh_id": "v1", "vhh_sequence": "QVQL", "antigen_label": "IL6",
                         "antigen_sequence": "MNSF", "label": "1"})
        writer.writerow({"vhh_id": "v2", "vhh_sequence": "EVQL", "antigen_label": "IL6",
                         "antigen_sequence": "MNSF", "label": "0"})
    loader = get_loader("avida", staged_csv=staged)
    examples = list(loader.load())
    assert len(examples) == 2
    assert {e.label for e in examples} == {"BIND", "NONBIND"}
    assert examples[0].binder_format == "vhh"
    assert examples[0].source == "avida-hil6"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_avida_loader.py -v`
Expected: FAIL (no `stage_avida.py`; no `avida` loader)

- [ ] **Step 3: Implement the staging script**

Create `scripts/stage_avida.py`:

```python
"""Normalize raw AVIDa-hIL6 files into one staged CSV the loader consumes.

Use::

    uv run python scripts/stage_avida.py \\
        --records ../abdisc-data/avida/raw/AVIDa-hIL6.csv \\
        --antigens ../abdisc-data/avida/raw/antigen_sequences.csv \\
        --output ../abdisc-data/avida/avida_staged.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def antigen_map(antigen_rows: list[dict[str, str]]) -> dict[str, str]:
    """Map antigen label -> sequence. Tolerates the two common column namings."""
    out: dict[str, str] = {}
    for row in antigen_rows:
        label = row.get("Ag_label") or row.get("antigen_label") or row.get("label", "")
        seq = row.get("antigen_sequence") or row.get("sequence", "")
        if label and seq:
            out[label] = seq
    return out


def build_rows(records: list[dict[str, str]], antigens: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, rec in enumerate(records):
        label = rec.get("label", "")
        ag_label = rec.get("Ag_label", "")
        seq = rec.get("VHH_sequence", "")
        antigen_seq = antigens.get(ag_label, "")
        if not seq or not antigen_seq or label not in ("0", "1"):
            continue
        rows.append({
            "vhh_id": f"avida-{i}",
            "vhh_sequence": seq,
            "antigen_label": ag_label,
            "antigen_sequence": antigen_seq,
            "label": label,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--antigens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_csv(args.records)
    antigens = antigen_map(read_csv(args.antigens))
    rows = build_rows(records, antigens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["vhh_id", "vhh_sequence", "antigen_label", "antigen_sequence", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} AVIDa rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement the loader**

Create `src/mirage/benchmark/avida.py`:

```python
"""AVIDa-hIL6 loader: sequence-only VHH / IL-6-family binding labels with REAL
assay-based negatives. Held-out orthogonal test for the mirage gate."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from mirage.benchmark._registry import AbstractLoader, register_loader
from mirage.scorers.base import BenchmarkExample


@register_loader("avida")
class AvidaLoader(AbstractLoader):
    """Reads the staged CSV produced by ``scripts/stage_avida.py``.

    Label mapping: ``1`` -> ``BIND``, ``0`` -> ``NONBIND`` (real assay negatives).
    """

    def __init__(self, staged_csv: str | Path) -> None:
        self.staged_csv = Path(staged_csv)
        if not self.staged_csv.is_file():
            raise FileNotFoundError(f"AVIDa staged CSV not found: {self.staged_csv}")

    def load(self) -> Iterator[BenchmarkExample]:
        with self.staged_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                yield BenchmarkExample(
                    id=row["vhh_id"],
                    label="BIND" if row["label"] == "1" else "NONBIND",
                    binder_chains=(row["vhh_sequence"],),
                    binder_format="vhh",
                    target_chains=(row["antigen_sequence"],),
                    target_name=row.get("antigen_label", "IL6-family"),
                    source="avida-hil6",
                    metadata={"raw_label": row["label"]},
                )
```

- [ ] **Step 5: Register the loader**

In `src/mirage/benchmark/__init__.py`, add the side-effect import line (match the existing style):

```python
import mirage.benchmark.avida  # noqa: F401
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_avida_loader.py -v`
Expected: PASS

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/benchmark/avida.py scripts/stage_avida.py src/mirage/benchmark/__init__.py tests/test_avida_loader.py
git commit -m "Add AVIDa-hIL6 loader + staging (orthogonal test with real negatives)"
```

---

## Task 10: Labeled-EpCAM (CAR-T killing) loader

**Files:**
- Create: `src/mirage/benchmark/epcam_killing.py`
- Modify: `src/mirage/benchmark/__init__.py`
- Test: `tests/test_epcam_killing_loader.py`

**Data provenance (manual, owner-supplied):** the killing assay labels map VHH
IDs to functional/non-functional. The sequences for those IDs come from the
SNAP panel. Produce a single CSV `epcam_killing_labels.csv` with columns
`vhh_id, vhh_sequence, label` where `label ∈ {Good, Bad}`:
- Good: 10, 25, 26, 34, 57, 61, 74, 86
- Bad: 14, 15, 16, 18, 21, 73
This mapping must be authored from the collaborator data + SNAP sequences; the
loader does not invent it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_epcam_killing_loader.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path

from mirage.benchmark._registry import get_loader


def _write_labels(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["vhh_id", "vhh_sequence", "label"])
        writer.writeheader()
        writer.writerow({"vhh_id": "10", "vhh_sequence": "QVQLAAAA", "label": "Good"})
        writer.writerow({"vhh_id": "14", "vhh_sequence": "EVQLBBBB", "label": "Bad"})


def test_epcam_killing_maps_good_bad_to_bind_nonbind(tmp_path: Path) -> None:
    labels = tmp_path / "epcam_killing_labels.csv"
    _write_labels(labels)
    loader = get_loader("epcam_killing", labels_csv=labels)
    examples = {e.id: e for e in loader.load()}
    assert examples["epcam-kill-10"].label == "BIND"
    assert examples["epcam-kill-14"].label == "NONBIND"
    assert examples["epcam-kill-10"].target_name == "EpCAM"
    assert examples["epcam-kill-10"].binder_format == "vhh"
    assert examples["epcam-kill-10"].metadata["assay"] == "cart_killing_aspc1"


def test_epcam_killing_requires_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    try:
        get_loader("epcam_killing", labels_csv=missing)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_epcam_killing_loader.py -v`
Expected: FAIL (no `epcam_killing` loader)

- [ ] **Step 3: Implement**

Create `src/mirage/benchmark/epcam_killing.py`:

```python
"""Labeled-EpCAM loader: designed VHHs with REAL CAR-T killing labels vs AsPC1
(EpCAM+). Held-out orthogonal test in the designed-binder deployment regime.

Label = functional killing (one step downstream of binding): ``Good`` -> ``BIND``,
``Bad`` -> ``NONBIND``. N is tiny (14) — test-only, report with wide CIs.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from mirage.benchmark._registry import AbstractLoader, register_loader
from mirage.benchmark.targets import EPCAM_ECD
from mirage.scorers.base import BenchmarkExample

_LABEL_MAP = {"Good": "BIND", "Bad": "NONBIND"}


@register_loader("epcam_killing")
class EpCAMKillingLoader(AbstractLoader):
    """Reads an owner-authored ``epcam_killing_labels.csv`` (vhh_id, vhh_sequence, label)."""

    def __init__(self, labels_csv: str | Path) -> None:
        self.labels_csv = Path(labels_csv)
        if not self.labels_csv.is_file():
            raise FileNotFoundError(f"EpCAM killing labels CSV not found: {self.labels_csv}")

    def load(self) -> Iterator[BenchmarkExample]:
        with self.labels_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                raw = row["label"].strip()
                if raw not in _LABEL_MAP:
                    raise ValueError(f"Unexpected EpCAM killing label {raw!r}; expected Good/Bad")
                yield BenchmarkExample(
                    id=f"epcam-kill-{row['vhh_id']}",
                    label=_LABEL_MAP[raw],
                    binder_chains=(row["vhh_sequence"],),
                    binder_format="vhh",
                    target_chains=(EPCAM_ECD,),
                    target_name="EpCAM",
                    source="epcam-killing",
                    target_pdb_id="4MZV",
                    metadata={"assay": "cart_killing_aspc1", "vhh_id": row["vhh_id"]},
                )
```

- [ ] **Step 4: Register the loader**

In `src/mirage/benchmark/__init__.py`, add:

```python
import mirage.benchmark.epcam_killing  # noqa: F401
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_epcam_killing_loader.py -v`
Expected: PASS

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/benchmark/epcam_killing.py src/mirage/benchmark/__init__.py tests/test_epcam_killing_loader.py
git commit -m "Add labeled-EpCAM (CAR-T killing) loader for designed-binder orthogonal test"
```

---

## Task 11: Frozen-gate orthogonal evaluation

**Files:**
- Create: `src/mirage/eval/orthogonal.py`, `scripts/analyze_ms_orthogonal.py`
- Test: `tests/test_orthogonal.py`

`evaluate_frozen_gate` is dataset-agnostic: given a frozen `MsModel` and an
iterable of `BenchmarkExample`, it builds Tier-S features, applies the frozen
threshold, and returns gate metrics + bootstrap CIs. The script runs it on AVIDa
and labeled-EpCAM and assembles the cross-regime precision-stability table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_orthogonal.py`:

```python
from __future__ import annotations

import numpy as np

from mirage.eval.orthogonal import evaluate_frozen_gate, features_for_examples
from mirage.model.ms import train_ms
from mirage.scorers.base import BenchmarkExample


def _example(binder: str, target: str, label: str) -> BenchmarkExample:
    return BenchmarkExample(
        id=f"{binder}-{label}", label=label, binder_chains=(binder,), binder_format="vhh",
        target_chains=(target,), target_name="X", source="test",
    )


def test_features_for_examples_shape() -> None:
    examples = [_example("KKKK", "DDDD", "BIND"), _example("DDDD", "KKKK", "NONBIND")]
    x, y, names = features_for_examples(examples, positive_label="BIND")
    assert x.shape[0] == 2
    assert list(y) == [1, 0]
    assert "binder_length" in names


def test_evaluate_frozen_gate_returns_metrics() -> None:
    # train a trivial model on synthetic separable features, then evaluate frozen
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.3, 40), rng.normal(2, 0.3, 40)])[:, None]
    y = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)
    model, _ = train_ms(x, y, feature_names=["binder_length"], l2=1.0,
                        target_precision=0.9, seed=0)
    # reuse the same x as "orthogonal" features for a smoke check
    result = evaluate_frozen_gate(model, x, y, n_boot=50, seed=1)
    assert "recall" in result["metrics"]
    assert "recall_ci" in result
    assert len(result["recall_ci"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orthogonal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mirage.eval.orthogonal'`

- [ ] **Step 3: Implement the harness**

Create `src/mirage/eval/orthogonal.py`:

```python
"""Apply a Champloo-frozen M-S gate to an orthogonal dataset, unchanged.

The headline mirage test: a threshold chosen on Champloo must hold its precision
on data it never saw. This module builds Tier-S features for any stream of
BenchmarkExamples and evaluates the frozen gate with bootstrap CIs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from mirage.eval.gate import bootstrap_ci, metrics_at_threshold
from mirage.features.sequence import FEATURE_NAMES, sequence_features
from mirage.model.ms import MsModel
from mirage.scorers.base import BenchmarkExample


def features_for_examples(
    examples: Iterable[BenchmarkExample], *, positive_label: str
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for ex in examples:
        feats = sequence_features(ex.binder_chains[0], ex.target_chains[0])
        rows.append([feats[name] for name in FEATURE_NAMES])
        labels.append(1 if ex.label == positive_label else 0)
    x = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    return x, y, FEATURE_NAMES


def evaluate_frozen_gate(
    model: MsModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score features with the frozen model + threshold; return metrics + CIs."""
    scores = model.predict_logit(x)
    metrics = metrics_at_threshold(scores, y, threshold=model.threshold)
    thr = model.threshold

    def _recall(s: np.ndarray, yy: np.ndarray) -> float:
        return metrics_at_threshold(s, yy, threshold=thr)["recall"]

    def _specificity(s: np.ndarray, yy: np.ndarray) -> float:
        return metrics_at_threshold(s, yy, threshold=thr)["specificity"]

    def _precision(s: np.ndarray, yy: np.ndarray) -> float:
        return metrics_at_threshold(s, yy, threshold=thr)["precision"]

    has_both = int((y == 1).sum()) > 0 and int((y == 0).sum()) > 0
    return {
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "metrics": metrics,
        "recall_ci": bootstrap_ci(_recall, scores, y, n_boot=n_boot, seed=seed)
        if (y == 1).sum() else (float("nan"), float("nan")),
        "specificity_ci": bootstrap_ci(_specificity, scores, y, n_boot=n_boot, seed=seed)
        if (y == 0).sum() else (float("nan"), float("nan")),
        "precision_ci": bootstrap_ci(_precision, scores, y, n_boot=n_boot, seed=seed)
        if has_both else (float("nan"), float("nan")),
    }
```

- [ ] **Step 4: Implement the script**

Create `scripts/analyze_ms_orthogonal.py`:

```python
"""Apply the Champloo-frozen M-S gate to the real-negative orthogonal sets
(AVIDa, labeled-EpCAM) and assemble the cross-regime precision-stability table.

Use::

    uv run python scripts/analyze_ms_orthogonal.py \\
        --model results/published/ms_model_af3.json \\
        --avida-csv ../abdisc-data/avida/avida_staged.csv \\
        --epcam-labels ../abdisc-data/epcam/epcam_killing_labels.csv \\
        --output results/published/mirage_ms_orthogonal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mirage.benchmark._registry import get_loader
from mirage.eval.orthogonal import evaluate_frozen_gate, features_for_examples
from mirage.model.ms import MsModel

# importing the package triggers loader self-registration
import mirage.benchmark  # noqa: F401,E402


def _evaluate(loader_name: str, model: MsModel, *, positive_label: str, **kwargs: Any) -> dict[str, Any]:
    loader = get_loader(loader_name, **kwargs)
    x, y, _names = features_for_examples(loader.load(), positive_label=positive_label)
    return evaluate_frozen_gate(model, x, y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--avida-csv", type=Path, default=None)
    parser.add_argument("--epcam-labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = MsModel.load(args.model)
    table: dict[str, Any] = {"threshold": model.threshold, "target_precision": model.target_precision, "regimes": {}}

    if args.avida_csv is not None:
        table["regimes"]["avida"] = _evaluate(
            "avida", model, positive_label="BIND", staged_csv=args.avida_csv
        )
    if args.epcam_labels is not None:
        table["regimes"]["epcam_killing"] = _evaluate(
            "epcam_killing", model, positive_label="BIND", labels_csv=args.epcam_labels
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))
    for regime, res in table["regimes"].items():
        m = res["metrics"]
        print(f"{regime}: n={res['n']} precision={m['precision']:.3f} "
              f"recall={m['recall']:.3f} specificity={m['specificity']:.3f}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_orthogonal.py -v`
Expected: PASS

- [ ] **Step 6: Run the real orthogonal evaluation (manual data step; requires staged AVIDa and/or EpCAM labels)**

Run:
```bash
uv run python scripts/analyze_ms_orthogonal.py \
  --model results/published/ms_model_af3.json \
  --avida-csv ../abdisc-data/avida/avida_staged.csv \
  --epcam-labels ../abdisc-data/epcam/epcam_killing_labels.csv \
  --output results/published/mirage_ms_orthogonal.json
```
Expected: prints per-regime precision/recall/specificity; writes JSON. (Omit a flag for any set not yet staged.)

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check && uv run ruff format && uv run mypy src/mirage
git add src/mirage/eval/orthogonal.py scripts/analyze_ms_orthogonal.py tests/test_orthogonal.py results/published/mirage_ms_orthogonal.json
git commit -m "Add frozen-gate orthogonal evaluation on AVIDa + labeled-EpCAM"
```

---

## Task 12: Phase A results summary

**Files:**
- Create: `results/published/mirage_phase_a_summary.md`

- [ ] **Step 1: Write the summary from the produced JSON artifacts**

Create `results/published/mirage_phase_a_summary.md` and fill the bracketed
numbers from `mirage_ms_indist_af3.json` and `mirage_ms_orthogonal.json`:

```markdown
# mirage Phase A — sequence-only gate (M-S) results

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
```

- [ ] **Step 2: Commit**

```bash
git add results/published/mirage_phase_a_summary.md
git commit -m "Add mirage Phase A results summary"
```

---

## Notes for the executor
- `data/staged/` and `data/raw/` are gitignored; staged feature CSVs are local artifacts. `results/published/` JSON + model artifacts are committed (small, citable).
- Manual data steps (Tasks 6/9/10 staging, and the real analysis runs) need the source data present: Champloo supp table (already in `abdisc-data/`), AVIDa raw CSVs (download), EpCAM killing-label CSV (author from collaborator data + SNAP sequences). The code tasks (tests) do not depend on those and can be completed first.
- Keep mirage dependency-free of GPU/ML-heavy packages: no sklearn, torch, or shap. Everything here is numpy/pandas/pure-Python.
- Out of scope (Phase B): Tier-C predictor-conditional features, AF3/structure generation for SAbDab and EpCAM, M-C, and the single-predictor-across-train/test constraint.

## Notes & wiki upkeep (standing instruction, all tasks)
- `mirage-notes/` and `mirage-wiki/` are Obsidian vaults, each its **own** git
  repo, separate from the code repo (bootstrapped in Task -1). Never commit
  notes/wiki files into the code repo or vice versa; commit each in its own repo.
- Maintain them **as you work**, at meaningful milestones — not just at the end,
  and not on every micro-step. Milestones = the code tasks landing as a batch;
  each real analysis run producing results; anything that gets blocked on data.
- **Progress log:** append a dated entry `YYYY-MM-DD - <short-desc>.md` under
  `mirage-notes/02 - Progress & Records/` — what landed, decisions made, what's
  blocked and why. The vault is empty (`.gitkeep` only), so you author the first
  entries (follow the abdisc-notes convention).
- **Current State:** keep `mirage-wiki/wiki/Current State.md` as the live
  "what's true right now" snapshot — create it and update it as Phase A
  progresses (what M-S is, what's built, what's pending). Synthesize current
  state; do not duplicate the plan.
- Commit notes/wiki changes in their own repos alongside the code milestones
  (concise messages, **no Claude trailer**), and **push each repo** after its
  commit — code repo (`origin`), `mirage-notes`, and `mirage-wiki` all have
  remotes wired in Task -1. If any `git push` is rejected for non-fast-forward,
  stop and surface it — do not force-push.

## Deferred: higher-capacity Tier-S backends (own phase, not Phase A)
Phase A commits the **bulk physicochemical Tier-S** features as the no-GPU
baseline floor — deliberately weak, deliberately defensible on 106 positives.
Capacity upgrades are a real roadmap item (esp. as a foundation for the
downstream RL-reward pipeline) but are **deferred to their own phase** so they
do not (a) break Phase A's no-GPU charter or (b) blur the weak-baseline-vs-M-C
comparison. Decided architecture for when they land:
- **ESM-2 frozen embeddings:** computed by a staging step that shells out to a
  separate torch/GPU env and caches `embeddings.npy` to disk (same pattern as
  GPU pose predictors) — the pure-Python mirage core reads the vectors as an
  alternate Tier-S backend feeding the **same** L2-logistic/GBT head. No torch
  in the uv env. End-to-end fine-tuning waits until the training reservoir
  (SAbDab/AVIDa folds) actually grows.
- **Cheaper interim option:** CDR-targeted features via ANARCI (already in the
  `mber` conda env; same shell-out pattern). CDR3 length + per-CDR composition
  attacks the real weakness — whole-VHH bulk composition dilutes the CDR binding
  signal — without torch and without the small-cohort overfit risk.
The feature backend is the swap point: keep `features/sequence.py` as one
backend behind a stable `FEATURE_NAMES`/`sequence_features`-style contract so a
new backend slots in without touching the model, gate, or orthogonal harness.
