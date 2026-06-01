# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mirage is a **binding-discrimination** model: given a *predicted* `(binder, target)` complex, it scores whether that complex reflects a **real** binding event. The *mirage score* estimates how likely an interface is a plausible-but-incorrect rendering rather than true binding. Scope is **general antibody–antigen** binding (VHH/nanobody in v1; Fab / scFv / minibinder / peptide are the committed format-generalization axis). mirage is the successor to **abdisc** (now preparatory baseline characterization) and **SNAP**.

**Predictor-conditional.** The discriminator *does* consume the pose predictor's own confidence outputs (ipTM / PAE / pLDDT) alongside the predicted complex — AF3 in practice, swappable. The earlier "predictor-agnostic" rule is **retired** as of the 2026-05-28 mirage reframe; format generalization is the committed evaluation axis instead.

**Ground truth & datasets.** Champloo / Smorodina (cognate VHH–Ag co-crystals + AF3 structures + ipTM) is the primary train / in-distribution set; SAbDab is the orthogonal true-positive reservoir and format axis; AVIDa-hIL6 and labeled-EpCAM (CAR-T killing) provide held-out **real-negative** tests. See `docs/datasets/dataset-registry.md`.

**Current work — Phase A (sequence-only gate, M-S).** Phase A builds a no-GPU, sequence-only L2-logistic binder/non-binder gate on Champloo Tier-S sequence features, evaluated as an FP-costly gate (sensitivity/specificity at a fixed-precision operating point, PPV-vs-prevalence sweep, bootstrap CIs) with a Champloo-*frozen* threshold validated on the orthogonal real-negative sets. M-S is the explicit **pre-structure baseline** — it does *not* consume predictor confidence and so does **not** answer mirage's headline question (that is M-C, Phase B); it establishes the data/feature/validation infrastructure and a sequence-only reference floor.

**Inherited scaffolding.** The codebase still carries the pose-prediction framework migrated from the predecessor project — pose predictors (AF2-M / …), crystal-RMSD / DockQ and AF2-M-confidence scorers, EpCAM / SAbDab loaders — retained for the structure track (Phase B) and the AF2-M confidence benchmark. New Phase A code is pure-Python / numpy under `src/mirage/{ml,features,eval,model}/`.

> Note: AF2-Multimer confidence metrics have ~0.50 AUROC for binder/non-binder discrimination (they model scaffold docking, not sequence-specific recognition) — see `snap/benchmark/RESULTS.md`. That negative result is *why* this line of work exists.

## Repository Structure

```
src/mirage/
├── scorers/
│   ├── base.py            # BenchmarkExample, Score, AbstractScorer — the contract
│   ├── _registry.py       # @register decorator + get_scorer + list_scorers
│   └── length.py          # LengthScorer — trivial smoke scorer (no GPU, no model)
├── benchmark/
│   ├── _registry.py       # AbstractLoader + @register_loader + get_loader + list_loaders
│   ├── loaders.py         # EpCAMLoader (more loaders land here as datasets get added)
│   └── targets.py         # Antigen sequence constants (EPCAM_ECD, ...)
├── eval/                  # Metrics + leaderboard — to be populated
├── design/                # Secondary track; placeholder, no concrete generator integrated
└── cli.py                 # Typer CLI (`mirage bench list-scorers | list-loaders | score`)
docs/superpowers/
├── specs/2026-05-28-mirage-design.md                     # mirage design (locked)
├── specs/2026-05-31-mirage-data-and-training-strategy.md # data + training strategy (Phase A spec)
└── plans/2026-05-31-mirage-phase-a-sequence-gate.md      # Phase A implementation plan
tests/                                        # pytest; conftest.py provides EpCAM path fixture
.github/workflows/ci.yml                      # uv sync → ruff → mypy → pytest on push/PR
```

## Build and Development Commands

This project uses **uv** (not conda). All commands run inside the uv-managed venv.

```bash
uv python install 3.11             # one-time
uv sync                            # install/refresh deps from pyproject.toml + uv.lock
uv run pre-commit install          # one-time hook setup

# Development loop
uv run ruff check                  # lint
uv run ruff format                 # autoformat
uv run mypy src/mirage             # type check (strict)
uv run pytest                      # tests
uv run mirage version              # CLI sanity check

# End-to-end smoke run against the SNAP EpCAM dataset
export MIRAGE_EPCAM_DATA="<path-to-SNAP>/binder-discrimination/data"
uv run mirage bench score --scorer length --loader epcam --output scores.csv
```

CI runs the same `ruff check && ruff format --check && mypy && pytest` battery on every push.

## Architecture

The framework is built around **one standardized result shape** so any scorer can be compared head-to-head with any other:

```
loader → Iterator[BenchmarkExample] → scorer.score() → Score → CSV row
```

Three core types in `src/mirage/scorers/base.py`:

- `BenchmarkExample` — one (binder, target) pair. Permissive: `binder_chains: tuple[str, ...]` and a free-form `binder_format: str` ("vhh", "scfv", "fab", "minibinder", "peptide", or whatever). Anything format-specific (CDR positions, resolution, source structure) rides in `metadata: dict[str, Any]`.
- `Score` — one scorer's output for one example. Headline `value: float` plus `extras: dict[str, float | str]` for auxiliary fields a scorer wants to expose.
- `AbstractScorer` — ABC with `score(example) -> Score` and a default `score_batch` iterator.

Two decorator-based registries follow the same pattern (in `scorers/_registry.py` and `benchmark/_registry.py`):

- `@register("name")` for scorers; `get_scorer(name, **kwargs)`; `list_scorers()`.
- `@register_loader("name")` for loaders; `get_loader(name, **kwargs)`; `list_loaders()`.

Built-in scorers/loaders self-register on package import via side-effect imports in `scorers/__init__.py` and `benchmark/__init__.py`. Do not break this — adding a new module to one of those packages requires adding it to the side-effect import line if you want it auto-registered.

## Adding a new scorer

1. Create `src/mirage/scorers/<name>.py`.
2. Subclass `AbstractScorer`, implement `score(example) -> Score`.
3. Decorate with `@register("<name>")`.
4. Add `import mirage.scorers.<name>  # noqa: F401` to `src/mirage/scorers/__init__.py`.
5. Add a test under `tests/test_<name>_scorer.py`.

The wrapper should be ~50 lines. GPU-dependent models (AF2-M, Protenix, Boltz, ...) do NOT install into the mirage uv env — they shell out to their own conda env or a SLURM job. Keep mirage's deps pure-Python and fast to resolve.

## Adding a new loader

Same pattern: `AbstractLoader` subclass with `load() -> Iterator[BenchmarkExample]`, `@register_loader("<name>")`, side-effect import in `benchmark/__init__.py`, a test. Loaders take config in `__init__` (e.g., `data_dir`); the CLI passes any `--data-dir`-style flags through `get_loader(name, **kwargs)`.

## Conventions

- **TDD-adjacent:** tests land in the same change as implementation. Pre-commit and CI both enforce them.
- **mypy strict** on `src/mirage/` — no `Any` returns, no implicit Optional. Test files are exempt.
- **Ruff:** `E, F, I, B, UP, N, RUF` selected. Format with `ruff format`. Use `Annotated[T, typer.Option(...)]` for Typer args (avoids B008).
- **No `from __future__ import annotations` needed for new files** unless you hit a forward-reference issue (Python 3.11 supports the union syntax natively).
- **Commits:** never include a `Co-Authored-By: Claude` (or any Claude/Anthropic) trailer. Pedram authors his commits alone.
- **Permissive binder schema:** when adding fields to `BenchmarkExample`, prefer optional or `metadata`-resident fields over required ones — diverse binder formats means anything format-specific should not be a required field.

## Datasets

**Committed for v1:**

- **EpCAM** — from SNAP's `binder-discrimination/data/`. 153 examples (43 POS + 86 SCR + 24 OFF VHHs vs. EpCAM ECD). Loaded via `EpCAMLoader`. Target sequence is UniProt P16422 residues 24-265 (4MZV chain A), constant in `src/mirage/benchmark/targets.py`.
- **SAbDab** — broader Ab-Ag complex DB. Loader **not yet built**.

**Likely to be added:** TBD after literature curation.

**Source repos (read-only references):**

- SNAP: `/Users/pedrambayat/Library/CloudStorage/GoogleDrive-pbayat123@gmail.com/My Drive/second brain/2 - Source Material/Research/SNAP/snap`. Data and analysis source for benchmark v1; do NOT modify.

## Out of Scope (do not introduce)

- **conda for mirage's own env** — uv only.
- **mBER as a committed dependency** — `src/mirage/design/` is a placeholder. No specific generator (mBER, BindCraft, BoltzDesign) is wired in yet, and any future integration must shell out to a separate env, not install into mirage.
- **GPU-dependent packages in pyproject.toml** — JAX-CUDA, PyTorch-CUDA wheels, ColabFold, etc. live in external envs.
- **DVC / Git LFS** — manifest is small; PDBs/predictions live in `data/raw/` (gitignored).
- **Splice-variant / CAR-T framing** — deprecated. Frame mirage as general antibody–antigen binding discrimination; CAR-T binder design is a downstream driver, not the scope.

## Reference Documents

- **mirage design (locked):** `docs/superpowers/specs/2026-05-28-mirage-design.md`.
- **Data & training strategy (Phase A spec):** `docs/superpowers/specs/2026-05-31-mirage-data-and-training-strategy.md`.
- **Phase A implementation plan:** `docs/superpowers/plans/2026-05-31-mirage-phase-a-sequence-gate.md`.
- **Dataset registry:** `docs/datasets/dataset-registry.md`.
- **Current state snapshot:** `../mirage-wiki/wiki/Current State.md` — live "what's true right now" pointer.
- **Progress log:** `../mirage-notes/02 - Progress & Records/` — dated entries.
- **Predecessor design docs (historical):** `docs/superpowers/specs/2026-05-10-abdisc-design.md`, `docs/superpowers/plans/2026-05-10-abdisc-week1-foundation.md` — abdisc-era, kept for lineage and superseded by the mirage docs above.

## When in doubt

- Ground claims about *current* state in `git log`, `pytest`, or the actual files — not in the design spec, which captures decisions made at a moment in time.
- The Current State wiki page is the canonical "what's true right now" pointer.
- When framing the project externally, lead with "general antibody–antigen binding discrimination" — not CAR-T, not splice-variant.
