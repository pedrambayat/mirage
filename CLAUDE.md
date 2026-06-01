# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

abdisc trains a **pose-pipeline-agnostic discriminator**. Given any predicted (binder, target) complex, it outputs a score for whether the binding event is real. "Is this binder real?" is operationalized as "is the predicted pose close to a crystal pose?" — measured by RMSD-to-crystal where a crystal exists.

**Ground truth source:** SAbDab. For each crystal-validated complex, run one or more pose predictors (AF2M / AF3 / Protenix / Boltz / …), compute RMSD between the predicted pose and the crystal pose. Small RMSD = real binding event well-recovered; large RMSD = predictor produced a plausible-looking but wrong complex.

**Positive control:** EpCAM (from SNAP). 153 VHHs against EpCAM ECD — see the EpCAM loader. Trained discriminator must call known binders' poses good.

**Key property:** the discriminator is pose-pipeline-agnostic. Same model scores Protenix and AF3 outputs equivalently. Do not encode predictor-specific signals into the discriminator's input.

Multiple binder formats are in scope (VHH/nanobody, scFv, Fab, minibinder, peptide). The original 2026-05-10 design spec's splice-variant / CAR-T framing is deprecated (see the 2026-05-11 and 2026-05-12 amendments). Design pipelines are a **secondary** investigation, not a co-equal track. Successor to the SNAP project.

**Two diagnostic analyses tracked alongside the headline discriminator** (not the headline themselves):
- **Pose-predictor concordance** — inter-predictor RMSD as a no-crystal proxy for pose correctness.
- **Hotspot agreement** — for designed binders, does the predictor place the interface on the intended hotspots? See `…/abdisc-wiki/wiki/Methods/Hotspot Agreement.md`.

## Repository Structure

```
src/abdisc/
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
└── cli.py                 # Typer CLI (`abdisc bench list-scorers | list-loaders | score`)
docs/superpowers/
├── specs/2026-05-10-abdisc-design.md       # Design doc (with 2026-05-11 amendment)
└── plans/2026-05-10-abdisc-week1-foundation.md  # Implementation plan (Tasks 1-2 bootstrapped)
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
uv run mypy src/abdisc             # type check (strict)
uv run pytest                      # tests
uv run abdisc version              # CLI sanity check

# End-to-end smoke run against the SNAP EpCAM dataset
export ABDISC_EPCAM_DATA="<path-to-SNAP>/binder-discrimination/data"
uv run abdisc bench score --scorer length --loader epcam --output scores.csv
```

CI runs the same `ruff check && ruff format --check && mypy && pytest` battery on every push.

## Architecture

The framework is built around **one standardized result shape** so any scorer can be compared head-to-head with any other:

```
loader → Iterator[BenchmarkExample] → scorer.score() → Score → CSV row
```

Three core types in `src/abdisc/scorers/base.py`:

- `BenchmarkExample` — one (binder, target) pair. Permissive: `binder_chains: tuple[str, ...]` and a free-form `binder_format: str` ("vhh", "scfv", "fab", "minibinder", "peptide", or whatever). Anything format-specific (CDR positions, resolution, source structure) rides in `metadata: dict[str, Any]`.
- `Score` — one scorer's output for one example. Headline `value: float` plus `extras: dict[str, float | str]` for auxiliary fields a scorer wants to expose.
- `AbstractScorer` — ABC with `score(example) -> Score` and a default `score_batch` iterator.

Two decorator-based registries follow the same pattern (in `scorers/_registry.py` and `benchmark/_registry.py`):

- `@register("name")` for scorers; `get_scorer(name, **kwargs)`; `list_scorers()`.
- `@register_loader("name")` for loaders; `get_loader(name, **kwargs)`; `list_loaders()`.

Built-in scorers/loaders self-register on package import via side-effect imports in `scorers/__init__.py` and `benchmark/__init__.py`. Do not break this — adding a new module to one of those packages requires adding it to the side-effect import line if you want it auto-registered.

## Adding a new scorer

1. Create `src/abdisc/scorers/<name>.py`.
2. Subclass `AbstractScorer`, implement `score(example) -> Score`.
3. Decorate with `@register("<name>")`.
4. Add `import abdisc.scorers.<name>  # noqa: F401` to `src/abdisc/scorers/__init__.py`.
5. Add a test under `tests/test_<name>_scorer.py`.

The wrapper should be ~50 lines. GPU-dependent models (AF2-M, Protenix, Boltz, ...) do NOT install into the abdisc uv env — they shell out to their own conda env or a SLURM job. Keep abdisc's deps pure-Python and fast to resolve.

## Adding a new loader

Same pattern: `AbstractLoader` subclass with `load() -> Iterator[BenchmarkExample]`, `@register_loader("<name>")`, side-effect import in `benchmark/__init__.py`, a test. Loaders take config in `__init__` (e.g., `data_dir`); the CLI passes any `--data-dir`-style flags through `get_loader(name, **kwargs)`.

## Conventions

- **TDD-adjacent:** tests land in the same change as implementation. Pre-commit and CI both enforce them.
- **mypy strict** on `src/abdisc/` — no `Any` returns, no implicit Optional. Test files are exempt.
- **Ruff:** `E, F, I, B, UP, N, RUF` selected. Format with `ruff format`. Use `Annotated[T, typer.Option(...)]` for Typer args (avoids B008).
- **No `from __future__ import annotations` needed for new files** unless you hit a forward-reference issue (Python 3.11 supports the union syntax natively).
- **Commits:** never include a `Co-Authored-By: Claude` (or any Claude/Anthropic) trailer. Pedram authors his commits alone.
- **Permissive binder schema:** when adding fields to `BenchmarkExample`, prefer optional or `metadata`-resident fields over required ones — diverse binder formats means anything format-specific should not be a required field.

## Datasets

**Committed for v1:**

- **EpCAM** — from SNAP's `binder-discrimination/data/`. 153 examples (43 POS + 86 SCR + 24 OFF VHHs vs. EpCAM ECD). Loaded via `EpCAMLoader`. Target sequence is UniProt P16422 residues 24-265 (4MZV chain A), constant in `src/abdisc/benchmark/targets.py`.
- **SAbDab** — broader Ab-Ag complex DB. Loader **not yet built**.

**Likely to be added:** TBD after literature curation.

**Source repos (read-only references):**

- SNAP: `/Users/pedrambayat/Library/CloudStorage/GoogleDrive-pbayat123@gmail.com/My Drive/second brain/2 - Source Material/Research/SNAP/snap`. Data and analysis source for benchmark v1; do NOT modify.

## Out of Scope (do not introduce)

- **conda for abdisc's own env** — uv only.
- **mBER as a committed dependency** — `src/abdisc/design/` is a placeholder. No specific generator (mBER, BindCraft, BoltzDesign) is wired in yet, and any future integration must shell out to a separate env, not install into abdisc.
- **GPU-dependent packages in pyproject.toml** — JAX-CUDA, PyTorch-CUDA wheels, ColabFold, etc. live in external envs.
- **DVC / Git LFS** — manifest is small; PDBs/predictions live in `data/raw/` (gitignored).
- **Splice-variant CAR-T framing** — see the 2026-05-11 amendment. CHL1 / MSLN remain available as motivating examples, not headline contributions.

## Reference Documents

- **Project design:** `docs/superpowers/specs/2026-05-10-abdisc-design.md` (read the 2026-05-11 amendment header first — sections 2, 3, 7 are stale; sections 4-6 remain authoritative).
- **Week 1 implementation plan:** `docs/superpowers/plans/2026-05-10-abdisc-week1-foundation.md` (Tasks 1-2 bootstrapped; Tasks 3-4 landed; Task 5+ remaining).
- **Current state snapshot:** `…/Research/abdisc/abdisc-wiki/wiki/Current State.md` — live, updated.
- **Wiki conventions:** `…/Research/abdisc/abdisc-wiki/schema/Schema.md`.
- **Stack bootstrap plan:** `~/.claude/plans/can-you-find-the-soft-moonbeam.md`.

## When in doubt

- Ground claims about *current* state in `git log`, `pytest`, or the actual files — not in the design spec, which captures decisions made at a moment in time.
- The Current State wiki page is the canonical "what's true right now" pointer.
- When framing the project externally, lead with "general binder discrimination" — not antibody-only, not CAR-T, not splice-variant.
