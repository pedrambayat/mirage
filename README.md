# mirage

A **pose-pipeline-agnostic discriminator** for predicted binder-target complexes. mirage takes any predicted complex and scores whether the binding event is real. Where a crystal exists, "real" is operationalized as "the predicted pose is close to the crystal pose" via RMSD-to-crystal and DockQ. Where no crystal exists, mirage can still run crystal-independent confidence and structural scorers.

The same discriminator should score Protenix, AF2-M, AF3, Boltz, and design-pipeline outputs equivalently, agnostic to which pipeline produced the pose. Multiple binder formats are in scope: VHH, scFv, Fab, minibinder, peptide. Successor to SNAP.

Current state: the SAbDab N=200 AF2-M baseline is published; CDR-scramble and wrong-target negative controls are completed and scored; numpy-only logistic confidence baselines and ablations are published; the EpCAM designed-binder AF2-M check is running as SLURM array `6133301`.

For durable synthesis, see `../mirage-wiki/wiki/Current State.md` and `../mirage-wiki/wiki/Explainer.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.11
uv sync
uv run pre-commit install
```

## Quick start

```bash
# List what's plugged in
uv run mirage bench list-scorers
uv run mirage bench list-loaders

# End-to-end smoke run against the SNAP EpCAM dataset (43 POS + 86 SCR + 24 OFF)
export ABDISC_EPCAM_DATA="<path-to-SNAP>/binder-discrimination/data"
uv run mirage bench score --scorer length --loader epcam --output scores.csv
```

Output CSV is one standardized row per example: `example_id, scorer_name, value, label, target_name, source, binder_format, split, extras_json`.

## Develop

```bash
uv run ruff check         # lint
uv run ruff format        # format
uv run mypy src/mirage    # type check
uv run pytest             # tests
```

## Layout

- `src/mirage/benchmark/` — loaders for EpCAM, SAbDab, and future datasets
- `src/mirage/pose_predictors/` — predictor wrappers and staging/submission lifecycle
- `src/mirage/scorers/` — scorer interface plus length, RMSD-to-crystal, and AF2-M confidence scorers
- `src/mirage/eval/` — metrics, leaderboard generation, plots
- `src/mirage/design/` — design-pipeline integrations (placeholder; specific generators wired in as the design investigation matures)
- `src/mirage/cli.py` — Typer CLI (`mirage bench …`, `mirage design …`, …)
- `scripts/` — staging, scoring, SLURM, and analysis scripts used for current baselines

GPU-dependent generators and scorers are intentionally NOT installed into the mirage uv env. When the project integrates one, it shells out to the generator's own environment (e.g., a separate conda env on a SLURM job) so mirage's deps stay pure-Python and fast to resolve.
