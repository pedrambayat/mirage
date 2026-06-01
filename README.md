# abdisc

A **pose-pipeline-agnostic discriminator** for predicted binder-target complexes. abdisc takes any predicted complex and scores whether the binding event is real. Where a crystal exists, "real" is operationalized as "the predicted pose is close to the crystal pose" via RMSD-to-crystal and DockQ. Where no crystal exists, abdisc can still run crystal-independent confidence and structural scorers.

The same discriminator should score Protenix, AF2-M, AF3, Boltz, and design-pipeline outputs equivalently, agnostic to which pipeline produced the pose. Multiple binder formats are in scope: VHH, scFv, Fab, minibinder, peptide. Successor to SNAP.

Current state: the SAbDab N=200 AF2-M baseline is published; CDR-scramble and wrong-target negative controls are completed and scored; numpy-only logistic confidence baselines and ablations are published; the EpCAM designed-binder AF2-M check is running as SLURM array `6133301`.

For durable synthesis, see `../abdisc-wiki/wiki/Current State.md` and `../abdisc-wiki/wiki/Explainer.md`.

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
uv run abdisc bench list-scorers
uv run abdisc bench list-loaders

# End-to-end smoke run against the SNAP EpCAM dataset (43 POS + 86 SCR + 24 OFF)
export ABDISC_EPCAM_DATA="<path-to-SNAP>/binder-discrimination/data"
uv run abdisc bench score --scorer length --loader epcam --output scores.csv
```

Output CSV is one standardized row per example: `example_id, scorer_name, value, label, target_name, source, binder_format, split, extras_json`.

## Develop

```bash
uv run ruff check         # lint
uv run ruff format        # format
uv run mypy src/abdisc    # type check
uv run pytest             # tests
```

## Layout

- `src/abdisc/benchmark/` — loaders for EpCAM, SAbDab, and future datasets
- `src/abdisc/pose_predictors/` — predictor wrappers and staging/submission lifecycle
- `src/abdisc/scorers/` — scorer interface plus length, RMSD-to-crystal, and AF2-M confidence scorers
- `src/abdisc/eval/` — metrics, leaderboard generation, plots
- `src/abdisc/design/` — design-pipeline integrations (placeholder; specific generators wired in as the design investigation matures)
- `src/abdisc/cli.py` — Typer CLI (`abdisc bench …`, `abdisc design …`, …)
- `scripts/` — staging, scoring, SLURM, and analysis scripts used for current baselines

GPU-dependent generators and scorers are intentionally NOT installed into the abdisc uv env. When the project integrates one, it shells out to the generator's own environment (e.g., a separate conda env on a SLURM job) so abdisc's deps stay pure-Python and fast to resolve.
