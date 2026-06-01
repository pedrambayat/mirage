# Repository Guidelines

## Project Structure & Module Organization

`abdisc` uses a Python `src/` layout. Core code lives in `src/abdisc/`: `benchmark/` contains dataset loaders and target metadata, `scorers/` contains scorer interfaces and implementations, `eval/` is for metrics and leaderboard work, `design/` is for design-pipeline integrations, and `cli.py` defines the Typer CLI. Tests live in `tests/` as focused `test_*.py` modules. Design notes and plans live under `docs/superpowers/`. Published outputs belong under `results/published/`; avoid committing large generated artifacts elsewhere.

## Build, Test, and Development Commands

Use `uv` for local development:

```bash
uv python install 3.11      # install the supported Python version
uv sync                     # create/update the locked environment
uv run abdisc version       # verify the CLI entry point
uv run pytest               # run the test suite
uv run ruff check           # lint Python files
uv run ruff format          # format Python files
uv run mypy src/abdisc      # strict type checking
uv run pre-commit install   # install local commit hooks
```

Benchmark examples include `uv run abdisc bench list-scorers`, `uv run abdisc bench list-loaders`, and `uv run abdisc bench score --scorer length --loader epcam --output scores.csv`.

## Coding Style & Naming Conventions

Target Python 3.11 and keep code typed. Ruff enforces rules `E`, `F`, `I`, `B`, `UP`, `N`, and `RUF`, with a 100-character line length and double quotes. Mypy runs in strict mode over `src/abdisc`. Use snake_case for functions, modules, variables, CLI callbacks, and test names. Use clear registry names such as `length` or `epcam` for scorers and loaders.

## Testing Guidelines

Pytest is configured with `testpaths = ["tests"]` and quiet reporting. Add tests in `tests/test_<feature>.py`. Prefer small unit tests for registries, scorer outputs, loader parsing, and CLI behavior. Run `uv run pytest` before opening a PR; run the CI-equivalent set (`ruff check`, `ruff format --check`, `mypy`, `pytest`) for shared interfaces or loader changes.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit subjects, often with an optional scope prefix, for example `README: emphasize general binder discrimination` or `Land core abstractions, EpCAM loader, LengthScorer, bench CLI`. Keep commits focused. PRs should describe intent, list validation commands, link related issues or docs, and include sample CLI output or result paths when behavior changes.

## Security & Configuration Tips

Do not commit private datasets, credentials, or bulky benchmark outputs. Pass external dataset paths with flags such as `--data-dir` or variables such as `ABDISC_EPCAM_DATA`. Keep GPU-dependent tools in separate environments.
