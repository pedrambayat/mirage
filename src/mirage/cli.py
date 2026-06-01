import csv
import json
from pathlib import Path
from typing import Annotated

import typer

from mirage import __version__
from mirage.benchmark import get_loader, list_loaders
from mirage.scorers import get_scorer, list_scorers

app = typer.Typer(help="Antibody-antigen discriminator benchmark.", no_args_is_help=True)
bench_app = typer.Typer(help="Benchmark commands.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")


@app.callback()
def _root() -> None:
    """Top-level mirage CLI; subcommands implement the actual work."""


@app.command()
def version() -> None:
    """Print the installed mirage version."""
    typer.echo(__version__)


@bench_app.command("list-scorers")
def bench_list_scorers() -> None:
    """List registered scorers."""
    for name in list_scorers():
        typer.echo(name)


@bench_app.command("list-loaders")
def bench_list_loaders() -> None:
    """List registered loaders."""
    for name in list_loaders():
        typer.echo(name)


@bench_app.command("score")
def bench_score(
    scorer: Annotated[str, typer.Option(help="Registered scorer name.")],
    loader: Annotated[str, typer.Option(help="Registered loader name.")],
    output: Annotated[Path, typer.Option(help="Output CSV path.")],
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Loader data dir.")] = None,
) -> None:
    """Run `scorer` over every example from `loader`; write a CSV of scores."""
    loader_kwargs: dict[str, Path] = {}
    if data_dir is not None:
        loader_kwargs["data_dir"] = data_dir
    loader_instance = get_loader(loader, **loader_kwargs)
    scorer_instance = get_scorer(scorer)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "example_id",
                "scorer_name",
                "value",
                "label",
                "target_name",
                "source",
                "binder_format",
                "split",
                "extras_json",
            ]
        )
        n = 0
        for example in loader_instance.load():
            result = scorer_instance.score(example)
            writer.writerow(
                [
                    example.id,
                    result.scorer_name,
                    result.value,
                    example.label,
                    example.target_name,
                    example.source,
                    example.binder_format,
                    example.split or "",
                    json.dumps(result.extras, sort_keys=True),
                ]
            )
            n += 1
    typer.echo(f"Wrote {n} scores to {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
