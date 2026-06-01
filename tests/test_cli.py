from __future__ import annotations

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mirage.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_version_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_bench_list_scorers(runner: CliRunner) -> None:
    result = runner.invoke(app, ["bench", "list-scorers"])
    assert result.exit_code == 0
    assert "length" in result.stdout


def test_bench_list_loaders(runner: CliRunner) -> None:
    result = runner.invoke(app, ["bench", "list-loaders"])
    assert result.exit_code == 0
    assert "epcam" in result.stdout


def test_bench_score_end_to_end(runner: CliRunner, epcam_data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "scores.csv"
    result = runner.invoke(
        app,
        [
            "bench",
            "score",
            "--scorer",
            "length",
            "--loader",
            "epcam",
            "--data-dir",
            str(epcam_data_dir),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output.is_file()
    with output.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 43 + 86 + 24
    assert {r["label"] for r in rows} == {"POS", "SCR", "OFF"}
    for row in rows:
        assert float(row["value"]) > 0
        assert row["scorer_name"] == "length"
