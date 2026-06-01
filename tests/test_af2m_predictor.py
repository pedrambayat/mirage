from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mirage.pose_predictors.af2m import (
    AF2MPosePredictor,
    _clean_sequence,
    _example_to_fasta,
    _parse_sbatch_job_id,
    af2m_from_env,
)
from mirage.scorers.base import BenchmarkExample


def _make_example(
    example_id: str = "test-1",
    binder: tuple[str, ...] = ("HQVQLV",),
) -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        label="POS",
        binder_chains=binder,
        binder_format="vhh",
        target_chains=("ATCDEFGHIKLMNPQRSTVWY",),
        target_name="Test antigen",
        source="test",
    )


def _make_predictor(tmp_path: Path) -> AF2MPosePredictor:
    return AF2MPosePredictor(
        output_root=tmp_path / "out",
        staged_root=tmp_path / "stage",
        slurm_script=tmp_path / "predict_af2m.slurm",
    )


def test_clean_sequence_strips_noncanonical() -> None:
    assert _clean_sequence(" acdef\n*X-hijk ") == "ACDEFHIK"


def test_example_to_fasta_vhh_two_chains() -> None:
    example = _make_example(binder=("HQVQLV",))
    fasta = _example_to_fasta(example)
    assert fasta.startswith(">test-1\n")
    assert "HQVQLV:ATCDEFGHIKLMNPQRSTVWY\n" in fasta


def test_example_to_fasta_fab_three_chains() -> None:
    example = BenchmarkExample(
        id="fab",
        label="POS",
        binder_chains=("HHHHH", "LLLLL"),
        binder_format="fab",
        target_chains=("AAAAA",),
        target_name="t",
        source="test",
    )
    assert "HHHHH:LLLLL:AAAAA" in _example_to_fasta(example)


def test_example_to_fasta_rejects_empty_after_cleaning() -> None:
    example = BenchmarkExample(
        id="bad",
        label="POS",
        binder_chains=("***",),
        binder_format="vhh",
        target_chains=("AAAA",),
        target_name="t",
        source="test",
    )
    with pytest.raises(ValueError):
        _example_to_fasta(example)


def test_results_for_returns_none_when_missing(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    assert pred.results_for(_make_example()) is None


def test_results_for_returns_path_when_cached(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    example = _make_example()
    target = pred.output_root / example.id / "rank1.pdb"
    target.parent.mkdir(parents=True)
    target.write_text("ATOM ...\n")
    assert pred.results_for(example) == target


def test_stage_writes_fasta_and_manifest(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [_make_example(f"ex-{i}") for i in range(3)]
    manifest = pred.stage(examples)
    assert manifest.n_rows == 3
    assert manifest.n_already_cached == 0
    assert manifest.path.is_file()
    rows = manifest.path.read_text().strip().splitlines()
    assert rows[0] == "example_id\tfasta_path\tout_dir"
    assert len(rows) == 4
    for example in examples:
        fasta = pred.staged_root / "fasta" / f"{example.id}.fasta"
        assert fasta.is_file()
        assert fasta.read_text().startswith(f">{example.id}\n")


def test_stage_skips_already_cached(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    cached = _make_example("cached")
    fresh = _make_example("fresh")
    target = pred.output_root / cached.id / "rank1.pdb"
    target.parent.mkdir(parents=True)
    target.write_text("ATOM ...\n")

    manifest = pred.stage([cached, fresh])
    assert manifest.n_rows == 1
    assert manifest.n_already_cached == 1
    body = manifest.path.read_text()
    assert "fresh" in body
    assert "cached" not in body


def test_stage_empty_manifest_when_all_cached(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    cached = _make_example("cached")
    target = pred.output_root / cached.id / "rank1.pdb"
    target.parent.mkdir(parents=True)
    target.write_text("ATOM ...\n")

    manifest = pred.stage([cached])
    assert manifest.n_rows == 0
    assert manifest.n_already_cached == 1
    assert not manifest.path.exists()


def test_sbatch_command_chunks_and_concurrency(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [_make_example(f"ex-{i}") for i in range(20)]
    manifest = pred.stage(examples)
    cmd = pred.sbatch_command(manifest, chunk_size=5, max_concurrent=4)
    assert cmd[0] == "sbatch"
    assert "--account=dbgoodma-goodman-laboratory" in cmd
    assert "--partition=b200-mig45" in cmd
    assert "--gres=gpu:1" in cmd
    assert "--array=0-3%4" in cmd  # 20 examples / 5 per task = 4 tasks
    assert cmd[-3] == str(manifest.path)
    assert cmd[-2] == "5"
    assert cmd[-1].endswith("run_af2m_chunk.py")


def test_sbatch_command_rounds_up_partial_chunk(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [_make_example(f"ex-{i}") for i in range(11)]
    manifest = pred.stage(examples)
    cmd = pred.sbatch_command(manifest, chunk_size=5, max_concurrent=4)
    assert "--array=0-2%4" in cmd  # 11/5 → 3 tasks (indices 0,1,2)


def test_submit_dry_run_does_not_subprocess(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    pred.slurm_script.write_text("#!/bin/bash\n")
    examples = [_make_example("ex-0")]
    manifest = pred.stage(examples)

    with patch("subprocess.run") as run_mock:
        result = pred.submit(manifest, chunk_size=5, max_concurrent=4, dry_run=True)
    assert result is None
    run_mock.assert_not_called()


def test_submit_no_op_on_empty_manifest(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    empty = pred.stage([])
    assert empty.n_rows == 0
    with patch("subprocess.run") as run_mock:
        assert pred.submit(empty) is None
    run_mock.assert_not_called()


def test_submit_parses_job_id(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    pred.slurm_script.write_text("#!/bin/bash\n")
    examples = [_make_example("ex-0")]
    manifest = pred.stage(examples)

    completed = subprocess.CompletedProcess(
        args=["sbatch"],
        returncode=0,
        stdout="Submitted batch job 9876543\n",
        stderr="",
    )
    with patch("subprocess.run", return_value=completed) as run_mock:
        job_id = pred.submit(manifest, chunk_size=2, max_concurrent=1)
    assert job_id == "9876543"
    args, _kwargs = run_mock.call_args
    assert args[0][0] == "sbatch"


def test_submit_raises_if_script_missing(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)  # slurm_script does NOT exist
    manifest = pred.stage([_make_example("ex-0")])
    with pytest.raises(FileNotFoundError):
        pred.submit(manifest)


def test_parse_sbatch_job_id_with_cluster_suffix() -> None:
    assert _parse_sbatch_job_id("Submitted batch job 12345 on cluster betty\n") == "12345"


def test_parse_sbatch_job_id_raises_on_bad_output() -> None:
    with pytest.raises(RuntimeError):
        _parse_sbatch_job_id("nope")


def test_score_pilot_manifest_parent_id_handles_wrong_targets() -> None:
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "score_pilot_manifest.py"
    spec = importlib.util.spec_from_file_location("score_pilot_manifest", script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._parent_id_for_manifest_id("sabdab-1abc-H-A-scramble01") == "sabdab-1abc-H-A"
    assert (
        module._parent_id_for_manifest_id("sabdab-1abc-H-A-wrongtarget02")
        == "sabdab-1abc-H-A"
    )
    assert (
        module._label_for_manifest_id(
            "sabdab-1abc-H-A-wrongtarget02", "sabdab-1abc-H-A", {"label": "POS"}
        )
        == "WRONG_TARGET"
    )


def test_af2m_from_env_picks_up_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRAGE_AF2M_OUTPUT_ROOT", str(tmp_path / "out_env"))
    monkeypatch.setenv("MIRAGE_AF2M_STAGED_ROOT", str(tmp_path / "stage_env"))
    pred = af2m_from_env()
    assert pred.output_root == tmp_path / "out_env"
    assert pred.staged_root == tmp_path / "stage_env"


def test_run_af2m_chunk_postprocess(tmp_path: Path) -> None:
    """Drive the runner's post-processing on a synthetic ColabFold output dir."""
    # Lazy import so the runner script's path resolution stays local to the test.
    import importlib.util

    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "run_af2m_chunk.py"
    spec = importlib.util.spec_from_file_location("run_af2m_chunk", runner_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out_dir = tmp_path / "ex-0"
    out_dir.mkdir()
    real_pdb = out_dir / "ex-0_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
    real_pdb.write_text("ATOM ...\n")
    (out_dir / "ex-0_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json").write_text(
        json.dumps({"iptm": 0.71, "ptm": 0.55, "plddt": [80.0, 90.0, 70.0], "max_pae": 12.5})
    )
    (out_dir / "ex-0_scores_rank_002_alphafold2_multimer_v3_model_3_seed_000.json").write_text(
        json.dumps({"iptm": 0.65, "ptm": 0.52, "plddt": [70.0, 80.0], "max_pae": 13.0})
    )

    module._post_process(out_dir, "ex-0")

    symlink = out_dir / "rank1.pdb"
    assert symlink.is_symlink()
    assert symlink.resolve() == real_pdb.resolve()

    scores = json.loads((out_dir / "scores.json").read_text())
    assert scores["example_id"] == "ex-0"
    assert scores["model_type"] == "alphafold2_multimer_v3"
    assert scores["rank1"]["iptm"] == pytest.approx(0.71)
    assert scores["rank1"]["mean_plddt"] == pytest.approx(80.0)
    assert [r["rank"] for r in scores["per_rank"]] == [1, 2]
