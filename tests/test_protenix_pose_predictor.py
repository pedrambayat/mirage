"""Tests for ProtenixPosePredictor and sequence_hash.

No GPU required — everything is pure-Python staging / command construction.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mirage.pose_predictors.protenix import (
    ProtenixPosePredictor,
    protenix_from_env,
    sequence_hash,
)
from mirage.scorers.base import BenchmarkExample

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(
    example_id: str = "test-1",
    binder: str = "QVQLVESGG",
    antigen: str = "ATCDEFGHIKLM",
) -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        label="POS",
        binder_chains=(binder,),
        binder_format="vhh",
        target_chains=(antigen,),
        target_name="Test antigen",
        source="test",
    )


def _make_predictor(tmp_path: Path) -> ProtenixPosePredictor:
    return ProtenixPosePredictor(
        output_root=tmp_path / "out",
        staged_root=tmp_path / "stage",
        msa_cache_dir=tmp_path / "msa_cache",
        slurm_script=tmp_path / "predict_protenix.slurm",
    )


def _load_stage_protenix_pairs():
    """Load scripts/stage_protenix_pairs.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "stage_protenix_pairs", REPO / "scripts" / "stage_protenix_pairs.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_protenix_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# sequence_hash
# ---------------------------------------------------------------------------


def test_sequence_hash_is_16_hex_chars() -> None:
    h = sequence_hash("QVQLVESGG")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_sequence_hash_is_deterministic() -> None:
    assert sequence_hash("QVQLVESGG") == sequence_hash("QVQLVESGG")


def test_sequence_hash_differs_for_different_seqs() -> None:
    assert sequence_hash("AAAA") != sequence_hash("CCCC")


def test_sequence_hash_matches_stage_protenix_pairs_re_export() -> None:
    """sequence_hash importable from stage_protenix_pairs re-export, returns identical values."""
    mod = _load_stage_protenix_pairs()
    seq = "EVQLLESGG"
    assert mod.sequence_hash(seq) == sequence_hash(seq)


# ---------------------------------------------------------------------------
# results_for
# ---------------------------------------------------------------------------


def test_results_for_returns_none_when_missing(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    assert pred.results_for(_make_example()) is None


def test_results_for_returns_path_when_rank1_cif_exists(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    example = _make_example()
    rank1 = pred.output_root / example.id / "rank1.cif"
    rank1.parent.mkdir(parents=True)
    rank1.write_text("data_\n")
    assert pred.results_for(example) == rank1


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


def test_stage_writes_manifest_header_and_one_json_per_example(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [
        _make_example(f"ex-{i}", binder=f"QVQ{'A' * i}", antigen=f"EVA{'L' * i}") for i in range(3)
    ]
    manifest = pred.stage(examples)

    assert manifest.n_rows == 3
    assert manifest.n_already_cached == 0
    assert manifest.path.is_file()

    rows = manifest.path.read_text().strip().splitlines()
    assert rows[0] == "example_id\tinput_path\tout_dir"
    assert len(rows) == 4  # header + 3 data rows

    # Verify each per-example JSON exists
    for example in examples:
        json_path = pred.staged_root / "inputs" / f"{example.id}.json"
        assert json_path.is_file(), f"missing {json_path}"


def test_stage_json_has_two_protein_chains_binder_first(tmp_path: Path) -> None:
    binder_seq = "QVQLVESGG"
    antigen_seq = "ATCDEFGHIKLM"
    pred = _make_predictor(tmp_path)
    example = _make_example("ex-0", binder=binder_seq, antigen=antigen_seq)
    pred.stage([example])

    json_path = pred.staged_root / "inputs" / "ex-0.json"
    payload = json.loads(json_path.read_text())

    assert payload["name"] == "ex-0"
    seqs = payload["sequences"]
    assert len(seqs) == 2

    binder_chain = seqs[0]["proteinChain"]
    antigen_chain = seqs[1]["proteinChain"]

    # Binder is first
    assert binder_chain["sequence"] == binder_seq
    assert antigen_chain["sequence"] == antigen_seq


def test_stage_json_msa_paths_are_absolute_with_correct_hash(tmp_path: Path) -> None:
    binder_seq = "QVQLVESGG"
    antigen_seq = "ATCDEFGHIKLM"
    pred = _make_predictor(tmp_path)
    example = _make_example("ex-0", binder=binder_seq, antigen=antigen_seq)
    pred.stage([example])

    json_path = pred.staged_root / "inputs" / "ex-0.json"
    payload = json.loads(json_path.read_text())
    seqs = payload["sequences"]

    binder_msa = seqs[0]["proteinChain"]["unpairedMsaPath"]
    antigen_msa = seqs[1]["proteinChain"]["unpairedMsaPath"]

    # Must be absolute paths
    assert Path(binder_msa).is_absolute()
    assert Path(antigen_msa).is_absolute()

    # Must end with <hash>.a3m
    assert binder_msa.endswith(f"/{sequence_hash(binder_seq)}.a3m")
    assert antigen_msa.endswith(f"/{sequence_hash(antigen_seq)}.a3m")


def test_stage_json_has_no_templates_key(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    example = _make_example("ex-0")
    pred.stage([example])

    json_path = pred.staged_root / "inputs" / "ex-0.json"
    payload = json.loads(json_path.read_text())
    seqs = payload["sequences"]

    for chain_entry in seqs:
        chain = chain_entry["proteinChain"]
        assert "templatesPath" not in chain
        assert "templates" not in chain


def test_stage_skips_already_cached(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    cached = _make_example("cached")
    fresh = _make_example("fresh")

    # Pre-create the rank1.cif for cached
    rank1 = pred.output_root / cached.id / "rank1.cif"
    rank1.parent.mkdir(parents=True)
    rank1.write_text("data_\n")

    manifest = pred.stage([cached, fresh])
    assert manifest.n_rows == 1
    assert manifest.n_already_cached == 1

    body = manifest.path.read_text()
    assert "fresh" in body
    assert "cached" not in body


def test_stage_all_cached_no_manifest_file(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    cached = _make_example("cached")
    rank1 = pred.output_root / cached.id / "rank1.cif"
    rank1.parent.mkdir(parents=True)
    rank1.write_text("data_\n")

    manifest = pred.stage([cached])
    assert manifest.n_rows == 0
    assert manifest.n_already_cached == 1
    assert not manifest.path.exists()


def test_stage_manifest_row_columns(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    example = _make_example("ex-0")
    manifest = pred.stage([example])

    with manifest.path.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert len(rows) == 1
    row = rows[0]
    assert row["example_id"] == "ex-0"
    assert Path(row["input_path"]).name == "ex-0.json"
    assert row["out_dir"] == str(pred.output_root / "ex-0")


# ---------------------------------------------------------------------------
# sbatch_command
# ---------------------------------------------------------------------------


def test_sbatch_command_contains_required_flags(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [_make_example(f"ex-{i}") for i in range(3)]
    manifest = pred.stage(examples)

    cmd = pred.sbatch_command(manifest, chunk_size=2, max_concurrent=4)

    assert cmd[0] == "sbatch"
    assert "--account=dbgoodma-goodman-laboratory" in cmd
    assert "--partition=dgx-b200" in cmd
    assert "--qos=dgx" in cmd
    assert "--gres=gpu:1" in cmd
    # 3 examples / chunk_size 2 → 2 tasks → array 0-1%4
    assert "--array=0-1%4" in cmd


def test_sbatch_command_runner_path_ends_with_run_protenix_chunk(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [_make_example("ex-0")]
    manifest = pred.stage(examples)
    cmd = pred.sbatch_command(manifest, chunk_size=2, max_concurrent=4)
    # Last positional arg is runner path
    assert cmd[-1].endswith("run_protenix_chunk.py")


def test_sbatch_command_manifest_chunk_runner_order(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [_make_example("ex-0")]
    manifest = pred.stage(examples)
    cmd = pred.sbatch_command(manifest, chunk_size=5, max_concurrent=8)
    # Positional args after slurm_script: manifest chunk_size runner
    script_idx = cmd.index(str(pred.slurm_script))
    assert cmd[script_idx + 1] == str(manifest.path)
    assert cmd[script_idx + 2] == "5"
    assert cmd[script_idx + 3].endswith("run_protenix_chunk.py")


def test_sbatch_command_array_rounds_up(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    examples = [_make_example(f"ex-{i}") for i in range(7)]
    manifest = pred.stage(examples)
    cmd = pred.sbatch_command(manifest, chunk_size=3, max_concurrent=2)
    # 7 / 3 → 3 tasks (0,1,2) → 0-2%2
    assert "--array=0-2%2" in cmd


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def test_submit_dry_run_does_not_subprocess(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    pred.slurm_script.write_text("#!/bin/bash\n")
    manifest = pred.stage([_make_example("ex-0")])

    with patch("subprocess.run") as run_mock:
        result = pred.submit(manifest, chunk_size=5, max_concurrent=4, dry_run=True)
    assert result is None
    run_mock.assert_not_called()


def test_submit_no_op_on_empty_manifest(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    empty = pred.stage([])
    with patch("subprocess.run") as run_mock:
        assert pred.submit(empty) is None
    run_mock.assert_not_called()


def test_submit_parses_job_id(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)
    pred.slurm_script.write_text("#!/bin/bash\n")
    manifest = pred.stage([_make_example("ex-0")])

    completed = subprocess.CompletedProcess(
        args=["sbatch"],
        returncode=0,
        stdout="Submitted batch job 1234567\n",
        stderr="",
    )
    with patch("subprocess.run", return_value=completed):
        job_id = pred.submit(manifest, chunk_size=2, max_concurrent=1)
    assert job_id == "1234567"


def test_submit_raises_if_script_missing(tmp_path: Path) -> None:
    pred = _make_predictor(tmp_path)  # slurm_script does NOT exist
    manifest = pred.stage([_make_example("ex-0")])
    with pytest.raises(FileNotFoundError):
        pred.submit(manifest)


# ---------------------------------------------------------------------------
# protenix_from_env
# ---------------------------------------------------------------------------


def test_protenix_from_env_picks_up_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIRAGE_PROTENIX_OUTPUT_ROOT", str(tmp_path / "out_env"))
    monkeypatch.setenv("MIRAGE_PROTENIX_STAGED_ROOT", str(tmp_path / "stage_env"))
    pred = protenix_from_env()
    assert pred.output_root == tmp_path / "out_env"
    assert pred.staged_root == tmp_path / "stage_env"
