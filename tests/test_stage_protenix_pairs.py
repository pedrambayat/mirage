"""Tests for scripts/stage_protenix_pairs.py — pure helpers only."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "stage_protenix_pairs", REPO / "scripts" / "stage_protenix_pairs.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_protenix_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_pairs_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["pair_id", "binder_seq", "antigen_seq", "label", "antigen_cluster", "fold"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_TWO_ROW_DATA = [
    {
        "pair_id": "pair_A",
        "binder_seq": "QVQ",
        "antigen_seq": "EVAL",
        "label": "1",
        "antigen_cluster": "0",
        "fold": "0",
    },
    {
        "pair_id": "pair_B",
        "binder_seq": "QVQ",
        "antigen_seq": "WXYZ",
        "label": "0",
        "antigen_cluster": "1",
        "fold": "0",
    },
]


def test_examples_from_pairs_csv_correct_ids_and_chains(tmp_path):
    mod = _load_module()
    csv_path = tmp_path / "pairs.csv"
    _write_pairs_csv(csv_path, _TWO_ROW_DATA)

    examples = list(mod.examples_from_pairs_csv(csv_path))

    assert len(examples) == 2
    ex_a = next(e for e in examples if e.id == "pair_A")
    assert ex_a.binder_chains == ("QVQ",)
    assert ex_a.target_chains == ("EVAL",)
    assert ex_a.label == "1"
    assert ex_a.binder_format == "vhh"


def test_examples_from_pairs_csv_metadata(tmp_path):
    mod = _load_module()
    csv_path = tmp_path / "pairs.csv"
    _write_pairs_csv(csv_path, _TWO_ROW_DATA)

    examples = list(mod.examples_from_pairs_csv(csv_path))
    ex_a = next(e for e in examples if e.id == "pair_A")
    assert ex_a.metadata["antigen_cluster"] == 0
    assert ex_a.metadata["fold"] == 0


def test_examples_from_pairs_csv_second_row(tmp_path):
    mod = _load_module()
    csv_path = tmp_path / "pairs.csv"
    _write_pairs_csv(csv_path, _TWO_ROW_DATA)

    examples = list(mod.examples_from_pairs_csv(csv_path))
    ex_b = next(e for e in examples if e.id == "pair_B")
    assert ex_b.binder_chains == ("QVQ",)
    assert ex_b.target_chains == ("WXYZ",)
    assert ex_b.label == "0"
    assert ex_b.metadata["antigen_cluster"] == 1
    assert ex_b.metadata["fold"] == 0


def test_unique_sequences_dedup_shared_binder(tmp_path):
    mod = _load_module()
    csv_path = tmp_path / "pairs.csv"
    _write_pairs_csv(csv_path, _TWO_ROW_DATA)

    # binder "QVQ" appears in both rows — deduplicated; antigens "EVAL" and "WXYZ" distinct
    uniq = mod.unique_sequences(csv_path)
    assert uniq == {"QVQ", "EVAL", "WXYZ"}


def test_sequence_hash_is_deterministic():
    mod = _load_module()
    h = mod.sequence_hash("QVQLVESGG")
    assert h == mod.sequence_hash("QVQLVESGG")


def test_sequence_hash_is_16_hex_chars():
    mod = _load_module()
    h = mod.sequence_hash("EVQLLESGG")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_sequence_hash_differs_for_different_seqs():
    mod = _load_module()
    assert mod.sequence_hash("AAAA") != mod.sequence_hash("CCCC")


# ---------------------------------------------------------------------------
# Multi-chain antigen tests
# ---------------------------------------------------------------------------

_MULTI_CHAIN_DATA = [
    {
        "pair_id": "pair_MC",
        "binder_seq": "QVQ",
        "antigen_seq": "AAAA:CCCC",
        "label": "1",
        "antigen_cluster": "0",
        "fold": "0",
    },
]


def test_examples_from_pairs_csv_multi_chain_antigen_splits(tmp_path):
    """A colon-delimited antigen_seq must yield separate target_chains."""
    mod = _load_module()
    csv_path = tmp_path / "pairs.csv"
    _write_pairs_csv(csv_path, _MULTI_CHAIN_DATA)

    examples = list(mod.examples_from_pairs_csv(csv_path))
    assert len(examples) == 1
    ex = examples[0]
    assert ex.target_chains == ("AAAA", "CCCC"), (
        f"Expected ('AAAA', 'CCCC'), got {ex.target_chains}"
    )
    # Binder is unchanged
    assert ex.binder_chains == ("QVQ",)


def test_unique_sequences_multi_chain_antigen_split_as_separate(tmp_path):
    """unique_sequences must return each chain of a colon-delimited antigen separately."""
    mod = _load_module()
    csv_path = tmp_path / "pairs.csv"
    _write_pairs_csv(csv_path, _MULTI_CHAIN_DATA)

    uniq = mod.unique_sequences(csv_path)
    # The joined string "AAAA:CCCC" must NOT appear; each chain must appear
    assert "AAAA:CCCC" not in uniq, "Joined antigen string must not be in unique_sequences"
    assert "AAAA" in uniq
    assert "CCCC" in uniq
    assert "QVQ" in uniq
    assert uniq == {"QVQ", "AAAA", "CCCC"}
