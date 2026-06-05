"""Tests for scripts/precompute_protenix_msa.py (pure parts; the protenix-msa
subprocess + ColabFold network call are operational, not unit-tested)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PM = _load_module("precompute_protenix_msa")

SEQ_A = "QVQLVESGGALVQPGGSLRLSCAASGRTFSDYAMGWFRQAPGKEREFVAAISRSGGSTYYADSVKG"
SEQ_B = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSY"


def test_local_hash_matches_package() -> None:
    """The script's sequence_hash MUST equal the package's, or cache keys desync."""
    from mirage.pose_predictors.protenix import sequence_hash as pkg_hash

    for seq in (SEQ_A, SEQ_B, "M", "QVQ"):
        assert PM.sequence_hash(seq) == pkg_hash(seq)


def test_cache_path_for(tmp_path: Path) -> None:
    p = PM.cache_path_for(tmp_path, SEQ_A)
    assert p.parent == tmp_path
    assert p.name == f"{PM.sequence_hash(SEQ_A)}.a3m"


def test_read_unique_manifest(tmp_path: Path) -> None:
    m = tmp_path / "unique_seqs.txt"
    m.write_text(f"{SEQ_A}\t{PM.sequence_hash(SEQ_A)}\n{SEQ_B}\t{PM.sequence_hash(SEQ_B)}\n\n")
    assert PM.read_unique_manifest(m) == [SEQ_A, SEQ_B]


def test_unmapped_skips_cached(tmp_path: Path) -> None:
    PM.cache_path_for(tmp_path, SEQ_A).write_text(">query\n" + SEQ_A + "\n")  # SEQ_A already cached
    assert PM.unmapped_sequences([SEQ_A, SEQ_B], tmp_path) == [SEQ_B]


def test_shard_partitions_cover_all_disjoint() -> None:
    seqs = [f"S{i}" for i in range(10)]
    shards = [PM.shard(seqs, i, 3) for i in range(3)]
    # disjoint
    flat = [s for sh in shards for s in sh]
    assert sorted(flat) == sorted(seqs)
    assert len(flat) == len(set(flat))
    # round-robin sizes (10 over 3 -> 4,3,3)
    assert sorted(len(sh) for sh in shards) == [3, 3, 4]


def test_write_fasta(tmp_path: Path) -> None:
    fasta = tmp_path / "x.fasta"
    PM.write_fasta([SEQ_A, SEQ_B], fasta)
    text = fasta.read_text()
    assert text == f">seq0\n{SEQ_A}\n>seq1\n{SEQ_B}\n"


def test_harvest_maps_by_query_not_index(tmp_path: Path) -> None:
    """Protenix reorders sequences, so harvest must key on each a3m's query line,
    NOT the output directory index. Here index 0 holds SEQ_B and index 1 holds
    SEQ_A (reversed) — harvest must still cache each under its own hash."""
    out = tmp_path / "out"
    (out / "0" / "0").mkdir(parents=True)
    (out / "1" / "1").mkdir(parents=True)
    # index 0 -> SEQ_B, index 1 -> SEQ_A (deliberately reversed)
    a3m_b = out / "0" / "0" / "non_pairing.a3m"
    a3m_a = out / "1" / "1" / "non_pairing.a3m"
    a3m_b.write_text(f">query\n{SEQ_B}\n>hitB\n{SEQ_B}\n")
    a3m_a.write_text(f">query\n{SEQ_A}\n>hitA\n{SEQ_A}\n")
    cache = tmp_path / "cache"

    n = PM.harvest(out, cache)
    assert n == 2
    link_a = PM.cache_path_for(cache, SEQ_A)
    link_b = PM.cache_path_for(cache, SEQ_B)
    assert link_a.is_symlink() and link_a.resolve() == a3m_a.resolve()
    assert link_b.is_symlink() and link_b.resolve() == a3m_b.resolve()
    # The cached a3m's query is the right sequence
    assert PM.a3m_query_sequence(link_a) == SEQ_A
    assert PM.a3m_query_sequence(link_b) == SEQ_B


def test_harvest_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "0" / "0").mkdir(parents=True)
    (out / "0" / "0" / "non_pairing.a3m").write_text(f">query\n{SEQ_A}\n")
    cache = tmp_path / "cache"
    assert PM.harvest(out, cache) == 1
    assert PM.harvest(out, cache) == 1  # re-run: refreshes the symlink, no error
    assert (
        PM.cache_path_for(cache, SEQ_A).resolve() == (out / "0" / "0" / "non_pairing.a3m").resolve()
    )


def test_main_dry_run_writes_fasta(tmp_path: Path) -> None:
    manifest = tmp_path / "unique_seqs.txt"
    manifest.write_text(f"{SEQ_A}\t{PM.sequence_hash(SEQ_A)}\n{SEQ_B}\t{PM.sequence_hash(SEQ_B)}\n")
    rc = PM.main(
        [
            "--unique-manifest",
            str(manifest),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--work-dir",
            str(tmp_path / "work"),
            "--n-shards",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (tmp_path / "work" / "shard_0.fasta").exists()
