"""Sequence-identity clustering for held-out-antigen splits.

Greedy k-mer-Jaccard clustering (pure Python, deterministic) with an optional
MMseqs2 fast path when the ``mmseqs`` binary is available. Used to group antigen
sequences so whole clusters can be held out together — no spike/HA variant
straddles a fold."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

_KMER_K = 5


def kmer_set(seq: str, k: int = _KMER_K) -> frozenset[str]:
    if len(seq) < k:
        return frozenset()
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def identity_to_jaccard_threshold(identity: float, k: int = _KMER_K) -> float:
    """Convert a target sequence identity into a k-mer Jaccard threshold.

    For sequences differing in fraction ``f = 1 - identity`` of positions, each
    mutation perturbs up to ``k`` k-mers, so intersection/union ≈ (1-kf)/(1+kf).
    A heuristic proxy, not a true identity calculation."""
    f = max(0.0, 1.0 - identity)
    raw = (1.0 - k * f) / (1.0 + k * f)
    return max(0.0, raw)


def greedy_cluster(seqs: list[str], *, max_identity: float = 0.9) -> list[int]:
    """Assign each sequence a cluster id by greedy single-linkage to cluster
    representatives on k-mer Jaccard. Deterministic in input order: a sequence
    joins the first cluster whose representative exceeds the threshold, else
    starts a new cluster. Returns cluster ids aligned to ``seqs``."""
    threshold = identity_to_jaccard_threshold(max_identity)
    rep_kmers: list[frozenset[str]] = []
    cluster_ids: list[int] = []
    for seq in seqs:
        ks = kmer_set(seq)
        assigned = -1
        for cid, rep in enumerate(rep_kmers):
            if jaccard(ks, rep) >= threshold:
                assigned = cid
                break
        if assigned < 0:
            assigned = len(rep_kmers)
            rep_kmers.append(ks)
        cluster_ids.append(assigned)
    return cluster_ids


def mmseqs_cluster(
    seqs: list[str], *, min_seq_id: float = 0.9, mmseqs_bin: str | None = None
) -> list[int] | None:
    """Cluster via ``mmseqs easy-cluster``. Returns cluster ids aligned to
    ``seqs``, or None if the mmseqs binary is unavailable."""
    binary = mmseqs_bin or shutil.which("mmseqs")
    if binary is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "in.fasta"
        with fasta.open("w") as fh:
            for i, seq in enumerate(seqs):
                fh.write(f">{i}\n{seq}\n")
        prefix = tmp_path / "clu"
        subprocess.run(
            [
                binary,
                "easy-cluster",
                str(fasta),
                str(prefix),
                str(tmp_path / "t"),
                "--min-seq-id",
                str(min_seq_id),
                "-c",
                "0.8",
                "--cov-mode",
                "0",
                "-v",
                "0",
            ],
            check=True,
            capture_output=True,
        )
        rep_to_cid: dict[str, int] = {}
        cid_of_member: dict[int, int] = {}
        with Path(f"{prefix}_cluster.tsv").open() as fh:
            for line in fh:
                rep, member = line.rstrip("\n").split("\t")
                if rep not in rep_to_cid:
                    rep_to_cid[rep] = len(rep_to_cid)
                cid_of_member[int(member)] = rep_to_cid[rep]
    return [cid_of_member[i] for i in range(len(seqs))]


def cluster_antigens(
    seqs: list[str], *, max_identity: float = 0.9, mmseqs_bin: str | None = None
) -> list[int]:
    """Cluster antigen sequences: MMseqs2 if available, else greedy k-mer."""
    via_mmseqs = mmseqs_cluster(seqs, min_seq_id=max_identity, mmseqs_bin=mmseqs_bin)
    if via_mmseqs is not None:
        return via_mmseqs
    return greedy_cluster(seqs, max_identity=max_identity)
