# SAbDab Sequence-Only Binding Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the strongest sequence-only VHH–antigen binding discriminator trainable on SAbDab cognate pairs, under a leakage-controlled held-out-antigen-cluster split, as a rigorous floor for mirage's M-S gate.

**Architecture:** Stage SAbDab positives (existing `SAbDabLoader`) → normalize → cluster antigens → assign antigen-cluster folds → build distribution-matched cross-cluster shuffled negatives. Embed unique sequences with ESM-2 650M in a *separate* GPU env (cached `.npy`); mirage stays torch-free and reads the cache. Train a four-rung model ladder — additive Tier-S, additive ESM-concat, diagonal bilinear (Hadamard, reuses the existing logistic), and a numpy low-rank bilinear two-tower — and evaluate with the existing gate metrics. AVIDa-hIL6 is a held-out same-antigen frozen-gate transfer, never training.

**Tech Stack:** Python 3.11, numpy, uv (mirage env); ESM-2 650M via `fair-esm`+torch in a separate `esm` conda env on SLURM (`dgx-b200`). Reuses `ml/core.py`, `model/ms.py`, `eval/gate.py`, `eval/orthogonal.py`, `features/normalize.py`, `benchmark/sabdab.py`.

**Spec:** `docs/superpowers/specs/2026-06-01-sabdab-sequence-baseline-design.md`

**Conventions:** TDD (test lands with code). `from __future__ import annotations` at top of each module. mypy strict on `src/mirage/` (use `np.ndarray[Any, Any]`); `scripts/` and `tests/` are not mypy-checked. Commits authored by Pedram — **never** add a `Co-Authored-By: Claude` trailer. Work on branch `sabdab-sequence-baseline`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/mirage/features/clustering.py` | Sequence-identity clustering (greedy k-mer Jaccard + optional MMseqs2) for held-out-antigen splits |
| `src/mirage/features/embeddings.py` | Read the ESM embedding cache; build per-rung paired feature matrices |
| `src/mirage/ml/bilinear.py` | numpy low-rank bilinear logistic trainer + grouped OOF |
| `src/mirage/model/bilinear.py` | Frozen `BilinearModel` artifact (save/load/predict_logit) |
| `src/mirage/eval/orthogonal.py` | EDIT: add `features_for_examples_embedding` (Tier-S path untouched) |
| `scripts/stage_sabdab_pairs.py` | Stage positives + negatives + folds → `sabdab_pairs.csv` + unique-seq manifest |
| `scripts/embed_sequences.py` | ESM-2 650M mean-pool embeddings (separate env); standalone, torch lazy |
| `scripts/slurm/embed_esm.slurm` | SLURM wrapper for the embed step |
| `scripts/analyze_sabdab_baseline.py` | Train the 4 rungs, OOF table under antigen-cluster folds, freeze best |
| `scripts/analyze_sabdab_orthogonal.py` | Apply the frozen model to AVIDa as held-out transfer |
| `results/published/sabdab_baseline_summary.md` | Results writeup (numbers transcribed post-run) |
| `tests/test_clustering.py`, `tests/test_sabdab_negatives.py`, `tests/test_embeddings_cache.py`, `tests/test_bilinear.py`, `tests/test_bilinear_model.py`, `tests/test_embed_windows.py`, `tests/test_orthogonal_embedding.py`, `tests/test_analyze_sabdab_baseline.py` | Tests |

---

## Task 1: Antigen clustering

**Files:**
- Create: `src/mirage/features/clustering.py`
- Test: `tests/test_clustering.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clustering.py
from __future__ import annotations

from mirage.features.clustering import greedy_cluster


_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVA"


def test_identical_sequences_one_cluster() -> None:
    assert greedy_cluster([_VHH, _VHH, _VHH]) == [0, 0, 0]


def test_unrelated_sequences_separate_clusters() -> None:
    other = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTS"
    ids = greedy_cluster([_VHH, other])
    assert ids[0] != ids[1]


def test_near_identical_merge_at_90pct() -> None:
    mutated = _VHH[:-1] + "A"  # single substitution
    assert greedy_cluster([_VHH, mutated], max_identity=0.9) == [0, 0]


def test_deterministic_in_input_order() -> None:
    seqs = [_VHH, "CCCCCCCCCCCCCCCCCCCC", _VHH]
    assert greedy_cluster(seqs) == greedy_cluster(seqs)
    assert greedy_cluster(seqs) == [0, 1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_clustering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mirage.features.clustering'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mirage/features/clustering.py
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
                binary, "easy-cluster", str(fasta), str(prefix), str(tmp_path / "t"),
                "--min-seq-id", str(min_seq_id), "-c", "0.8", "--cov-mode", "0", "-v", "0",
            ],
            check=True, capture_output=True,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_clustering.py -v && uv run mypy src/mirage/features/clustering.py && uv run ruff check src/mirage/features/clustering.py`
Expected: tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/mirage/features/clustering.py tests/test_clustering.py
git commit -m "Add antigen sequence clustering for held-out-antigen splits"
```

---

## Task 2: SAbDab pair staging (positives + leakage-safe negatives)

**Files:**
- Create: `scripts/stage_sabdab_pairs.py`
- Test: `tests/test_sabdab_negatives.py`

The pure, testable core is `build_pairs`; `main` wires the loader, normalization, clustering, and fold assignment around it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sabdab_negatives.py
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "stage_sabdab_pairs", REPO / "scripts" / "stage_sabdab_pairs.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _uniform_positives(mod, n_clusters=4, per_cluster=25):
    pos = []
    for c in range(n_clusters):
        for j in range(per_cluster):
            ag = f"AG_{c}_{j}"  # distinct antigen per (cluster, j)
            pos.append(
                mod.Positive(
                    pair_id=f"p_{c}_{j}",
                    binder_seq=f"VHH_{c}_{j}",
                    antigen_seq=ag,
                    antigen_cluster=c,
                    fold=0,
                )
            )
    return pos


def test_negatives_are_cross_cluster_and_fold_consistent():
    mod = _load_module()
    pos = _uniform_positives(mod)
    rows = mod.build_pairs(pos, k=5, seed=7)
    by_id = {p.pair_id: p for p in pos}
    for r in rows:
        if r["label"] == "0":
            parent = by_id[r["pair_id"].split("__neg")[0]]
            assert int(r["antigen_cluster"]) != parent.antigen_cluster
            assert int(r["fold"]) == parent.fold


def test_one_positive_plus_k_negatives_each():
    mod = _load_module()
    pos = _uniform_positives(mod)
    rows = mod.build_pairs(pos, k=5, seed=7)
    assert sum(1 for r in rows if r["label"] == "1") == len(pos)
    assert sum(1 for r in rows if r["label"] == "0") == len(pos) * 5


def test_negative_cluster_marginal_matches_positive_marginal():
    # Uniform positive clusters -> negative cluster marginal ~uniform (each
    # cluster excluded equally often), confirming distribution-matching.
    mod = _load_module()
    pos = _uniform_positives(mod)
    rows = mod.build_pairs(pos, k=40, seed=11)
    neg_clusters = [int(r["antigen_cluster"]) for r in rows if r["label"] == "0"]
    counts = Counter(neg_clusters)
    total = sum(counts.values())
    for c in range(4):
        assert abs(counts[c] / total - 0.25) < 0.05


def test_build_pairs_is_deterministic():
    mod = _load_module()
    pos = _uniform_positives(mod)
    assert mod.build_pairs(pos, k=5, seed=3) == mod.build_pairs(pos, k=5, seed=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sabdab_negatives.py -v`
Expected: FAIL — `scripts/stage_sabdab_pairs.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/stage_sabdab_pairs.py
"""Stage SAbDab VHH-antigen pairs for the sequence-only baseline.

Loads cognate positives via the existing SAbDabLoader (VHH-only), normalizes
sequences to mature domains, clusters antigens by sequence identity, assigns
held-out-antigen-cluster folds, and constructs distribution-matched,
cross-cluster, fold-consistent shuffled negatives. Emits a flat pair CSV plus a
unique-sequence manifest for the embedding step.

Use::

    uv run python scripts/stage_sabdab_pairs.py \\
        --data-dir ../abdisc-data/sabdab \\
        --output data/staged/sabdab/sabdab_pairs.csv \\
        --manifest data/staged/sabdab/sabdab_unique_seqs.txt
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Positive:
    pair_id: str
    binder_seq: str
    antigen_seq: str
    antigen_cluster: int
    fold: int


def _antigens_by_cluster(positives: list[Positive], fold: int) -> dict[int, list[str]]:
    """Unique antigen sequences present in ``fold``, grouped by cluster."""
    out: dict[int, list[str]] = {}
    seen: set[tuple[int, str]] = set()
    for p in positives:
        if p.fold != fold:
            continue
        key = (p.antigen_cluster, p.antigen_seq)
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(p.antigen_cluster, []).append(p.antigen_seq)
    return out


def build_pairs(positives: list[Positive], *, k: int, seed: int) -> list[dict[str, str]]:
    """Emit positive rows + k distribution-matched shuffled negatives each.

    For a positive in fold ``f`` with antigen cluster ``c_i``, each negative
    antigen is drawn by (1) sampling a cluster from the fold's positive
    cluster-frequency distribution restricted to clusters != c_i (so the
    negative cluster marginal matches the positive marginal and antigen
    popularity carries no label signal), then (2) a uniform antigen within that
    cluster. Every negative is cross-cluster and fold-consistent by construction.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for fold in sorted({p.fold for p in positives}):
        fold_pos = [p for p in positives if p.fold == fold]
        ag_by_cluster = _antigens_by_cluster(positives, fold)
        cluster_counts = Counter(p.antigen_cluster for p in fold_pos)
        clusters = np.array(sorted(cluster_counts))
        weights = np.array([cluster_counts[int(c)] for c in clusters], dtype=float)
        for p in fold_pos:
            rows.append(
                {
                    "pair_id": p.pair_id,
                    "binder_seq": p.binder_seq,
                    "antigen_seq": p.antigen_seq,
                    "label": "1",
                    "antigen_cluster": str(p.antigen_cluster),
                    "fold": str(fold),
                }
            )
            keep = (clusters != p.antigen_cluster) & np.array(
                [int(c) in ag_by_cluster for c in clusters]
            )
            cand = clusters[keep]
            cand_w = weights[keep]
            if cand.size == 0:
                continue
            probs = cand_w / cand_w.sum()
            for j in range(k):
                cj = int(rng.choice(cand, p=probs))
                pool = ag_by_cluster[cj]
                ag = pool[int(rng.integers(len(pool)))]
                rows.append(
                    {
                        "pair_id": f"{p.pair_id}__neg{j}",
                        "binder_seq": p.binder_seq,
                        "antigen_seq": ag,
                        "label": "0",
                        "antigen_cluster": str(cj),
                        "fold": str(fold),
                    }
                )
    return rows


_FIELDNAMES = ["pair_id", "binder_seq", "antigen_seq", "label", "antigen_cluster", "fold"]


def _build_positives(data_dir: Path, *, n_splits: int, seed: int) -> list[Positive]:
    from mirage.benchmark.sabdab import SAbDabLoader
    from mirage.features.clustering import cluster_antigens
    from mirage.features.normalize import normalize_antigen, normalize_binder
    from mirage.ml.core import assign_folds

    loader = SAbDabLoader(data_dir=data_dir, use_anarci=True)
    raw: list[tuple[str, str, str]] = []  # (pair_id, binder_seq, antigen_seq)
    for ex in loader.load():
        if ex.binder_format != "vhh":
            continue
        binder = normalize_binder(ex.binder_chains[0])
        antigen = ":".join(normalize_antigen(c) for c in ex.target_chains)
        if not binder or not antigen.replace(":", ""):
            continue
        raw.append((ex.id, binder, antigen))

    unique_ag = sorted({a for _, _, a in raw})
    ag_cluster = dict(
        zip(
            unique_ag,
            cluster_antigens([a.replace(":", "") for a in unique_ag]),
            strict=True,
        )
    )
    clusters = np.array([ag_cluster[a] for _, _, a in raw])
    folds = assign_folds(clusters, n_splits=n_splits, seed=seed)
    return [
        Positive(pid, b, a, int(ag_cluster[a]), int(f))
        for (pid, b, a), f in zip(raw, folds, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    positives = _build_positives(args.data_dir, n_splits=args.n_splits, seed=args.seed)
    rows = build_pairs(positives, k=args.k, seed=args.seed)

    # Invariant checks before writing (cross-cluster + fold-consistent).
    by_id = {p.pair_id: p for p in positives}
    for r in rows:
        if r["label"] == "0":
            parent = by_id[r["pair_id"].split("__neg")[0]]
            assert int(r["antigen_cluster"]) != parent.antigen_cluster
            assert int(r["fold"]) == parent.fold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    uniques = sorted({r["binder_seq"] for r in rows} | {r["antigen_seq"] for r in rows})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(uniques) + "\n")

    n_pos = sum(1 for r in rows if r["label"] == "1")
    print(
        f"positives={n_pos} negatives={len(rows) - n_pos} "
        f"unique_clusters={len({p.antigen_cluster for p in positives})} "
        f"unique_seqs={len(uniques)}"
    )
    print(f"Wrote {len(rows)} rows to {args.output} and {len(uniques)} seqs to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sabdab_negatives.py -v && uv run ruff check scripts/stage_sabdab_pairs.py`
Expected: 4 tests PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/stage_sabdab_pairs.py tests/test_sabdab_negatives.py
git commit -m "Stage SAbDab VHH pairs with leakage-safe shuffled negatives"
```

- [ ] **Step 6: Run staging on PARCC (produces real artifacts)**

Run: `uv run python scripts/stage_sabdab_pairs.py --data-dir ../abdisc-data/sabdab --output data/staged/sabdab/sabdab_pairs.csv --manifest data/staged/sabdab/sabdab_unique_seqs.txt`
Expected: prints positive/negative counts; writes `data/staged/sabdab/sabdab_pairs.csv` and `sabdab_unique_seqs.txt`. (This runs the full ANARCI pass over SAbDab heavy chains — minutes, needs the local HMMER. `data/staged/` is gitignored.)

---

## Task 3: ESM-2 650M embedding script (separate env) + window helper test

**Files:**
- Create: `scripts/embed_sequences.py`
- Create: `scripts/slurm/embed_esm.slurm`
- Test: `tests/test_embed_windows.py`

The script is standalone (torch imported lazily) so its pure window helper is importable and testable in the mirage env without torch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed_windows.py
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "embed_sequences", REPO / "scripts" / "embed_sequences.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_windows_cover_sequence_without_overlap():
    mod = _load()
    assert mod.iter_windows(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_short_sequence_single_window():
    mod = _load()
    assert mod.iter_windows(3, 1022) == [(0, 3)]


def test_exact_multiple():
    mod = _load()
    assert mod.iter_windows(8, 4) == [(0, 4), (4, 8)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embed_windows.py -v`
Expected: FAIL — `scripts/embed_sequences.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/embed_sequences.py
"""Embed unique sequences with ESM-2 650M and cache mean-pooled vectors.

Runs in a SEPARATE env (torch + fair-esm), NOT the mirage uv env. Reads a
manifest (one normalized sequence per line; multi-chain antigens are
':'-joined), mean-pools the final-layer per-residue representations per chain
(chunking sequences longer than the model context into non-overlapping windows,
length-weighted), averages chains length-weighted, and writes ``embeddings.npy``
(N x 1280, float32) plus ``keys.txt`` (the manifest lines, in row order). The
reader keys on the raw sequence string, so no hashing is involved.

Use (inside the esm env)::

    python scripts/embed_sequences.py \\
        --manifest data/staged/sabdab/sabdab_unique_seqs.txt \\
        --out-embeddings data/staged/sabdab/embeddings.npy \\
        --out-keys data/staged/sabdab/keys.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_MAX_LEN = 1022  # ESM-2 context (1024 minus BOS/EOS)


def iter_windows(n: int, max_len: int = _MAX_LEN) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) windows covering [0, n)."""
    if n <= 0:
        return [(0, 0)]
    return [(s, min(s + max_len, n)) for s in range(0, n, max_len)]


def _embed_one(seq: str, model, alphabet, batch_converter, device) -> np.ndarray:  # noqa: ANN001
    import torch

    vecs: list[np.ndarray] = []
    lengths: list[int] = []
    for chain in seq.split(":"):
        if not chain:
            continue
        for start, end in iter_windows(len(chain)):
            window = chain[start:end]
            _, _, toks = batch_converter([("q", window)])
            toks = toks.to(device)
            with torch.no_grad():
                out = model(toks, repr_layers=[33], return_contacts=False)
            rep = out["representations"][33][0, 1 : len(window) + 1].mean(0)
            vecs.append(rep.float().cpu().numpy())
            lengths.append(len(window))
    weights = np.array(lengths, dtype=float)
    return np.average(np.stack(vecs), axis=0, weights=weights).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-embeddings", type=Path, required=True)
    parser.add_argument("--out-keys", type=Path, required=True)
    args = parser.parse_args()

    import esm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    seqs = [s for s in args.manifest.read_text().splitlines() if s]
    embeddings = np.zeros((len(seqs), 1280), dtype=np.float32)
    for i, seq in enumerate(seqs):
        embeddings[i] = _embed_one(seq, model, alphabet, batch_converter, device)
        if (i + 1) % 200 == 0:
            print(f"embedded {i + 1}/{len(seqs)}", flush=True)

    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_embeddings, embeddings)
    args.out_keys.write_text("\n".join(seqs) + "\n")
    print(f"Wrote {embeddings.shape} to {args.out_embeddings} and keys to {args.out_keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write the SLURM wrapper**

```bash
# scripts/slurm/embed_esm.slurm
#!/bin/bash
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=esm_embed
#SBATCH --output=slurm-%j.out
set -euo pipefail
# Run in the dedicated ESM env (torch + fair-esm). Create once:
#   conda create -n esm python=3.11 -y && conda activate esm
#   pip install torch fair-esm numpy
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esm
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
python scripts/embed_sequences.py \
  --manifest data/staged/sabdab/sabdab_unique_seqs.txt \
  --out-embeddings data/staged/sabdab/embeddings.npy \
  --out-keys data/staged/sabdab/keys.txt
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_embed_windows.py -v && uv run ruff check scripts/embed_sequences.py`
Expected: 3 tests PASS, ruff clean. (ruff `ANN001` is not in the selected ruleset; the `# noqa` is harmless.)

- [ ] **Step 6: Commit**

```bash
git add scripts/embed_sequences.py scripts/slurm/embed_esm.slurm tests/test_embed_windows.py
git commit -m "Add ESM-2 650M embedding script + SLURM wrapper (separate env)"
```

- [ ] **Step 7: Submit the embed job on PARCC**

Run: `sbatch scripts/slurm/embed_esm.slurm` then monitor with `squeue -u $USER` / `sacct -j <id>`.
Expected: produces `data/staged/sabdab/embeddings.npy` + `keys.txt` covering every manifest line. (Create the `esm` conda env once as noted in the wrapper if absent.)

---

## Task 4: Embedding cache reader + paired-matrix builder

**Files:**
- Create: `src/mirage/features/embeddings.py`
- Test: `tests/test_embeddings_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embeddings_cache.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mirage.features.embeddings import load_embedding_cache, paired_matrix


def _write_cache(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    keys = ["AAAA", "CCCC", "DDDD"]
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    npy = tmp_path / "emb.npy"
    kf = tmp_path / "keys.txt"
    np.save(npy, arr)
    kf.write_text("\n".join(keys) + "\n")
    return npy, kf, {k: arr[i] for i, k in enumerate(keys)}


def test_load_roundtrip(tmp_path: Path):
    npy, kf, expected = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    assert set(cache) == set(expected)
    assert np.allclose(cache["CCCC"], [3.0, 4.0])


def test_paired_matrix_concat(tmp_path: Path):
    npy, kf, _ = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    x = paired_matrix([("AAAA", "CCCC")], cache, layout="concat")
    assert x.shape == (1, 4)
    assert np.allclose(x[0], [1.0, 2.0, 3.0, 4.0])


def test_paired_matrix_hadamard(tmp_path: Path):
    npy, kf, _ = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    x = paired_matrix([("AAAA", "CCCC")], cache, layout="hadamard")
    assert np.allclose(x[0], [1.0 * 3.0, 2.0 * 4.0])


def test_missing_key_raises(tmp_path: Path):
    npy, kf, _ = _write_cache(tmp_path)
    cache = load_embedding_cache(npy, kf)
    with pytest.raises(KeyError):
        paired_matrix([("AAAA", "ZZZZ")], cache, layout="concat")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings_cache.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mirage/features/embeddings.py
"""Read the ESM embedding cache and build per-rung paired feature matrices.

numpy-only — mirage never imports torch. The cache is produced by the separate
``scripts/embed_sequences.py`` and keyed on the raw (normalized) sequence
string, so reader and writer agree without hashing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_embedding_cache(
    npy_path: Path, keys_path: Path
) -> dict[str, np.ndarray[Any, Any]]:
    """Map each cached sequence to its embedding row."""
    keys = [k for k in keys_path.read_text().splitlines() if k]
    arr = np.load(npy_path)
    if arr.shape[0] != len(keys):
        raise ValueError(
            f"cache mismatch: {arr.shape[0]} embeddings vs {len(keys)} keys"
        )
    return {k: arr[i] for i, k in enumerate(keys)}


def paired_matrix(
    pairs: Sequence[tuple[str, str]],
    cache: dict[str, np.ndarray[Any, Any]],
    *,
    layout: str,
) -> np.ndarray[Any, Any]:
    """Build a feature matrix for (binder_seq, antigen_seq) pairs.

    ``layout="concat"`` -> ``[e_ab | e_ag]`` (2d-wide; used by the additive
    logistic and split back into halves by the bilinear model).
    ``layout="hadamard"`` -> ``e_ab * e_ag`` (d-wide; diagonal bilinear).
    Raises KeyError if a sequence is absent from the cache (embed it first)."""
    rows: list[np.ndarray[Any, Any]] = []
    for binder, antigen in pairs:
        e_b = cache[binder]
        e_g = cache[antigen]
        if layout == "concat":
            rows.append(np.concatenate([e_b, e_g]))
        elif layout == "hadamard":
            rows.append(e_b * e_g)
        else:
            raise ValueError(f"unknown layout: {layout!r}")
    return np.asarray(rows, dtype=float)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings_cache.py -v && uv run mypy src/mirage/features/embeddings.py && uv run ruff check src/mirage/features/embeddings.py`
Expected: 5 tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/mirage/features/embeddings.py tests/test_embeddings_cache.py
git commit -m "Add ESM embedding cache reader + paired-matrix builder"
```

---

## Task 5: numpy low-rank bilinear trainer

**Files:**
- Create: `src/mirage/ml/bilinear.py`
- Test: `tests/test_bilinear.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bilinear.py
from __future__ import annotations

import numpy as np

from mirage.eval.gate import auroc
from mirage.ml.bilinear import bilinear_oof_scores, fit_bilinear, predict_bilinear
from mirage.ml.core import apply_standardizer, fit_logistic_regression, standardizer


def _planted_interaction(n=400, d=8, seed=0):
    rng = np.random.default_rng(seed)
    xa = rng.normal(size=(n, d))
    xg = rng.normal(size=(n, d))
    m = rng.normal(size=(d, d))
    score = np.sum((xa @ m) * xg, axis=1)
    y = (score > np.median(score)).astype(int)
    return xa, xg, y


def test_bilinear_recovers_planted_interaction():
    xa, xg, y = _planted_interaction()
    pa, pg, b = fit_bilinear(xa, xg, y, rank=8, l2=1e-3, lr=0.1, n_iter=3000, seed=1)
    logits = predict_bilinear(xa, xg, pa, pg, b)
    assert auroc(logits, y) > 0.8


def test_additive_logistic_is_chance_on_pure_interaction():
    xa, xg, y = _planted_interaction()
    x = np.concatenate([xa, xg], axis=1)
    mean, std = standardizer(x)
    xs = apply_standardizer(x, mean, std)
    ic, coef = fit_logistic_regression(xs, y.astype(float), l2=1.0)
    logits = ic + xs @ coef
    assert abs(auroc(logits, y) - 0.5) < 0.1


def test_fit_bilinear_is_deterministic():
    xa, xg, y = _planted_interaction()
    a = fit_bilinear(xa, xg, y, rank=4, l2=1.0, lr=0.05, n_iter=200, seed=2)
    b = fit_bilinear(xa, xg, y, rank=4, l2=1.0, lr=0.05, n_iter=200, seed=2)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and a[2] == b[2]


def test_oof_scores_are_finite_and_grouped():
    xa, xg, y = _planted_interaction()
    folds = np.array([i % 5 for i in range(y.size)])
    oof = bilinear_oof_scores(xa, xg, y, folds, rank=8, l2=1e-3, lr=0.1, n_iter=1000, seed=1)
    assert np.isfinite(oof).all()
    assert auroc(oof, y) > 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bilinear.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mirage/ml/bilinear.py
"""numpy low-rank bilinear logistic model: score = (P_a e_a) . (P_g e_g) + b.

The minimal model that can separate cognate from shuffled pairs, where an
additive model is provably ~chance. Trained by gradient descent on the logistic
loss with L2 on the projections. Pure numpy — mirage stays torch-free."""

from __future__ import annotations

from typing import Any

import numpy as np

from mirage.ml.core import apply_standardizer, standardizer


def _sigmoid(z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def predict_bilinear(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    proj_a: np.ndarray[Any, Any],
    proj_g: np.ndarray[Any, Any],
    intercept: float,
) -> np.ndarray[Any, Any]:
    ua = xa @ proj_a.T
    ug = xg @ proj_g.T
    out: np.ndarray[Any, Any] = np.sum(ua * ug, axis=1) + intercept
    return out


def fit_bilinear(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    *,
    rank: int,
    l2: float,
    lr: float,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    """Fit projections (P_a, P_g) and bias by full-batch gradient descent.

    Inputs are assumed already standardized by the caller. Returns
    (proj_a [rank x d_a], proj_g [rank x d_g], intercept)."""
    rng = np.random.default_rng(seed)
    n, da = xa.shape
    dg = xg.shape[1]
    proj_a = rng.normal(0.0, 1.0 / np.sqrt(da), size=(rank, da))
    proj_g = rng.normal(0.0, 1.0 / np.sqrt(dg), size=(rank, dg))
    intercept = 0.0
    yf = y.astype(float)
    for _ in range(n_iter):
        ua = xa @ proj_a.T  # n x rank
        ug = xg @ proj_g.T  # n x rank
        logits = np.sum(ua * ug, axis=1) + intercept
        resid = _sigmoid(logits) - yf  # n
        g_proj_a = (resid[:, None] * ug).T @ xa / n + l2 * proj_a
        g_proj_g = (resid[:, None] * ua).T @ xg / n + l2 * proj_g
        proj_a -= lr * g_proj_a
        proj_g -= lr * g_proj_g
        intercept -= lr * float(resid.mean())
    return proj_a, proj_g, float(intercept)


def bilinear_oof_scores(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    folds: np.ndarray[Any, Any],
    *,
    rank: int,
    l2: float,
    lr: float,
    n_iter: int,
    seed: int,
) -> np.ndarray[Any, Any]:
    """Out-of-fold bilinear logits, standardizing on each fold's train rows so
    held-out scores never see test statistics."""
    out: np.ndarray[Any, Any] = np.full(y.shape, np.nan, dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        if np.unique(y[train]).size < 2:
            continue
        ma, sa = standardizer(xa[train])
        mg, sg = standardizer(xg[train])
        pa, pg, b = fit_bilinear(
            apply_standardizer(xa[train], ma, sa),
            apply_standardizer(xg[train], mg, sg),
            y[train],
            rank=rank,
            l2=l2,
            lr=lr,
            n_iter=n_iter,
            seed=seed,
        )
        out[test] = predict_bilinear(
            apply_standardizer(xa[test], ma, sa),
            apply_standardizer(xg[test], mg, sg),
            pa,
            pg,
            b,
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bilinear.py -v && uv run mypy src/mirage/ml/bilinear.py && uv run ruff check src/mirage/ml/bilinear.py`
Expected: 4 tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/mirage/ml/bilinear.py tests/test_bilinear.py
git commit -m "Add numpy low-rank bilinear logistic trainer + grouped OOF"
```

---

## Task 6: Frozen BilinearModel artifact

**Files:**
- Create: `src/mirage/model/bilinear.py`
- Test: `tests/test_bilinear_model.py`

`predict_logit(x)` takes the concat layout `x = [e_a | e_g]` (raw, 2d-wide) so the existing `evaluate_frozen_gate(model, x, y)` works unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bilinear_model.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from mirage.ml.bilinear import fit_bilinear, predict_bilinear
from mirage.ml.core import apply_standardizer, standardizer
from mirage.model.bilinear import BilinearModel


def _fit_small():
    rng = np.random.default_rng(0)
    n, d = 200, 6
    xa = rng.normal(size=(n, d))
    xg = rng.normal(size=(n, d))
    m = rng.normal(size=(d, d))
    y = (np.sum((xa @ m) * xg, axis=1) > 0).astype(int)
    ma, sa = standardizer(xa)
    mg, sg = standardizer(xg)
    pa, pg, b = fit_bilinear(
        apply_standardizer(xa, ma, sa), apply_standardizer(xg, mg, sg),
        y, rank=d, l2=1e-3, lr=0.1, n_iter=500, seed=1,
    )
    model = BilinearModel(
        feature_dim=d, rank=d,
        mean_a=ma.tolist(), std_a=sa.tolist(),
        mean_g=mg.tolist(), std_g=sg.tolist(),
        proj_a=pa.tolist(), proj_g=pg.tolist(),
        intercept=b, threshold=0.0, target_precision=0.9,
    )
    return model, xa, xg, (ma, sa, mg, sg, pa, pg, b)


def test_predict_logit_matches_manual_pipeline():
    model, xa, xg, (ma, sa, mg, sg, pa, pg, b) = _fit_small()
    x = np.concatenate([xa, xg], axis=1)
    manual = predict_bilinear(
        apply_standardizer(xa, ma, sa), apply_standardizer(xg, mg, sg), pa, pg, b
    )
    assert np.allclose(model.predict_logit(x), manual)


def test_save_load_roundtrip(tmp_path: Path):
    model, xa, xg, _ = _fit_small()
    x = np.concatenate([xa, xg], axis=1)
    path = tmp_path / "bilinear.json"
    model.save(path)
    loaded = BilinearModel.load(path)
    assert loaded.feature_dim == model.feature_dim
    assert np.allclose(loaded.predict_logit(x), model.predict_logit(x))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bilinear_model.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mirage/model/bilinear.py
"""Frozen low-rank bilinear gate artifact.

Mirrors model/ms.py::MsModel: standardizer stats + projections + threshold,
serialized to JSON, applied unchanged to held-out sets. ``predict_logit`` takes
the concat layout x = [e_a | e_g] (raw), so the existing orthogonal harness
(eval/orthogonal.py::evaluate_frozen_gate) works without modification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mirage.ml.bilinear import predict_bilinear
from mirage.ml.core import apply_standardizer


@dataclass(frozen=True)
class BilinearModel:
    feature_dim: int
    rank: int
    mean_a: list[float]
    std_a: list[float]
    mean_g: list[float]
    std_g: list[float]
    proj_a: list[list[float]]
    proj_g: list[list[float]]
    intercept: float
    threshold: float
    target_precision: float

    def predict_logit(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        d = self.feature_dim
        xa = apply_standardizer(x[:, :d], np.asarray(self.mean_a), np.asarray(self.std_a))
        xg = apply_standardizer(x[:, d:], np.asarray(self.mean_g), np.asarray(self.std_g))
        return predict_bilinear(
            xa, xg, np.asarray(self.proj_a), np.asarray(self.proj_g), self.intercept
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> BilinearModel:
        data: dict[str, Any] = json.loads(path.read_text())
        return cls(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bilinear_model.py -v && uv run mypy src/mirage/model/bilinear.py && uv run ruff check src/mirage/model/bilinear.py`
Expected: 2 tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/mirage/model/bilinear.py tests/test_bilinear_model.py
git commit -m "Add frozen BilinearModel artifact"
```

---

## Task 7: Baseline analysis — train the four-rung ladder

**Files:**
- Create: `scripts/analyze_sabdab_baseline.py`
- Test: `tests/test_analyze_sabdab_baseline.py`

The pure rung-runners are tested on synthetic arrays; `main` wires data loading, featurization, and artifact writing around them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyze_sabdab_baseline.py
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_sabdab_baseline", REPO / "scripts" / "analyze_sabdab_baseline.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _interaction_data(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    xa = rng.normal(size=(n, d))
    xg = rng.normal(size=(n, d))
    m = rng.normal(size=(d, d))
    y = (np.sum((xa @ m) * xg, axis=1) > 0).astype(int)
    folds = np.array([i % 5 for i in range(n)])
    return xa, xg, y, folds


def test_linear_rung_returns_summary_and_model():
    mod = _load()
    xa, xg, y, folds = _interaction_data()
    x = np.concatenate([xa, xg], axis=1)
    summary, model = mod.run_linear_rung(
        x, y, folds, l2=1.0, target_precision=0.9, seed=1
    )
    assert "auroc" in summary and "metrics" in summary
    assert hasattr(model, "predict_logit")


def test_bilinear_rung_beats_linear_on_interaction():
    mod = _load()
    xa, xg, y, folds = _interaction_data()
    x = np.concatenate([xa, xg], axis=1)
    lin_summary, _ = mod.run_linear_rung(x, y, folds, l2=1.0, target_precision=0.9, seed=1)
    bil_summary, bil_model = mod.run_bilinear_rung(
        xa, xg, y, folds, rank=6, l2=1e-3, lr=0.1, n_iter=1500,
        target_precision=0.9, seed=1,
    )
    assert bil_summary["auroc"] > lin_summary["auroc"]
    assert bil_summary["auroc"] > 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyze_sabdab_baseline.py -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/analyze_sabdab_baseline.py
"""Train the four-rung sequence-only ladder on SAbDab and report gate metrics.

Rungs: 0 additive Tier-S, 1 additive ESM-concat, 2 diagonal bilinear (Hadamard,
reusing the existing logistic), 3 low-rank bilinear two-tower. OOF scores use the
pre-assigned antigen-cluster folds; the head-to-head AUROC / recall@precision /
PPV-sweep table is written to JSON and the best interaction rung is frozen.

Use::

    uv run python scripts/analyze_sabdab_baseline.py \\
        --pairs data/staged/sabdab/sabdab_pairs.csv \\
        --embeddings data/staged/sabdab/embeddings.npy \\
        --keys data/staged/sabdab/keys.txt \\
        --output results/published/sabdab_baseline.json \\
        --model-out results/published/sabdab_bilinear_model.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.gate import auroc, choose_threshold_for_precision, summary_dict
from mirage.features.embeddings import load_embedding_cache, paired_matrix
from mirage.features.sequence import FEATURE_NAMES, sequence_features
from mirage.ml.bilinear import bilinear_oof_scores, fit_bilinear, predict_bilinear
from mirage.ml.core import apply_standardizer, standardizer
from mirage.model.bilinear import BilinearModel
from mirage.model.ms import MsModel, train_ms


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def run_linear_rung(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    folds: np.ndarray[Any, Any],
    *,
    l2: float,
    target_precision: float,
    seed: int,
) -> tuple[dict[str, Any], MsModel]:
    """Fit an additive/diagonal logistic rung. Passing the pre-assigned fold
    column as ``groups`` (with n_splits = #folds) makes train_ms's grouped OOF
    reproduce exactly those antigen-cluster folds."""
    names = [f"f{i}" for i in range(x.shape[1])]
    n_splits = int(np.unique(folds).size)
    model, oof = train_ms(
        x, y, feature_names=names, l2=l2, target_precision=target_precision,
        seed=seed, groups=folds.astype(str), n_splits=n_splits,
    )
    finite = np.isfinite(oof)
    thr = choose_threshold_for_precision(
        oof[finite], y[finite], target_precision=target_precision
    )
    summary = summary_dict(oof[finite], y[finite], threshold=thr)
    summary["auroc"] = auroc(oof[finite], y[finite])
    return summary, model


def run_bilinear_rung(
    xa: np.ndarray[Any, Any],
    xg: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    folds: np.ndarray[Any, Any],
    *,
    rank: int,
    l2: float,
    lr: float,
    n_iter: int,
    target_precision: float,
    seed: int,
) -> tuple[dict[str, Any], BilinearModel]:
    oof = bilinear_oof_scores(
        xa, xg, y, folds, rank=rank, l2=l2, lr=lr, n_iter=n_iter, seed=seed
    )
    finite = np.isfinite(oof)
    thr_oof = choose_threshold_for_precision(
        oof[finite], y[finite], target_precision=target_precision
    )
    summary = summary_dict(oof[finite], y[finite], threshold=thr_oof)
    summary["auroc"] = auroc(oof[finite], y[finite])

    ma, sa = standardizer(xa)
    mg, sg = standardizer(xg)
    pa, pg, b = fit_bilinear(
        apply_standardizer(xa, ma, sa), apply_standardizer(xg, mg, sg),
        y, rank=rank, l2=l2, lr=lr, n_iter=n_iter, seed=seed,
    )
    full = predict_bilinear(
        apply_standardizer(xa, ma, sa), apply_standardizer(xg, mg, sg), pa, pg, b
    )
    thr_full = choose_threshold_for_precision(full, y, target_precision=target_precision)
    model = BilinearModel(
        feature_dim=int(xa.shape[1]), rank=rank,
        mean_a=ma.tolist(), std_a=sa.tolist(), mean_g=mg.tolist(), std_g=sg.tolist(),
        proj_a=pa.tolist(), proj_g=pg.tolist(),
        intercept=b, threshold=float(thr_full), target_precision=target_precision,
    )
    return summary, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--bilinear-l2", type=float, default=1e-3)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    rows = read_csv(args.pairs)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    folds = np.array([int(r["fold"]) for r in rows], dtype=int)
    binders = [r["binder_seq"] for r in rows]
    antigens = [r["antigen_seq"] for r in rows]
    pairs = list(zip(binders, antigens, strict=True))

    cache = load_embedding_cache(args.embeddings, args.keys)
    x_concat = paired_matrix(pairs, cache, layout="concat")
    x_hadamard = paired_matrix(pairs, cache, layout="hadamard")
    d = x_hadamard.shape[1]
    xa, xg = x_concat[:, :d], x_concat[:, d:]

    x_tiers = np.array(
        [
            [sequence_features(b, a)[name] for name in FEATURE_NAMES]
            for b, a in pairs
        ],
        dtype=float,
    )

    tp, seed = args.target_precision, args.seed
    results: dict[str, Any] = {
        "n": int(y.size), "n_positive": int((y == 1).sum()), "target_precision": tp,
    }
    results["rung0_tier_s"], _ = run_linear_rung(x_tiers, y, folds, l2=args.l2, target_precision=tp, seed=seed)
    results["rung1_esm_concat"], _ = run_linear_rung(x_concat, y, folds, l2=args.l2, target_precision=tp, seed=seed)
    results["rung2_hadamard"], diag_model = run_linear_rung(x_hadamard, y, folds, l2=args.l2, target_precision=tp, seed=seed)
    results["rung3_bilinear"], bil_model = run_bilinear_rung(
        xa, xg, y, folds, rank=args.rank, l2=args.bilinear_l2, lr=args.lr,
        n_iter=args.n_iter, target_precision=tp, seed=seed,
    )

    # Freeze the strongest interaction rung by OOF AUROC.
    if results["rung3_bilinear"]["auroc"] >= results["rung2_hadamard"]["auroc"]:
        results["frozen_rung"] = "rung3_bilinear"
        bil_model.save(args.model_out)
    else:
        results["frozen_rung"] = "rung2_hadamard"
        diag_model.save(args.model_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    for name in ("rung0_tier_s", "rung1_esm_concat", "rung2_hadamard", "rung3_bilinear"):
        print(f"{name}: AUROC={results[name]['auroc']:.3f} recall={results[name]['metrics']['recall']:.3f}")
    print(f"Froze {results['frozen_rung']} to {args.model_out}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_analyze_sabdab_baseline.py -v && uv run ruff check scripts/analyze_sabdab_baseline.py`
Expected: 2 tests PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_sabdab_baseline.py tests/test_analyze_sabdab_baseline.py
git commit -m "Add SAbDab four-rung baseline analysis (additive -> bilinear ladder)"
```

- [ ] **Step 6: Run the baseline on PARCC (after staging + embedding)**

Run: `uv run python scripts/analyze_sabdab_baseline.py --pairs data/staged/sabdab/sabdab_pairs.csv --embeddings data/staged/sabdab/embeddings.npy --keys data/staged/sabdab/keys.txt --output results/published/sabdab_baseline.json --model-out results/published/sabdab_bilinear_model.json`
Expected: rung 0/1 AUROC ≈ chance, rungs 2/3 strictly higher; writes the JSON + frozen artifact.

---

## Task 8: AVIDa held-out transfer (guardrail)

**Files:**
- Modify: `src/mirage/eval/orthogonal.py` (add `features_for_examples_embedding`; leave the Tier-S `features_for_examples` untouched)
- Create: `scripts/analyze_sabdab_orthogonal.py`
- Test: `tests/test_orthogonal_embedding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orthogonal_embedding.py
from __future__ import annotations

import numpy as np

from mirage.eval.orthogonal import features_for_examples_embedding
from mirage.scorers.base import BenchmarkExample


def test_features_for_examples_embedding_concat():
    # cache keyed on NORMALIZED sequences; toy seqs pass normalize unchanged.
    cache = {"AAAA": np.array([1.0, 2.0]), "CCCC": np.array([3.0, 4.0])}
    ex = BenchmarkExample(
        id="e1", label="POS", binder_chains=("AAAA",), binder_format="vhh",
        target_chains=("CCCC",), target_name="t", source="test",
    )
    x, y = features_for_examples_embedding([ex], cache, positive_label="POS", layout="concat")
    assert x.shape == (1, 4)
    assert np.allclose(x[0], [1.0, 2.0, 3.0, 4.0])
    assert y.tolist() == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orthogonal_embedding.py -v`
Expected: FAIL — `features_for_examples_embedding` does not exist.

- [ ] **Step 3: Add the embedding feature builder to `eval/orthogonal.py`**

Append this function (the existing `features_for_examples` and `evaluate_frozen_gate` are unchanged):

```python
def features_for_examples_embedding(
    examples: Iterable[BenchmarkExample],
    cache: dict[str, np.ndarray[Any, Any]],
    *,
    positive_label: str,
    layout: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build embedding paired features for a stream of examples, normalizing each
    sequence to its mature domain first (so lookups hit the cache, which is keyed
    on normalized sequences). Raises KeyError if a sequence was not embedded."""
    from mirage.features.embeddings import paired_matrix

    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    for ex in examples:
        binder = normalize_binder(ex.binder_chains[0])
        antigen = ":".join(normalize_antigen(c) for c in ex.target_chains)
        pairs.append((binder, antigen))
        labels.append(1 if ex.label == positive_label else 0)
    if not pairs:
        raise ValueError("features_for_examples_embedding received no examples")
    x = paired_matrix(pairs, cache, layout=layout)
    y = np.array(labels, dtype=int)
    return x, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orthogonal_embedding.py -v && uv run mypy src/mirage/eval/orthogonal.py && uv run ruff check src/mirage/eval/orthogonal.py`
Expected: PASS, mypy clean, ruff clean.

- [ ] **Step 5: Write the AVIDa transfer script**

```python
# scripts/analyze_sabdab_orthogonal.py
"""Apply the frozen SAbDab gate to AVIDa-hIL6 as a held-out same-antigen transfer.

AVIDa is NEVER training data — this is the orthogonal real-negative canary. The
frozen model (BilinearModel for rung 3, MsModel for rung 2) is applied unchanged
through the existing evaluate_frozen_gate harness. AVIDa's unique normalized
sequences must already be present in the embedding cache (embed them first).

Use::

    uv run python scripts/analyze_sabdab_orthogonal.py \\
        --avida-csv data/staged/avida/avida_staged.csv \\
        --embeddings data/staged/sabdab/embeddings.npy \\
        --keys data/staged/sabdab/keys.txt \\
        --model results/published/sabdab_bilinear_model.json --model-type bilinear \\
        --layout concat --output results/published/sabdab_orthogonal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mirage.benchmark._registry import get_loader  # AvidaLoader self-registers as "avida"
from mirage.eval.orthogonal import evaluate_frozen_gate, features_for_examples_embedding
from mirage.features.embeddings import load_embedding_cache
from mirage.model.bilinear import BilinearModel
from mirage.model.ms import MsModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avida-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-type", choices=["bilinear", "ms"], required=True)
    parser.add_argument("--layout", choices=["concat", "hadamard"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-label", default="BIND")
    args = parser.parse_args()

    model = (
        BilinearModel.load(args.model)
        if args.model_type == "bilinear"
        else MsModel.load(args.model)
    )
    cache = load_embedding_cache(args.embeddings, args.keys)
    examples = list(get_loader("avida", staged_csv=args.avida_csv).load())
    x, y = features_for_examples_embedding(
        examples, cache, positive_label=args.positive_label, layout=args.layout
    )
    result = evaluate_frozen_gate(model, x, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["metrics"], indent=2, default=str))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Verified:** AVIDa is streamed via the registered `AvidaLoader` (`get_loader("avida", staged_csv=...).load()`), positive label `"BIND"` — matching `scripts/analyze_ms_orthogonal.py`. `AvidaLoader.__init__` takes `staged_csv`.

- [ ] **Step 6: Run ruff on the new script**

Run: `uv run ruff check scripts/analyze_sabdab_orthogonal.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/mirage/eval/orthogonal.py scripts/analyze_sabdab_orthogonal.py tests/test_orthogonal_embedding.py
git commit -m "Add embedding-based AVIDa transfer for the frozen SAbDab gate"
```

- [ ] **Step 8: Embed AVIDa uniques + run the transfer (PARCC)**

Add AVIDa's unique normalized sequences to a manifest and embed them (extend the manifest used in Task 3 or run a second embed job appending to the cache), then:
Run: `uv run python scripts/analyze_sabdab_orthogonal.py --avida-csv data/staged/avida/avida_staged.csv --embeddings data/staged/sabdab/embeddings.npy --keys data/staged/sabdab/keys.txt --model results/published/sabdab_bilinear_model.json --model-type bilinear --layout concat --output results/published/sabdab_orthogonal.json`
Expected: writes the AVIDa transfer metrics JSON.

---

## Task 9: Results writeup + doc updates

**Files:**
- Create: `results/published/sabdab_baseline_summary.md`
- Modify: `CLAUDE.md` (Datasets section: SAbDab loader now built + baseline)

- [ ] **Step 1: Transcribe the real numbers into the summary**

After Tasks 7–8 have produced `results/published/sabdab_baseline.json` and `sabdab_orthogonal.json`, write `results/published/sabdab_baseline_summary.md` mirroring `mirage_phase_a_summary.md`. Required sections, each filled from the JSON (no fabricated numbers):
  - **Status / caveat** — M-S is the pre-structure baseline; does not consume predictor confidence.
  - **In-distribution head-to-head table** — one row per rung (0 Tier-S, 1 ESM-concat, 2 Hadamard, 3 bilinear): AUROC, recall@P=0.9, specificity, precision, threshold; plus the PPV-prevalence sweep for the frozen rung.
  - **Read** — does the interaction term (rungs 2/3) clear the additive floor (rungs 0/1)? Did held-out-antigen generalization hold or collapse? State plainly.
  - **AVIDa transfer row** — frozen-gate metrics on AVIDa with the held-out-same-antigen caveat.
  - **How to reproduce** — the `stage_sabdab_pairs.py` → `embed_esm.slurm` → `analyze_sabdab_baseline.py` → `analyze_sabdab_orthogonal.py` command chain.

- [ ] **Step 2: Update `CLAUDE.md` Datasets section**

In the **SAbDab** bullet, change "Loader **not yet built**." to note the loader is built and the sequence-only baseline lives at `results/published/sabdab_baseline_summary.md`.

- [ ] **Step 3: Full battery + commit**

```bash
uv run ruff check && uv run ruff format --check && uv run mypy src/mirage && uv run pytest
git add results/published/sabdab_baseline_summary.md results/published/sabdab_baseline.json results/published/sabdab_orthogonal.json results/published/sabdab_bilinear_model.json CLAUDE.md
git commit -m "Publish SAbDab sequence-only baseline results"
```

Expected: full battery green; results committed. (If a results JSON path differs from the frozen rung chosen, commit the actual artifact produced.)

---

## Verification (end-to-end)

1. `uv run pytest` — full battery green; ANARCI/HMMER and mmseqs tests skip gracefully where those tools are absent.
2. Staging produces `sabdab_pairs.csv` with the cross-cluster + fold-consistent invariants asserted in-script.
3. The SLURM embed job produces `embeddings.npy` + `keys.txt` covering every manifest line (`load_embedding_cache` raises on a count mismatch).
4. `analyze_sabdab_baseline.py`: rung 0 ≈ chance, rung 1 ≈ chance, rungs 2/3 strictly above — the interaction term earns its keep — and the frozen artifact is written.
5. `analyze_sabdab_orthogonal.py` produces the AVIDa transfer row via the frozen model through the existing harness.
6. Re-running staging → embed → analyze reproduces the committed `sabdab_baseline_summary.md` numbers from `abdisc-data/sabdab/` + the cache.

## Self-Review notes (resolved)

- **Spec coverage:** every spec section maps to a task — clustering (T1), staging+negatives (T2), embeddings+SLURM (T3), cache reader (T4), bilinear trainer (T5), frozen artifact (T6), four-rung analysis (T7), AVIDa transfer (T8), writeup+docs (T9).
- **Type consistency:** `predict_bilinear(xa, xg, proj_a, proj_g, intercept)`, `fit_bilinear(...) -> (proj_a, proj_g, intercept)`, and `BilinearModel.predict_logit(x=[e_a|e_g])` use consistent names across T5/T6/T7. `paired_matrix(pairs, cache, layout=...)` and `load_embedding_cache(npy, keys)` are consistent across T4/T7/T8. `run_linear_rung`/`run_bilinear_rung` signatures match between T7 implementation and test.
- **Placeholders:** none — all steps carry complete code. The AVIDa entry point was verified against `benchmark/avida.py` + `analyze_ms_orthogonal.py`: `get_loader("avida", staged_csv=...).load()`, positive label `"BIND"`.
