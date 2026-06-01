"""SAbDab loader — crystal-validated antibody-antigen complexes.

Reads an OPIG SAbDab summary TSV plus the chothia-renumbered PDB structures
fetched by `sabdab_downloader.py`, and yields one `BenchmarkExample` per
accepted (binder, target) pair. The resulting examples carry the crystal
PDB path (via both the canonical `complex_pdb_path` field and a
`metadata["crystal_pdb_path"]` mirror), which downstream RMSD-to-crystal
scorers will use as ground truth.

Expected `data_dir` layout::

    summary.tsv                                         # OPIG export
    sabdab_dataset/<pdb>/structure/chothia/<pdb>.pdb    # chothia-renumbered

Filters (design spec §5.1):

* resolution <= ``max_resolution`` (default 3.0 Å)
* antigen chain length sum >= ``min_antigen_length`` (default 30 residues)
* binder heavy chain is ANARCI-resolvable (chothia scheme)
* binder dedupe at k-mer Jaccard >= identity-derived threshold
  (heuristic Python-only proxy for >``max_identity`` sequence identity)
* permissive binder format: VHH, scFv, and Fab all admitted

Set ``use_anarci=False`` to skip the ANARCI gate (useful for unit tests
that don't have HMMER installed).
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirage.benchmark._registry import AbstractLoader, register_loader
from mirage.scorers.base import BenchmarkExample

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESOLUTION = 3.0
_DEFAULT_MIN_ANTIGEN_LENGTH = 30
_DEFAULT_MAX_IDENTITY = 0.9
_KMER_K = 5

_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}  # fmt: skip


@dataclass(frozen=True)
class _Candidate:
    """Intermediate record carried across the loader's filtering phases."""

    row: dict[str, str]
    pdb_path: Path
    h_seq: str
    l_seq: str | None
    target_seqs: tuple[str, ...]
    target_chain_ids: tuple[str, ...]
    resolution: float


@contextmanager
def _prepend_path(directory: str) -> Iterator[None]:
    """Temporarily prepend a directory to PATH, restoring on exit."""
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{directory}{os.pathsep}{old}" if old else directory
    try:
        yield
    finally:
        os.environ["PATH"] = old


def _resolve_hmmer_bin(override: str | None = None) -> str:
    if override:
        return override
    from_env = os.environ.get("MIRAGE_HMMER_BIN")
    if from_env:
        return from_env
    on_path = shutil.which("hmmscan")
    if on_path:
        return str(Path(on_path).parent)
    mber_bin = Path.home() / "miniconda3" / "envs" / "mber" / "bin"
    if (mber_bin / "hmmscan").is_file():
        return str(mber_bin)
    raise RuntimeError(
        "Could not locate hmmscan. Install HMMER or set MIRAGE_HMMER_BIN to a "
        "directory containing hmmscan (or pass hmmer_bin= to SAbDabLoader)."
    )


def _extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    """Return the single-letter sequence for one chain.

    Reads CA atoms of standard amino acids from ATOM records. Skips HETATM
    and any non-standard residue. De-duplicates by (resseq, icode) so
    alternate locations don't double-count.
    """
    seen: set[tuple[int, str]] = set()
    chars: list[str] = []
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain_id:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26]
            key = (resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa = _THREE_TO_ONE.get(line[17:20].strip())
            if aa:
                chars.append(aa)
    return "".join(chars)


def _parse_antigen_chains(field: str) -> tuple[str, ...]:
    if not field or field == "NA":
        return ()
    return tuple(c.strip() for c in field.split("|") if c.strip())


def _binder_format(row: dict[str, str]) -> str:
    if row.get("Lchain", "NA").strip() == "NA":
        return "vhh"
    if row.get("scfv", "False").strip().lower() == "true":
        return "scfv"
    return "fab"


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _kmer_set(seq: str, k: int = _KMER_K) -> frozenset[str]:
    if len(seq) < k:
        return frozenset()
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _identity_to_jaccard_threshold(identity: float, k: int = _KMER_K) -> float:
    """Convert a target sequence identity into a k-mer Jaccard threshold.

    Heuristic: for length-L sequences differing in F fractional positions
    (identity = 1-F), each mutation perturbs up to k k-mers, so the
    intersection / union of k-mer sets is roughly (1 - kF) / (1 + kF) for
    small F. We invert this; the cap at 0 prevents degenerate cases when
    identity is very low. This is a proxy, not a true identity calculation
    (see the module docstring).
    """
    f = max(0.0, 1.0 - identity)
    raw = (1.0 - k * f) / (1.0 + k * f)
    return max(0.0, raw)


@register_loader("sabdab")
class SAbDabLoader(AbstractLoader):
    """Load SAbDab antibody-antigen complexes as `BenchmarkExample`s.

    See the module docstring for the expected data_dir layout and the
    filter pipeline.
    """

    SUMMARY_FILE = "summary.tsv"
    STRUCTURE_SUBDIR = "sabdab_dataset"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        max_resolution: float = _DEFAULT_MAX_RESOLUTION,
        min_antigen_length: int = _DEFAULT_MIN_ANTIGEN_LENGTH,
        max_identity: float = _DEFAULT_MAX_IDENTITY,
        use_anarci: bool = True,
        hmmer_bin: str | None = None,
    ) -> None:
        resolved = data_dir if data_dir is not None else os.environ.get("MIRAGE_SABDAB_DATA")
        if resolved is None:
            raise ValueError(
                "SAbDabLoader needs data_dir or MIRAGE_SABDAB_DATA env var pointing "
                "at the directory containing summary.tsv and sabdab_dataset/."
            )
        self.data_dir = Path(resolved)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        self.summary_path = self.data_dir / self.SUMMARY_FILE
        if not self.summary_path.is_file():
            raise FileNotFoundError(f"summary file missing: {self.summary_path}")

        self.structure_root = self.data_dir / self.STRUCTURE_SUBDIR

        self.max_resolution = max_resolution
        self.min_antigen_length = min_antigen_length
        self.max_identity = max_identity
        self.use_anarci = use_anarci
        self._hmmer_bin = hmmer_bin

    def load(self) -> Iterator[BenchmarkExample]:
        candidates = list(self._iter_structural_candidates())
        logger.info("SAbDab: %d candidates after TSV+structure filters", len(candidates))

        if self.use_anarci:
            candidates = self._anarci_filter(candidates)
            logger.info("SAbDab: %d candidates after ANARCI filter", len(candidates))

        threshold = _identity_to_jaccard_threshold(self.max_identity)
        accepted_kmers: list[frozenset[str]] = []
        for cand in candidates:
            ks = _kmer_set(cand.h_seq)
            if any(_jaccard(ks, prev) >= threshold for prev in accepted_kmers):
                continue
            accepted_kmers.append(ks)
            yield self._make_example(cand)

    def _iter_structural_candidates(self) -> Iterator[_Candidate]:
        with self.summary_path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                cand = self._row_to_candidate(row)
                if cand is not None:
                    yield cand

    def _row_to_candidate(self, row: dict[str, str]) -> _Candidate | None:
        res = _safe_float(row.get("resolution", ""))
        if res is None or res <= 0 or res > self.max_resolution:
            return None
        if "protein" not in row.get("antigen_type", ""):
            return None
        antigen_chains = _parse_antigen_chains(row.get("antigen_chain", ""))
        if not antigen_chains:
            return None

        pdb = row["pdb"].lower()
        pdb_path = self.structure_root / pdb / "structure" / "chothia" / f"{pdb}.pdb"
        if not pdb_path.is_file():
            return None

        h_chain = row["Hchain"].strip()
        if not h_chain or h_chain == "NA":
            return None
        h_seq = _extract_chain_sequence(pdb_path, h_chain)
        if not h_seq:
            return None

        l_seq: str | None = None
        l_chain = row.get("Lchain", "NA").strip()
        if l_chain and l_chain != "NA":
            extracted = _extract_chain_sequence(pdb_path, l_chain)
            if extracted:
                l_seq = extracted

        target_seqs: list[str] = []
        target_chain_ids: list[str] = []
        for ag in antigen_chains:
            ag_seq = _extract_chain_sequence(pdb_path, ag)
            if ag_seq:
                target_seqs.append(ag_seq)
                target_chain_ids.append(ag)
        if not target_seqs:
            return None
        if sum(len(s) for s in target_seqs) < self.min_antigen_length:
            return None

        return _Candidate(
            row=row,
            pdb_path=pdb_path,
            h_seq=h_seq,
            l_seq=l_seq,
            target_seqs=tuple(target_seqs),
            target_chain_ids=tuple(target_chain_ids),
            resolution=res,
        )

    def _anarci_filter(self, candidates: list[_Candidate]) -> list[_Candidate]:
        if not candidates:
            return []
        from anarci import anarci  # type: ignore[import-untyped]

        hmmer_bin = _resolve_hmmer_bin(self._hmmer_bin)
        queries = [(f"q{i}", c.h_seq) for i, c in enumerate(candidates)]
        # ANARCI 2026.x returns (numbering, hit_info, hmm_alignment_details).
        # `numbering[i]` is None when the i-th query is not an antibody V-domain.
        # We prepend hmmer_bin to PATH because some of anarci's internal
        # multi-domain refinement calls re-invoke `hmmscan` without
        # propagating the `hmmerpath=` argument.
        with _prepend_path(hmmer_bin):
            numbering = anarci(queries, scheme="chothia", hmmerpath=hmmer_bin)[0]
        return [c for c, num in zip(candidates, numbering, strict=True) if num is not None]

    def _make_example(self, cand: _Candidate) -> BenchmarkExample:
        row = cand.row
        fmt = _binder_format(row)
        binder_chains: tuple[str, ...]
        if fmt == "fab" and cand.l_seq is not None:
            binder_chains = (cand.h_seq, cand.l_seq)
        else:
            binder_chains = (cand.h_seq,)

        pdb_code = row["pdb"].lower()
        example_id = f"sabdab-{pdb_code}-{row['Hchain']}-{cand.target_chain_ids[0]}"

        metadata: dict[str, Any] = {
            "crystal_pdb_path": str(cand.pdb_path),
            "Hchain": row["Hchain"],
            "Lchain": row.get("Lchain", "NA"),
            # `antigen_chain` reflects what was actually staged into target_chains:
            # rows like 9u5p ("A | R", protein | nucleic-acid) drop the RNA chain, so
            # the field stored here is the protein-only subset. The raw row value is
            # preserved under `antigen_chain_raw` for traceability.
            "antigen_chain": "|".join(cand.target_chain_ids),
            "antigen_chain_raw": row.get("antigen_chain", ""),
            "target_chain_ids": cand.target_chain_ids,
            "resolution": cand.resolution,
            "method": row.get("method", ""),
            "antigen_species": row.get("antigen_species", ""),
            "heavy_species": row.get("heavy_species", ""),
            "scfv": row.get("scfv", ""),
            "date": row.get("date", ""),
        }
        if cand.l_seq is not None and fmt != "fab":
            metadata["light_chain_sequence"] = cand.l_seq

        return BenchmarkExample(
            id=example_id,
            label="POS",
            binder_chains=binder_chains,
            binder_format=fmt,
            target_chains=cand.target_seqs,
            target_name=row.get("antigen_name", "") or row.get("compound", ""),
            source="sabdab",
            target_pdb_id=pdb_code.upper(),
            complex_pdb_path=cand.pdb_path,
            metadata=metadata,
        )
