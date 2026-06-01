from __future__ import annotations

from pathlib import Path

import pytest

from mirage.benchmark import get_loader, list_loaders
from mirage.benchmark.sabdab import (
    SAbDabLoader,
    _binder_format,
    _extract_chain_sequence,
    _identity_to_jaccard_threshold,
    _jaccard,
    _kmer_set,
    _parse_antigen_chains,
)


def test_sabdab_registered_by_default() -> None:
    assert "sabdab" in list_loaders()


def test_sabdab_loader_requires_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIRAGE_SABDAB_DATA", raising=False)
    with pytest.raises(ValueError, match="data_dir"):
        get_loader("sabdab")


def test_sabdab_loader_rejects_missing_dir() -> None:
    with pytest.raises(FileNotFoundError):
        get_loader("sabdab", data_dir=Path("/nonexistent/mirage-sabdab-test"))


def test_sabdab_loader_rejects_missing_summary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="summary"):
        get_loader("sabdab", data_dir=tmp_path)


# -- helper unit tests --------------------------------------------------------


def test_parse_antigen_chains_handles_single_and_multi() -> None:
    assert _parse_antigen_chains("B") == ("B",)
    assert _parse_antigen_chains("B | C") == ("B", "C")
    assert _parse_antigen_chains("") == ()
    assert _parse_antigen_chains("NA") == ()


def test_binder_format_inference() -> None:
    assert _binder_format({"Lchain": "NA", "scfv": "False"}) == "vhh"
    assert _binder_format({"Lchain": "L", "scfv": "True"}) == "scfv"
    assert _binder_format({"Lchain": "L", "scfv": "False"}) == "fab"


def test_kmer_jaccard_identical_sequences() -> None:
    s = "QVQLVESGGGLVQPGGSLRLSCAAS"
    ks = _kmer_set(s)
    assert _jaccard(ks, ks) == 1.0


def test_kmer_jaccard_disjoint_sequences() -> None:
    a = _kmer_set("AAAAAA")
    b = _kmer_set("CCCCCC")
    assert _jaccard(a, b) == 0.0


def test_identity_to_jaccard_monotonic() -> None:
    # Stricter identity should give a stricter (higher) Jaccard threshold.
    low = _identity_to_jaccard_threshold(0.5)
    high = _identity_to_jaccard_threshold(0.95)
    assert high > low
    assert _identity_to_jaccard_threshold(1.0) == 1.0


# -- synthetic PDB / TSV unit test -------------------------------------------


_MINI_PDB = """\
REMARK   1 Synthetic test PDB for SAbDabLoader.
ATOM      1  N   GLN H   1      11.000  12.000  13.000  1.00 20.00           N
ATOM      2  CA  GLN H   1      11.500  12.500  13.500  1.00 20.00           C
ATOM      3  CA  VAL H   2      12.000  13.000  14.000  1.00 20.00           C
ATOM      4  CA  GLN H   3      12.500  13.500  14.500  1.00 20.00           C
ATOM      5  CA  LEU H   4      13.000  14.000  15.000  1.00 20.00           C
ATOM      6  CA  GLU H   5      13.500  14.500  15.500  1.00 20.00           C
ATOM      7  CA  SER H   6      14.000  15.000  16.000  1.00 20.00           C
ATOM      8  CA  GLY H   7      14.500  15.500  16.500  1.00 20.00           C
ATOM      9  CA  GLY H   8      15.000  16.000  17.000  1.00 20.00           C
ATOM     10  CA  GLY H   9      15.500  16.500  17.500  1.00 20.00           C
ATOM     11  CA  LEU H  10      16.000  17.000  18.000  1.00 20.00           C
ATOM     12  CA  MET A   1      20.000  20.000  20.000  1.00 20.00           C
ATOM     13  CA  GLU A   2      21.000  20.000  20.000  1.00 20.00           C
ATOM     14  CA  THR A   3      22.000  20.000  20.000  1.00 20.00           C
ATOM     15  CA  ALA A   4      23.000  20.000  20.000  1.00 20.00           C
ATOM     16  CA  ARG A   5      24.000  20.000  20.000  1.00 20.00           C
ATOM     17  CA  GLY A   6      25.000  20.000  20.000  1.00 20.00           C
ATOM     18  CA  LYS A   7      26.000  20.000  20.000  1.00 20.00           C
ATOM     19  CA  PHE A   8      27.000  20.000  20.000  1.00 20.00           C
ATOM     20  CA  TYR A   9      28.000  20.000  20.000  1.00 20.00           C
ATOM     21  CA  TRP A  10      29.000  20.000  20.000  1.00 20.00           C
ATOM     22  CA  ASN A  11      30.000  20.000  20.000  1.00 20.00           C
ATOM     23  CA  GLN A  12      31.000  20.000  20.000  1.00 20.00           C
ATOM     24  CA  ASP A  13      32.000  20.000  20.000  1.00 20.00           C
ATOM     25  CA  HIS A  14      33.000  20.000  20.000  1.00 20.00           C
ATOM     26  CA  ILE A  15      34.000  20.000  20.000  1.00 20.00           C
ATOM     27  CA  LYS A  16      35.000  20.000  20.000  1.00 20.00           C
ATOM     28  CA  LEU A  17      36.000  20.000  20.000  1.00 20.00           C
ATOM     29  CA  MET A  18      37.000  20.000  20.000  1.00 20.00           C
ATOM     30  CA  ASN A  19      38.000  20.000  20.000  1.00 20.00           C
ATOM     31  CA  PRO A  20      39.000  20.000  20.000  1.00 20.00           C
ATOM     32  CA  ARG A  21      40.000  20.000  20.000  1.00 20.00           C
ATOM     33  CA  GLN A  22      41.000  20.000  20.000  1.00 20.00           C
ATOM     34  CA  SER A  23      42.000  20.000  20.000  1.00 20.00           C
ATOM     35  CA  THR A  24      43.000  20.000  20.000  1.00 20.00           C
ATOM     36  CA  VAL A  25      44.000  20.000  20.000  1.00 20.00           C
ATOM     37  CA  TRP A  26      45.000  20.000  20.000  1.00 20.00           C
ATOM     38  CA  TYR A  27      46.000  20.000  20.000  1.00 20.00           C
ATOM     39  CA  ALA A  28      47.000  20.000  20.000  1.00 20.00           C
ATOM     40  CA  CYS A  29      48.000  20.000  20.000  1.00 20.00           C
ATOM     41  CA  ASP A  30      49.000  20.000  20.000  1.00 20.00           C
ATOM     42  CA  GLU A  31      50.000  20.000  20.000  1.00 20.00           C
ATOM     43  CA  ARG A  32      51.000  20.000  20.000  1.00 20.00           C
HETATM   44  X   HOH A 100      99.000  99.000  99.000  1.00 30.00           O
"""


def _make_fixture_dir(root: Path, pdb_code: str = "test") -> Path:
    pdb_dir = root / "sabdab_dataset" / pdb_code / "structure" / "chothia"
    pdb_dir.mkdir(parents=True)
    (pdb_dir / f"{pdb_code}.pdb").write_text(_MINI_PDB)
    summary = root / "summary.tsv"
    summary.write_text(
        "pdb\tHchain\tLchain\tmodel\tantigen_chain\tantigen_type\t"
        "antigen_name\tresolution\tmethod\tscfv\n"
        f"{pdb_code}\tH\tNA\t0\tA\tprotein\ttest_antigen\t2.00\t"
        "X-RAY DIFFRACTION\tFalse\n"
    )
    return root


def test_extract_chain_sequence_from_synthetic_pdb(tmp_path: Path) -> None:
    _make_fixture_dir(tmp_path)
    pdb_path = tmp_path / "sabdab_dataset" / "test" / "structure" / "chothia" / "test.pdb"
    h_seq = _extract_chain_sequence(pdb_path, "H")
    assert h_seq == "QVQLESGGGL"
    a_seq = _extract_chain_sequence(pdb_path, "A")
    assert len(a_seq) == 32  # 32 antigen CA atoms; HETATM water ignored
    assert a_seq.startswith("MET")


def test_loader_yields_from_synthetic_fixture(tmp_path: Path) -> None:
    _make_fixture_dir(tmp_path)
    loader = SAbDabLoader(data_dir=tmp_path, use_anarci=False)
    examples = list(loader.load())
    assert len(examples) == 1
    ex = examples[0]
    assert ex.id == "sabdab-test-H-A"
    assert ex.label == "POS"
    assert ex.binder_format == "vhh"
    assert ex.binder_chains == ("QVQLESGGGL",)
    assert len(ex.target_chains) == 1
    assert len(ex.target_chains[0]) == 32
    assert ex.target_pdb_id == "TEST"
    assert ex.source == "sabdab"
    assert ex.complex_pdb_path is not None
    assert ex.complex_pdb_path.name == "test.pdb"
    assert ex.metadata["crystal_pdb_path"].endswith("test.pdb")
    assert ex.metadata["resolution"] == 2.0


def test_loader_filters_on_resolution(tmp_path: Path) -> None:
    _make_fixture_dir(tmp_path)
    # max_resolution=1.0 is stricter than the 2.0 in the fixture row
    loader = SAbDabLoader(data_dir=tmp_path, max_resolution=1.0, use_anarci=False)
    assert list(loader.load()) == []


def test_loader_filters_on_antigen_length(tmp_path: Path) -> None:
    _make_fixture_dir(tmp_path)
    loader = SAbDabLoader(data_dir=tmp_path, min_antigen_length=100, use_anarci=False)
    assert list(loader.load()) == []


# -- integration tests against the real on-disk dataset ---------------------


def test_loader_integration_yields_examples(sabdab_data_dir: Path) -> None:
    # Skip ANARCI to keep the test fast (HMMER may or may not be on PATH).
    loader = SAbDabLoader(data_dir=sabdab_data_dir, use_anarci=False)
    examples = list(loader.load())
    assert len(examples) > 0
    formats = {ex.binder_format for ex in examples}
    assert formats <= {"vhh", "fab", "scfv"}
    ex = examples[0]
    assert ex.source == "sabdab"
    assert ex.label == "POS"
    assert len(ex.binder_chains) >= 1
    assert len(ex.target_chains) >= 1
    assert ex.complex_pdb_path is not None
    assert ex.complex_pdb_path.is_file()
    assert "crystal_pdb_path" in ex.metadata


def test_loader_normalizes_antigen_chain_to_staged_subset(tmp_path: Path) -> None:
    """When antigen_chain lists a chain with no extractable CAs (e.g. the
    9u5p case where the row carries ``"A | R"`` with antigen_type
    ``"protein | nucleic-acid"`` but the PDB has no protein CA atoms for the
    RNA chain), the staged ``target_chains`` only contains the protein chain
    and the metadata fields downstream of the loader must reflect that
    subset — otherwise the RMSD scorer mistakenly expects two crystal target
    chains and fails with ``target_chain_count_mismatch``.
    """
    _make_fixture_dir(tmp_path)
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "pdb\tHchain\tLchain\tmodel\tantigen_chain\tantigen_type\t"
        "antigen_name\tresolution\tmethod\tscfv\n"
        # antigen_chain lists chain A (present in the PDB) and chain R
        # (not present), to mirror the 9u5p protein + RNA case after the
        # nucleic-acid chain falls out of CA-only sequence extraction.
        "test\tH\tNA\t0\tA | R\tprotein | nucleic-acid\ttest_antigen\t2.00\t"
        "X-RAY DIFFRACTION\tFalse\n"
    )
    loader = SAbDabLoader(data_dir=tmp_path, use_anarci=False)
    examples = list(loader.load())
    assert len(examples) == 1
    ex = examples[0]
    assert ex.target_chains == (ex.target_chains[0],)  # exactly one staged chain
    assert ex.metadata["target_chain_ids"] == ("A",)
    assert ex.metadata["antigen_chain"] == "A"
    assert ex.metadata["antigen_chain_raw"] == "A | R"
    # The example id should use the first staged chain, not necessarily the
    # first raw-row chain (here they happen to agree).
    assert ex.id == "sabdab-test-H-A"
