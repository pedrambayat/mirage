"""Tests for ProtenixConfidenceScorer.

Covers:
- Synthetic unit tests for `interface_pae_stats` and `interface_plddt_value`
  with hand-computed expected values (no file I/O).
- Synthetic unit test for `build_binder_token_mask` (size-mismatch sentinel).
- 3-chain antigen correctness: antigen-antigen pairs excluded from interface.
- End-to-end test on the committed fixture at tests/fixtures/protenix/3OGO__3OGO/.
- Missing-prediction sentinel.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mirage.scorers.base import BenchmarkExample
from mirage.scorers.protenix_confidence import (
    ProtenixConfidenceScorer,
    build_binder_token_mask,
    interface_pae_stats,
    interface_plddt_value,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "protenix"

# Sequences extracted from the CIF fixture (chain A = VHH binder, chain B = GFP)
# These match SEQ_A / SEQ_B in test_chain_role_resolver.py.
SEQ_A = "MQVQLVESGGALVQPGGSLRLSCAASGFPVNRYSMRWYRQAPGKEREWVAGMSSAGDRSSYEDSVKGRFTISRDDARNTVYLQMNSLKPEDTAVYYCNVNVGFEYWGQGTQVTVSSKHHHHHH"  # noqa: E501
SEQ_B = "MAHHHHHHSSGVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"  # noqa: E501


def _example(
    example_id: str = "3OGO__3OGO",
    binder_chains: tuple[str, ...] = (SEQ_A,),
    target_chains: tuple[str, ...] = (SEQ_B,),
) -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        label="1",
        binder_chains=binder_chains,
        binder_format="vhh",
        target_chains=target_chains,
        target_name="test",
        source="fixture",
    )


# ---------------------------------------------------------------------------
# Unit tests: interface_pae_stats
# ---------------------------------------------------------------------------


class TestInterfacePaeStats:
    """interface_pae_stats returns (mean_interface_pae, min_interface_pae)."""

    def test_two_role_known_values(self) -> None:
        # mask: tokens 0,1 = binder; tokens 2,3 = antigen
        # Binder x antigen pairs: (0,2),(0,3),(1,2),(1,3) and reverse (2,0),(2,1),(3,0),(3,1) = 8
        pae = np.array(
            [
                [0.0, 0.1, 2.0, 3.0],  # row 0
                [0.2, 0.0, 4.0, 5.0],  # row 1
                [6.0, 7.0, 0.0, 0.3],  # row 2
                [8.0, 9.0, 0.4, 0.0],  # row 3
            ],
            dtype=float,
        )
        mask = np.array([True, True, False, False])
        mean_pae, min_pae = interface_pae_stats(pae, mask)

        # Binder x antigen values: 2,3,4,5 (rows 0,1 -> cols 2,3) + 6,7,8,9 (rows 2,3 -> cols 0,1)
        expected_mean = (2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0) / 8.0
        expected_min = 2.0
        assert mean_pae == pytest.approx(expected_mean, abs=1e-6)
        assert min_pae == pytest.approx(expected_min, abs=1e-6)

    def test_all_binder_returns_nan(self) -> None:
        """When all tokens are binder tokens there are no interface pairs."""
        pae = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]], dtype=float)
        mask = np.array([True, True, True])
        mean_pae, min_pae = interface_pae_stats(pae, mask)
        assert math.isnan(mean_pae)
        assert math.isnan(min_pae)

    def test_all_antigen_returns_nan(self) -> None:
        """When all tokens are antigen tokens there are no interface pairs."""
        pae = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        mask = np.array([False, False])
        mean_pae, min_pae = interface_pae_stats(pae, mask)
        assert math.isnan(mean_pae)
        assert math.isnan(min_pae)

    def test_three_chain_excludes_antigen_antigen(self) -> None:
        """3-chain case: 1 binder (token 0) + 2 antigen chains (tokens 1,2,3).

        token 0 = binder; tokens 1,2 = antigen chain X; token 3 = antigen chain Y.

        The antigen-antigen pairs (1-2, 1-3, 2-1, 2-3, 3-1, 3-2) must be
        EXCLUDED.  Only binder(0) x antigen(1,2,3) pairs count = 6 pairs:
        (0->1), (0->2), (0->3), (1->0), (2->0), (3->0).

        We set antigen-antigen entries to a large distinct value (999) so any
        accidental inclusion would shift the mean noticeably.
        """
        # fmt: off
        pae = np.array([
            [  0.0,  2.0,  4.0,  6.0],  # row 0 (binder)
            [ 10.0,999.0,999.0,999.0],  # row 1 (antigen X)
            [ 20.0,999.0,999.0,999.0],  # row 2 (antigen X)
            [ 30.0,999.0,999.0,999.0],  # row 3 (antigen Y)
        ], dtype=float)
        # fmt: on
        # token 0 = binder, tokens 1,2,3 = antigen (regardless of their asym_id)
        mask = np.array([True, False, False, False])
        mean_pae, min_pae = interface_pae_stats(pae, mask)

        # Included pairs (i→j where mask[i] != mask[j]):
        # (0→1)=2, (0→2)=4, (0→3)=6, (1→0)=10, (2→0)=20, (3→0)=30
        expected_mean = (2.0 + 4.0 + 6.0 + 10.0 + 20.0 + 30.0) / 6.0
        expected_min = 2.0
        assert mean_pae == pytest.approx(expected_mean, abs=1e-6)
        assert min_pae == pytest.approx(expected_min, abs=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: interface_plddt_value
# ---------------------------------------------------------------------------


class TestInterfacePlddtValue:
    """interface_plddt_value returns mean per-token pLDDT (0-100) over interface tokens."""

    def _make_inputs(
        self,
    ) -> tuple[
        np.ndarray[int, np.dtype[np.float64]],
        np.ndarray[int, np.dtype[np.int64]],
        np.ndarray[int, np.dtype[np.float64]],
        np.ndarray[int, np.dtype[np.bool_]],
    ]:
        """
        4 tokens (0..3), 6 atoms.
        Binder tokens: 0, 1  (mask=True)
        Antigen tokens: 2, 3  (mask=False)

        Atom assignments:
          atom 0,1 → token 0  (binder)
          atom 2,3 → token 1  (binder)
          atom 4   → token 2  (antigen)
          atom 5   → token 3  (antigen)

        atom_plddt (0-1 scale):
          atoms 0,1 -> 0.80  per-token pLDDT token0 = 0.80
          atoms 2,3 -> 0.60  per-token pLDDT token1 = 0.60
          atom  4   -> 0.90  per-token pLDDT token2 = 0.90
          atom  5   -> 0.70  per-token pLDDT token3 = 0.70

        contact_probs (4x4): interface = cross-role, max >= 0.5
          token 0 vs antigen: max(cp[0,2], cp[0,3]) = max(0.8, 0.1) = 0.8 -> interface
          token 1 vs antigen: max(cp[1,2], cp[1,3]) = max(0.3, 0.2) = 0.3 -> NOT interface
          token 2 vs binder:  max(cp[2,0], cp[2,1]) = max(0.8, 0.3) = 0.8 -> interface
          token 3 vs binder:  max(cp[3,0], cp[3,1]) = max(0.1, 0.2) = 0.2 -> NOT interface

        Interface tokens: 0 (plddt=80) and 2 (plddt=90).
        Expected: (80 + 90) / 2 = 85.0
        """
        atom_plddt = np.array([0.80, 0.80, 0.60, 0.60, 0.90, 0.70], dtype=float)
        atom_to_token = np.array([0, 0, 1, 1, 2, 3], dtype=int)
        contact_probs = np.array(
            [
                [1.0, 0.9, 0.8, 0.1],  # token 0
                [0.9, 1.0, 0.3, 0.2],  # token 1
                [0.8, 0.3, 1.0, 0.9],  # token 2
                [0.1, 0.2, 0.9, 1.0],  # token 3
            ],
            dtype=float,
        )
        binder_mask = np.array([True, True, False, False])
        return atom_plddt, atom_to_token, contact_probs, binder_mask

    def test_known_value(self) -> None:
        atom_plddt, atom_to_token, contact_probs, binder_mask = self._make_inputs()
        result = interface_plddt_value(atom_plddt, atom_to_token, contact_probs, binder_mask)
        assert result == pytest.approx(85.0, abs=1e-5)

    def test_no_interface_returns_nan(self) -> None:
        # All contact_probs below threshold
        atom_plddt = np.array([0.80, 0.60], dtype=float)
        atom_to_token = np.array([0, 1], dtype=int)
        contact_probs = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=float)
        binder_mask = np.array([True, False])
        result = interface_plddt_value(atom_plddt, atom_to_token, contact_probs, binder_mask)
        assert math.isnan(result)

    def test_all_binder_returns_nan(self) -> None:
        atom_plddt = np.array([0.80, 0.90, 0.70], dtype=float)
        atom_to_token = np.array([0, 1, 2], dtype=int)
        contact_probs = np.ones((3, 3), dtype=float)
        binder_mask = np.array([True, True, True])
        result = interface_plddt_value(atom_plddt, atom_to_token, contact_probs, binder_mask)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Unit tests: build_binder_token_mask
# ---------------------------------------------------------------------------


class TestBuildBinderTokenMask:
    """build_binder_token_mask returns None on size mismatch / token index overflow."""

    def test_size_mismatch_returns_none(self) -> None:
        """If atom_to_token_idx length != number of atoms in structure, return None."""
        from mirage.scorers._structure import load_structure

        cif = FIXTURE_ROOT / "3OGO__3OGO" / "3OGO__3OGO_sample_0.cif"
        struct = load_structure(cif)

        # The trimmed fixture has only 200 atoms in atom_to_token_idx (24 tokens)
        # but the CIF has 2951 atoms — sizes are inconsistent.
        atom_to_token_trimmed = np.arange(200, dtype=int) % 24
        result = build_binder_token_mask(struct, ["A"], atom_to_token_trimmed, n_tokens=24)
        assert result is None

    def test_token_index_overflow_returns_none(self) -> None:
        """If max(atom_to_token_idx) >= n_tokens, return None."""
        from mirage.scorers._structure import load_structure

        cif = FIXTURE_ROOT / "3OGO__3OGO" / "3OGO__3OGO_sample_0.cif"
        struct = load_structure(cif)

        # n_tokens deliberately too small relative to the max index
        atom_to_token = np.zeros(2951, dtype=int)
        atom_to_token[-1] = 5  # max index = 5, but n_tokens = 3 → overflow
        result = build_binder_token_mask(struct, ["A"], atom_to_token, n_tokens=3)
        assert result is None


# ---------------------------------------------------------------------------
# End-to-end: fixture
# ---------------------------------------------------------------------------


class TestProtenixConfidenceScorerFixture:
    def test_score_value_and_extras(self) -> None:
        """3OGO end-to-end: scalars correct; interface keys present (NaN OK).

        The fixture's full_data JSON is trimmed (24 tokens / 200 atoms) while
        the CIF has 2951 atoms — build_binder_token_mask detects the size
        mismatch and returns None, so interface features are NaN.
        """
        scorer = ProtenixConfidenceScorer(predictions_root=FIXTURE_ROOT)
        example = _example("3OGO__3OGO")
        score = scorer.score(example)

        assert score.scorer_name == "protenix_confidence"
        assert score.example_id == "3OGO__3OGO"

        # Headline value = iptm
        assert score.value == pytest.approx(0.9382603, abs=1e-4)

        # Scalar extras from summary JSON
        assert score.extras["iptm"] == pytest.approx(0.9382603, abs=1e-4)
        assert score.extras["ptm"] == pytest.approx(0.9282569, abs=1e-4)
        assert score.extras["mean_plddt"] == pytest.approx(91.768, abs=1e-2)

        # Interface keys must be present; values are NaN due to trimmed fixture
        assert "interface_pae" in score.extras
        assert "min_interface_pae" in score.extras
        assert "interface_plddt" in score.extras

    def test_registered_by_name(self) -> None:
        from mirage.scorers import get_scorer

        scorer = get_scorer("protenix_confidence", predictions_root=str(FIXTURE_ROOT))
        assert scorer.name == "protenix_confidence"


# ---------------------------------------------------------------------------
# Missing prediction
# ---------------------------------------------------------------------------


class TestMissingPrediction:
    def test_missing_returns_nan_score(self, tmp_path: Path) -> None:
        scorer = ProtenixConfidenceScorer(predictions_root=tmp_path)
        example = _example("NONEXISTENT__ID")
        score = scorer.score(example)
        assert math.isnan(score.value)
        assert score.extras.get("missing") == "prediction"

    def test_corrupt_summary_returns_nan_score(self, tmp_path: Path) -> None:
        example_dir = tmp_path / "BAD__EXAMPLE"
        example_dir.mkdir()
        # Write a corrupt summary JSON
        bad_summary = example_dir / "BAD__EXAMPLE_summary_confidence_sample_0.json"
        bad_summary.write_text("{not valid json")
        scorer = ProtenixConfidenceScorer(predictions_root=tmp_path)
        example = _example("BAD__EXAMPLE")
        score = scorer.score(example)
        assert math.isnan(score.value)
        assert "error" in score.extras
