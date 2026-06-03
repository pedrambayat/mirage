"""Tests for ProtenixConfidenceScorer.

Covers:
- Synthetic unit tests for `interface_pae_stats` and `interface_plddt_value`
  with hand-computed expected values (no file I/O).
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
    interface_pae_stats,
    interface_plddt_value,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "protenix"


def _example(example_id: str = "3OGO__3OGO") -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        label="1",
        binder_chains=("Q",),
        binder_format="vhh",
        target_chains=("E",),
        target_name="test",
        source="fixture",
    )


# ---------------------------------------------------------------------------
# Unit tests: interface_pae_stats
# ---------------------------------------------------------------------------


class TestInterfacePaeStats:
    """interface_pae_stats returns (mean_interface_pae, min_interface_pae)."""

    def test_two_chain_known_values(self) -> None:
        # token_asym_id: [0, 0, 1, 1] → cross-chain pairs are (0,2),(0,3),(1,2),(1,3)
        # and symmetric counterparts (2,0),(2,1),(3,0),(3,1) — 8 pairs total.
        pae = np.array(
            [
                [0.0, 0.1, 2.0, 3.0],  # row 0
                [0.2, 0.0, 4.0, 5.0],  # row 1
                [6.0, 7.0, 0.0, 0.3],  # row 2
                [8.0, 9.0, 0.4, 0.0],  # row 3
            ],
            dtype=float,
        )
        asym = np.array([0, 0, 1, 1])
        mean_pae, min_pae = interface_pae_stats(pae, asym)

        # Cross-chain values: 2,3,4,5 (rows 0,1 → cols 2,3) + 6,7,8,9 (rows 2,3 → cols 0,1)
        expected_mean = (2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0) / 8.0
        expected_min = 2.0
        assert mean_pae == pytest.approx(expected_mean, abs=1e-6)
        assert min_pae == pytest.approx(expected_min, abs=1e-6)

    def test_same_chain_only_returns_nan(self) -> None:
        pae = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]], dtype=float)
        asym = np.array([0, 0, 0])
        mean_pae, min_pae = interface_pae_stats(pae, asym)
        assert math.isnan(mean_pae)
        assert math.isnan(min_pae)

    def test_three_chains_cross_chain_pairs(self) -> None:
        # Chains 0 and 1 each have 1 token; chain 2 has 1 token.
        # All cross-chain pairs contribute: (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) = 6 pairs.
        pae = np.array(
            [
                [0.0, 10.0, 20.0],
                [30.0, 0.0, 40.0],
                [50.0, 60.0, 0.0],
            ],
            dtype=float,
        )
        asym = np.array([0, 1, 2])
        mean_pae, min_pae = interface_pae_stats(pae, asym)
        expected_mean = (10.0 + 20.0 + 30.0 + 40.0 + 50.0 + 60.0) / 6.0
        assert mean_pae == pytest.approx(expected_mean, abs=1e-6)
        assert min_pae == pytest.approx(10.0, abs=1e-6)


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
        np.ndarray[int, np.dtype[np.int64]],
    ]:
        """
        4 tokens (0..3), 6 atoms, 2 chains (asym 0 = tokens 0,1; asym 1 = tokens 2,3).

        Atom assignments:
          atom 0,1 → token 0  (chain 0)
          atom 2,3 → token 1  (chain 0)
          atom 4   → token 2  (chain 1)
          atom 5   → token 3  (chain 1)

        atom_plddt (0-1 scale):
          atoms 0,1 -> 0.80  per-token pLDDT token0 = 0.80
          atoms 2,3 -> 0.60  per-token pLDDT token1 = 0.60
          atom  4   -> 0.90  per-token pLDDT token2 = 0.90
          atom  5   -> 0.70  per-token pLDDT token3 = 0.70

        contact_probs (4x4): interface = cross-chain, max >= 0.5
          token 0 vs chain 1: max(cp[0,2], cp[0,3]) = max(0.8, 0.1) = 0.8 -> interface
          token 1 vs chain 1: max(cp[1,2], cp[1,3]) = max(0.3, 0.2) = 0.3 -> NOT interface
          token 2 vs chain 0: max(cp[2,0], cp[2,1]) = max(0.8, 0.3) = 0.8 -> interface
          token 3 vs chain 0: max(cp[3,0], cp[3,1]) = max(0.1, 0.2) = 0.2 -> NOT interface

        Interface tokens: 0 (plddt=0.80*100=80) and 2 (plddt=0.90*100=90).
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
        asym = np.array([0, 0, 1, 1], dtype=int)
        return atom_plddt, atom_to_token, contact_probs, asym

    def test_known_value(self) -> None:
        atom_plddt, atom_to_token, contact_probs, asym = self._make_inputs()
        result = interface_plddt_value(atom_plddt, atom_to_token, contact_probs, asym)
        assert result == pytest.approx(85.0, abs=1e-5)

    def test_no_interface_returns_nan(self) -> None:
        # All contact_probs below threshold
        atom_plddt = np.array([0.80, 0.60], dtype=float)
        atom_to_token = np.array([0, 1], dtype=int)
        contact_probs = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=float)
        asym = np.array([0, 1], dtype=int)
        result = interface_plddt_value(atom_plddt, atom_to_token, contact_probs, asym)
        assert math.isnan(result)

    def test_same_chain_only_returns_nan(self) -> None:
        atom_plddt = np.array([0.80, 0.90, 0.70], dtype=float)
        atom_to_token = np.array([0, 1, 2], dtype=int)
        contact_probs = np.ones((3, 3), dtype=float)
        asym = np.array([0, 0, 0], dtype=int)
        result = interface_plddt_value(atom_plddt, atom_to_token, contact_probs, asym)
        # No cross-chain pairs, so no interface tokens possible
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# End-to-end: fixture
# ---------------------------------------------------------------------------


class TestProtenixConfidenceScorerFixture:
    def test_score_value_and_extras(self) -> None:
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

        # Keys from full-data must be present (values will be NaN on trimmed fixture
        # because token_asym_id is all 0 — no cross-chain pairs)
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
