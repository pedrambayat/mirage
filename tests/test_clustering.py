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
