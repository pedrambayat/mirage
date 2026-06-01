from __future__ import annotations

from mirage.features.sequence import FEATURE_NAMES, sequence_features


def test_feature_names_match_dict_keys() -> None:
    feats = sequence_features("ACDEFGHIK", "KKKKDDDD")
    assert list(feats.keys()) == list(FEATURE_NAMES)


def test_lengths_are_reported() -> None:
    feats = sequence_features("AAAA", "GGGGGG")
    assert feats["binder_length"] == 4.0
    assert feats["target_length"] == 6.0


def test_net_charge_sign() -> None:
    # all-lysine binder is strongly positive; all-aspartate is negative
    assert sequence_features("KKKK", "A")["binder_net_charge"] > 0
    assert sequence_features("DDDD", "A")["binder_net_charge"] < 0


def test_fractions_in_unit_interval() -> None:
    feats = sequence_features("FWYAVLIMCDEKR", "ACDEFGHIKLMNPQRSTVWY")
    for key, value in feats.items():
        if key.endswith("_frac"):
            assert 0.0 <= value <= 1.0


def test_empty_sequence_is_safe() -> None:
    feats = sequence_features("", "")
    assert feats["binder_length"] == 0.0
    assert feats["binder_aromatic_frac"] == 0.0
