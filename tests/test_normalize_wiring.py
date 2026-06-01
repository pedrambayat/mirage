from __future__ import annotations

from mirage.eval.orthogonal import features_for_examples
from mirage.scorers.base import BenchmarkExample


def test_features_for_examples_normalizes_antigen_signal_peptide() -> None:
    # Two examples identical except one antigen carries the IL-6 signal peptide;
    # after normalization their target features must be identical.
    mature = "VPPGEDSKDVAAPHRQ"
    ex_raw = BenchmarkExample(
        id="raw",
        label="BIND",
        binder_chains=("QVQL",),
        binder_format="vhh",
        target_chains=("MNSFSTSAFGPVAFSLGLLLVLPAAFPAP" + mature,),
        target_name="IL6",
        source="",
    )
    ex_mature = BenchmarkExample(
        id="mat",
        label="BIND",
        binder_chains=("QVQL",),
        binder_format="vhh",
        target_chains=(mature,),
        target_name="IL6",
        source="",
    )
    x, _y, _names = features_for_examples([ex_raw, ex_mature], positive_label="BIND")
    assert list(x[0]) == list(x[1])
