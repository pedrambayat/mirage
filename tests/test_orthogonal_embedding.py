from __future__ import annotations

import numpy as np

from mirage.eval.orthogonal import features_for_examples_embedding
from mirage.scorers.base import BenchmarkExample


def test_features_for_examples_embedding_concat():
    # cache keyed on NORMALIZED sequences; toy seqs pass normalize unchanged.
    cache = {"AAAA": np.array([1.0, 2.0]), "CCCC": np.array([3.0, 4.0])}
    ex = BenchmarkExample(
        id="e1",
        label="POS",
        binder_chains=("AAAA",),
        binder_format="vhh",
        target_chains=("CCCC",),
        target_name="t",
        source="test",
    )
    x, y = features_for_examples_embedding([ex], cache, positive_label="POS", layout="concat")
    assert x.shape == (1, 4)
    assert np.allclose(x[0], [1.0, 2.0, 3.0, 4.0])
    assert y.tolist() == [1]
