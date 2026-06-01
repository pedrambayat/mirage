import importlib.util
from pathlib import Path

import numpy as np


def _load_analysis_module():
    script_path = Path(__file__).parents[1] / "scripts" / "analyze_af2m_confidence_vs_rmsd.py"
    spec = importlib.util.spec_from_file_location("analyze_af2m_confidence_vs_rmsd", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_logistic_loo_scores_separate_simple_signal():
    mod = _load_analysis_module()
    x = np.asarray([[-4.0], [-3.0], [-2.0], [2.0], [3.0], [4.0]])
    y = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    scores = mod._logistic_loo_scores(x, y, l2=1.0)

    assert np.isfinite(scores).all()
    assert mod._auroc(scores, y.astype(int)) == 1.0


def test_stratified_bootstrap_metric_ci_is_deterministic_and_bounded():
    mod = _load_analysis_module()
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])

    ci1 = mod._stratified_bootstrap_metric_ci(scores, labels, mod._auroc, n_bootstrap=100, seed=123)
    ci2 = mod._stratified_bootstrap_metric_ci(scores, labels, mod._auroc, n_bootstrap=100, seed=123)

    assert ci1 == ci2
    assert 0.0 <= ci1[0] <= ci1[1] <= 1.0


def test_logistic_ablation_feature_sets_cover_singletons_and_pae_plddt_pairs():
    mod = _load_analysis_module()

    feature_sets = mod._logistic_ablation_feature_sets()

    assert ("iptm",) in feature_sets
    assert ("pae_interface_mean", "plddt_interface_mean") in feature_sets
    assert mod._DEFAULT_LOGISTIC_FEATURES in feature_sets
    assert len(feature_sets) == len(set(feature_sets))


def test_negative_vs_real_label_keeps_only_near_native_reals_and_negatives():
    mod = _load_analysis_module()
    rmsd = np.asarray([2.0, 5.0, 9.0])

    labels, keep = mod._negative_vs_real_label(rmsd, n_negative=2, rmsd_cutoff=4.0)

    np.testing.assert_array_equal(labels, np.asarray([1.0, np.nan, np.nan, 0.0, 0.0]))
    np.testing.assert_array_equal(keep, np.asarray([True, False, False, True, True]))
