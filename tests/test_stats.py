import math

from artifact_signals.stats import (binom_test, brier, cluster_bootstrap, holm, mcnemar, point_biserial, reliability,
                                    risk_coverage_auc, spearman, tv_distance)


def test_binom_and_mcnemar():
    assert abs(binom_test(5, 10, 0.5) - 1.0) < 1e-9
    assert binom_test(10, 10, 0.5, "greater") == 2 ** -10
    assert abs(binom_test(0, 10, 0.5) - 2 * 2 ** -10) < 1e-12
    m = mcnemar([True] * 10 + [False] * 10, [True] * 5 + [False] * 15)
    assert m["a_only"] == 5 and m["b_only"] == 0 and m["p"] == 2 * 0.5 ** 5


def test_bootstrap_and_holm():
    est, lo, hi = cluster_bootstrap([1, 0, 1, 1, 0, 1, 1, 1], list("aabbccdd"), n_boot=200, seed=1)
    assert lo <= est <= hi and est == 0.75
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_correlations_and_distances():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert point_biserial([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) > 0.9
    assert tv_distance({"a": 0.75, "b": 0.25}, {"a": 0.5, "b": 0.5}) == 0.25


def test_brier_reliability_and_risk_coverage():
    assert brier([1.0, 0.0], [True, False]) == 0.0 and brier([0.5], [True]) == 0.25
    r = reliability([0.1, 0.9, 0.95], [False, True, True], bins=2)
    assert r[0]["n"] == 1 and r[1]["n"] == 2 and r[1]["hit_rate"] == 1.0
    assert risk_coverage_auc([0.9, 0.8, 0.7, 0.6], [True, True, False, False]) < risk_coverage_auc([0.6, 0.7, 0.8, 0.9], [True, True, False, False])
    assert math.isnan(risk_coverage_auc([], []))
