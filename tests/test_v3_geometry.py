import numpy as np
import pytest
from scipy.linalg import solve

from prism.design import objective_gradient, proxy
from prism.v3.artifacts import Run
from prism.v3.data import prepare_data
from prism.v3.features import extract_features, merge_features
from prism.v3.geometry import (blocked_objective_gradient, build_geometry, design_report,
                               distribution_diagnostics, pair_geometry, solve_design)
from test_v3_data import FakeTokenizer, tiny_config, write_sources
from test_v3_features import deterministic_scorer


def test_full_pair_proxy_matches_independent_augmented_least_squares_and_covariance():
    rng = np.random.default_rng(87)
    phi, ell, ridge = rng.normal(size=(17, 2, 5)), rng.normal(size=(17, 2)), .03
    phi[..., -1] = phi[..., 0]
    actual = pair_geometry(phi, ell, ridge, block_size=3)
    X = phi.reshape(-1, 5)
    target = (ell - ell.mean(axis=1, keepdims=True)).ravel()
    fitted = np.linalg.lstsq(np.vstack((X, np.sqrt(ridge) * np.eye(5))),
                             np.r_[target, np.zeros(5)], rcond=None)[0]
    np.testing.assert_allclose(actual["b"], fitted, atol=1e-12)
    independent = np.zeros((5, 5))
    for responses, probability in zip(phi, actual["proxy_probabilities"]):
        mean = probability @ responses
        for response, p in zip(responses, probability):
            independent += p * np.outer(response - mean, response - mean) / len(phi)
    np.testing.assert_allclose(actual["K"], independent, atol=1e-13)
    legacy = proxy(phi, ell, ridge)
    for expected, key in zip(legacy, ("b", "proxy_probabilities", "K")):
        np.testing.assert_allclose(actual[key], expected, atol=1e-12)
    reversed_pairs = pair_geometry(phi[:, ::-1], ell[:, ::-1], ridge, block_size=4)
    np.testing.assert_allclose(actual["K"], reversed_pairs["K"], atol=1e-13)
    np.testing.assert_allclose(actual["v"], -reversed_pairs["v"], atol=0)


def test_blocked_original_objective_and_gradient_match_direct_linear_solves():
    rng = np.random.default_rng(5)
    v, q = rng.normal(size=(19, 6)), rng.dirichlet(np.ones(19))
    factor = rng.normal(size=(6, 3))
    K = factor @ factor.T
    value, gradient = blocked_objective_gradient(v, K, q, .01, 3)
    legacy_value, legacy_gradient = objective_gradient(v, K, q, .01)
    direct_A = .01 * np.eye(6) + sum(w * np.outer(edge, edge) for w, edge in zip(q, v))
    assert abs(value - np.trace(solve(direct_A, K))) < 1e-10
    np.testing.assert_allclose(value, legacy_value, atol=1e-10)
    np.testing.assert_allclose(gradient, legacy_gradient, atol=1e-10)


def test_sparse_support_and_ess_are_distinguished_without_thresholding():
    q = np.array([.9, .1 - 1e-20, 1e-20, 0.])
    original = q.copy()
    diagnostics = distribution_diagnostics(q)
    assert diagnostics["positive_support"] == 3
    assert diagnostics["effective_sample_size"] < 2
    assert diagnostics["positive_below_1e_15"] == 1
    assert diagnostics["mass_95_coverage_pairs"] == 2
    assert not diagnostics["ess_is_wor_budget_limit"]
    np.testing.assert_array_equal(q, original)
    with pytest.raises(ValueError):
        distribution_diagnostics([-.1, 1.1])


def test_tiny_full_pool_three_designs_are_certified_and_budget_stays_user_owned(tmp_path):
    c = tiny_config(tmp_path)
    prepare_data(c, write_sources(c), tokenizer=FakeTokenizer())
    extract_features(c, scorer=deterministic_scorer)
    merge_features(c, 1)
    report = build_geometry(c)
    assert not report["human_labels_used"] and report["new_sft_responses_for_K"] == 0
    assert report["rows"] == 18
    for value in c["design"]["lambdas"]:
        result = solve_design(c, value)
        assert result["converged"]
        assert result["independent_relative_gap"] <= 1e-4
        assert result["objective_ratio_to_empirical_same_lambda"] <= 1 + 1e-8
        assert result["budget"] is None
    report = design_report(c)
    assert report["all_converged"] and report["budget"] is None
    assert not report["automatic_budget_selection"]
    Run(c).require("design-report")
