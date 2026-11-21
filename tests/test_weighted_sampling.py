from collections import Counter
from itertools import permutations

import numpy as np
import pytest

from prism.weighted_sampling import fixed_size_dependent_rounding, weighted_without_replacement


def test_exact_prefix_replay_no_duplicates_and_zero_exclusion():
    weights = np.array([0., 1., 2., 0., 4., 3.])
    original = weights.copy()
    full = weighted_without_replacement(weights, 4, 17)
    assert set(full["ordered_indices"]) == {1, 2, 4, 5}
    assert len(set(full["ordered_indices"])) == 4
    assert full == weighted_without_replacement(weights, 4, 17)
    for size in range(5):
        short = weighted_without_replacement(weights, size, 17)
        for key in ("ordered_indices", "base_log_probabilities", "conditional_log_probabilities"):
            assert short[key] == full[key][:size]
    np.testing.assert_array_equal(weights, original)


def test_recorded_probabilities_match_direct_sequential_deletion():
    weights = np.array([.1, .3, .2, .4])
    sample = weighted_without_replacement(weights, 4, 23)
    remaining = weights.copy()
    for i, edge in enumerate(sample["ordered_indices"]):
        assert np.exp(sample["base_log_probabilities"][i]) == pytest.approx(weights[edge] / weights.sum())
        assert np.exp(sample["conditional_log_probabilities"][i]) == pytest.approx(remaining[edge] / remaining.sum())
        remaining[edge] = 0
    assert sample["conditional_log_probabilities"][-1] == 0
    assert sample["inclusion_probabilities"] is None
    # Selecting the entire support includes every item with probability one,
    # even though the initial categorical weights are unequal.
    assert not np.allclose(np.exp(sample["base_log_probabilities"]), 1)


def test_ordered_distribution_matches_exact_two_draw_law():
    weights = np.array([1., 2., 3.])
    trials = 12000
    counts = Counter(tuple(weighted_without_replacement(weights, 2, seed)["ordered_indices"])
                     for seed in range(trials))
    for first, second in permutations(range(3), 2):
        probability = weights[first] / weights.sum() * weights[second] / (weights.sum() - weights[first])
        standard_error = np.sqrt(probability * (1 - probability) / trials)
        assert abs(counts[first, second] / trials - probability) < 7 * standard_error + 1 / trials


def test_log_domain_survives_extreme_weight_ranges():
    sample = weighted_without_replacement([1e308, 1e-320, 2e-320], 3, 19)
    assert sample["ordered_indices"][0] == 0
    assert np.isfinite(sample["base_log_probabilities"]).all()
    assert np.isfinite(sample["conditional_log_probabilities"]).all()
    assert sample["conditional_log_probabilities"][-1] == 0


def test_common_weight_scale_does_not_change_sample_law_or_probabilities():
    weights = np.array([1., 2., 3., 4.])
    a = weighted_without_replacement(weights, 3, 7)
    b = weighted_without_replacement(weights * 1e200, 3, 7)
    assert a["ordered_indices"] == b["ordered_indices"]
    np.testing.assert_allclose(a["conditional_log_probabilities"], b["conditional_log_probabilities"], atol=1e-12)


def test_does_not_consume_global_numpy_rng():
    original = np.random.get_state()
    try:
        np.random.seed(492)
        before = np.random.get_state()
        weighted_without_replacement([1., 2., 3.], 2, 17)
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
    finally:
        np.random.set_state(original)


def test_dependent_rounding_is_exact_replay_fixed_size_and_nonmutating():
    probabilities = np.array([1., .8, .7, .3, .2, 0.])
    original = probabilities.copy()
    sample = fixed_size_dependent_rounding(probabilities, 31)
    assert sample == fixed_size_dependent_rounding(probabilities, 31)
    assert sample["size"] == 3
    assert len(sample["ordered_indices"]) == 3
    assert len(set(sample["ordered_indices"])) == 3
    assert 0 in sample["ordered_indices"]
    assert 5 not in sample["ordered_indices"]
    assert sample["fixed_size"] and not sample["replacement"]
    np.testing.assert_array_equal(probabilities, original)


def test_dependent_rounding_preserves_first_order_marginals():
    probabilities = np.array([1., .8, .7, .3, .2, 0.])
    trials = 8000
    counts = np.zeros(len(probabilities), dtype=np.int64)
    for seed in range(trials):
        counts[fixed_size_dependent_rounding(probabilities, seed)["ordered_indices"]] += 1
    standard_errors = np.sqrt(probabilities * (1 - probabilities) / trials)
    assert np.all(np.abs(counts / trials - probabilities) < 7 * standard_errors + 2 / trials)


def test_dependent_rounding_does_not_consume_global_numpy_rng():
    original = np.random.get_state()
    try:
        np.random.seed(871)
        before = np.random.get_state()
        fixed_size_dependent_rounding([.8, .7, .3, .2], 17)
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
    finally:
        np.random.set_state(original)


@pytest.mark.parametrize("weights,size,seed", [
    ([], 0, 1), ([0., 0.], 0, 1), ([1., -1.], 1, 1),
    ([np.nan, 1.], 1, 1), ([np.inf, 1.], 1, 1), ([[1., 2.]], 1, 1),
    ([1., 0.], 2, 1), ([1., 2.], -1, 1), ([1., 2.], True, 1),
    ([1., 2.], 1, -1), ([1., 2.], 1, True),
])
def test_rejects_invalid_inputs_without_changing_distribution(weights, size, seed):
    with pytest.raises(ValueError):
        weighted_without_replacement(weights, size, seed)


@pytest.mark.parametrize("probabilities,seed", [
    ([], 1), ([.5, .5, .5], 1), ([1.1, -.1], 1),
    ([np.nan, 1.], 1), ([np.inf, 0.], 1), ([[.5, .5]], 1),
    ([.5, .5], -1), ([.5, .5], True),
])
def test_dependent_rounding_rejects_invalid_inputs(probabilities, seed):
    with pytest.raises(ValueError):
        fixed_size_dependent_rounding(probabilities, seed)
