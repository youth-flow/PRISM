"""Finite-pool weighted sampling without replacement for V3 preparation.

Gumbel ranking produces the ordered law of repeatedly selecting in proportion
to the remaining weights (Plackett--Luce), not an inclusion-probability design.
For fixed pool/weights/seed, different budgets share an exact ordered prefix.
This module does not change the frozen V2 pipeline or read preference labels.

Reference: https://proceedings.mlr.press/v97/kool19a.html
"""
import hashlib

import numpy as np


PROTOCOL = "weighted_without_replacement_gumbel_v1"
DEPENDENT_ROUNDING_PROTOCOL = "fixed_size_dependent_rounding_v1"


def weighted_without_replacement(weights, size, seed):
    """Return a JSON-safe ordered sample and its sequential probabilities.

    Zero weights are excluded. Insufficient positive support is an error;
    no floor, uniform mixing, replacement, or post-hoc deduplication is used.
    Logs retain probabilities beyond the range of ordinary float arithmetic.
    ``base_log_probabilities`` describe the normalized *initial* weights;
    ``conditional_log_probabilities`` describe each actual sequential draw.
    Neither field contains marginal inclusion probabilities.
    """
    if type(size) is not int or size < 0:
        raise ValueError("size must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be a nonempty finite nonnegative vector")
    positive = np.flatnonzero(w > 0)
    support = len(positive)
    if not support:
        raise ValueError("At least one weight must be positive")
    if size > support:
        raise ValueError(f"Requested {size} distinct pairs but positive-weight support is {support}")

    log_weights = np.log(w[positive])
    noise = np.random.default_rng(seed).gumbel(size=support)
    if not np.isfinite(noise).all():
        raise RuntimeError("Nonfinite random key; preserve the seed and inspect the RNG")
    order = positive[np.argsort(-(log_weights + noise), kind="stable")]
    ordered_log_weights = np.log(w[order])
    # The full ranking lets us sum remaining mass once, without cancellation
    # when a large weight is removed and tiny positive weights remain.
    remaining_log_mass = np.logaddexp.accumulate(ordered_log_weights[::-1])[::-1]
    selected = order[:size]
    base = ordered_log_weights[:size] - remaining_log_mass[0]
    conditional = ordered_log_weights[:size] - remaining_log_mass[:size]
    if not np.isfinite(base).all() or not np.isfinite(conditional).all():
        raise RuntimeError("Invalid sampled probability accounting")
    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "size": size,
        "pool_size": len(w),
        "positive_support": support,
        "weights_sha256": hashlib.sha256(w.astype("<f8", copy=False).tobytes()).hexdigest(),
        "ordered_indices": selected.tolist(),
        "base_log_probabilities": base.tolist(),
        "conditional_log_probabilities": conditional.tolist(),
        "inclusion_probabilities": None,
    }


def fixed_size_dependent_rounding(inclusion_probabilities, seed):
    """Round first-order inclusion probabilities to one fixed-size sample.

    The input must be in [0,1] and sum to an integer N.  Randomized pairwise
    dependent rounding preserves every first-order marginal in expectation,
    preserves the sum on every update, and returns exactly N distinct items.
    The final selected set is put in a separate seeded random order so Reward
    training does not inherit pool-index order.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    original = np.asarray(inclusion_probabilities, dtype=np.float64)
    if (original.ndim != 1 or not len(original) or not np.isfinite(original).all()
            or np.any(original < -1e-12) or np.any(original > 1 + 1e-12)):
        raise ValueError("inclusion probabilities must be a finite vector in [0,1]")
    total = float(original.sum(dtype=np.float64))
    size = int(round(total))
    if not np.isclose(total, size, rtol=0, atol=1e-9) or not 0 <= size <= len(original):
        raise ValueError("inclusion probabilities must sum to an integer sample size")
    probabilities = np.clip(original.copy(), 0.0, 1.0)
    probabilities[probabilities < 1e-14] = 0.0
    probabilities[probabilities > 1 - 1e-14] = 1.0
    rng = np.random.default_rng(seed)
    pending = np.flatnonzero((probabilities > 0) & (probabilities < 1)).tolist()
    rng.shuffle(pending)
    random_updates = 0
    while len(pending) >= 2:
        i, j = pending.pop(), pending.pop()
        alpha = min(1 - probabilities[i], probabilities[j])
        beta = min(probabilities[i], 1 - probabilities[j])
        denominator = alpha + beta
        if denominator <= 0 or not np.isfinite(denominator):
            raise RuntimeError("Dependent rounding reached an invalid fractional pair")
        if rng.random() < beta / denominator:
            probabilities[i] += alpha
            probabilities[j] -= alpha
        else:
            probabilities[i] -= beta
            probabilities[j] += beta
        random_updates += 1
        for index in (i, j):
            if probabilities[index] < 1e-12:
                probabilities[index] = 0.0
            elif probabilities[index] > 1 - 1e-12:
                probabilities[index] = 1.0
            else:
                pending.append(index)
    if pending:
        index = pending.pop()
        required = size - int(np.count_nonzero(probabilities == 1))
        if required not in (0, 1) or not np.isclose(probabilities[index], required, atol=1e-9):
            raise RuntimeError("Dependent rounding left a nonintegral residual")
        probabilities[index] = float(required)
    selected = np.flatnonzero(probabilities == 1)
    if len(selected) != size or np.any((probabilities != 0) & (probabilities != 1)):
        raise RuntimeError("Dependent rounding did not return the fixed sample size")
    ordered = rng.permutation(selected)
    return {
        "protocol": DEPENDENT_ROUNDING_PROTOCOL,
        "seed": seed,
        "size": size,
        "pool_size": len(original),
        "inclusion_probabilities_sha256": hashlib.sha256(
            original.astype("<f8", copy=False).tobytes()).hexdigest(),
        "ordered_indices": ordered.tolist(),
        "selected_inclusion_probabilities": original[ordered].tolist(),
        "marginals": "first_order_preserved_in_expectation",
        "fixed_size": True,
        "replacement": False,
        "ordering": "seeded_uniform_permutation_after_rounding",
        "random_pair_updates": random_updates,
    }
