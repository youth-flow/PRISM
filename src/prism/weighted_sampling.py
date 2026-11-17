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
