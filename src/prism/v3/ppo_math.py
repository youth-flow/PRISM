"""Small NumPy reference equations for the V3 single-card PPO executor.

The terminal-score/padding convention follows TRL 0.15.2.  These references
are also usable on the login/local host without importing Torch or any model.
"""
import math

import numpy as np


def learning_rate_multiplier(round_index, total_rounds, warmup_ratio=.05, minimum_ratio=.1):
    if not 1 <= round_index <= total_rounds or not 0 <= warmup_ratio < 1 or not 0 <= minimum_ratio <= 1:
        raise ValueError("Invalid PPO learning-rate horizon")
    warmup = math.ceil(total_rounds * warmup_ratio)
    if warmup and round_index <= warmup:
        return round_index / warmup
    progress = (round_index - warmup) / max(1, total_rounds - warmup)
    return minimum_ratio + (1 - minimum_ratio) * .5 * (1 + math.cos(math.pi * progress))


def gae_reference(rewards, values, gamma=1., lam=.95):
    rewards, values = np.asarray(rewards, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if rewards.shape != values.shape or rewards.ndim != 2:
        raise ValueError("GAE expects equal [batch,time] rewards and values")
    advantages = np.zeros_like(values)
    carry = np.zeros(len(values))
    for t in range(values.shape[1] - 1, -1, -1):
        following = values[:, t + 1] if t + 1 < values.shape[1] else 0.
        carry = rewards[:, t] + gamma * following - values[:, t] + gamma * lam * carry
        advantages[:, t] = carry
    return advantages, advantages + values


def value_statistics(predictions, targets):
    pred, target = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if pred.shape != target.shape or not pred.size or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Expected finite value predictions and targets")
    error = target - pred
    variance = float(target.var())
    return {"mse": float(np.mean(error ** 2)), "bias": float(np.mean(pred - target)),
            "target_variance": variance,
            "explained_variance": float(1 - error.var() / variance) if variance > 1e-12 else None,
            "constant_baseline_mse": variance}
