# V3 14k frozen-linear Reward PPO results (2026-11-20)

## Scope and frozen recipe

This experiment compares the original V3 `prism` and `empirical` preference
subsets at the same user-selected budget of 14,000 pairs.  Each policy uses its
method-specific frozen SFT-backbone plus linear Reward head (selected L2 0.1),
with an independent SFT-backbone Critic LoRA and normalized linear value head.
Reward parameters never share trainable storage with the Critic.

The only executor changes from the V3 recipe are the Reward loader, Critic
initialization, four-GPU homogeneous distributed execution, and the explicitly
selected formal stop at 16,384 episodes.  The scheduler deliberately retains
the original 65,536-episode trajectory.  The unchanged scientific parameters
are global rollout 128, minibatch 64, four PPO epochs, Actor LR `1e-5`, Critic
LoRA/head LR `3e-5`/`1e-4`, KL coefficient 0.05, FP32 with TF32 disabled, and a
512-token response cap.  Each method ran on four RTX 4090 GPUs concurrently.

## PPO audit

Both policies completed 16,384 episodes, 128 rollout rounds, and 1,024 real
Adam updates for each of Actor, Critic LoRA, and Critic head.  The Reward base
and head remained bitwise frozen; checkpoint finiteness, optimizer counters,
parameter coverage, distributed synchronization, and fixed train/dev probes
passed independently.

| Method | 16k dev Reward delta | 16k dev KL-adjusted objective delta | Dev sequence KL | Critic EV | Final checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| Empirical | +1.682819 | +1.053022 | 12.595945 | 0.655023 | `3d3873976d73ccdd80066da154d1579ea5dac3bbd2279d8d240e0aee3d092f2a` |
| PRISM | +1.376111 | +0.840096 | 10.720301 | 0.662630 | `9c7fb1630f157d544c0791ff9db9287ab64952a003830b75dd0409f50b72c093` |

The two Reward scales are method-specific, so their internal deltas are not a
valid cross-method quality comparison.  The fixed external evaluation below is
the comparison that matters.

## Final HH preference evaluation

Luna medium judged the fixed, previously unseen 500-prompt HH evaluation set.
Every comparison used both AB and BA order, ties scored 0.5, for 3,000 valid
judgments total.  Intervals are 2,000-draw prompt bootstraps conditional on the
single training seed.

| Comparison (left wins) | Win rate | 95% CI | Helpful | Harmless |
|---|---:|---:|---:|---:|
| PRISM vs Empirical | **59.20%** | **55.35–63.15%** | 56.76% | 65.86% |
| PRISM vs SFT | **71.35%** | **67.75–75.10%** | 71.65% | 70.52% |
| Empirical vs SFT | **66.90%** | **63.05–70.65%** | 67.55% | 65.11% |

All three intervals are wholly above 50%.  Therefore the experiment gives
positive evidence for both claims that a 14k frozen linear Reward can drive a
useful HH PPO update and that PRISM sampling outperforms Empirical sampling at
the same preference-label budget.

## Generation drift and IFEval

| Model | Sequence KL vs SFT | Mean response tokens | Truncation | IFEval strict prompt | Delta vs SFT (95% CI) |
|---|---:|---:|---:|---:|---:|
| SFT | 0 | 109.236 | 2.4% | 45.47% | — |
| Empirical PPO | 12.7732 | 185.390 | 1.2% | 41.96% | -3.51pp (-7.39, +0.18) |
| PRISM PPO | 10.5526 | 167.810 | 1.2% | 40.48% | -4.99pp (-9.06, -1.29) |

The HH preference improvement is not free: both policies become longer and
lose IFEval instruction-following accuracy.  Empirical's strict-prompt decline
is not significant at 95%; PRISM's is.  The correct conclusion is therefore
task-conditional: PRISM is the better 14k HH Reward/PPO path, but this 16k PPO
checkpoint should not be presented as an unconditional general-capability
upgrade.  A later experiment that wants a better Pareto point should predefine
stronger KL/instruction-retention gates or externally evaluate earlier PPO
checkpoints; these results do not authorize selecting them after the fact.

## Reproducibility and accounting

- PPO jobs: Empirical `1994613`, PRISM `1994617`; independent audits
  `1994620` and `1994621`, all `COMPLETED/0`.
- Final Judge import/aggregate: `1995168`/`1995169`, both `COMPLETED/0`.
- Aggregate JSON SHA256:
  `eb7a7e79f5c89a3055155aed70fdc371f19d1c87af2a73c790ece4030dc04cce`.
- Aggregate receipt SHA256:
  `5cf5a6630416d52c40c18d0fb7e4d2c1c7530a7a610b26967715d9a0860f62a2`.
- All 48 top-level jobs completed with exit zero.  The full chain consumed
  52.884167 GPU-card hours and 409.581667 CPU-core hours from
  2026-11-19 17:40:50 to 2026-11-20 00:54:15 China Standard Time.
- Luna: 3,000 valid judgments from 3,002 dispatches; two invalid attempts were
  retried.  Known cost is USD 0.364637 with USD 0.007640 unresolved reservation,
  below the isolated USD 100 hard cap.  There are no orphan dispatches.
- Receipt-verified local copies are under
  `local/v3-execution/lora-ppo/linear-final`; runtime artifacts remain ignored
  and are not committed.
