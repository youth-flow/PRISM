# V6: budget-capped PRISM Design

V6 is an isolated continuation of V5. It changes the PRISM Design feasible
region and the corresponding PRISM sampling law; the SFT model, HH data,
frozen EOS-2048 linear Reward, Reward fitting, PPO recipe, and evaluation
suites remain byte-for-byte equal to `configs/experiment.v5.yaml`.

The prepared first budget is `N=20,000`, extending the completed V5 budgets
above 14k. Preparing this branch does not start geometry, Reward training,
PPO, or paid evaluation.

## Optimization problem

For the unchanged trace-inverse Design objective, V6 solves

\[
  \min_q\; \mathrm{tr}\!\left(K\left(\lambda I +
  \sum_i q_i v_i v_i^\top\right)^{-1}\right),\qquad
  \sum_i q_i=1,\quad 0\le q_i\le 1/N.
\]

The upper bound makes `pi_i = N q_i` a valid first-order inclusion
probability. It also guarantees that every feasible Design has support on at
least `N` distinct pairs, eliminating the previous positive-support boundary
for larger budgets.

## Solver

V6 uses capped-simplex away-step Frank-Wolfe. The linear minimization oracle is
exact: it puts mass `1/N` on the `N` coordinates with the smallest current
gradient. Away steps remove obsolete active atoms, and exact scalar line search
is applied to the original objective. Acceptance never relies on an optimizer
status flag; the implementation recomputes the global capped-simplex
Frank-Wolfe gap in blocks and repeats the certificate with an independent LU
factorization before study freeze.

This is preferable here to generic constrained L-BFGS/SLSQP because the full
HH problem has roughly 160k probability variables, while the capped oracle is
closed-form and never projects or violates feasibility. Small numerical tests
compare the result against an independent SLSQP solution.

## Sampling adjustment

Empirical remains uniform sampling without replacement. PRISM no longer uses
the old sequential Plackett--Luce/Gumbel law in V6. Instead, it applies seeded
pairwise dependent rounding to `pi=Nq`:

- exactly `N` distinct pairs are returned;
- no replacement or post-hoc deduplication is used;
- every item's first-order inclusion probability is preserved in expectation;
- fixed zero/one inclusions remain fixed;
- a separate seeded permutation determines training order;
- an independent implementation replays and audits the exact selection.

Thus the sampling procedure implements the meaning of the new cap instead of
treating `q` as another sequential draw weight.

## Frozen V5 components

The V6 validator rejects any undeclared change outside the solver,
constraint declaration, PRISM sampling protocol, V6 identity/output, and
precommitted budget. In particular it preserves:

- OLMo-2-0425-1B-SFT revision `0d85a3d...ac27f` and HH revision
  `09be8c5b...a1fa`;
- Design lambdas and all other geometry settings;
- frozen bias-free linear Reward and common `L2=0.1`;
- formal PPO horizon 16,384 with the 65,536-episode learning-rate clock;
- Luna medium AB/BA, IFEval, and sequence-KL evaluation settings;
- paid API calls disabled in configuration.

## Entry points

Validate or inspect the plan without starting compute:

```powershell
prism-v6 validate
prism-v6 plan
```

Execution stages remain behind the normal explicit stage commands and V6
guards. Study freeze requires budget 20,000 and Design lambda 0.001; primary
linear PPO must stop at episode 16,384.

For efficient execution, `import-upstream` may hash-recertify only the V5
`data`, `features`, `geometry`, `geometry-audit`, and `shared-sft` stages. It
explicitly excludes every Design distribution and all sampling, Reward, PPO,
and evaluation artifacts because those are budget-dependent or downstream of
the new constrained solution.
