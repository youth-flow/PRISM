import numpy as np
import pytest
from scipy.optimize import minimize

from prism.design import (capped_away_frank_wolfe, capped_simplex_gap,
                          objective_gradient)
from prism.v3.reward import fit_rewards
from prism.v3.study import independent_design_audit
from prism.v3.study import freeze_study, sample
from prism.v3.training_audit import audit_training
from v3_helpers import configuration, numerical_run


@pytest.mark.parametrize("seed", [3, 19, 41])
def test_capped_away_frank_wolfe_matches_independent_constrained_solver(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(18, 5))
    target = rng.normal(size=(5, 5))
    K = target.T @ target / 5
    budget = 7
    initial = np.full(len(v), 1 / len(v))
    q, report = capped_away_frank_wolfe(
        v, K, initial, .07, budget, max_iter=1500, relative_gap=2e-7,
        block_size=5)

    def objective(q_value):
        return objective_gradient(v, K, q_value, .07)

    reference = minimize(
        objective, initial, jac=True, method="SLSQP",
        bounds=[(0, 1 / budget)] * len(v),
        constraints={"type": "eq", "fun": lambda value: value.sum() - 1,
                     "jac": lambda value: np.ones_like(value)},
        options={"ftol": 1e-12, "maxiter": 3000})
    assert reference.success, reference.message
    assert report["converged"]
    assert report["solver"] == "capped_simplex_away_step_frank_wolfe"
    assert q.sum() == pytest.approx(1, abs=1e-10)
    assert q.min() >= -1e-12
    assert q.max() <= 1 / budget + 1e-12
    assert np.count_nonzero(q > 1e-12) >= budget
    assert report["objective"] == pytest.approx(reference.fun, rel=3e-6, abs=2e-8)
    value, gradient = objective_gradient(v, K, q, .07)
    gap, indices = capped_simplex_gap(q, gradient, budget)
    assert gap == pytest.approx(report["fw_gap"], abs=1e-12)
    assert len(indices) == budget
    assert gap <= 2e-7 * value
    audited = independent_design_audit(v, K, q, .07, 2e-7, budget=budget)
    assert audited["passed"] and audited["budget"] == budget


def test_capped_design_handles_zero_target_and_rejects_infeasible_inputs():
    v = np.eye(5)
    q, report = capped_away_frank_wolfe(
        v, np.zeros((5, 5)), np.full(5, .2), .1, 3)
    np.testing.assert_allclose(q, .2)
    assert report["converged"] and report["objective"] == 0
    with pytest.raises(ValueError, match="outside"):
        capped_away_frank_wolfe(v, np.eye(5), [1, 0, 0, 0, 0], .1, 3)
    with pytest.raises(ValueError, match="budget-capped simplex"):
        capped_simplex_gap(np.array([.5, .125, .125, .125, .125]),
                           np.arange(5.), 3)


def test_v6_sampling_and_independent_training_audit_end_to_end(tmp_path):
    c = configuration(tmp_path)
    c["v6"] = {"preference_budget": 4}
    c["sampling"]["protocol"] = "v6_budget_capped_exact_without_replacement_v1"
    numerical_run(c)
    frozen = freeze_study(c, 4, user_confirmed=True)
    assert frozen["independent_design_audit"]["feasible_region"] == "budget_capped_simplex"
    summary = sample(c)
    assert summary["unique_pairs"] == {"empirical": 4, "prism": 4}
    assert summary["method_protocols"]["empirical"] == "weighted_without_replacement_gumbel_v1"
    assert summary["method_protocols"]["prism"] == "fixed_size_dependent_rounding_v1"
    fit_rewards(c)
    verified = audit_training(c)
    assert verified["passed"]
    assert verified["sampling"]["prism"]["sampling_protocol"] == "fixed_size_dependent_rounding_v1"
