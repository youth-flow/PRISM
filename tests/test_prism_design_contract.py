"""Independent identities and analytic optima for the PRISM design core."""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from prism import pipeline
from prism.design import _line_objective, frank_wolfe, objective_gradient, proxy
from prism.io import receipt, root, write_rows
from test_contracts import fixture_config


def test_proxy_against_augmented_least_squares_and_pairwise_covariance_identity():
    rng = np.random.default_rng(107)
    phi, ell, ridge = rng.normal(size=(5, 4, 6)), rng.normal(size=(5, 4)), 0.07
    # Rank-deficient raw features still have a unique ridge solution.
    phi[..., -1] = phi[..., 0]
    b, p, K = proxy(phi, ell, ridge)
    X, target = phi.reshape(-1, 6), (ell-ell.mean(axis=1, keepdims=True)).ravel()
    reference = np.linalg.lstsq(np.vstack((X, np.sqrt(ridge)*np.eye(6))),
                                 np.concatenate((target, np.zeros(6))), rcond=None)[0]
    np.testing.assert_allclose(b, reference, atol=1e-12)
    pairwise = np.zeros((6, 6))
    for i in range(len(phi)):
        for a, other in combinations(range(phi.shape[1]), 2):
            diff = phi[i, a]-phi[i, other]
            pairwise += p[i, a]*p[i, other]*np.outer(diff, diff)/len(phi)
    np.testing.assert_allclose(K, pairwise, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_fw_matches_closed_form_and_handles_a_drop_step_without_coverage_floor():
    v, K, ridge = np.diag([2., 1.]), np.diag([3., 1.]), 0.1
    # Min 3/(lambda+4q) + 1/(lambda+1-q), q in [0,1].
    a, b = np.sqrt(12.), 1.
    qstar = (a*(ridge+1)-b*ridge)/(a+4*b)
    q, report = frank_wolfe(v, K, [0.2, 0.8], ridge, relative_gap=1e-6)
    np.testing.assert_allclose(q, [qstar, 1-qstar], atol=2e-7)
    assert report["converged"]
    initial_value, grad = objective_gradient(v, K, np.array([.2, .8]), ridge)
    assert initial_value-report["objective"] <= np.array([.2, .8]) @ grad-grad.min()+1e-10
    # The optimum has zero mass on edge 1. A valid away/drop step must allow it.
    boundary, report = frank_wolfe(np.eye(2), np.diag([1., 0.]), [.999, .001], ridge)
    np.testing.assert_allclose(boundary, [1, 0], atol=1e-12)
    assert report["converged"]
    zero, report = frank_wolfe(np.eye(2), np.zeros((2, 2)), [.25, .75], ridge)
    np.testing.assert_array_equal(zero, [.25, .75])
    assert report["objective"] == report["fw_gap"] == 0 and report["converged"]


def test_fw_matches_independent_slsqp_on_rank_deficient_and_duplicate_edges():
    for seed in range(12):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(10, 4))
        v[1:3] = v[0]
        factor = rng.normal(size=(4, 1 if seed % 2 else 4))
        K, ridge, initial = factor @ factor.T, .05, np.full(10, .1)
        def direct(q):
            A = ridge*np.eye(4)
            for mass, edge in zip(q, v):
                A += mass*np.outer(edge, edge)
            return float(np.trace(np.linalg.solve(A, K)))
        reference = minimize(direct, initial, method="SLSQP", bounds=[(0, 1)]*10,
                             constraints={"type": "eq", "fun": lambda q: q.sum()-1},
                             options={"maxiter": 2000, "ftol": 1e-10})
        assert reference.success
        q, report = frank_wolfe(v, K, initial, ridge, max_iter=2000, relative_gap=1e-6)
        assert report["converged"]
        assert abs(direct(q)-reference.fun) <= 2e-6*max(1., abs(reference.fun))
        assert np.all(np.diff([r["objective"] for r in report["history"]]) <= 1e-9)


def test_unconverged_design_cannot_spend_labels(tmp_path):
    c = fixture_config(tmp_path)
    c["methods"] = ["empirical", "prism"]
    c["design"].update(fw_max_iter=1, fw_relative_gap=1e-14)
    p = root(c)
    rng = np.random.default_rng(39)
    np.savez(p/"pool.npz", phi=rng.normal(size=(8, 4, 3)), logp=rng.normal(size=(8, 4)))
    write_rows(p/"pool.jsonl", [])
    receipt(p/"pool.receipt.json", [p/"pool.npz", p/"pool.jsonl", p/"experiment.json"])
    pipeline.design(c)
    import pytest
    with pytest.raises(RuntimeError, match="FW did not converge"):
        pipeline.sample(c, "prism", 1000, 7)
    assert not (p/"prism/n1000/seed7/queries.jsonl").exists()


def test_dense_initializer_drops_redundant_mass_with_frozen_iteration_budget():
    # Thousands of initially positive zero edges must not consume one away step
    # each. Compare with the analytic two-informative-edge optimum.
    v = np.vstack((100 * np.eye(2), np.zeros((18000, 2))))
    K, ridge = np.diag([1., 4.]), .01
    q, report = frank_wolfe(v, K, np.full(len(v), 1 / len(v)), ridge, 500, 1e-4)
    qstar = (1 + 2 * ridge / 10000) / 3 - ridge / 10000
    assert report["converged"] and len(report["history"]) <= 500
    np.testing.assert_allclose(q[:2], [qstar, 1-qstar], atol=1e-6)
    assert q[2:].sum() < 1e-10
    assert any(r.get("direction") == "away_initial" for r in report["history"])
    value, gradient = objective_gradient(v, K, q, ridge)
    assert q @ gradient - gradient.min() <= 1e-4 * value


def test_dense_atom_line_matches_direct_inverse_in_both_directions():
    rng = np.random.default_rng(907)
    v = rng.normal(size=(20, 5))
    initial, q = rng.dirichlet(np.ones(20)), rng.dirichlet(np.ones(20))
    factor = rng.normal(size=(5, 3))
    K, ridge = factor @ factor.T, .3
    mixture, current = v.T @ (initial[:, None]*v), v.T @ (q[:, None]*v)
    A = ridge*np.eye(5) + current
    line = _line_objective(A, K, ridge, mixture=mixture)
    for gamma in (-.1, 0, .25, .9):
        direct = float(np.trace(np.linalg.solve(A + gamma*(mixture-current), K)))
        np.testing.assert_allclose(line(gamma), direct, rtol=1e-12, atol=1e-12)
