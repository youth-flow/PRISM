"""Joint solver: original objective, global certificate, and cache provenance."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

from prism import design as numerics, pipeline
from prism.io import digest, receipt, root, verify_receipt, write_rows
from test_contracts import fixture_config


def test_normalized_gradient_and_scale_invariance():
    rng = np.random.default_rng(71)
    v, factor = rng.normal(size=(13, 4)), rng.normal(size=(4, 3))
    K, weights = factor @ factor.T, rng.uniform(.2, 2, size=13)
    value, gradient, q, _ = numerics._normalized_objective(v, K, weights, .1)
    np.testing.assert_allclose(value, numerics.objective_gradient(v, K, q, .1)[0])
    np.testing.assert_allclose(numerics._normalized_objective(v, K, weights * 37, .1)[0], value)
    for i in range(len(weights)):
        delta = np.eye(len(weights))[i] * 1e-6
        numeric = (numerics._normalized_objective(v, K, weights + delta, .1)[0]
                   - numerics._normalized_objective(v, K, weights - delta, .1)[0]) / 2e-6
        np.testing.assert_allclose(gradient[i], numeric, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("seed", range(10))
def test_joint_design_against_independent_original_problem(seed):
    rng = np.random.default_rng(seed)
    v, factor = rng.normal(size=(20, 5)), rng.normal(size=(5, 1 if seed % 2 else 5))
    K, initial = factor @ factor.T, np.full(20, .05)
    # This reference uses neither the normalized coordinates nor our kernel.
    def direct(q):
        A = .1 * np.eye(5)
        for mass, edge in zip(q, v):
            A += mass * np.outer(edge, edge)
        return float(np.trace(np.linalg.solve(A, K)))
    reference = minimize(direct, initial, method="SLSQP", bounds=[(0., 1.)] * 20,
                         constraints={"type": "eq", "fun": lambda q: q.sum() - 1},
                         options={"maxiter": 2000, "ftol": 1e-11})
    q, report = numerics.joint_design(v, K, initial, .1, 500, 1e-4)
    assert reference.success and report["converged"]
    np.testing.assert_allclose(report["objective"], reference.fun, rtol=1e-4, atol=1e-8)
    A = .1 * np.eye(5) + v.T @ (q[:, None] * v)
    solved = np.linalg.solve(A, v.T)
    gradient = -np.sum(solved * (K @ solved), axis=0)
    gap = float(q @ gradient - gradient.min())
    assert gap <= 1e-4 * direct(q)
    np.testing.assert_allclose(report["fw_gap"], gap, atol=1e-10)
    assert q.min() >= 0 and abs(q.sum() - 1) < 1e-12


def test_zero_target_and_boundary_mass_have_no_probability_floor():
    q, report = numerics.joint_design(np.eye(2), np.zeros((2, 2)), [.25, .75], .1)
    assert report["converged"] and report["objective"] == report["fw_gap"] == 0
    np.testing.assert_array_equal(q, [.25, .75])
    q, report = numerics.joint_design(np.eye(2), np.diag([1., 0.]), [.5, .5], .1)
    assert report["converged"]
    np.testing.assert_allclose(q, [1., 0.], atol=1e-12)


def test_optimizer_success_flag_is_not_a_design_certificate(monkeypatch):
    monkeypatch.setattr(numerics, "minimize", lambda fun, x0, **kw:
                        SimpleNamespace(x=x0, success=True, message="optimizer says success"))
    q, report = numerics.joint_design(np.eye(2), np.diag([1., 0.]), [.5, .5], .1)
    assert not report["converged"]
    assert report["optimizer_message"] == "optimizer says success"


def archived_attempt(tmp_path):
    c = fixture_config(tmp_path)
    c["methods"] = ["empirical", "prism"]
    c["design"].update(fw_max_iter=1, fw_relative_gap=1e-14)
    p = root(c)
    rng = np.random.default_rng(39)
    np.savez(p / "pool.npz", phi=rng.normal(size=(8, 4, 3)), logp=rng.normal(size=(8, 4)))
    write_rows(p / "pool.jsonl", [])
    receipt(p / "pool.receipt.json", [p / "pool.npz", p / "pool.jsonl", p / "experiment.json"])
    pipeline.design(c)
    archive = p / "design_attempts/original"
    archive.mkdir(parents=True)
    for name in ("design.npz", "design.report.json", "design.receipt.json"):
        (p / name).rename(archive / name)
    return c, p, archive


def test_cached_geometry_is_exact_and_bound_to_original_pool(tmp_path, monkeypatch):
    c, p, archive = archived_attempt(tmp_path)
    config_hash = digest(p / "experiment.json")
    with np.load(archive / "design.npz", allow_pickle=False) as a:
        original = {k: a[k] for k in ("index", "v", "b", "proxy_probabilities", "K", "empirical")}
    monkeypatch.setattr(pipeline, "proxy", lambda *a: pytest.fail("Cached b/K must not be recomputed"))
    pipeline.design(c, solver="joint-lbfgs", geometry=archive / "design.npz", warm_start=True)
    assert digest(p / "experiment.json") == config_hash
    with np.load(p / "design.npz", allow_pickle=False) as a:
        for name, values in original.items():
            np.testing.assert_array_equal(a[name], values)
    metadata = verify_receipt(p / "design.receipt.json")
    assert metadata["geometry_reused"] and metadata["warm_start"]
    assert str(archive / "design.npz") in metadata["input_sha256"]
    # Deliberately inadequate numerical budget still cannot write label queries.
    with pytest.raises(RuntimeError, match="FW did not converge"):
        pipeline.sample(c, "prism", 1000, 7)
    assert not (p / "prism/n1000/seed7/queries.jsonl").exists()


@pytest.mark.parametrize("changed", ["design.npz", "design.report.json", "pool.receipt.json"])
def test_cached_geometry_refuses_changed_source_or_pool(tmp_path, changed):
    c, p, archive = archived_attempt(tmp_path)
    path = p / changed if changed == "pool.receipt.json" else archive / changed
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed"):
        pipeline.design(c, solver="joint-lbfgs", geometry=archive / "design.npz", warm_start=True)
    assert not (p / "design.npz").exists()
