import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from prism.design import edges, empirical, frank_wolfe, objective_gradient, proxy
from prism.reward import fit_bt


def test_proxy_matches_pdf_normal_equations_and_covariance():
    rng = np.random.default_rng(2)
    phi, ell = rng.normal(size=(7, 4, 3)), rng.normal(size=(7, 4))
    b, p, K = proxy(phi, ell, 0.3)
    X = phi.reshape(-1, 3)
    target = (ell - ell.mean(axis=1, keepdims=True)).ravel()
    np.testing.assert_allclose((X.T @ X + 0.3*np.eye(3)) @ b, X.T @ target)
    manual = np.zeros((3, 3))
    for f, mass in zip(phi, p):
        centered = f - mass @ f
        manual += centered.T @ (mass[:, None] * centered) / len(phi)
    np.testing.assert_allclose(K, manual)
    assert np.linalg.eigvalsh(K).min() >= -1e-12
    # Prompt shifts in ell disappear; feature shifts do NOT silently add an intercept.
    np.testing.assert_allclose(proxy(phi, ell + np.arange(7)[:, None], 0.3)[0], b)


def test_fw_gradient_gap_and_independent_solver():
    rng = np.random.default_rng(10)
    v = rng.normal(size=(8, 3))
    K = np.diag([0.3, 1, 5])
    q = np.full(8, 1/8)
    val, grad = objective_gradient(v, K, q, 0.2)
    eps = 1e-6
    for i in range(8):
        delta = np.eye(8)[i]*eps
        numeric = (objective_gradient(v, K, q+delta, 0.2)[0] - objective_gradient(v, K, q-delta, 0.2)[0])/(2*eps)
        np.testing.assert_allclose(grad[i], numeric, rtol=1e-6)
    actual, report = frank_wolfe(v, K, q, 0.2, 1000, 1e-5)
    reference = minimize(lambda z: objective_gradient(v, K, z, 0.2), q, jac=True, method="SLSQP", bounds=[(0, 1)]*8, constraints={"type": "eq", "fun": lambda z: z.sum()-1, "jac": lambda z: np.ones(8)}, options={"ftol": 1e-12, "maxiter": 1000})
    assert reference.success
    assert report["objective"] - reference.fun < 2e-4
    assert report["objective"] <= val
    assert actual.min() >= 0 and np.isclose(actual.sum(), 1)
    assert np.all(np.diff([r["objective"] for r in report["history"]]) <= 1e-9)
    _, final_grad = objective_gradient(v, K, actual, 0.2)
    np.testing.assert_allclose(report["fw_gap"], actual @ final_grad - final_grad.min())


def test_empirical_weights_prompts_not_unequal_edge_counts():
    idx = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 2], [1, 1, 2]])
    np.testing.assert_allclose(empirical(idx), [1/2, 1/6, 1/6, 1/6])


def test_edge_orientation_and_bt_recovery():
    idx, v = edges(np.arange(12).reshape(2, 3, 2))
    np.testing.assert_array_equal(idx[0], [0, 0, 1])
    np.testing.assert_array_equal(v[0], [-2, -2])
    rng = np.random.default_rng(41)
    X = rng.normal(size=(20000, 3))
    truth = np.array([0.4, -0.7, 1.0])
    labels = rng.binomial(1, expit(X @ truth))
    w, report = fit_bt(X, labels)
    np.testing.assert_allclose(w, truth, atol=0.06)
    assert report["regularization"] == 0
    inverse, _ = fit_bt(-X, 1-labels)
    np.testing.assert_allclose(inverse, w, atol=1e-7)
