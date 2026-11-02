"""PDF A3-A5: target-centered loglinear proxy and damped optimal design."""
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def proxy(phi, logp, ridge):
    # phi [prompts,candidates,hidden], logp [prompts,candidates].
    phi, logp = np.asarray(phi, dtype=np.float64), np.asarray(logp, dtype=np.float64)
    if phi.ndim != 3 or logp.shape != phi.shape[:2] or ridge <= 0 or not np.isfinite(phi).all() or not np.isfinite(logp).all():
        raise ValueError("Invalid proxy inputs")
    X = phi.reshape(-1, phi.shape[-1])
    t = (logp - logp.mean(axis=1, keepdims=True)).ravel()
    # EXACT PDF formula: center targets only. No implicit intercept/PCA/whitening.
    b = cho_solve(cho_factor(X.T @ X + ridge * np.eye(X.shape[1])), X.T @ t)
    p = softmax(phi @ b, axis=1)
    centered = phi - np.einsum("ij,ijd->id", p, phi)[:, None, :]
    K = np.einsum("ij,ijd,ije->de", p, centered, centered) / len(phi)
    return b, p, (K + K.T) / 2


def edges(phi):
    index = np.array([(i, a, b) for i in range(len(phi)) for a, b in combinations(range(phi.shape[1]), 2)], dtype=np.int64)
    v = phi[index[:, 0], index[:, 1]] - phi[index[:, 0], index[:, 2]]
    return index, np.asarray(v, dtype=np.float64)


def empirical(index):
    # Empirical prompt mass 1/M; uniform unordered edges conditional on prompt.
    ids, counts = np.unique(index[:, 0], return_counts=True)
    weights = dict(zip(ids, 1 / (len(ids) * counts)))
    return np.array([weights[i] for i in index[:, 0]], dtype=np.float64)


def objective_gradient(v, K, q, ridge):
    A = ridge * np.eye(v.shape[1]) + v.T @ (q[:, None] * v)
    inv = cho_solve(cho_factor(A), np.eye(len(A)))
    B = inv @ K @ inv
    sensitivity = np.einsum("ij,jk,ik->i", v, B, v)
    return float(np.sum(K * inv.T)), -sensitivity


def frank_wolfe(v, K, initial, ridge, max_iter=500, relative_gap=1e-4):
    v, K, q = np.asarray(v), np.asarray(K), np.asarray(initial, dtype=np.float64).copy()
    if ridge <= 0 or max_iter < 1 or relative_gap <= 0 or np.any(q < 0) or not np.isclose(q.sum(), 1):
        raise ValueError("Invalid FW inputs")
    if K.shape != (v.shape[1], v.shape[1]) or not np.allclose(K, K.T) or np.linalg.eigvalsh(K).min() < -1e-8 * max(1, np.linalg.norm(K, 2)):
        raise ValueError("K must be positive semidefinite")
    history = []
    for iteration in range(max_iter):
        value, grad = objective_gradient(v, K, q, ridge)
        vertex = int(grad.argmin())
        gap = max(0.0, float(q @ grad - grad[vertex]))
        history.append({"iteration": iteration, "objective": value, "gap": gap})
        if gap <= relative_gap * max(value, 1e-15):
            break
        # Away-step Frank-Wolfe removes unhelpful mass faster than vanilla FW.
        # It optimizes the SAME simplex objective; no coverage floor is added.
        active = np.flatnonzero(q > 0)
        away_vertex = int(active[np.argmax(grad[active])])
        away_gap = float(grad[away_vertex] - q @ grad)
        lower, upper = 0.0, 1.0
        if away_gap > gap and q[away_vertex] < 1 - 1e-14:
            vertex = away_vertex
            lower, upper = -q[vertex] / (1 - q[vertex]), 0.0
        direction = -q.copy()
        direction[vertex] += 1
        # A(gamma)=(1-gamma)A + gamma*(ridge I + v_e v_e^T).
        # Diagonalize A once: bounded line search then uses Sherman-Morrison O(d).
        A = ridge * np.eye(v.shape[1]) + v.T @ (q[:, None] * v)
        eigen, U = np.linalg.eigh(A)
        z, T = U.T @ v[vertex], U.T @ K @ U
        def line(gamma):
            invdiag = 1 / ((1 - gamma) * eigen + gamma * ridge)
            a = invdiag * z
            return float(np.diag(T) @ invdiag - gamma * (a @ T @ a) / (1 + gamma * (z @ a)))
        result = minimize_scalar(line, bounds=(lower, upper), method="bounded", options={"xatol": 1e-9})
        step = min((lower, float(result.x), upper), key=line)
        if step == 0:
            break
        q += step * direction
        q = np.maximum(q, 0)
        q /= q.sum()
    value, grad = objective_gradient(v, K, q, ridge)
    gap = max(0.0, float(q @ grad - grad.min()))
    return q, {"objective": value, "fw_gap": gap, "converged": gap <= relative_gap * max(value, 1e-15), "solver": "away_step_frank_wolfe", "history": history}
