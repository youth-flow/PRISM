"""PDF A3-A5: target-centered loglinear proxy and damped optimal design."""
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize, minimize_scalar
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
    K = np.einsum("ij,ijd,ije->de", p, centered, centered, optimize=True) / len(phi)
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
    sensitivity = np.einsum("ij,jk,ik->i", v, B, v, optimize=True)
    return float(np.sum(K * inv.T)), -sensitivity


def _normalized_objective(v, K, weights, ridge):
    """Original objective in nonnegative coordinates q=w/sum(w), without floors."""
    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(total) or np.any(weights < 0):
        raise ValueError("Nonnegative design weights need finite positive total mass")
    q = weights / total
    value, gradient = objective_gradient(v, K, q, ridge)
    return value, (gradient - float(q @ gradient)) / total, q, gradient


def joint_design(v, K, initial, ridge, max_iter=500, relative_gap=1e-4):
    """Joint L-BFGS-B search; only the original global FW gap certifies success.

    Every original simplex point is represented, including zero coordinates.
    The coordinate objective is not asserted convex; its stopping flag is never
    a certificate. Convexity of F(q) supplies the independently recomputed gap.
    """
    v, K, initial = (np.asarray(a, dtype=np.float64) for a in (v, K, initial))
    if (v.ndim != 2 or not len(v) or initial.shape != (len(v),)
            or not np.isfinite(v).all() or not np.isfinite(initial).all()
            or np.any(initial < 0) or not np.isclose(initial.sum(), 1)
            or not np.isfinite(ridge) or ridge <= 0
            or type(max_iter) is not int or max_iter < 1
            or not np.isfinite(relative_gap) or relative_gap <= 0):
        raise ValueError("Invalid joint Design inputs")
    if (K.shape != (v.shape[1], v.shape[1]) or not np.isfinite(K).all()
            or not np.allclose(K, K.T)
            or np.linalg.eigvalsh(K).min() < -1e-8 * max(1, np.linalg.norm(K, 2))):
        raise ValueError("K must be positive semidefinite")
    # Positive constant scaling conditions the numerical coordinates; neither
    # the minimizers nor the unscaled objective/gap used for acceptance change.
    scale = len(initial)
    state = {"evaluations": 0, "iterations": 0, "history": []}

    class Certified(Exception):
        pass

    def objective(weights):
        if weights.sum() <= 0:
            # The origin has no associated q. Reject it in line search instead
            # of introducing a probability floor or accepting a fake solution.
            return float("inf"), np.zeros_like(weights)
        value, gradient, q, original_gradient = _normalized_objective(v, K, weights, ridge)
        gap = max(0., float(q @ original_gradient - original_gradient.min()))
        state.update(q=q.copy(), value=value, gap=gap)
        state["evaluations"] += 1
        if gap <= relative_gap * max(value, 1e-15):
            raise Certified()
        return scale * value, scale * gradient

    def callback(weights):
        state["iterations"] += 1
        state["history"].append({"iteration": state["iterations"], "objective": state["value"], "gap": state["gap"]})
        if state["iterations"] % 10 == 0:
            print(f"design-joint iteration={state['iterations']} evaluations={state['evaluations']} objective={state['value']:.10g} relative_gap={state['gap']/max(state['value'], 1e-15):.7g}", flush=True)

    result = None
    try:
        result = minimize(objective, scale * initial, jac=True, method="L-BFGS-B",
                          bounds=[(0., None)] * scale, callback=callback,
                          options={"maxiter": max_iter, "maxfun": max_iter * 20,
                                   "maxls": 50, "ftol": np.finfo(float).eps, "gtol": 1e-12})
        weights = np.maximum(result.x, 0.)
        q = weights / weights.sum()
    except Certified:
        q = state["q"]
    # Fresh evaluation in original q coordinates; no optimizer success flag or
    # normalized-coordinate gradient norm can bypass this acceptance condition.
    value, gradient = objective_gradient(v, K, q, ridge)
    gap = max(0., float(q @ gradient - gradient.min()))
    return q, {"objective": value, "fw_gap": gap,
               "converged": bool(gap <= relative_gap * max(value, 1e-15)),
               "solver": "normalized_nonnegative_lbfgsb", "max_iter": max_iter,
               "relative_gap_tolerance": relative_gap, "coordinate_scale": scale,
               "iterations": state["iterations"], "evaluations": state["evaluations"],
               "optimizer_message": "original_global_gap_certificate" if result is None else str(result.message),
               "history": state["history"]}


def _line_objective(A, K, ridge, *, vertex=None, mixture=None):
    """Exact trace objective along (1-gamma)q + gamma*atom, including ridge."""
    if vertex is not None:
        eigen, U = np.linalg.eigh(A)
        z, T = U.T @ vertex, U.T @ K @ U
        def line(gamma):
            invdiag = 1 / ((1 - gamma) * eigen + gamma * ridge)
            a = invdiag * z
            return float(np.diag(T) @ invdiag - gamma * (a @ T @ a) / (1 + gamma * (z @ a)))
        return line
    # The dense initial distribution is a redundant, full-rank atom. Whiten
    # A+gamma*(M_initial-M_q) once; each scalar evaluation is then O(d).
    L = np.linalg.cholesky(A)
    def whiten(matrix):
        left = solve_triangular(L, matrix, lower=True, check_finite=False)
        return solve_triangular(L, left.T, lower=True, check_finite=False).T
    delta = whiten(mixture - A + ridge * np.eye(len(A)))
    eigen, U = np.linalg.eigh((delta + delta.T) / 2)
    projected = whiten(K) @ U
    diagonal = np.einsum("ij,ij->j", U, projected)
    def line(gamma):
        denominators = 1 + gamma * eigen
        if np.any(denominators <= 0):
            return float("inf")
        return float(np.sum(diagonal / denominators))
    return line


def frank_wolfe(v, K, initial, ridge, max_iter=500, relative_gap=1e-4):
    v, K, q = np.asarray(v), np.asarray(K), np.asarray(initial, dtype=np.float64).copy()
    if ridge <= 0 or max_iter < 1 or relative_gap <= 0 or np.any(q < 0) or not np.isclose(q.sum(), 1):
        raise ValueError("Invalid FW inputs")
    if K.shape != (v.shape[1], v.shape[1]) or not np.allclose(K, K.T) or np.linalg.eigvalsh(K).min() < -1e-8 * max(1, np.linalg.norm(K, 2)):
        raise ValueError("K must be positive semidefinite")
    # Represent the dense initializer as ONE redundant atom, plus the original
    # edge vertices. Their convex hull remains exactly the original simplex.
    # Away steps can reduce its whole mass instead of dropping thousands of
    # tiny initial edge weights one at a time. Its coefficient may reach zero;
    # there is no fixed coverage floor or feature/rank approximation.
    initial = q.copy()
    coefficients = np.zeros(len(q) + 1)
    coefficients[0] = 1
    initial_matrix, history = None, []
    for iteration in range(max_iter):
        value, grad = objective_gradient(v, K, q, ridge)
        vertex = int(grad.argmin())
        gap = max(0.0, float(q @ grad - grad[vertex]))
        record = {"iteration": iteration, "objective": value, "gap": gap,
                  "initial_atom_weight": float(coefficients[0]),
                  "active_atoms": int(np.count_nonzero(coefficients))}
        history.append(record)
        if iteration % 10 == 0 or gap <= relative_gap * max(value, 1e-15):
            print(f"design-fw iteration={iteration} objective={value:.9g} relative_gap={gap/max(value, 1e-15):.6g} initial_atom={coefficients[0]:.6g}", flush=True)
        if gap <= relative_gap * max(value, 1e-15):
            break
        atom_gradient = np.concatenate(([initial @ grad], grad))
        active = np.flatnonzero(coefficients > 0)
        away_atom = int(active[np.argmax(atom_gradient[active])])
        away_gap = float(atom_gradient[away_atom] - q @ grad)
        atom = vertex + 1
        lower, upper = 0.0, 1.0
        if away_gap > gap and coefficients[away_atom] < 1 - 1e-14:
            atom = away_atom
            lower, upper = -coefficients[atom] / (1 - coefficients[atom]), 0.0
        A = ridge * np.eye(v.shape[1]) + v.T @ (q[:, None] * v)
        if atom == 0:
            if initial_matrix is None:
                initial_matrix = v.T @ (initial[:, None] * v)
            line = _line_objective(A, K, ridge, mixture=initial_matrix)
        else:
            line = _line_objective(A, K, ridge, vertex=v[atom - 1])
        result = minimize_scalar(line, bounds=(lower, upper), method="bounded", options={"xatol": 1e-9})
        step = min((lower, float(result.x), upper), key=line)
        if step == 0:
            break
        record.update(direction="away_initial" if atom == 0 else ("away_vertex" if step < 0 else "forward"), step=float(step))
        coefficients *= 1 - step
        coefficients[atom] += step
        if step == lower and lower < 0:
            coefficients[atom] = 0.0  # Exact drop: do not retain roundoff ghosts.
        coefficients = np.maximum(coefficients, 0)
        coefficients /= coefficients.sum()
        q = coefficients[0] * initial + coefficients[1:]
    value, grad = objective_gradient(v, K, q, ridge)
    gap = max(0.0, float(q @ grad - grad.min()))
    return q, {"objective": value, "fw_gap": gap, "converged": gap <= relative_gap * max(value, 1e-15), "solver": "away_step_frank_wolfe", "initial_representation": "one_redundant_mixture_atom", "history": history}
