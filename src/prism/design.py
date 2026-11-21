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
    return objective_gradient_from_information(v, K, A)


def objective_gradient_from_information(v, K, information):
    """Evaluate the objective/gradient from an already accumulated matrix."""
    v, K, information = (np.asarray(a, dtype=np.float64) for a in (v, K, information))
    if (v.ndim != 2 or K.shape != (v.shape[1], v.shape[1])
            or information.shape != K.shape or not all(np.isfinite(a).all()
                                                        for a in (v, K, information))):
        raise ValueError("Invalid Design information matrix")
    A = information
    inv = cho_solve(cho_factor(A), np.eye(len(A)))
    B = inv @ K @ inv
    sensitivity = np.einsum("ij,jk,ik->i", v, B, v, optimize=True)
    return float(np.sum(K * inv.T)), -sensitivity


def capped_simplex_lmo(gradient, budget):
    """Linear minimizer for sum(q)=1 and 0<=q_i<=1/budget.

    Every vertex is uniform on exactly ``budget`` coordinates.  Lexicographic
    tie-breaking makes the numerical solver and its independent audit stable
    across repeated runs.
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    if (gradient.ndim != 1 or not len(gradient) or not np.isfinite(gradient).all()
            or type(budget) is not int or not 1 <= budget <= len(gradient)):
        raise ValueError("Invalid capped-simplex linear minimization inputs")
    indices = np.lexsort((np.arange(len(gradient), dtype=np.int64), gradient))[:budget]
    indices.sort()
    return indices, float(np.sum(gradient[indices], dtype=np.float64) / budget)


def capped_simplex_gap(q, gradient, budget):
    """Return the global Frank--Wolfe gap on the budget-capped simplex."""
    q, gradient = (np.asarray(a, dtype=np.float64) for a in (q, gradient))
    if q.shape != gradient.shape or not np.isfinite(q).all():
        raise ValueError("Invalid capped-simplex gap inputs")
    upper = 1.0 / budget
    if (np.any(q < -1e-12) or np.any(q > upper + 1e-12)
            or not np.isclose(q.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("Design weights violate the budget-capped simplex")
    indices, minimum = capped_simplex_lmo(gradient, budget)
    return max(0.0, float(q @ gradient - minimum)), indices


def _weighted_information(v, weights, block_size):
    matrix = np.zeros((v.shape[1], v.shape[1]), dtype=np.float64)
    for start in range(0, len(v), block_size):
        part = v[start:start + block_size]
        matrix += part.T @ (weights[start:start + len(part), None] * part)
    return matrix


def _vertex_information(v, indices, budget, block_size):
    matrix = np.zeros((v.shape[1], v.shape[1]), dtype=np.float64)
    scale = 1.0 / budget
    for start in range(0, len(indices), block_size):
        part = v[indices[start:start + block_size]]
        matrix += scale * (part.T @ part)
    return matrix


def capped_away_frank_wolfe(v, K, initial, ridge, budget, max_iter=500,
                            relative_gap=1e-4, block_size=4096):
    """Solve the budget-aware Design on a capped probability simplex.

    The feasible region is ``sum(q)=1, 0<=q_i<=1/budget``.  Its vertices are
    uniform distributions over exactly ``budget`` pool items, so the cap is a
    direct continuous relaxation of selecting ``budget`` distinct pairs.  The
    dense empirical initializer is retained as one redundant feasible atom;
    away steps can remove its entire coefficient rather than one coordinate at
    a time.  Only the capped-simplex global FW gap certifies convergence.
    """
    v, K, initial = (np.asarray(a, dtype=np.float64) for a in (v, K, initial))
    if (v.ndim != 2 or not len(v) or initial.shape != (len(v),)
            or not all(np.isfinite(a).all() for a in (v, K, initial))
            or type(budget) is not int or not 1 <= budget <= len(v)
            or type(max_iter) is not int or max_iter < 1
            or type(block_size) is not int or block_size < 1
            or not np.isfinite(ridge) or ridge <= 0
            or not np.isfinite(relative_gap) or relative_gap <= 0):
        raise ValueError("Invalid budget-capped Design inputs")
    if (K.shape != (v.shape[1], v.shape[1]) or not np.allclose(K, K.T)
            or np.linalg.eigvalsh(K).min() < -1e-8 * max(1, np.linalg.norm(K, 2))):
        raise ValueError("K must be positive semidefinite")
    upper = 1.0 / budget
    if (np.any(initial < 0) or np.any(initial > upper + 1e-14)
            or not np.isclose(initial.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Initial Design is outside the budget-capped simplex")

    q = initial.copy()
    identity = np.eye(v.shape[1])
    initial_matrix = _weighted_information(v, initial, block_size)
    information = ridge * identity + initial_matrix
    # None denotes the dense initializer; every other atom is a sorted int32
    # array describing a capped-simplex vertex.
    atoms, coefficients = [None], np.array([1.0], dtype=np.float64)
    atom_lookup, history = {}, []

    for iteration in range(max_iter):
        value, gradient = objective_gradient_from_information(v, K, information)
        gap, toward_indices = capped_simplex_gap(q, gradient, budget)
        atom_gradients = np.empty(len(atoms), dtype=np.float64)
        atom_gradients[0] = float(initial @ gradient)
        for atom_index, indices in enumerate(atoms[1:], start=1):
            atom_gradients[atom_index] = float(np.mean(gradient[indices]))
        active = np.flatnonzero(coefficients > 0)
        away_atom = int(active[np.argmax(atom_gradients[active])])
        away_gap = max(0.0, float(atom_gradients[away_atom] - q @ gradient))
        record = {
            "iteration": iteration,
            "objective": value,
            "gap": gap,
            "relative_gap": gap / max(value, 1e-15),
            "initial_atom_weight": float(coefficients[0]),
            "active_atoms": int(len(active)),
            "maximum_weight": float(q.max()),
        }
        history.append(record)
        if iteration % 10 == 0 or gap <= relative_gap * max(value, 1e-15):
            print("design-capped-fw iteration={} objective={:.9g} relative_gap={:.6g} "
                  "initial_atom={:.6g} active_atoms={}".format(
                      iteration, value, record["relative_gap"], coefficients[0], len(active)),
                  flush=True)
        if gap <= relative_gap * max(value, 1e-15):
            break

        key = toward_indices.astype("<i4", copy=False).tobytes()
        toward_atom = atom_lookup.get(key)
        if toward_atom is None:
            toward_atom = len(atoms)
            atoms.append(toward_indices.astype(np.int32, copy=True))
            coefficients = np.append(coefficients, 0.0)
            atom_lookup[key] = toward_atom

        atom = toward_atom
        lower, upper_step = 0.0, 1.0
        direction = "forward"
        if away_gap > gap and coefficients[away_atom] < 1 - 1e-14:
            atom = away_atom
            lower = -float(coefficients[atom] / (1 - coefficients[atom]))
            upper_step = 0.0
            direction = "away_initial" if atom == 0 else "away_vertex"
        if atom == 0:
            atom_matrix = initial_matrix
        else:
            atom_matrix = _vertex_information(v, atoms[atom], budget, block_size)
        line = _line_objective(information, K, ridge, mixture=atom_matrix)
        result = minimize_scalar(line, bounds=(lower, upper_step), method="bounded",
                                 options={"xatol": 1e-10})
        step = min((lower, float(result.x), upper_step), key=line)
        if abs(step) <= 1e-15:
            record.update(direction=direction, step=0.0, stalled=True)
            break
        record.update(direction=direction, step=float(step), away_gap=away_gap)

        coefficients *= 1 - step
        coefficients[atom] += step
        if step == lower and lower < 0:
            coefficients[atom] = 0.0
        coefficients[np.abs(coefficients) < 1e-15] = 0.0
        if np.any(coefficients < -1e-12) or not np.isclose(coefficients.sum(), 1, atol=1e-10):
            raise RuntimeError("Capped FW active-set coefficients became invalid")
        coefficients = np.maximum(coefficients, 0)
        coefficients /= coefficients.sum()
        q *= 1 - step
        if atom == 0:
            q += step * initial
        else:
            q[atoms[atom]] += step / budget
        information = (1 - step) * information + step * (ridge * identity + atom_matrix)
        information = (information + information.T) / 2

    # Reconstruct once from the active decomposition and independently
    # recompute the original objective/gradient, eliminating recurrence drift.
    q = coefficients[0] * initial
    for coefficient, indices in zip(coefficients[1:], atoms[1:]):
        if coefficient:
            q[indices] += coefficient / budget
    q[np.abs(q) < 1e-16] = 0.0
    value, gradient = objective_gradient(v, K, q, ridge)
    gap, _ = capped_simplex_gap(q, gradient, budget)
    return q, {
        "objective": value,
        "fw_gap": gap,
        "converged": bool(gap <= relative_gap * max(value, 1e-15)),
        "solver": "capped_simplex_away_step_frank_wolfe",
        "budget": budget,
        "weight_upper_bound": 1.0 / budget,
        "initial_representation": "one_redundant_feasible_empirical_atom",
        "max_iter": max_iter,
        "relative_gap_tolerance": relative_gap,
        "iterations": len(history),
        "active_atoms": int(np.count_nonzero(coefficients)),
        "history": history,
    }


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
