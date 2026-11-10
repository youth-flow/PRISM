"""Unweighted Bernoulli BT likelihood with optional explicit fixed L2."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_bt(v, labels, max_iter=2000, gradient_tolerance=1e-6, max_weight_norm=1000,
           l2_regularization=0.0):
    v, a = np.asarray(v, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if v.ndim != 2 or a.shape != (len(v),) or not len(v) or not np.isin(a, [0, 1]).all() or not np.isfinite(v).all():
        raise ValueError("Invalid binary preference data")
    if not np.isfinite(l2_regularization) or l2_regularization < 0:
        raise ValueError("l2_regularization must be finite and nonnegative")
    def loss_grad(w):
        z = v @ w
        loss = np.mean(np.logaddexp(0, z) - a * z) + 0.5 * l2_regularization * (w @ w)
        grad = v.T @ (expit(z) - a) / len(a) + l2_regularization * w
        return float(loss), grad
    result = minimize(loss_grad, np.zeros(v.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "gtol": gradient_tolerance, "ftol": 1e-14, "maxls": 50, "maxcor": 50})
    loss, grad = loss_grad(result.x)
    norm = float(np.linalg.norm(result.x))
    report = {"loss": loss, "gradient_inf": float(np.abs(grad).max()), "weight_norm": norm, "iterations": int(result.nit), "solver_message": str(result.message), "regularization": float(l2_regularization)}
    # Finite MLE may fail under separation; never silently add a penalty or publish NaNs.
    if not np.isfinite(result.x).all() or norm >= max_weight_norm or report["gradient_inf"] > gradient_tolerance * 10:
        raise RuntimeError(f"BT optimization failed; inspect separation/conditioning: {report}")
    report["finite_mle_certified"] = False
    report["fit_type"] = "l2_regularized_bt" if l2_regularization > 0 else "unregularized_bt"
    report["finite_unique_penalized_minimizer_exists"] = bool(l2_regularization > 0)
    report["note"] = (
        "Positive fixed L2 makes the objective strongly convex and coercive; the numerical fit is approximate, not an unregularized MLE."
        if l2_regularization > 0 else
        "Numerical stationary fit; a small gradient alone does not certify existence of a finite MLE."
    )
    return result.x, report
