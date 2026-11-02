"""Unweighted Bernoulli BT likelihood. No pseudo logits or regularization."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_bt(v, labels, max_iter=2000, gradient_tolerance=1e-6, max_weight_norm=1000):
    v, a = np.asarray(v, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if v.ndim != 2 or a.shape != (len(v),) or not len(v) or not np.isin(a, [0, 1]).all() or not np.isfinite(v).all():
        raise ValueError("Invalid binary preference data")
    def loss_grad(w):
        z = v @ w
        return float(np.mean(np.logaddexp(0, z) - a * z)), v.T @ (expit(z) - a) / len(a)
    result = minimize(loss_grad, np.zeros(v.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "gtol": gradient_tolerance, "ftol": 1e-14, "maxls": 50, "maxcor": 50})
    loss, grad = loss_grad(result.x)
    norm = float(np.linalg.norm(result.x))
    report = {"loss": loss, "gradient_inf": float(np.abs(grad).max()), "weight_norm": norm, "iterations": int(result.nit), "solver_message": str(result.message), "regularization": 0.0}
    # Finite MLE may fail under separation; never silently add a penalty or publish NaNs.
    if not np.isfinite(result.x).all() or norm >= max_weight_norm or report["gradient_inf"] > gradient_tolerance * 10:
        raise RuntimeError(f"BT optimization failed; inspect separation/conditioning: {report}")
    report["finite_mle_certified"] = False
    report["note"] = "Numerical stationary fit; a small gradient alone does not certify existence of a finite MLE."
    return result.x, report
