"""Full HH finite-pair loglinear geometry and certified Design solvers.

No label is read here. Matrix accumulation uses float64 blocks, while frozen
model features originate from FP32. V3/V5 retain the original joint solver;
V6 opts into a budget-capped feasible region through its explicit contract.
"""
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax

from prism.design import (capped_away_frank_wolfe, capped_simplex_gap,
                          joint_design)
from .artifacts import Run, read_json, write_json
from .recovery import recover_stage, seal_and_publish, stage_lock


def _sources():
    return [Path(__file__), Path(__file__).resolve().parent.parent / "design.py"]


def pair_geometry(phi, logp, ridge=0.01, block_size=4096):
    """Exact original proxy formula with two existing candidates per record."""
    phi, logp = np.asarray(phi), np.asarray(logp)
    if (phi.ndim != 3 or phi.shape[1] != 2 or not len(phi)
            or logp.shape != phi.shape[:2] or not np.isfinite(phi).all()
            or not np.isfinite(logp).all() or type(block_size) is not int or block_size < 1
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Finite nonempty two-candidate features and positive ridge required")
    count, _, dimension = phi.shape
    normal, target = ridge * np.eye(dimension), np.zeros(dimension)
    for start in range(0, count, block_size):
        part = np.asarray(phi[start:start + block_size], dtype=np.float64)
        probabilities = np.asarray(logp[start:start + block_size], dtype=np.float64)
        X = part.reshape(-1, dimension)
        centered = (probabilities - probabilities.mean(axis=1, keepdims=True)).ravel()
        normal += X.T @ X
        target += X.T @ centered
    b = cho_solve(cho_factor(normal), target)
    v = np.empty((count, dimension), dtype=np.float64)
    proxy_probabilities = np.empty((count, 2), dtype=np.float64)
    K = np.zeros((dimension, dimension))
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        part = np.asarray(phi[start:stop], dtype=np.float64)
        p = softmax(part @ b, axis=1)
        delta = part[:, 0] - part[:, 1]
        v[start:stop], proxy_probabilities[start:stop] = delta, p
        weighted = delta * np.sqrt(p[:, 0] * p[:, 1] / count)[:, None]
        K += weighted.T @ weighted
    K = (K + K.T) / 2
    if not all(np.isfinite(a).all() for a in (v, K, b, proxy_probabilities)):
        raise ValueError("Nonfinite full-pool geometry")
    return {"v": v, "K": K, "b": b, "proxy_probabilities": proxy_probabilities,
            "empirical": np.full(count, 1.0 / count)}


def build_geometry(c):
    with stage_lock(Run(c), "geometry"):
        return _build_geometry(c)


def _build_geometry(c):
    run = Run(c)
    run.require("features")
    if run.completed("geometry"):
        return read_json(run.file("geometry/diagnostics.json"))
    path, marker = run.file("geometry/geometry.npz"), run.file("geometry/diagnostics.json")
    dependencies = [run.receipt_path("features")]
    if recover_stage(run, "geometry", [path, marker], dependencies, code_paths=_sources()):
        return read_json(marker)
    started = time.monotonic()
    with np.load(run.file("features/pool.npz"), allow_pickle=False) as features:
        phi, logp, ids = features["phi"], features["logp"], features["ids"]
        result = pair_geometry(phi, logp, c["design"]["surrogate_ridge"], c["design"]["block_size"])
    eigenvalues = np.linalg.eigvalsh(result["K"])
    tolerance = max(1e-12, float(eigenvalues[-1]) * 1e-10)
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("Proxy K is not positive semidefinite")
    p = result["proxy_probabilities"]
    weight = p[:, 0] * p[:, 1]
    squared = float(eigenvalues @ eigenvalues)
    report = {"protocol": "hh_full_pool_pair_covariance_v1", "rows": len(ids), "dimension": len(result["b"]),
              "target": "finite_hh_pair_loglinear_covariance_proxy", "human_labels_used": False,
              "new_sft_responses_for_K": 0, "surrogate_ridge": c["design"]["surrogate_ridge"],
              "normal_equation": "sum_XtX_plus_ridge; only_logp_targets_centered_per_pair",
              "K_eigen_min": float(eigenvalues[0]), "K_eigen_max": float(eigenvalues[-1]),
              "K_trace": float(eigenvalues.sum()), "K_numerical_rank": int(np.count_nonzero(eigenvalues > tolerance)),
              "K_effective_rank": float(eigenvalues.sum() ** 2 / squared) if squared else 0.0,
              "pair_probability_product_quantiles": np.quantile(weight, [0, .01, .5, .99, 1]).tolist(),
              "zero_pair_probability_products": int(np.count_nonzero(weight == 0)),
              "elapsed_seconds": time.monotonic() - started}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **result, ids=ids)
    write_json(marker, report)
    seal_and_publish(run, "geometry", [path, marker], dependencies, metadata=report, code_paths=_sources())
    return report


def distribution_diagnostics(q):
    q = np.asarray(q, dtype=np.float64)
    if (q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any(q < 0)
            or not np.isclose(q.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("Design diagnostic requires a probability simplex vector")
    descending = np.sort(q)[::-1]
    cumulative = np.cumsum(descending)
    coverage = lambda mass: min(len(q), int(np.searchsorted(cumulative, mass, side="left")) + 1)
    return {"pool_size": len(q), "positive_support": int(np.count_nonzero(q > 0)),
            "maximum_weight": float(q.max()), "effective_sample_size": float(1 / (q @ q)),
            "top_1_percent_mass": float(descending[:max(1, math.ceil(len(q) * .01))].sum()),
            "top_10_percent_mass": float(descending[:max(1, math.ceil(len(q) * .1))].sum()),
            "mass_95_coverage_pairs": coverage(.95), "mass_99_coverage_pairs": coverage(.99),
            "positive_below_1e_15": int(np.count_nonzero((q > 0) & (q < 1e-15))),
            "mass_below_1e_15": float(q[q < 1e-15].sum()),
            "positive_below_millionth_uniform": int(np.count_nonzero((q > 0) & (q < 1e-6 / len(q)))),
            "weights_thresholded_or_modified": False,
            "support_is_wor_hard_limit": True, "ess_is_wor_budget_limit": False}


def blocked_objective_gradient(v, K, q, ridge, block_size=4096):
    """Independently recompute the original q-space objective/global FW gap."""
    dimension = v.shape[1]
    A = ridge * np.eye(dimension)
    for start in range(0, len(v), block_size):
        part = np.asarray(v[start:start + block_size], dtype=np.float64)
        A += part.T @ (q[start:start + block_size, None] * part)
    inverse = cho_solve(cho_factor(A), np.eye(dimension))
    B, gradient = inverse @ K @ inverse, np.empty(len(v))
    for start in range(0, len(v), block_size):
        part = np.asarray(v[start:start + block_size], dtype=np.float64)
        gradient[start:start + len(part)] = -np.einsum("ij,ij->i", part @ B, part)
    return float(np.sum(K * inverse.T)), gradient


def lambda_tag(design_lambda):
    return str(float(design_lambda))


def solve_design(c, design_lambda):
    stage = f"design-{lambda_tag(design_lambda)}"
    with stage_lock(Run(c), stage):
        return _solve_design(c, design_lambda)


def _solve_design(c, design_lambda):
    if type(design_lambda) not in (int, float) or design_lambda not in c["design"]["lambdas"]:
        raise ValueError("Design lambda must be one of the declared sensitivity values")
    run = Run(c)
    run.require("geometry")
    tag = lambda_tag(design_lambda)
    stage = f"design-{tag}"
    marker = run.file(f"design/lambda-{tag}/report.json")
    if run.completed(stage):
        return read_json(marker)
    path = run.file(f"design/lambda-{tag}/q.npy")
    dependencies = [run.receipt_path("geometry")]
    v6 = c.get("v6") if isinstance(c.get("v6"), dict) else None
    budget = v6.get("preference_budget") if v6 else None
    if recover_stage(run, stage, [path, marker], dependencies, code_paths=_sources(),
                     context={"lambda": float(design_lambda), "budget": budget}):
        return read_json(marker)
    started = time.monotonic()
    with np.load(run.file("geometry/geometry.npz"), allow_pickle=False) as geometry:
        v, K, empirical = geometry["v"], geometry["K"], geometry["empirical"]
    if c["design"].get("solver", "joint-lbfgs") == "capped-away-fw":
        q, original = capped_away_frank_wolfe(
            v, K, empirical, float(design_lambda), budget,
            c["design"]["max_iter"], c["design"]["relative_gap"],
            c["design"]["block_size"])
    else:
        q, original = joint_design(v, K, empirical, float(design_lambda),
                                   c["design"]["max_iter"], c["design"]["relative_gap"])
    value, gradient = blocked_objective_gradient(v, K, q, float(design_lambda), c["design"]["block_size"])
    baseline, _ = blocked_objective_gradient(v, K, empirical, float(design_lambda), c["design"]["block_size"])
    gap = (capped_simplex_gap(q, gradient, budget)[0] if v6 else
           max(0., float(q @ gradient - gradient.min())))
    independently_converged = gap <= c["design"]["relative_gap"] * max(value, 1e-15)
    original_agreement = abs(value - original["objective"]) <= 1e-8 * max(1., abs(value))
    report = {**original, **distribution_diagnostics(q), "design_lambda": float(design_lambda),
              "independent_objective": value, "independent_global_fw_gap": gap,
              "independent_relative_gap": gap / max(value, 1e-15),
              "original_objective_agreement": bool(original_agreement),
              "converged": bool(original["converged"] and independently_converged and original_agreement),
              "empirical_objective_same_lambda": baseline,
              "objective_ratio_to_empirical_same_lambda": value / baseline if baseline else None,
              "elapsed_seconds": time.monotonic() - started, "budget": budget,
              "extra_constraints": ("sum_q=1_and_0<=q_i<=1/N" if v6 else None),
              "probability_floor": 0.0,
              "acceptance": ("budget_capped_simplex_global_FW_gap_with_independent_blocked_recompute"
                             if v6 else
                             "original_simplex_global_FW_gap_with_independent_blocked_recompute")}
    if v6:
        upper = 1.0 / budget
        report.update(weight_upper_bound=upper,
                      maximum_cap_violation=max(0.0, float(q.max() - upper)),
                      support_covers_budget=bool(np.count_nonzero(q > 0) >= budget))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, q, allow_pickle=False)
    write_json(marker, report)
    seal_and_publish(run, stage, [path, marker], dependencies, code_paths=_sources(),
                     context={"lambda": float(design_lambda), "budget": budget}, metadata={
        "design_lambda": float(design_lambda), "budget": budget,
        "converged": report["converged"]})
    return report


def design_report(c):
    with stage_lock(Run(c), "design-report"):
        return _design_report(c)


def _design_report(c):
    run = Run(c)
    if run.completed("design-report"):
        return read_json(run.file("design/report.json"))
    dependencies, reports = [], []
    for value in c["design"]["lambdas"]:
        tag = lambda_tag(value)
        run.require(f"design-{tag}")
        dependencies.append(run.receipt_path(f"design-{tag}"))
        reports.append(read_json(run.file(f"design/lambda-{tag}/report.json")))
    path = run.file("design/report.json")
    if recover_stage(run, "design-report", [path], dependencies, code_paths=_sources()):
        return read_json(path)
    v6 = c.get("v6") if isinstance(c.get("v6"), dict) else None
    budget = v6.get("preference_budget") if v6 else None
    result = {"designs": reports, "budget": budget,
              "budget_authority": ("v6_precommitted_budget_contract" if v6 else
                                   "user_after_design_report"),
              "automatic_budget_selection": False, "all_converged": all(r["converged"] for r in reports),
              "note": ("Compare within-lambda objective ratios under the same budget cap; raw objectives across lambdas are not comparable."
                       if v6 else
                       "Compare within-lambda objective ratios and sparsity; raw objectives across lambdas are not comparable.")}
    write_json(path, result)
    seal_and_publish(run, "design-report", [path], dependencies, code_paths=_sources(),
                     metadata={"all_converged": result["all_converged"], "budget": budget})
    return result
