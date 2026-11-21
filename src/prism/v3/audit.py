"""Honest partial readiness and full formal V3 completion audits."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import logsumexp

from .artifacts import Run, read_json, write_json
from .study import independent_design_audit, lambda_name


def required_stages(c):
    return ["data", "features", "geometry", "geometry-audit"] + ["design-" + lambda_name(v) for v in c["design"]["lambdas"]] + [
        "design-report", "study-selection", "sampling", "reward", "training-audit", "shared-sft", "normalization",
        "critic-warmup-empirical", "critic-warmup-prism", "ppo-common-acceptance",
        "policy-empirical", "policy-prism", "eval-sft", "eval-empirical", "eval-prism",
        "judge-export", "judge-import", "aggregate"]


def audit_geometry(c):
    """Independent LU fit and explicit centered covariance, rather than pApB VVt."""
    run = Run(c)
    if run.completed("geometry-audit"):
        return read_json(run.file("geometry/independent.audit.json"))
    run.require("geometry")
    run.require("features")
    with np.load(run.file("features/pool.npz"), allow_pickle=False) as file:
        phi, logp, feature_ids = file["phi"], file["logp"], file["ids"].astype(str)
    with np.load(run.file("geometry/geometry.npz"), allow_pickle=False) as file:
        v, K, b, p, ids = (file[k] for k in ("v", "K", "b", "proxy_probabilities", "ids"))
    if (phi.ndim != 3 or phi.shape[1] != 2 or v.shape != (len(phi), phi.shape[2])
            or not np.array_equal(feature_ids, ids.astype(str)) or len(set(feature_ids)) != len(phi)):
        raise ValueError("Independent geometry audit found candidate identity/shape mismatch")
    dimension, block_size = phi.shape[-1], c["design"]["block_size"]
    normal = c["design"]["surrogate_ridge"] * np.eye(dimension)
    target = np.zeros(dimension)
    for start in range(0, len(phi), block_size):
        block = np.asarray(phi[start:start + block_size], dtype=np.float64)
        ell = np.asarray(logp[start:start + block_size], dtype=np.float64)
        X = block.reshape(-1, dimension)
        normal += X.T @ X
        target += X.T @ (ell - ell.mean(axis=1, keepdims=True)).ravel()
        if not np.array_equal(v[start:start + len(block)], block[:, 0] - block[:, 1]):
            raise ValueError("Human comparison features do not match Design vectors")
    independent_b = lu_solve(lu_factor(normal), target)
    b_error = float(np.linalg.norm(independent_b - b) / max(np.linalg.norm(b), 1e-15))
    independent_K, probability_error = np.zeros_like(K), 0.0
    for start in range(0, len(phi), block_size):
        block = np.asarray(phi[start:start + block_size], dtype=np.float64)
        logits = block @ independent_b
        probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        probability_error = max(probability_error, float(np.max(np.abs(probabilities - p[start:start + len(block)]))))
        center = np.einsum("ij,ijd->id", probabilities, block)
        for candidate in range(2):
            delta = block[:, candidate] - center
            independent_K += delta.T @ (probabilities[:, candidate, None] * delta) / len(phi)
    K_error = float(np.linalg.norm(independent_K - K) / max(np.linalg.norm(K), 1e-15))
    if (not all(np.isfinite(x) for x in (b_error, K_error, probability_error))
            or b_error > 1e-7 or probability_error > 1e-7 or K_error > 1e-6):
        raise ValueError(f"Independent HH proxy geometry mismatch: b={b_error}, p={probability_error}, K={K_error}")
    value = {"protocol": "v3_independent_proxy_geometry_v1", "passed": True, "rows": len(phi),
             "dimension": dimension, "b_relative_error": b_error, "probability_max_abs_error": probability_error,
             "K_relative_error": K_error, "b_solver": "LU", "covariance": "explicit_centered_two_candidate",
             "label_values_read": False, "b_tolerance": 1e-7, "p_tolerance": 1e-7, "K_tolerance": 1e-6}
    path = run.file("geometry/independent.audit.json")
    write_json(path, value)
    run.publish("geometry-audit", [path], [run.receipt_path("geometry"), run.receipt_path("features")])
    return value


def status(c):
    run = Run(c, create=False)
    done, missing = [], []
    for stage in required_stages(c):
        (done if run.completed(stage) else missing).append(stage)
    selection = run.selection() if "study-selection" in done else None
    return {"complete": not missing, "completed_stages": done, "missing_stages": missing,
            "selected_preference_budget": selection["budget"] if selection else None,
            "waiting_for_user_budget": "design-report" in done and selection is None,
            "paid_calls_enabled": c["execution"]["allow_paid_api_calls"]}


def audit(c, *, require_complete=False):
    run = Run(c)
    snapshot = status(c)
    if require_complete and not snapshot["complete"]:
        raise ValueError("V3 is not complete: " + ", ".join(snapshot["missing_stages"]))
    checks = {"all_available_receipts_verified": True}
    if "study-selection" in snapshot["completed_stages"]:
        selection = run.selection()
        with np.load(run.file("geometry/geometry.npz"), allow_pickle=False) as array:
            v, K = array["v"], array["K"]
        q = np.load(run.file(f"design/lambda-{lambda_name(selection['design_lambda'])}/q.npy"), allow_pickle=False)
        checks["selected_original_design"] = independent_design_audit(
            v, K, q, selection["design_lambda"], c["design"]["relative_gap"],
            c["design"]["block_size"],
            budget=selection["budget"] if isinstance(c.get("v6"), dict) else None)
    for method in c["study"]["methods"]:
        if "policy-" + method in snapshot["completed_stages"]:
            report = read_json(run.file(f"policy/{method}/report.json"))
            total = c["ppo"]["total_episodes"]
            rounds = total // c["ppo"]["rollout_batch_size"]
            expected_steps = rounds * c["ppo"]["num_ppo_epochs"] * c["ppo"]["num_mini_batches"]
            actual = report.get("actual_optimizer_steps", {})
            if (report.get("completed") is not True or report.get("completed_episodes") != total
                    or report.get("total_episodes") != total or report.get("rollout_rounds") != rounds
                    or set(actual) != {"actor", "critic_lora", "critic_head"}
                    or any(r.get("step_min") != expected_steps or r.get("step_max") != expected_steps
                           or r.get("tensor_count", 0) < 1 for r in actual.values())
                    or report.get("gradient_norm_clipping_executed") is not True
                    or len(report.get("probability_checks", [])) != rounds):
                raise ValueError("A partial policy cannot count as a completed V3 unit")
            for check in report["probability_checks"]:
                if (check.get("max_abs_logp_error", float("inf")) > c["ppo"]["max_logp_error"]
                        or check.get("zero_update_approxkl", float("inf")) > c["ppo"]["max_zero_update_approxkl"]):
                    raise ValueError("Completed V3 policy failed its actual probability path checks")
            checks["policy_" + method] = {"episodes": total, "rollout_rounds": rounds,
                                         "actual_steps": expected_steps, "hashes_verified": True}
    if "judge-import" in snapshot["completed_stages"]:
        checks["judge"] = "hash-bound offline import verified; final aggregate retains actual dispatch accounting"
    value = {"protocol": "v3_final_audit_v1", **snapshot, "checks": checks,
             "status": "formal_experiment_complete" if snapshot["complete"] else "partial_engineering_readiness"}
    target = run.file("audits/final.json" if snapshot["complete"] else "audits/readiness.json")
    write_json(target, value, overwrite=not snapshot["complete"])
    if snapshot["complete"]:
        run.publish("final-audit", [target], [run.receipt_path(stage) for stage in required_stages(c)])
    return value
