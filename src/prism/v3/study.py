"""Explicit user budget freeze and label-blind finite-pool sampling."""
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from prism.io import digest, rows, write_rows
from prism.weighted_sampling import weighted_without_replacement
from .artifacts import Run, read_json, write_json


def lambda_name(value):
    return str(float(value))


def independent_design_audit(v, K, q, ridge, tolerance, block_size=4096):
    """LU rather than the solver's Cholesky path; certify the original simplex."""
    v, K, q = (np.asarray(x, dtype=np.float64) for x in (v, K, q))
    if (v.ndim != 2 or q.shape != (len(v),) or K.shape != (v.shape[1], v.shape[1])
            or not all(np.isfinite(x).all() for x in (v, K, q)) or np.any(q < 0)
            or not np.isclose(q.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("Invalid original-simplex Design artifacts")
    if ridge <= 0 or tolerance <= 0 or block_size < 1:
        raise ValueError("Invalid Design audit settings")
    A = ridge * np.eye(v.shape[1])
    for start in range(0, len(v), block_size):
        block = v[start:start + block_size]
        A += block.T @ (q[start:start + block_size, None] * block)
    inverse = lu_solve(lu_factor(A), np.eye(len(A)))
    B = inverse @ K @ inverse
    gradient = np.empty(len(v))
    for start in range(0, len(v), block_size):
        block = v[start:start + block_size]
        gradient[start:start + len(block)] = -np.einsum("ij,ij->i", block @ B, block)
    objective = float(np.sum(K * inverse.T))
    gap = max(0.0, float(q @ gradient - gradient.min()))
    relative = gap / max(objective, 1e-15)
    if not np.isfinite(relative) or relative > tolerance:
        raise ValueError(f"Selected Design failed independent original-space gap: {relative}")
    return {"objective": objective, "fw_gap": gap, "relative_gap": relative,
            "tolerance": tolerance, "factorization": "independent_LU", "passed": True}


def freeze_study(c, budget, design_lambda=None, *, user_confirmed=False):
    if user_confirmed is not True:
        raise ValueError("The user must choose the preference budget after the Design report")
    if type(budget) is not int or budget < 1:
        raise ValueError("Choose one positive integer preference budget")
    selected = c["design"]["preferred_lambda"] if design_lambda is None else design_lambda
    if type(selected) not in (int, float) or selected not in c["design"]["lambdas"]:
        raise ValueError("Select one of the three diagnosed Design ridge values")
    run = Run(c)
    if run.completed("study-selection"):
        current = run.selection()
        if current["budget"] != budget or current["design_lambda"] != selected:
            raise ValueError("Preserve the frozen V3 study; use a separate run for a changed decision")
        return current
    run.require("geometry")
    run.require("geometry-audit")
    dependencies = [run.receipt_path("geometry"), run.receipt_path("geometry-audit")]
    for candidate in c["design"]["lambdas"]:
        stage = "design-" + lambda_name(candidate)
        run.require(stage)
        dependencies.append(run.receipt_path(stage))
    run.require("design-report")
    dependencies.append(run.receipt_path("design-report"))
    selected_dir = run.file("design/lambda-" + lambda_name(selected))
    report = read_json(selected_dir / "report.json")
    if report.get("converged") is not True:
        raise ValueError("Cannot freeze a study from an unconverged Design attempt")
    with np.load(run.file("geometry/geometry.npz"), allow_pickle=False) as array:
        v, K, ids = array["v"], array["K"], array["ids"].astype(str)
    q = np.load(selected_dir / "q.npy", allow_pickle=False)
    if budget > int(np.count_nonzero(q > 0)):
        raise ValueError("Preference budget exceeds the actual positive-weight support; no mass will be added")
    audit = independent_design_audit(v, K, q, selected, c["design"]["relative_gap"], c["design"]["block_size"])
    value = {"protocol": "v3_user_budget_selection_v1", "budget": budget,
             "budget_authority": "user_after_design_report", "user_budget_confirmed": True,
             "design_lambda": float(selected), "design_lambda_authority": "assistant_diagnostics",
             "seed": c["study"]["seed"], "methods": c["study"]["methods"],
             "pool_size": len(ids), "positive_support": int(np.count_nonzero(q > 0)),
             "q_sha256": digest(selected_dir / "q.npy"), "independent_design_audit": audit}
    target = run.file("study/selection.json")
    write_json(target, value)
    run.publish("study-selection", [target], dependencies + [selected_dir / "q.npy", selected_dir / "report.json"])
    return value


def sample(c):
    run = Run(c)
    if run.completed("sampling"):
        return read_json(run.file("sampling/summary.json"))
    selection = run.selection()
    run.require("geometry")
    run.require("data")
    geometry_path = run.file("geometry/geometry.npz")
    with np.load(geometry_path, allow_pickle=False) as array:
        ids = array["ids"].astype(str)
        empirical = array["empirical"].astype(np.float64)
    if len(set(ids)) != len(ids) or not np.allclose(empirical, np.full(len(ids), 1 / len(ids)), rtol=0, atol=1e-14):
        raise ValueError("HH Empirical must be uniform over distinct eligible canonical pairs")
    label_path = run.file("data/labels.jsonl")
    labels = {}
    for row in rows(label_path):
        if row["id"] in labels or type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("Invalid or duplicate human preference identity")
        labels[row["id"]] = row["label"]
    if not set(ids).issubset(labels):
        raise ValueError("Missing human preference labels for eligible Design pairs")
    q_path = run.file("design/lambda-" + lambda_name(selection["design_lambda"]) + "/q.npy")
    if digest(q_path) != selection["q_sha256"]:
        raise ValueError("Selected Design weights changed")
    distributions = {"empirical": empirical, "prism": np.load(q_path, allow_pickle=False)}
    outputs, selected_ids = [], {}
    for method in c["study"]["methods"]:
        result = weighted_without_replacement(distributions[method], selection["budget"], c["study"]["seed"])
        indices = result["ordered_indices"]
        result.update(method=method, pair_ids=ids[indices].tolist(), common_gumbel_keys=False,
                      shared_rng_seed=True, gumbel_key_alignment="positive_support_order",
                      labels_are_existing_human_observations=True)
        marker = run.file(f"sampling/{method}.json")
        write_json(marker, result)
        target = run.file(f"sampling/{method}.jsonl")
        records = [{"id": str(ids[index]), "geometry_index": index, "selection_index": position,
                    "label": labels[str(ids[index])]} for position, index in enumerate(indices)]
        if target.exists():
            if rows(target) != records:
                raise ValueError("Preserve existing V3 preference selection")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_rows(target, records)
        selected_ids[method] = set(result["pair_ids"])
        outputs.extend([marker, target])
    summary = {"budget": selection["budget"], "seed": c["study"]["seed"], "formal_policy_units": 2,
               "unique_pairs": {m: len(v) for m, v in selected_ids.items()},
               "shared_pairs": len(set.intersection(*selected_ids.values())),
               "union_existing_human_labels": len(set.union(*selected_ids.values())),
               "new_annotation_api_calls": 0, "sampling_protocol": c["sampling"]["protocol"]}
    summary_path = run.file("sampling/summary.json")
    write_json(summary_path, summary)
    run.publish("sampling", outputs + [summary_path],
                [run.receipt_path("study-selection"), run.receipt_path("geometry"), run.receipt_path("data"), label_path, q_path])
    return summary
