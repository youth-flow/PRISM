"""Isolated full-training-pool frozen-feature BT diagnostic, never a PPO input."""
from pathlib import Path
import time

import numpy as np
from scipy.special import expit

from prism.io import rows
from prism.reward import fit_bt
from .artifacts import Run, read_json, write_json
from .recovery import recover_stage, seal_and_publish, stage_lock
from .reward import prediction_metrics, save_npz
from .study import lambda_name


PREFIX = "diagnostics/full-pool-reward"


def _sources():
    folder = Path(__file__).resolve().parent
    return [Path(__file__), folder / "reward.py", folder.parent / "reward.py"]


def _inputs(c, run):
    stages = ("data", "features", "cohort-audit") if c.get("protocol") == "hh_finite_pool_v4" else ("data", "features", "geometry")
    for stage in stages:
        run.require(stage)
    if "geometry" in stages:
        with np.load(run.file("geometry/geometry.npz"), allow_pickle=False) as file:
            v, ids = file["v"], file["ids"].astype(str)
    else:
        with np.load(run.file("features/pool.npz"), allow_pickle=False) as file:
            phi = file["phi"].astype(np.float64)
            v, ids = phi[:, 0] - phi[:, 1], file["ids"].astype(str)
        del phi
    label_rows = rows(run.file("data/labels.jsonl"))
    labels = {r["id"]: r["label"] for r in label_rows}
    if (v.shape != (len(ids), c["data"]["feature_dimension"]) or len(set(ids)) != len(ids)
            or len(labels) != len(label_rows) or set(labels) != set(ids)
            or any(type(value) is not int or value not in (0, 1) for value in labels.values())
            or not np.isfinite(v).all()):
        raise ValueError("Full-pool Reward requires every valid training pair and its exact human label")
    y = np.asarray([labels[i] for i in ids], dtype=np.float64)
    with np.load(run.file("features/reward_dev.npz"), allow_pickle=False) as file:
        phi, dev_ids = file["phi"].astype(np.float64), file["ids"].astype(str)
    dev = rows(run.file("data/reward_dev.jsonl"))
    if (phi.shape != (len(dev), 2, v.shape[1]) or dev_ids.tolist() != [r["id"] for r in dev]
            or len(set(dev_ids)) != len(dev_ids) or set(dev_ids) & set(ids) or not np.isfinite(phi).all()):
        raise ValueError("Full-pool Reward development data are misaligned or overlap training")
    dev_v, dev_y = phi[:, 0]-phi[:, 1], np.asarray([r["label"] for r in dev], dtype=np.float64)
    dependencies = [run.receipt_path(s) for s in stages]
    dependencies.extend((run.file("data/labels.jsonl"), run.file("data/reward_dev.jsonl")))
    return v, y, dev_v, dev_y, dev, dependencies


def _verify(v, y, w, penalty, recorded, tolerance):
    z = v @ w
    gradient = v.T @ (expit(z)-y)/len(y) + penalty*w
    loss = float(np.mean(np.logaddexp(0, z)-y*z) + .5*penalty*(w @ w))
    residual = float(np.max(np.abs(gradient)))
    if (not np.isfinite(w).all() or not np.isfinite(residual) or residual > tolerance
            or not np.isclose(loss, recorded["loss"], rtol=0, atol=1e-10)):
        raise ValueError("Independent full-pool penalized-BT gradient/objective check failed")
    return {"passed": True, "gradient_inf": residual, "penalized_loss": loss, "tolerance": tolerance}


def _subsets(dev_v, dev_y, dev, w):
    subsets = {r["source"]["subset"] for r in dev}
    result = {}
    for subset in sorted(subsets | {"helpful", "harmless"}):
        mask = np.array([r["source"]["subset"].startswith(subset) if subset in ("helpful", "harmless")
                         else r["source"]["subset"] == subset for r in dev])
        if mask.any():
            result[subset] = prediction_metrics(dev_v[mask], dev_y[mask], w)
    return result


def fit_full_pool_candidate(c, penalty):
    if penalty not in c["reward"]["l2_candidates"]:
        raise ValueError("Use the same predeclared Reward regularization grid")
    run = Run(c)
    tag = lambda_name(penalty)
    stage = f"full-pool-reward-l2-{tag}"
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(run.file(f"{PREFIX}/l2-{tag}/report.json"))
        v, y, dev_v, dev_y, dev, dependencies = _inputs(c, run)
        weight_path, marker = (run.file(f"{PREFIX}/l2-{tag}/{name}") for name in ("weights.npz", "report.json"))
        context = {"l2": float(penalty), "all_training_pairs": True, "backbone_trainable": False}
        if recover_stage(run, stage, [weight_path, marker], dependencies, code_paths=_sources(), context=context):
            return read_json(marker)
        print(f"full-pool-reward start rows={len(v)} dimension={v.shape[1]} l2={penalty}", flush=True)
        started = time.monotonic()
        options = c["reward"]
        w, fit = fit_bt(v, y, max_iter=options["max_iter"], gradient_tolerance=options["gradient_tolerance"],
                       max_weight_norm=options["max_weight_norm"], l2_regularization=penalty)
        independent = _verify(v, y, w, penalty, fit, options["gradient_tolerance"])
        edition = "v4" if c.get("protocol") == "hh_finite_pool_v4" else "v3"
        report = {"protocol": edition + "_full_pool_frozen_reward_diagnostic_v1", "training_pairs": len(y),
                  "development_pairs": len(dev_y), "feature_dimension": v.shape[1], "l2": float(penalty),
                  "fit": fit, "independent_fit_check": independent,
                  "training": prediction_metrics(v, y, w), "development": prediction_metrics(dev_v, dev_y, w),
                  "development_subsets": _subsets(dev_v, dev_y, dev, w),
                  "training_label_one_fraction": float(y.mean()), "development_label_one_fraction": float(dev_y.mean()),
                  "backbone_trainable": False, "feature_map": "unchanged_frozen_SFT_FP32_EOS",
                  "mean_loss_regularization": True, "formal_N14000_artifacts_modified": False,
                  "final_test_used": False, "api_calls": 0, "elapsed_seconds": time.monotonic()-started}
        save_npz(weight_path, w=w)
        write_json(marker, report)
        seal_and_publish(run, stage, [weight_path, marker], dependencies, metadata={"l2": float(penalty), "rows": len(y)},
                         code_paths=_sources(), context=context)
        return report


def summarize_full_pool(c):
    run = Run(c)
    stage = "full-pool-reward-report"
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(run.file(f"{PREFIX}/summary.json"))
        v, y, dev_v, dev_y, dev, dependencies = _inputs(c, run)
        reports = []
        for penalty in c["reward"]["l2_candidates"]:
            tag = lambda_name(penalty)
            name = f"full-pool-reward-l2-{tag}"
            run.require(name)
            dependencies.append(run.receipt_path(name))
            report = read_json(run.file(f"{PREFIX}/l2-{tag}/report.json"))
            with np.load(run.file(f"{PREFIX}/l2-{tag}/weights.npz"), allow_pickle=False) as file:
                w = file["w"]
            _verify(v, y, w, penalty, report["fit"], c["reward"]["gradient_tolerance"])
            for split, vv, yy in (("training", v, y), ("development", dev_v, dev_y)):
                actual = prediction_metrics(vv, yy, w)
                for key in ("accuracy", "bt_nll", "brier", "confidence_ece"):
                    if not np.isclose(actual[key], report[split][key], rtol=0, atol=1e-10):
                        raise ValueError("Full-pool summary independent metric recomputation differs")
            reports.append(report)
        best = min(reports, key=lambda r: (r["development"]["bt_nll"], -r["l2"]))
        reference = {}
        if run.completed("reward"):
            existing = read_json(run.file("reward/selection.json"))
            original = next(r for r in existing["candidates"] if r["l2_regularization"] == existing["selected_l2"])
            reference = {"budget": existing["budget"], "l2": existing["selected_l2"], "methods": original["methods"]}
            dependencies.append(run.receipt_path("reward"))
        edition = "v4" if c.get("protocol") == "hh_finite_pool_v4" else "v3"
        result = {"protocol": edition + "_full_pool_frozen_reward_diagnostic_v1", "diagnostic_only": True,
                  "training_pairs": len(y), "development_pairs": len(dev_y), "candidates": reports,
                  "selected_l2": best["l2"], "selection_metric": "same_development_BT_NLL",
                  "full_pool_selected": {k: best[k] for k in ("training", "development", "development_subsets")},
                  "N14000_reference": reference, "backbone_updated": False, "PPO_started": False,
                  "formal_reward_replaced": False, "final_test_used": False, "api_calls": 0,
                  "independent_audit_passed": True}
        target = run.file(f"{PREFIX}/summary.json")
        write_json(target, result)
        seal_and_publish(run, stage, [target], dependencies, metadata={"selected_l2": best["l2"], "rows": len(y)},
                         code_paths=_sources())
        return result
