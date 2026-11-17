"""Common-development selection of frozen-feature human BT reward heads."""
from io import BytesIO

import numpy as np
from scipy.special import expit

from prism.io import digest, rows
from prism.reward import fit_bt
from .artifacts import Run, read_json, write_json
from .study import lambda_name


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], np.asarray(v)) for k, v in arrays.items()):
                raise ValueError(f"Preserve existing V3 numerical artifact: {path}")
        return
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())


def prediction_metrics(v, labels, w):
    v, labels, w = (np.asarray(x, dtype=np.float64) for x in (v, labels, w))
    if v.ndim != 2 or labels.shape != (len(v),) or w.shape != (v.shape[1],) or not len(v):
        raise ValueError("Invalid Reward validation arrays")
    if not np.isfinite(v).all() or not np.isfinite(w).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Reward validation needs finite features and binary labels")
    logits = v @ w
    probabilities = expit(logits)
    correct = np.where(logits == 0, 0.5, (logits > 0) == labels)
    confidence = np.maximum(probabilities, 1 - probabilities)
    bins, ece = [], 0.0
    for lower in np.linspace(0.5, 0.95, 10):
        upper = lower + 0.05
        mask = (confidence >= lower) & (confidence < upper if upper < 1 - 1e-12 else confidence <= 1)
        if not mask.any():
            continue
        accuracy, mean_confidence = float(correct[mask].mean()), float(confidence[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - mean_confidence)
        bins.append({"lower": float(lower), "upper": min(1.0, float(upper)), "count": int(mask.sum()),
                     "accuracy": accuracy, "mean_confidence": mean_confidence})
    return {"pairs": len(v), "bt_nll": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
            "accuracy": float(correct.mean()), "brier": float(np.mean((probabilities - labels) ** 2)),
            "confidence_ece": ece, "confidence_bins": bins}


def _inputs(c, run):
    run.selection()
    run.require("sampling")
    run.require("features")
    run.require("geometry")
    geometry_path = run.file("geometry/geometry.npz")
    with np.load(geometry_path, allow_pickle=False) as array:
        v, ids = array["v"].astype(np.float64), array["ids"].astype(str)
    dev_path = run.file("features/reward_dev.npz")
    with np.load(dev_path, allow_pickle=False) as array:
        phi, dev_ids = array["phi"].astype(np.float64), array["ids"].astype(str)
    if phi.ndim != 3 or phi.shape[1:] != (2, v.shape[1]) or len(dev_ids) != len(phi) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("Reward development features have invalid candidate shape or identities")
    dev_rows = rows(run.file("data/reward_dev.jsonl"))
    if [r["id"] for r in dev_rows] != dev_ids.tolist() or set(ids) & set(dev_ids):
        raise ValueError("Reward development features are misaligned or overlap training")
    dev_labels = np.asarray([r["label"] for r in dev_rows], dtype=np.float64)
    selections = {}
    for method in c["study"]["methods"]:
        records = rows(run.file(f"sampling/{method}.jsonl"))
        indices = np.asarray([r["geometry_index"] for r in records], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(v)) or len(set(indices)) != len(indices):
            raise ValueError("Reward training selection contains invalid or repeated pairs")
        if ids[indices].tolist() != [r["id"] for r in records]:
            raise ValueError("Reward training features do not match selected human comparisons")
        selections[method] = (v[indices], np.asarray([r["label"] for r in records], dtype=np.float64))
    return selections, phi[:, 0] - phi[:, 1], dev_labels, [geometry_path, dev_path, run.file("data/reward_dev.jsonl")]


def fit_rewards(c):
    run = Run(c)
    if run.completed("reward"):
        return read_json(run.file("reward/selection.json"))
    selections, dev_v, dev_labels, dependencies = _inputs(c, run)
    all_candidates = []
    options = c["reward"]
    for penalty in options["l2_candidates"]:
        stage = "reward-candidate-" + lambda_name(penalty)
        directory = run.file("reward/candidates/l2-" + lambda_name(penalty))
        if run.completed(stage):
            candidate = read_json(directory / "report.json")
        else:
            candidate = {"l2_regularization": float(penalty), "methods": {}}
            outputs = []
            for method, (v, labels) in selections.items():
                w, report = fit_bt(v, labels, max_iter=options["max_iter"],
                                  gradient_tolerance=options["gradient_tolerance"],
                                  max_weight_norm=options["max_weight_norm"], l2_regularization=penalty)
                if report["gradient_inf"] > options["gradient_tolerance"]:
                    raise ValueError(f"Reward candidate has not reached the V3 gradient target: {report}")
                path = directory / (method + ".npz")
                save_npz(path, w=w)
                candidate["methods"][method] = {"fit": report,
                    "training": prediction_metrics(v, labels, w),
                    "development": prediction_metrics(dev_v, dev_labels, w)}
                outputs.append(path)
            candidate["mean_development_bt_nll"] = float(np.mean([
                m["development"]["bt_nll"] for m in candidate["methods"].values()]))
            report_path = directory / "report.json"
            write_json(report_path, candidate)
            run.publish(stage, outputs + [report_path], dependencies + [run.receipt_path("sampling"), run.receipt_path("features"), run.receipt_path("geometry")])
        all_candidates.append(candidate)
    selected = min(all_candidates, key=lambda row: (row["mean_development_bt_nll"], -row["l2_regularization"]))
    penalty = selected["l2_regularization"]
    outputs = []
    for method in c["study"]["methods"]:
        source = run.file(f"reward/candidates/l2-{lambda_name(penalty)}/{method}.npz")
        with np.load(source, allow_pickle=False) as array:
            w = array["w"]
        path = run.file(f"reward/{method}.npz")
        save_npz(path, w=w)
        outputs.append(path)
    report = {"protocol": "v3_shared_development_l2_v1", "budget": run.selection()["budget"],
              "selected_l2": penalty, "selection_metric": "equal_method_mean_development_bt_nll",
              "tie_break": "larger_L2", "candidates": all_candidates,
              "reward_backbone_trainable": False, "new_annotation_api_calls": 0}
    target = run.file("reward/selection.json")
    write_json(target, report)
    run.publish("reward", outputs + [target], [run.receipt_path("reward-candidate-" + lambda_name(v)) for v in options["l2_candidates"]])
    return report


def normalize_rewards(c):
    run = Run(c)
    if run.completed("normalization"):
        return read_json(run.file("normalization/summary.json"))
    run.require("reward")
    run.require("shared-sft")
    path = run.file("shared/calibration.features.npz")
    with np.load(path, allow_pickle=False) as array:
        phi, ids = array["phi"].astype(np.float32), array["ids"].astype(str)
    if phi.ndim != 2 or len(phi) != c["reward"]["calibration_responses"] or len(ids) != len(phi) or len(set(ids)) != len(ids) or not np.isfinite(phi).all():
        raise ValueError("Invalid common frozen-SFT calibration features")
    summary, outputs = {}, []
    for method in c["study"]["methods"]:
        with np.load(run.file(f"reward/{method}.npz"), allow_pickle=False) as array:
            w = array["w"].astype(np.float32)
        if w.shape != (phi.shape[1],):
            raise ValueError("Calibration and Reward use different feature maps")
        scores = (phi @ w).astype(np.float64)
        mean, std = float(scores.mean()), float(scores.std(ddof=0))
        if not np.isfinite(scores).all() or std <= 1e-8:
            raise ValueError("Reward calibration has no usable finite score scale")
        normalized = (scores - mean) / std
        value = {"protocol": "v3_fixed_sft_mean_std_v1", "method": method, "mean": mean, "std": std,
                 "calibration_responses": len(phi), "calibration_sha256": digest(path),
                 "score_dtype": "float32", "statistics_dtype": "float64", "ddof": 0,
                 "reference_normalized_mean": float(normalized.mean()),
                 "reference_normalized_std": float(normalized.std(ddof=0)), "frozen_after_fit": True}
        target = run.file(f"normalization/{method}.json")
        write_json(target, value)
        score_path = run.file(f"normalization/{method}.scores.npz")
        save_npz(score_path, ids=ids, scores=scores, normalized_scores=normalized)
        summary[method] = value
        outputs.extend([target, score_path])
    target = run.file("normalization/summary.json")
    write_json(target, summary)
    run.publish("normalization", outputs + [target], [run.receipt_path("reward"), run.receipt_path("shared-sft"), path])
    return summary
