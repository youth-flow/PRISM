"""Frozen 7B linear Reward learning curve with an untouched HH test split."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit

from prism.io import rows, write_rows
from prism.v3.artifacts import Run, hash_json, read_json, write_json
from prism.v3.data import _manifest_files, canonical_pair, encode_pair, load_tokenizer, tokenizer_fingerprint
from prism.v3.features import HFPairScorer, _arrays, _sources as feature_sources
from prism.v3.full_pool_reward import _inputs
from prism.v3.recovery import recover_stage, seal_and_publish, stage_lock
from prism.v3.reward import prediction_metrics, save_npz


PREFIX = "diagnostics/linear-reward-curve"
PARTIAL_BUDGETS = (10000, 50000, 100000)
BATCH = 256
LEARNING_RATE = 1e-3
L2 = 0.01


def _sources():
    shared = Path(__file__).resolve().parents[1]
    return [Path(__file__), *feature_sources(), shared / "v3" / "full_pool_reward.py",
            shared / "v3" / "reward.py", shared / "reward.py"]


def _test_rows(c, sources, blocked_prompts):
    pairs, conflicts, counts = {}, set(), Counter()
    for spec, path in sources:
        if spec["split"] != "test":
            continue
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                counts["raw_rows"] += 1
                source = {"subset": spec["subset"], "split": "test", "line": line,
                          "raw_sha256": hashlib.sha256(text.encode()).hexdigest()}
                try:
                    raw = json.loads(text)
                    pair, label = canonical_pair(raw, source)
                except (ValueError, TypeError, AttributeError):
                    counts["invalid_rows"] += 1
                    continue
                if pair["prompt_id"] in blocked_prompts:
                    counts["excluded_context_overlap_rows"] += 1
                    continue
                pair.update(label=label["label"])
                first = pairs.setdefault(pair["id"], pair)
                if first is not pair:
                    counts["duplicate_rows"] += 1
                    if first["label"] != pair["label"]:
                        conflicts.add(pair["id"])
    if counts["raw_rows"] != c["data"]["expected_test_rows"]:
        raise ValueError("Pinned HH official test row count differs")
    counts["conflicting_pairs"] = len(conflicts)
    selected = [pairs[identity] for identity in sorted(pairs.keys() - conflicts)]
    counts["test_pairs"] = len(selected)
    if not selected:
        raise ValueError("No isolated official HH test pairs")
    return selected, dict(counts)


def prepare_test(c, source_manifest):
    if not source_manifest:
        raise ValueError("Provide the pinned HH raw --source-manifest")
    run, stage = Run(c), "linear-test-data"
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(run.file(f"{PREFIX}/test-data.json"))
        run.require("cohort-audit")
        run.require("data")
        manifest, sources = _manifest_files(c, source_manifest)
        inputs = [run.receipt_path("data"), run.receipt_path("cohort-audit"), manifest]
        inputs.extend(path for spec, path in sources if spec["split"] == "test")
        data, report = (run.file(f"{PREFIX}/{name}") for name in ("test.jsonl", "test-data.json"))
        if recover_stage(run, stage, [data, report], inputs, code_paths=_sources()):
            return read_json(report)
        blocked = {r["prompt_id"] for name in ("pool", "reward_dev")
                   for r in rows(run.file(f"data/{name}.jsonl"))}
        blocked.update(r["id"] for name in ("ppo_dev", "evaluation")
                       for r in rows(run.file(f"data/{name}.jsonl")))
        selected, counts = _test_rows(c, sources, blocked)
        tokenizer = load_tokenizer(c)
        if tokenizer_fingerprint(tokenizer) != read_json(run.file("data/manifest.json"))["tokenizer_semantics"]:
            raise ValueError("Test tokenizer semantics differ from the training feature map")
        maximum = 0
        for record in selected:
            prompt, answers = encode_pair(record, tokenizer, c["data"]["context_tokens"])
            maximum = max(maximum, len(prompt) + max(map(len, answers)))
        result = {**counts, "maximum_pair_tokens": maximum, "source_revision": c["data"]["revision"],
                  "test_order_sha256": hash_json([r["id"] for r in selected]),
                  "test_unique_contexts": len({r["prompt_id"] for r in selected}),
                  "sources": dict(Counter(r["source"]["subset"] for r in selected))}
        write_rows(data, selected)
        write_json(report, result)
        seal_and_publish(run, stage, [data, report], inputs, code_paths=_sources(), metadata=result)
        return result


def extract_test_features(c):
    run, stage = Run(c), "linear-test-features"
    with stage_lock(run, stage):
        run.require("linear-test-data")
        run.require("model-smoke")
        target, marker = (run.file(f"{PREFIX}/{name}") for name in ("test-features.npz", "test-features.json"))
        if run.completed(stage):
            return read_json(marker)
        inputs = [run.receipt_path(s) for s in ("linear-test-data", "model-smoke")]
        if recover_stage(run, stage, [target, marker], inputs, code_paths=_sources()):
            return read_json(marker)
        test = rows(run.file(f"{PREFIX}/test.jsonl"))
        arrays = _arrays(test, range(len(test)), HFPairScorer(c), c["data"]["feature_dimension"], progress="v4 held-out test")
        result = {"test_pairs": len(test), "model": c["model"], "dtype": "float32",
                  "test_order_sha256": hash_json(arrays["ids"].tolist())}
        save_npz(target, **arrays)
        write_json(marker, result)
        seal_and_publish(run, stage, [target, marker], inputs, code_paths=_sources(), metadata=result)
        return result


def _train_order(run, seed):
    with np.load(run.file("features/pool.npz"), allow_pickle=False) as file:
        identities = file["ids"].tolist()
    order = np.random.default_rng(seed).permutation(len(identities))
    return order, [identities[i] for i in order]


def _budgets(run):
    run.require("features")
    full = read_json(run.file("features/manifest.json"))["pool_rows"]
    if full <= PARTIAL_BUDGETS[-1]:
        raise ValueError("Full valid HH training pool must exceed the final partial checkpoint")
    return (*PARTIAL_BUDGETS, full)


def _checkpoint_paths(run, budget):
    return [run.file(f"{PREFIX}/checkpoints/{budget}/{name}") for name in ("weights.npz", "train.json")]


def _expected_steps(budgets):
    result, previous, steps = {}, 0, 0
    for budget in budgets:
        steps += (budget - previous + BATCH - 1) // BATCH
        result[budget] = steps
        previous = budget
    return result


def _load_checkpoint(paths, budget, steps, prefix, options, dimension):
    weight, marker = paths
    report = read_json(marker)
    expected = {"training_pairs": budget, "optimizer_steps": steps, "training_prefix_sha256": prefix,
                "optimizer": options}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint training prefix or optimizer contract differs")
    with np.load(weight, allow_pickle=False) as data:
        arrays = [data[key] for key in ("w", "adam_first_moment", "adam_second_moment")]
        if data["step"].dtype != np.int64 or data["step"].shape != () or data["step"].item() != steps:
            raise ValueError("Checkpoint optimizer counter differs")
    if (any(a.dtype != np.float64 or a.shape != (dimension,) or not np.isfinite(a).all() for a in arrays)
            or np.any(arrays[2] < 0)):
        raise ValueError("Checkpoint head or moments are invalid")
    return arrays


def _optimizer(c):
    return {"name": "adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH,
            "l2": L2, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "order_seed": c["study"]["seed"]}


def _advance(features, labels, indices, arrays, step, options):
    w, first, second = arrays
    v, y = features[indices], labels[indices]
    grad = v.T @ (expit(v @ w) - y) / len(y) + options["l2"] * w
    beta1, beta2 = options["beta1"], options["beta2"]
    first *= beta1
    first += (1 - beta1) * grad
    second *= beta2
    second += (1 - beta2) * np.square(grad)
    denominator = np.sqrt(second / (1 - beta2**step)) + options["epsilon"]
    w -= options["learning_rate"] * first / ((1 - beta1**step) * denominator)
    if not np.isfinite(w).all() or not np.isfinite(second).all():
        raise FloatingPointError("Nonfinite linear Reward update")


def train(c):
    run = Run(c)
    with stage_lock(run, "linear-reward-train"):
        budgets = _budgets(run)
        features, labels, _, _, _, inputs = _inputs(c, run)
        options = _optimizer(c)
        order, identities = _train_order(run, options["order_seed"])
        dimension = features.shape[1]
        arrays = [np.zeros(dimension, dtype=np.float64) for _ in range(3)]
        expected_steps = _expected_steps(budgets)
        beginning = step = 0
        reports = []
        for budget in budgets:
            stage = f"linear-reward-{budget}"
            paths = _checkpoint_paths(run, budget)
            prefix = hash_json(identities[:budget])
            context = {"training_pairs": budget, "optimizer": options, "training_prefix_sha256": prefix}
            if run.completed(stage) or recover_stage(run, stage, paths, inputs, code_paths=_sources(), context=context):
                step = expected_steps[budget]
                arrays = _load_checkpoint(paths, budget, step, prefix, options, dimension)
                report = read_json(paths[1])
            else:
                started = time.monotonic()
                for start in range(beginning, budget, BATCH):
                    step += 1
                    _advance(features, labels, order[start:min(start+BATCH, budget)], arrays, step, options)
                if step != expected_steps[budget]:
                    raise RuntimeError("Linear Reward training did not consume the planned exact prefix")
                w, first, second = arrays
                report = {**context, "optimizer_steps": step, "weight_norm": float(np.linalg.norm(w)),
                          "interval_seconds": time.monotonic()-started}
                save_npz(paths[0], w=w, adam_first_moment=first, adam_second_moment=second,
                         step=np.asarray(step, dtype=np.int64))
                write_json(paths[1], report)
                seal_and_publish(run, stage, paths, inputs, code_paths=_sources(), context=context, metadata=report)
                print(f"linear-reward checkpoint pairs={budget} optimizer_steps={step}", flush=True)
            reports.append(report)
            inputs = [run.receipt_path(stage)]
            beginning = budget
        return reports


def summarize(c):
    run, stage = Run(c), "linear-reward-report"
    with stage_lock(run, stage):
        summary, curve = (run.file(f"{PREFIX}/{name}") for name in ("summary.json", "learning-curve.csv"))
        if run.completed(stage):
            return read_json(summary)
        budgets = _budgets(run)
        features, labels, dev_v, dev_y, _, inputs = _inputs(c, run)
        for dependency in ("linear-test-data", "linear-test-features", *(f"linear-reward-{n}" for n in budgets)):
            run.require(dependency)
            inputs.append(run.receipt_path(dependency))
        if recover_stage(run, stage, [summary, curve], inputs, code_paths=_sources()):
            return read_json(summary)
        test = rows(run.file(f"{PREFIX}/test.jsonl"))
        test_y = np.asarray([row["label"] for row in test], dtype=np.float64)
        with np.load(run.file(f"{PREFIX}/test-features.npz"), allow_pickle=False) as data:
            phi = data["phi"]
            if (data["ids"].tolist() != [row["id"] for row in test]
                    or not np.array_equal(data["indices"], np.arange(len(test)))
                    or phi.shape != (len(test), 2, features.shape[1]) or phi.dtype != np.float32
                    or not np.isfinite(phi).all() or not np.isin(test_y, [0, 1]).all()):
                raise ValueError("Independent test features do not match the labeled test cohort")
            test_v = np.subtract(phi[:, 0], phi[:, 1], dtype=np.float64)
        results, csv_rows = [], []
        options = _optimizer(c)
        order, identities = _train_order(run, options["order_seed"])
        step_counts = _expected_steps(budgets)
        for budget in budgets:
            paths = _checkpoint_paths(run, budget)
            prefix = hash_json(identities[:budget])
            w, _, _ = _load_checkpoint(paths, budget, step_counts[budget], prefix, options, features.shape[1])
            if budget == len(labels):
                train_v, train_y = features, labels
            else:
                train_v, train_y = features[order[:budget]], labels[order[:budget]]
            metrics = {"training": prediction_metrics(train_v, train_y, w),
                       "development": prediction_metrics(dev_v, dev_y, w),
                       "test": prediction_metrics(test_v, test_y, w)}
            results.append({**read_json(paths[1]), "checkpoint": paths[0].relative_to(run.path).as_posix(), **metrics})
            csv_rows.append({"training_pairs": budget, "optimizer_steps": step_counts[budget],
                             **{f"{name}_{key}": value for name, scores in metrics.items()
                                for key, value in scores.items() if key != "confidence_bins"}})
        result = {"protocol": "v4_frozen_linear_reward_learning_curve_v1", "model": c["model"],
                  "training_epochs": 1, "optimizer": options, "test_used_for_training_or_selection": False,
                  "development_pairs": len(dev_y), "test_cohort": read_json(run.file(f"{PREFIX}/test-data.json")),
                  "test_majority_class_accuracy": float(max(test_y.mean(), 1 - test_y.mean())),
                  "checkpoints": results, "independent_audit_passed": True}
        with curve.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        write_json(summary, result)
        seal_and_publish(run, stage, [summary, curve], inputs, code_paths=_sources(), metadata={"budgets": budgets})
        return result
