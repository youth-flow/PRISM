"""Independent human-label, sampling-law and penalized-BT numerical checks."""
import hashlib

import numpy as np
from scipy.special import expit

from prism.io import digest, rows
from .artifacts import Run, read_json, write_json
from .reward import prediction_metrics
from .study import lambda_name


def _independent_dependent_rounding(inclusion, seed):
    """Independent replay of V6 pairwise rounding and its output ordering."""
    values = np.asarray(inclusion, dtype=np.float64).copy()
    size = int(round(float(values.sum())))
    values = np.clip(values, 0.0, 1.0)
    values[values < 1e-14] = 0.0
    values[values > 1 - 1e-14] = 1.0
    rng = np.random.default_rng(seed)
    pending = np.flatnonzero((values > 0) & (values < 1)).tolist()
    rng.shuffle(pending)
    while len(pending) >= 2:
        i, j = pending.pop(), pending.pop()
        up, down = min(1 - values[i], values[j]), min(values[i], 1 - values[j])
        if rng.random() < down / (up + down):
            values[i] += up
            values[j] -= up
        else:
            values[i] -= down
            values[j] += down
        for index in (i, j):
            if values[index] < 1e-12:
                values[index] = 0.0
            elif values[index] > 1 - 1e-12:
                values[index] = 1.0
            else:
                pending.append(index)
    if pending:
        index = pending.pop()
        values[index] = float(size - int(np.count_nonzero(values == 1)))
    selected = np.flatnonzero(values == 1)
    if len(selected) != size:
        raise ValueError("Independent V6 dependent-rounding replay failed")
    return rng.permutation(selected)


def audit_training(c):
    run = Run(c)
    if run.completed("training-audit"):
        return read_json(run.file("audits/training.json"))
    selection = run.selection()
    for stage in ("sampling", "reward", "features", "geometry"):
        run.require(stage)
    dependencies = [run.receipt_path(stage) for stage in ("study-selection", "sampling", "reward", "features", "geometry")]
    with np.load(run.file("geometry/geometry.npz"), allow_pickle=False) as f:
        ids, v, empirical = f["ids"].astype(str), f["v"], f["empirical"]
    labels_rows = rows(run.file("data/labels.jsonl"))
    labels = {r["id"]: r["label"] for r in labels_rows}
    if len(labels) != len(labels_rows) or len(set(ids)) != len(ids):
        raise ValueError("Training audit found duplicate pair/label identities")
    path = run.file("design/lambda-" + lambda_name(selection["design_lambda"]) + "/q.npy")
    if digest(path) != selection["q_sha256"]:
        raise ValueError("Training audit Design identity changed")
    weights = {"empirical": empirical, "prism": np.load(path, allow_pickle=False)}
    v6 = c.get("v6") if isinstance(c.get("v6"), dict) else None
    selected, sample_reports, previous_inputs = {}, {}, []
    for method in c["study"]["methods"]:
        records = rows(run.file(f"sampling/{method}.jsonl"))
        metadata = read_json(run.file(f"sampling/{method}.json"))
        index = np.asarray([r["geometry_index"] for r in records], dtype=np.int64)
        if (len(records) != selection["budget"] or len(set(index)) != len(index)
                or np.any(index < 0) or np.any(index >= len(ids))
                or [r["selection_index"] for r in records] != list(range(len(records)))
                or [r["id"] for r in records] != ids[index].tolist()
                or any(type(r["label"]) is not int or r["label"] != labels[r["id"]] for r in records)):
            raise ValueError("Training audit human labels, direction or distinct sampling identity changed")
        w = weights[method]
        positive = np.flatnonzero(w > 0)
        if np.any(w[index] <= 0):
            raise ValueError("Training audit found a sampled zero-weight pair")
        if v6 and method == "prism":
            inclusion = selection["budget"] * w
            ranking = _independent_dependent_rounding(inclusion, selection["seed"])
            inclusion_hash = hashlib.sha256(
                inclusion.astype("<f8", copy=False).tobytes()).hexdigest()
            if (not np.array_equal(index, ranking)
                    or metadata.get("protocol") != "fixed_size_dependent_rounding_v1"
                    or metadata.get("ordered_indices") != index.tolist()
                    or metadata.get("pair_ids") != ids[index].tolist()
                    or metadata.get("inclusion_probabilities_sha256") != inclusion_hash
                    or not np.allclose(metadata.get("selected_inclusion_probabilities"),
                                       inclusion[index], rtol=0, atol=1e-12)
                    or metadata.get("fixed_size") is not True
                    or metadata.get("replacement") is not False
                    or metadata.get("design_weights_are_inclusion_probabilities_divided_by_budget") is not True):
                raise ValueError("Training audit V6 dependent-rounding record changed")
        else:
            noise = np.random.default_rng(selection["seed"]).gumbel(size=len(positive))
            scores = np.log(w[positive]) + noise
            # Lexicographic tie breaking independently reproduces stable Gumbel rank.
            ranking = positive[np.lexsort((positive, -scores))]
            if (not np.array_equal(index, ranking[:len(index)]) or metadata["ordered_indices"] != index.tolist()
                    or metadata["pair_ids"] != ids[index].tolist() or metadata.get("common_gumbel_keys") is not False
                    or metadata.get("shared_rng_seed") is not True or metadata.get("gumbel_key_alignment") != "positive_support_order"):
                raise ValueError("Training audit weighted-WOR ordering or coupling record changed")
            # Independent extended-precision suffix sums, rather than logaddexp scan.
            suffix = np.cumsum(w[ranking[::-1]].astype(np.longdouble))[::-1]
            log_remaining = np.log(suffix).astype(np.float64)
            base = np.log(w[index]) - log_remaining[0]
            conditional = np.log(w[index]) - log_remaining[:len(index)]
            if (not np.allclose(base, metadata["base_log_probabilities"], rtol=0, atol=1e-10)
                    or not np.allclose(conditional, metadata["conditional_log_probabilities"], rtol=0, atol=1e-10)
                    or metadata.get("inclusion_probabilities") is not None):
                raise ValueError("Training audit sequential probabilities differ from remaining-weight law")
        selected[method] = (v[index], np.array([r["label"] for r in records], dtype=float))
        previous = run.file(f"attempts/sampling-metadata-20261117/sampling/{method}.jsonl")
        same_previous = None
        if previous.exists():
            if previous.read_bytes() != run.file(f"sampling/{method}.jsonl").read_bytes():
                raise ValueError("Sampling metadata correction changed the original scientific selection")
            previous_inputs.append(previous)
            same_previous = True
        sample_reports[method] = {"distinct_pairs": len(index), "positive_support": len(positive),
                                  "human_labels_exact": True, "independent_wor_replay_exact": True,
                                  "sampling_protocol": metadata["protocol"],
                                  "corrected_metadata_selection_unchanged": same_previous}
    with np.load(run.file("features/reward_dev.npz"), allow_pickle=False) as f:
        phi, dev_ids = f["phi"].astype(np.float64), f["ids"].astype(str)
    dev = rows(run.file("data/reward_dev.jsonl"))
    if dev_ids.tolist() != [r["id"] for r in dev] or set(dev_ids) & set(ids):
        raise ValueError("Independent Reward development identities overlap or are misaligned")
    dev_v = phi[:, 0] - phi[:, 1]
    dev_labels = np.asarray([r["label"] for r in dev], dtype=float)
    candidates = []
    for penalty in c["reward"]["l2_candidates"]:
        report = read_json(run.file(f"reward/candidates/l2-{lambda_name(penalty)}/report.json"))
        checks = {}
        for method, (train_v, train_labels) in selected.items():
            with np.load(run.file(f"reward/candidates/l2-{lambda_name(penalty)}/{method}.npz"), allow_pickle=False) as f:
                w = f["w"].astype(np.float64)
            if w.shape != (v.shape[1],) or not np.isfinite(w).all():
                raise ValueError("Reward audit found invalid head weights")
            z = train_v @ w
            gradient = train_v.T @ (expit(z)-train_labels) / len(z) + penalty*w
            loss = float(np.mean(np.logaddexp(0, z)-train_labels*z) + .5*penalty*(w @ w))
            norm = float(np.max(np.abs(gradient)))
            recorded = report["methods"][method]
            if norm > c["reward"]["gradient_tolerance"] or not np.isclose(loss, recorded["fit"]["loss"], rtol=0, atol=1e-10):
                raise ValueError("Independent penalized-BT gradient/objective failed")
            for split, vv, yy in (("training", train_v, train_labels), ("development", dev_v, dev_labels)):
                actual = prediction_metrics(vv, yy, w)
                for key in ("bt_nll", "accuracy", "brier", "confidence_ece"):
                    if not np.isclose(actual[key], recorded[split][key], rtol=0, atol=1e-10):
                        raise ValueError("Reward audit metric differs from stored human-label predictions")
            checks[method] = {"gradient_inf": norm, "penalized_loss": loss,
                              "training": recorded["training"], "development": recorded["development"]}
        mean = float(np.mean([row["development"]["bt_nll"] for row in checks.values()]))
        candidates.append({"l2": float(penalty), "mean_development_bt_nll": mean, "methods": checks})
    best = min(candidates, key=lambda row: (row["mean_development_bt_nll"], -row["l2"]))
    chosen = read_json(run.file("reward/selection.json"))
    if chosen["selected_l2"] != best["l2"]:
        raise ValueError("Common Reward regularizer does not minimize shared development NLL")
    for method in c["study"]["methods"]:
        with np.load(run.file(f"reward/{method}.npz")) as a, np.load(run.file(f"reward/candidates/l2-{lambda_name(best['l2'])}/{method}.npz")) as b:
            if not np.array_equal(a["w"], b["w"]):
                raise ValueError("Selected Reward head differs from the accepted common-L2 fit")
    value = {"protocol": "v3_independent_training_data_and_bt_audit_v1", "passed": True,
             "budget": selection["budget"], "sampling": sample_reports, "selected_l2": best["l2"],
             "candidates": candidates, "reward_backbone_trainable": False, "api_calls": 0}
    target = run.file("audits/training.json")
    write_json(target, value)
    run.publish("training-audit", [target], dependencies+previous_inputs)
    return value
