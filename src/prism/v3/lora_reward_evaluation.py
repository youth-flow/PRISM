"""Independent checkpoint scoring of the exact consumed training prefixes.

Evaluation is separate from the immutable training executor so pending and
running training jobs keep their original source and resume contract.
"""
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit

from prism.io import digest, rows
from .artifacts import Run, hash_json, read_json, write_json
from .lora_reward import (CACHE_PREFIX, PREFIX, RECIPE, STAGE_PREFIX, PackedPairs, _load,
                          _restore, batch_boundaries, frozen_hash, predict, sources as model_sources)
from .recovery import seal_and_publish, stage_lock
from .reward import save_npz


def sources():
    return [Path(__file__), *model_sources()]


def checkpoint_targets(training):
    """The four requested snapshots plus the development-selected snapshot."""
    count = training["training_pairs"]
    batches = batch_boundaries(count)
    targets = {}
    milestones = sorted({n for n in RECIPE["checkpoint_pairs"] if n < count} | {count})
    for seen in milestones:
        matches = [r for r in training["history"] if r["pairs_seen"] == seen]
        if len(matches) != 1:
            raise ValueError("Missing or repeated exact Reward milestone")
        row = matches[0]
        step = row["step"]
        if not 1 <= step <= len(batches) or batches[step-1][1] != seen:
            raise ValueError("Reward milestone step/cursor mismatch")
        targets[str(seen)] = {"step": step, "pairs_seen": seen,
                              "checkpoint": training["milestone_checkpoints"][str(seen)],
                              "development": row["development"]}
    selected = next(r for r in training["history"] if r["step"] == training["best_step"])
    if selected["pairs_seen"] != (batches[selected["step"]-1][1] if selected["step"] else 0):
        raise ValueError("Selected Reward cursor mismatch")
    targets["selected"] = {"step": selected["step"], "pairs_seen": selected["pairs_seen"],
                           "checkpoint": training["selected_checkpoint"], "development": selected["development"]}
    return targets


def independent_metrics(scores, labels):
    """Recompute pair metrics independently of the training metric helper."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if (scores.shape != (len(labels), 2) or not np.isfinite(scores).all()
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Invalid checkpoint prediction arrays")
    if not len(labels):
        return None  # The initial checkpoint has consumed no training labels.
    z = scores[:, 0]-scores[:, 1]
    signed = (2*labels-1)*z
    correct = np.where(z == 0., .5, (z > 0) == labels)
    return {"pairs": len(labels), "accuracy": float(np.mean(correct)),
            "bt_nll": float(-np.mean(log_expit(signed))),
            "brier": float(np.mean((expit(z)-labels)**2))}


def validate_checkpoint(checkpoint, target, contract):
    import torch
    if (checkpoint["step"] != target["step"] or checkpoint["seen"] != target["pairs_seen"]
            or checkpoint["contract"] != contract):
        raise ValueError("Reward checkpoint source/input/cursor mismatch")
    states = checkpoint["optimizer"]["state"]
    if target["step"] and len(states) != len(checkpoint["trainable"]):
        raise ValueError("Reward Adam state does not cover all trainable tensors")
    for state in states.values():
        if int(state["step"].item()) != target["step"]:
            raise ValueError("Reward true Adam step count disagrees with checkpoint")
        if any(not torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
            raise ValueError("Nonfinite Reward optimizer state")


def score(c, shard_index, shards):
    import torch
    if type(shards) is not int or not 1 <= shards <= 50 or not 0 <= shard_index < shards:
        raise ValueError("Invalid independent Reward scoring shard coverage")
    run, stage = Run(c), f"{STAGE_PREFIX}-score-{shard_index:03d}-of-{shards:03d}"
    marker = run.file(f"{PREFIX}/evaluation/part-{shard_index:03d}-of-{shards:03d}.json")
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(marker)
        run.require(f"{STAGE_PREFIX}-train")
        training = read_json(run.file(f"{PREFIX}/training.json"))
        targets = checkpoint_targets(training)
        run, inputs, model, data, dev, _ = _load(c)
        if frozen_hash(model) != read_json(run.file(f"{PREFIX}/smoke-micro4.json"))["frozen_base_sha256"]:
            raise ValueError("Reward scoring base weights differ from the GPU acceptance")
        contract = {"recipe_sha256": hash_json(RECIPE), "input_receipt": digest(run.receipt_path("lora-reward-data")),
                    "sources": {p.name: digest(p) for p in model_sources()}, "config": run.fingerprint}
        order = np.random.default_rng(RECIPE["seed"]).permutation(len(data))
        outputs, snapshots = [], {}
        for step in sorted({v["step"] for v in targets.values()}):
            target = next(v for v in targets.values() if v["step"] == step)
            checkpoint = torch.load(run.file(target["checkpoint"]), map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, target, contract)
            _restore(model, checkpoint)
            arrays = {}
            for name, dataset, indices in (
                ("training", data, order[:target["pairs_seen"]][shard_index::shards]),
                ("development", dev, np.arange(shard_index, len(dev), shards))):
                arrays.update({name+"_indices": indices, name+"_ids": dataset.ids[indices],
                               name+"_scores": predict(model, dataset, indices, inputs["pad"])})
            path = run.file(f"{PREFIX}/evaluation/step-{step:06d}-part-{shard_index:03d}-of-{shards:03d}.npz")
            save_npz(path, **arrays)
            outputs.append(path)
            snapshots[str(step)] = {"path": str(path.relative_to(run.path)), "pairs_seen": target["pairs_seen"],
                                    "checkpoint_sha256": digest(run.file(target["checkpoint"])),
                                    "optimizer_state_passed": True}
            print(f"reward-checkpoint-score shard={shard_index}/{shards} step={step} seen={target['pairs_seen']}", flush=True)
        report = {"shard_index": shard_index, "shards": shards, "snapshots": snapshots,
                  "training_scope": "exact_consumed_shuffle_prefix", "device": torch.cuda.get_device_name(0),
                  "targets": targets, "PPO_started": False}
        write_json(marker, report)
        seal_and_publish(run, stage, [*outputs, marker], [run.receipt_path(f"{STAGE_PREFIX}-train")],
                         metadata={"passed": True, "shards": shards, "index": shard_index},
                         code_paths=sources(), context=RECIPE)
        return report


def summarize(c, shards):
    run, stage = Run(c), f"{STAGE_PREFIX}-report"
    with stage_lock(run, stage):
        marker = run.file(f"{PREFIX}/summary.json")
        if run.completed(stage):
            return read_json(marker)
        verified, digests = {}, {}
        run.require(f"{STAGE_PREFIX}-train", _verified=verified, _digests=digests)
        training = read_json(run.file(f"{PREFIX}/training.json"))
        targets = checkpoint_targets(training)
        data, dev = (PackedPairs(run.file(f"{CACHE_PREFIX}/{s}")) for s in ("train", "dev"))
        order = np.random.default_rng(RECIPE["seed"]).permutation(len(data))
        dev_rows = rows(run.file("data/reward_dev.jsonl"))
        dependencies, parts = [run.receipt_path(f"{STAGE_PREFIX}-train")], []
        for index in range(shards):
            name = f"{STAGE_PREFIX}-score-{index:03d}-of-{shards:03d}"
            run.require(name, _verified=verified, _digests=digests)
            dependencies.append(run.receipt_path(name))
            part = read_json(run.file(f"{PREFIX}/evaluation/part-{index:03d}-of-{shards:03d}.json"))
            if part["shard_index"] != index or part["shards"] != shards or part["targets"] != targets:
                raise ValueError("Checkpoint score partition/target manifest mismatch")
            parts.append(part)
        results = {}
        for key, target in targets.items():
            seen, step = target["pairs_seen"], target["step"]
            checkpoint_sha = digests[run.file(target["checkpoint"])]
            metrics, dev_scores = {}, None
            for split, dataset, expected in (("training", data, order[:seen]),
                                             ("development", dev, np.arange(len(dev)))):
                merged, cover = np.full((len(expected), 2), np.nan), np.zeros(len(expected), dtype=np.int8)
                for index, part in enumerate(parts):
                    snapshot = part["snapshots"][str(step)]
                    if (snapshot["pairs_seen"] != seen or not snapshot["optimizer_state_passed"]
                            or snapshot["checkpoint_sha256"] != checkpoint_sha):
                        raise ValueError("Checkpoint scoring provenance mismatch")
                    positions = np.arange(index, len(expected), shards)
                    with np.load(run.file(snapshot["path"]), allow_pickle=False) as f:
                        if (not np.array_equal(f[split+"_indices"], expected[positions])
                                or f[split+"_ids"].tolist() != dataset.ids[expected[positions]].tolist()):
                            raise ValueError("Checkpoint score prefix order/identity mismatch")
                        merged[positions] = f[split+"_scores"]
                    cover[positions] += 1
                if not np.all(cover == 1) or not np.isfinite(merged).all():
                    raise ValueError("Checkpoint score coverage or finiteness failed")
                metrics[split] = independent_metrics(merged, dataset.labels[expected])
                if split == "development":
                    dev_scores = merged
            if abs(metrics["development"]["bt_nll"]-target["development"]["bt_nll"]) > 2e-5:
                raise ValueError("Independent checkpoint development NLL differs from training-time evaluation")
            subsets = {}
            for subset in sorted({r["source"]["subset"] for r in dev_rows}):
                mask = np.asarray([r["source"]["subset"] == subset for r in dev_rows])
                subsets[subset] = independent_metrics(dev_scores[mask], dev.labels[mask])
            results[key] = {"step": step, "pairs_seen": seen, **metrics, "development_subsets": subsets,
                            "checkpoint": target["checkpoint"]}
        baseline = read_json(run.file("diagnostics/full-pool-reward/l2-0.01/report.json"))
        report = {"recipe": RECIPE, "learning_curve": [results[k] for k in targets if k != "selected"],
                  "selected": results["selected"], "linear_full_pool_reference":
                      {s: baseline[s] for s in ("training", "development")},
                  "independent_shard_coverage_passed": True, "optimizer_state_passed": True,
                  "training_accuracy_scope": "only_pairs_consumed_at_each_checkpoint",
                  "fresh_head_no_full_pool_label_warmstart": True, "single_epoch": True,
                  "full_schedule_continuous_checkpoints": True, "diagnostic_only": True,
                  "PPO_started": False, "final_test_used": False, "api_calls": 0}
        write_json(marker, report)
        seal_and_publish(run, stage, [marker], dependencies, metadata={"passed": True}, code_paths=sources(), context=RECIPE)
        return report
