"""Isolated full-HH Reward LoRA diagnostic; never admitted as a policy reward.

The original V3 config/geometry/heads stay immutable. This module has its own
recipe, packed complete-answer inputs, checkpoints, predictions and receipts.
"""
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from prism.io import digest, rows
from .artifacts import Run, hash_json, read_json, write_json
from .data import encode_pair, tokenizer_fingerprint
from .recovery import seal_and_publish, stage_lock
from .reward import prediction_metrics, save_npz


PREFIX = "diagnostics/full-pool-lora-reward"
RECIPE = {
    "protocol": "v3_full_pool_reward_lora_v1", "seed": 11,
    "rank": 16, "alpha": 32, "dropout": 0.,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "epochs": 1, "effective_pairs": 64, "micro_pairs": 4,
    "adapter_learning_rate": 2e-5, "head_learning_rate": 1e-4,
    "head_l2": .01, "adapter_weight_decay": .01,
    "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
    "warmup_ratio": .03, "minimum_learning_rate_ratio": .1,
    "gradient_norm": 1., "checkpoint_steps": 256,
    "precision": "FP32_TF32_disabled", "gradient_checkpointing": "non_reentrant",
    "head_bias": False, "initialization": "full_pool_development_selected_linear_head_zero_LoRA",
    "selection": "development_mean_BT_NLL_including_initial_checkpoint",
    "truncate_human_answers": False, "final_policy_evaluation_used": False,
    "formal_PPO_reward": False, "api_calls": 0,
}


def sources():
    folder = Path(__file__).resolve().parent
    return [Path(__file__), folder / "data.py", folder / "ppo.py", folder / "ppo_trainer.py",
            folder / "ppo_math.py", folder.parent / "numerics.py"]


def logit_metrics(logits, labels):
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return prediction_metrics(z[:, None], y, np.ones(1))


def _pack(path, records, labels, tokenizer, context):
    """Packed int32 sequences, no pickle, padding, truncation or label remapping."""
    offsets, cursor, lengths = [], 0, []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tokens").open("xb") as stream:
        for r in records:
            prompt, answers = encode_pair(r, tokenizer, context)
            a, b = (np.asarray(prompt + answer, dtype=np.int32) for answer in answers)
            offsets.append((cursor, cursor + len(a), cursor + len(a) + len(b)))
            a.tofile(stream)
            b.tofile(stream)
            cursor += len(a) + len(b)
            lengths.append(max(len(a), len(b)))
    save_npz(path.with_suffix(".npz"), offsets=np.asarray(offsets, dtype=np.int64),
             ids=np.asarray([r["id"] for r in records]), labels=np.asarray(labels, dtype=np.int64))
    return {"pairs": len(records), "tokens": cursor, "max_sequence_tokens": max(lengths),
            "mean_longer_sequence_tokens": float(np.mean(lengths)),
            "longest_pair_index": int(np.argmax(lengths))}


class PackedPairs:
    def __init__(self, path):
        path = Path(path)
        self.tokens = np.memmap(path.with_suffix(".tokens"), mode="r", dtype=np.int32)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as f:
            self.offsets, self.ids, self.labels = f["offsets"], f["ids"].astype(str), f["labels"]
        if (self.offsets.shape != (len(self.ids), 3) or len(set(self.ids)) != len(self.ids)
                or self.labels.shape != (len(self.ids),) or not np.isin(self.labels, [0, 1]).all()
                or self.offsets[0, 0] != 0 or self.offsets[-1, -1] != len(self.tokens)
                or not np.all(np.diff(self.offsets, axis=1) > 0)
                or not np.array_equal(self.offsets[1:, 0], self.offsets[:-1, 2])):
            raise ValueError("Invalid packed complete HH pairs")

    def __len__(self):
        return len(self.ids)

    def pair(self, index):
        start, middle, end = self.offsets[index]
        return [self.tokens[start:middle].tolist(), self.tokens[middle:end].tolist()]


def prepare(c):
    run, stage = Run(c), "lora-reward-data"
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(run.file(f"{PREFIX}/data.json"))
        for name in ("data", "full-pool-reward-report"):
            run.require(name)
        from .ppo import load_tokenizer
        tokenizer = load_tokenizer(c)
        if tokenizer_fingerprint(tokenizer) != read_json(run.file("data/manifest.json"))["tokenizer_semantics"]:
            raise ValueError("LoRA tokenization differs from the frozen HH experiment")
        baseline = read_json(run.file("diagnostics/full-pool-reward/summary.json"))
        if baseline["selected_l2"] != RECIPE["head_l2"]:
            raise ValueError("Review LoRA initialization if the full-pool linear selection changes")
        pool, dev = rows(run.file("data/pool.jsonl")), rows(run.file("data/reward_dev.jsonl"))
        labels = {r["id"]: r["label"] for r in rows(run.file("data/labels.jsonl"))}
        if set(labels) != {r["id"] for r in pool} or {r["prompt_id"] for r in pool} & {r["prompt_id"] for r in dev}:
            raise ValueError("LoRA train/development labels or context separation failed")
        report = {"recipe": RECIPE, "recipe_sha256": hash_json(RECIPE), "eos": tokenizer.eos_token_id,
                  "pad": tokenizer.pad_token_id, "tokenizer_semantics": tokenizer_fingerprint(tokenizer),
                  "training": _pack(run.file(f"{PREFIX}/train"), pool, [labels[r["id"]] for r in pool], tokenizer,
                                    c["data"]["context_tokens"]),
                  "development": _pack(run.file(f"{PREFIX}/dev"), dev, [r["label"] for r in dev], tokenizer,
                                       c["data"]["context_tokens"])}
        write_json(run.file(f"{PREFIX}/data.json"), report)
        outputs = [run.file(f"{PREFIX}/{s}.{e}") for s in ("train", "dev") for e in ("tokens", "npz")]
        outputs.append(run.file(f"{PREFIX}/data.json"))
        seal_and_publish(run, stage, outputs, [run.receipt_path("data"), run.receipt_path("full-pool-reward-report")],
                         metadata=report, code_paths=sources(), context=RECIPE)
        return report


def make_model(backbone, weights, recipe=RECIPE):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    backbone.requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    adapted = get_peft_model(backbone, LoraConfig(r=recipe["rank"], lora_alpha=recipe["alpha"],
                             target_modules=recipe["target_modules"], lora_dropout=recipe["dropout"],
                             bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapted.config.use_cache = False

    class RewardModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = adapted
            self.head = torch.nn.Linear(len(weights), 1, bias=False, dtype=torch.float32,
                                        device=next(backbone.parameters()).device)
            with torch.no_grad():
                self.head.weight.copy_(torch.as_tensor(np.asarray(weights).copy(), dtype=torch.float32)[None])

        def forward(self, sequences, pad):
            from .ppo_trainer import left_pad, prepare_inputs
            tokens = left_pad(sequences, pad, self.head.weight.device)
            output = self.backbone(**prepare_inputs(tokens, pad))
            return self.head(output.last_hidden_state[:, -1].float()).squeeze(-1)

    model = RewardModel()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Reward LoRA must remain full FP32")
    return model


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer(model, recipe=RECIPE):
    import torch
    adapter = [p for p in model.backbone.parameters() if p.requires_grad]
    if not adapter or any("lora_" not in n for n, p in model.backbone.named_parameters() if p.requires_grad):
        raise ValueError("Only LoRA and the scalar head may be trainable")
    return torch.optim.AdamW([
        {"params": adapter, "lr": recipe["adapter_learning_rate"], "weight_decay": recipe["adapter_weight_decay"]},
        {"params": model.head.parameters(), "lr": recipe["head_learning_rate"], "weight_decay": 0.}],
        betas=tuple(recipe["adam_betas"]), eps=recipe["adam_epsilon"])


def update(model, opt, data, indices, pad, recipe=RECIPE):
    """Exact pair-mean BT loss; the final short update keeps its true divisor."""
    import torch
    from torch.nn import functional as F
    model.train()
    opt.zero_grad(set_to_none=True)
    total = 0.
    # Sorting only within this optimizer batch reduces padding. Every pair
    # retains exactly 1/len(indices) weight and remains in the same update.
    ordered = sorted(map(int, indices), key=lambda i: max(map(len, data.pair(i))))
    for start in range(0, len(ordered), recipe["micro_pairs"]):
        micro = ordered[start:start+recipe["micro_pairs"]]
        sequences = [seq for i in micro for seq in data.pair(i)]
        scores = model(sequences, pad).reshape(-1, 2)
        z = scores[:, 0] - scores[:, 1]
        y = torch.as_tensor(data.labels[micro], device=z.device, dtype=torch.float32)
        loss = (F.softplus(z)-y*z).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Reward pair loss")
        total += loss.detach().item()
        (loss/len(indices)).backward()
    penalty = .5*recipe["head_l2"]*model.head.weight.square().sum()
    penalty.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_norm"], error_if_nonfinite=True)
    opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
        raise ValueError("Nonfinite Reward LoRA update")
    return {"pair_mean_bt_nll": total/len(indices), "head_penalty": penalty.detach().item(),
            "gradient_norm_before_clip": norm.item(), "pairs": len(indices)}


def predict(model, data, indices, pad):
    import torch
    was_training = model.training
    model.eval()
    indices = list(map(int, indices))
    result = np.empty((len(indices), 2), dtype=np.float32)
    ordered = sorted(range(len(indices)), key=lambda j: max(map(len, data.pair(indices[j]))))
    with torch.no_grad():
        for start in range(0, len(ordered), RECIPE["micro_pairs"]):
            locations = ordered[start:start+RECIPE["micro_pairs"]]
            sequences = [seq for j in locations for seq in data.pair(indices[j])]
            result[locations] = model(sequences, pad).reshape(-1, 2).cpu().numpy()
    model.train(was_training)
    scores = result
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite Reward predictions")
    return scores


def _load(c):
    from .ppo import load_model
    run = Run(c)
    run.require("lora-reward-data")
    report = read_json(run.file(f"{PREFIX}/data.json"))
    original_recipe = report["recipe"]
    if (report["recipe_sha256"] != hash_json(original_recipe)
            or {k:v for k,v in original_recipe.items() if k != "micro_pairs"}
            != {k:v for k,v in RECIPE.items() if k != "micro_pairs"}):
        raise ValueError("Packed Reward inputs belong to another recipe")
    with np.load(run.file("diagnostics/full-pool-reward/l2-0.01/weights.npz"), allow_pickle=False) as f:
        weights = f["w"]
    from .ppo_trainer import seed_everything
    seed_everything(RECIPE["seed"])
    model = make_model(load_model(c, causal=False), weights)
    train, dev = (PackedPairs(run.file(f"{PREFIX}/{s}")) for s in ("train", "dev"))
    return run, report, model, train, dev, weights


def smoke(c):
    import torch
    run, stage = Run(c), f"lora-reward-smoke-micro{RECIPE['micro_pairs']}"
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(run.file(f"{PREFIX}/smoke-micro{RECIPE['micro_pairs']}.json"))
        run, inputs, model, train, dev, weights = _load(c)
        before = frozen_hash(model)
        count = min(8, len(dev))
        scores = predict(model, dev, range(count), inputs["pad"])
        with np.load(run.file("features/reward_dev.npz"), allow_pickle=False) as f:
            if f["ids"][:count].tolist() != dev.ids[:count].tolist():
                raise ValueError("Reward baseline identity mismatch")
            reference = f["phi"][:count].astype(np.float64) @ weights
        error = float(np.max(np.abs(scores-reference)))
        if error > 2e-4:
            raise ValueError(f"Zero-adapter baseline scores differ: {error}")
        initial = trainable_state(model)
        opt = optimizer(model)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        started = time.monotonic()
        # Include the longest real pair in backward, then a second independent update.
        history = [update(model, opt, train, [inputs["training"]["longest_pair_index"]]*RECIPE["micro_pairs"], inputs["pad"]),
                   update(model, opt, train, list(range(RECIPE["micro_pairs"])), inputs["pad"])]
        final = trainable_state(model)
        changed = [name for name in initial if not torch.equal(initial[name], final[name])]
        if (before != frozen_hash(model) or "head.weight" not in changed
                or not any("lora_B" in n for n in changed)):
            raise ValueError("LoRA diagnostic did not preserve base weights/update the adapter and head")
        report = {"passed": True, "baseline_max_score_error": error, "baseline_tolerance": 2e-4,
                  "frozen_base_sha256": before, "base_weights_unchanged": True,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "adapter_layers": [n for n, m in model.backbone.named_modules() if hasattr(m, "lora_A")],
                  "changed_tensors": changed, "updates": history, "recipe": RECIPE,
                  "backward_seconds": time.monotonic()-started,
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                  "longest_sequence_tokens": inputs["training"]["max_sequence_tokens"],
                  "PPO_started": False}
        path = run.file(f"{PREFIX}/smoke-micro{RECIPE['micro_pairs']}.json")
        write_json(path, report)
        seal_and_publish(run, stage, [path], [run.receipt_path("lora-reward-data"), run.receipt_path("features")],
                         metadata={"passed": True}, code_paths=sources(), context=RECIPE)
        return report


def _save_checkpoint(path, model, opt, step, seen, best, history, contract):
    import torch
    value = {"trainable": trainable_state(model), "optimizer": opt.state_dict(), "step": step,
             "seen": seen, "best_step": best[0], "best_nll": best[1], "history": history,
             "contract": contract, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if path.exists():
        raise FileExistsError(f"Preserve existing Reward checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
    temporary.replace(path)


def _restore(model, checkpoint):
    import torch
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(checkpoint["trainable"]) != expected:
        raise ValueError("Reward checkpoint parameter coverage changed")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                value = checkpoint["trainable"][name]
                if value.dtype != torch.float32 or value.shape != p.shape or not torch.isfinite(value).all():
                    raise ValueError("Invalid Reward checkpoint tensor")
                p.copy_(value)


def train(c, *, resume=False):
    import torch
    from .ppo_math import learning_rate_multiplier
    run, stage = Run(c), "lora-reward-train"
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(run.file(f"{PREFIX}/training.json"))
        smoke_stage = f"lora-reward-smoke-micro{RECIPE['micro_pairs']}"
        run.require(smoke_stage)
        run, inputs, model, data, dev, _ = _load(c)
        directory = run.file(f"{PREFIX}/checkpoints")
        contract = {"recipe_sha256": hash_json(RECIPE), "input_receipt": digest(run.receipt_path("lora-reward-data")),
                    "sources": {p.name: digest(p) for p in sources()}, "config": run.fingerprint}
        order = np.random.default_rng(RECIPE["seed"]).permutation(len(data))
        total_steps = math.ceil(len(data)/RECIPE["effective_pairs"])
        opt, step, seen, history = optimizer(model), 0, 0, []
        existing = sorted(directory.glob("step-*.pt")) if directory.exists() else []
        if existing:
            if not resume:
                raise FileExistsError("Reward LoRA checkpoints exist; inspect then explicitly resume")
            checkpoint = torch.load(existing[-1], map_location=model.head.weight.device, weights_only=False)
            if checkpoint["contract"] != contract:
                raise ValueError("Reward resume source/input/recipe changed")
            _restore(model, checkpoint)
            opt.load_state_dict(checkpoint["optimizer"])
            step, seen, history = checkpoint["step"], checkpoint["seen"], checkpoint["history"]
            best = (checkpoint["best_step"], checkpoint["best_nll"])
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
            if seen != min(step*RECIPE["effective_pairs"], len(data)):
                raise ValueError("Reward resume data cursor is inconsistent")
        else:
            scores = predict(model, dev, range(len(dev)), inputs["pad"])
            initial = logit_metrics(scores[:, 0]-scores[:, 1], dev.labels)
            best = (0, initial["bt_nll"])
            history = [{"step": 0, "pairs_seen": 0, "development": initial}]
            _save_checkpoint(directory/"step-000000.pt", model, opt, 0, 0, best, history, contract)
            print(f"reward-lora initial {initial}", flush=True)
        started, initial_seen = time.monotonic(), seen
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        while seen < len(data):
            step += 1
            indices = order[seen:seen+RECIPE["effective_pairs"]]
            multiplier = learning_rate_multiplier(step, total_steps, RECIPE["warmup_ratio"], RECIPE["minimum_learning_rate_ratio"])
            for group, rate in zip(opt.param_groups, (RECIPE["adapter_learning_rate"], RECIPE["head_learning_rate"])):
                group["lr"] = rate*multiplier
            statistics = update(model, opt, data, indices, inputs["pad"])
            seen += len(indices)
            progress = {"step": step, "total_steps": total_steps, "pairs_seen": seen, "training_pairs": len(data),
                        "last_update": statistics, "elapsed_seconds": time.monotonic()-started,
                        "pairs_per_second": (seen-initial_seen)/(time.monotonic()-started), "best_step": best[0]}
            if step % 16 == 0 or seen == len(data):
                write_json(run.file(f"{PREFIX}/progress.json"), progress, overwrite=True)
                print(f"reward-lora progress {progress}", flush=True)
            if step % RECIPE["checkpoint_steps"] == 0 or seen == len(data):
                scores = predict(model, dev, range(len(dev)), inputs["pad"])
                metrics = logit_metrics(scores[:, 0]-scores[:, 1], dev.labels)
                if metrics["bt_nll"] < best[1]:
                    best = (step, metrics["bt_nll"])
                history.append({"step": step, "pairs_seen": seen, "development": metrics})
                _save_checkpoint(directory/f"step-{step:06d}.pt", model, opt, step, seen, best, history, contract)
                print(f"reward-lora development step={step} {metrics}", flush=True)
        report = {"recipe": RECIPE, "training_pairs": len(data), "pairs_seen": seen, "optimizer_steps": step,
                  "best_step": best[0], "best_development_bt_nll": best[1], "history": history,
                  "selected_checkpoint": f"{PREFIX}/checkpoints/step-{best[0]:06d}.pt",
                  "last_checkpoint": f"{PREFIX}/checkpoints/step-{step:06d}.pt",
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                  "PPO_started": False, "final_test_used": False}
        marker = run.file(f"{PREFIX}/training.json")
        write_json(marker, report)
        seal_and_publish(run, stage, [marker, *sorted(directory.glob("step-*.pt"))],
                         [run.receipt_path("lora-reward-data"), run.receipt_path(smoke_stage)],
                         metadata={"best_step": best[0], "steps": step}, code_paths=sources(), context=RECIPE)
        return report


def score(c, shard_index, shards):
    import torch
    if type(shards) is not int or not 1 <= shards <= 50 or not 0 <= shard_index < shards:
        raise ValueError("Invalid Reward scoring shard coverage")
    run, stage = Run(c), f"lora-reward-score-{shard_index:03d}-of-{shards:03d}"
    marker = run.file(f"{PREFIX}/scores/part-{shard_index:03d}-of-{shards:03d}.npz")
    with stage_lock(run, stage):
        if run.completed(stage):
            return {"completed": stage}
        run.require("lora-reward-train")
        training = read_json(run.file(f"{PREFIX}/training.json"))
        run, inputs, model, data, dev, _ = _load(c)
        checkpoint = torch.load(run.file(training["selected_checkpoint"]), map_location="cpu", weights_only=False)
        _restore(model, checkpoint)
        arrays = {}
        for split, dataset in (("train", data), ("dev", dev)):
            indices = np.arange(shard_index, len(dataset), shards, dtype=np.int64)
            arrays.update({f"{split}_indices": indices, f"{split}_ids": dataset.ids[indices],
                           f"{split}_scores": predict(model, dataset, indices, inputs["pad"])})
        save_npz(marker, **arrays)
        seal_and_publish(run, stage, [marker], [run.receipt_path("lora-reward-train")],
                         metadata={"checkpoint_step": training["best_step"]}, code_paths=sources(),
                         context={"shards": shards, "index": shard_index})
        return {"completed": stage, "training_pairs": len(arrays["train_indices"])}


def summarize(c, shards):
    run, stage = Run(c), "lora-reward-report"
    with stage_lock(run, stage):
        marker = run.file(f"{PREFIX}/summary.json")
        if run.completed(stage):
            return read_json(marker)
        training = read_json(run.file(f"{PREFIX}/training.json"))
        inputs = [run.receipt_path("lora-reward-train")]
        datasets = {s: PackedPairs(run.file(f"{PREFIX}/{p}")) for s, p in (("training", "train"), ("development", "dev"))}
        merged = {s: np.full((len(d), 2), np.nan) for s, d in datasets.items()}
        coverage = {s: np.zeros(len(d), dtype=int) for s, d in datasets.items()}
        for index in range(shards):
            name = f"lora-reward-score-{index:03d}-of-{shards:03d}"
            run.require(name)
            inputs.append(run.receipt_path(name))
            with np.load(run.file(f"{PREFIX}/scores/part-{index:03d}-of-{shards:03d}.npz"), allow_pickle=False) as f:
                for split, prefix in (("training", "train"), ("development", "dev")):
                    expected = np.arange(index, len(datasets[split]), shards)
                    if (not np.array_equal(f[f"{prefix}_indices"], expected)
                            or f[f"{prefix}_ids"].tolist() != datasets[split].ids[expected].tolist()):
                        raise ValueError("Reward prediction shard order/identities mismatch")
                    merged[split][expected] = f[f"{prefix}_scores"]
                    coverage[split][expected] += 1
        if any(not np.all(a == 1) for a in coverage.values()) or any(not np.isfinite(a).all() for a in merged.values()):
            raise ValueError("Reward prediction merge is incomplete/nonfinite")
        metrics = {s: logit_metrics(a[:, 0]-a[:, 1], datasets[s].labels) for s, a in merged.items()}
        selected = next(r for r in training["history"] if r["step"] == training["best_step"])
        if abs(metrics["development"]["bt_nll"]-selected["development"]["bt_nll"]) > 2e-5:
            raise ValueError("Independent selected-checkpoint development predictions disagree")
        dev_rows = rows(run.file("data/reward_dev.jsonl"))
        subsets = {}
        for subset in sorted({r["source"]["subset"] for r in dev_rows}):
            mask = np.array([r["source"]["subset"] == subset for r in dev_rows])
            a = merged["development"][mask]
            subsets[subset] = logit_metrics(a[:, 0]-a[:, 1], datasets["development"].labels[mask])
        baseline = read_json(run.file("diagnostics/full-pool-reward/l2-0.01/report.json"))
        report = {"recipe": RECIPE, **metrics, "development_subsets": subsets, "best_step": training["best_step"],
                  "linear_baseline": {s: baseline[s] for s in datasets},
                  "development_accuracy_change": metrics["development"]["accuracy"]-baseline["development"]["accuracy"],
                  "full_training_rescored_at_one_selected_checkpoint": True, "independent_shard_coverage_passed": True,
                  "diagnostic_only": True, "PPO_started": False, "final_test_used": False}
        write_json(marker, report)
        seal_and_publish(run, stage, [marker], inputs, metadata={"passed": True}, code_paths=sources(), context=RECIPE)
        return report
