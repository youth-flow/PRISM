"""V3 model adapters, common SFT trajectories, critic prefit and policy stages.

Imports are deliberately lazy: V3 configuration/planning/CPU reward fitting do
not require Torch, Transformers, PEFT, model weights, a GPU or any API key.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from prism.io import digest, rows
from .artifacts import Run, hash_json, read_json, write_json


def executor_sources():
    """Resume is bound to the numerical implementation, not merely its name."""
    folder = Path(__file__).resolve().parent
    return {name: digest(folder / name) for name in ("ppo.py", "ppo_trainer.py", "ppo_math.py")}


def hf_config(c):
    return {"choices": {"sft": {"model": c["model"]["id"], "revision": c["model"]["revision"]}},
            "runtime": c["runtime"]}


def configure_torch(c):
    import torch
    from prism.numerics import configure_numerics
    if c["runtime"]["dtype"] != "float32" or c["runtime"].get("allow_tf32", False):
        raise ValueError("V3 models require full FP32 and disabled TF32")
    configure_numerics(c)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if c["runtime"]["device"] == "cuda" and (not torch.cuda.is_available()
            or (torch.cuda.device_count() != 1 and not distributed)):
        raise ValueError("Run one V3 process per GPU, or initialize the audited process group first")
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))


def load_model(c, *, causal=True):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM
    configure_torch(c)
    loader = AutoModelForCausalLM if causal else AutoModel
    model = loader.from_pretrained(c["model"]["id"], revision=c["model"]["revision"],
                                  torch_dtype=torch.float32, attn_implementation="eager",
                                  local_files_only=c["runtime"]["local_files_only"], trust_remote_code=False)
    model.to(c["runtime"]["device"]).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    return model


def load_tokenizer(c):
    from prism.hf import tokenizer
    return tokenizer(hf_config(c))


def make_critic(backbone, weights, mean, std, settings):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Reward normalization must be finite with positive standard deviation")
    backbone.requires_grad_(False)
    adapted = get_peft_model(backbone, LoraConfig(r=settings["critic_lora_r"],
                             lora_alpha=settings["critic_lora_alpha"], target_modules="all-linear",
                             lora_dropout=0., bias="none", task_type=TaskType.FEATURE_EXTRACTION))

    class ValueModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = adapted
            self.config = backbone.config
            dimension = getattr(self.config, "hidden_size", None) or self.config.n_embd
            self.score = torch.nn.Linear(dimension, 1, bias=True, dtype=torch.float32)
            if np.asarray(weights).shape != (dimension,):
                raise ValueError("Reward/critic feature dimensions differ")
            with torch.no_grad():
                self.score.weight.copy_(torch.as_tensor(np.asarray(weights).copy(), dtype=torch.float32)[None] / std)
                self.score.bias.fill_(-mean / std)
            self.to(next(backbone.parameters()).device)

        def forward(self, **inputs):
            output = self.backbone(**inputs)
            return self.score(output.last_hidden_state.float()).squeeze(-1)

    return ValueModel().eval()


def make_actor(backbone, settings):
    from peft import LoraConfig, TaskType, get_peft_model
    return get_peft_model(backbone, LoraConfig(r=settings["lora_r"], lora_alpha=settings["lora_alpha"],
                         target_modules=settings["lora_target_modules"], lora_dropout=settings["lora_dropout"],
                         bias="none", task_type=TaskType.CAUSAL_LM)).eval()


class FrozenNormalizedReward:
    """Canonical EOS frozen features; reward and critic never share parameters."""
    def __init__(self, backbone, weights, mean, std, *, eos, pad, context_tokens=4096, batch_size=4):
        import torch
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("Invalid fixed SFT reward normalization")
        self.backbone = backbone.requires_grad_(False).eval()
        self.device = next(backbone.parameters()).device
        self.weights = torch.as_tensor(np.asarray(weights).copy(), dtype=torch.float32, device=self.device)
        self.mean, self.std, self.eos, self.pad = float(mean), float(std), eos, pad
        self.context_tokens, self.batch_size = context_tokens, batch_size

    def score(self, prompts, responses):
        import torch
        from .ppo_trainer import left_pad, prepare_inputs
        if len(prompts) != len(responses) or not prompts:
            raise ValueError("Reward needs matched nonempty prompt/response lists")
        result = []
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                sequences = []
                for prompt, response in zip(prompts[start:start + self.batch_size], responses[start:start + self.batch_size]):
                    if not response:
                        raise ValueError("Empty reward response")
                    tokens = list(prompt) + list(response)
                    if tokens[-1] != self.eos:
                        tokens.append(self.eos)
                    if len(tokens) > self.context_tokens:
                        raise ValueError("Canonical reward sequence exceeds model context")
                    sequences.append(tokens)
                tokens = left_pad(sequences, self.pad, self.device)
                output = self.backbone(**prepare_inputs(tokens, self.pad))
                phi = output.last_hidden_state[:, -1].float()
                if phi.shape[1:] != self.weights.shape:
                    raise ValueError("Frozen reward feature dimension changed")
                result.append((phi @ self.weights - self.mean) / self.std)
        return torch.cat(result)

    def frozen_parameters(self):
        """Expose immutable tensors so PPO can detect accidental Reward updates."""
        return self.backbone.parameters()


def _save_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if rows(path) != records:
            raise ValueError(f"Preserve differing shared trajectory artifact: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _save_features(path, features, ids):
    path = Path(path)
    array = np.asarray(features, dtype=np.float32)
    id_array = np.asarray(ids, dtype=str)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if not np.array_equal(existing["phi"], array) or not np.array_equal(existing["ids"], id_array):
                raise ValueError("Preserve differing shared feature artifact")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, phi=array, ids=id_array)


def prepare_shared(c):
    from filelock import FileLock
    run = Run(c)
    marker = run.file("shared/.execution.lock")
    marker.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(marker), timeout=0):
        return _prepare_shared(c)


def _prepare_shared(c):
    """Generate common fixed SFT trajectories, resumable per deterministic item.

    This stage is independent of N. Its progress stream contains no paid API
    response. Each item's seed is stable and no already recorded item is sent
    through the model again. The train probe reuses calibration's first items.
    """
    import torch
    from prism.hf import prompt_ids
    from .ppo_trainer import left_pad, prepare_inputs, seed_everything, stream_generate
    run = Run(c)
    if run.completed("shared-sft"):
        return run.require("shared-sft")
    run.require("data")
    policy, development = rows(run.file("data/policy.jsonl")), rows(run.file("data/ppo_dev.jsonl"))
    count = c["reward"]["calibration_responses"]
    if len(policy) < count or count <= c["ppo"]["critic_warmup_episodes"]:
        raise ValueError("Need distinct shared SFT calibration/critic-heldout trajectories")
    selected = [("calibration", i, record) for i, record in enumerate(policy[:count])]
    selected += [("dev_probe", i, record) for i, record in enumerate(development)]
    progress = run.file("shared/generation.progress.jsonl")
    write_json(run.file("shared/generation.contract.json"), {"config_sha256": run.fingerprint,
               "policy_sha256": digest(run.file("data/policy.jsonl")),
               "dev_sha256": digest(run.file("data/ppo_dev.jsonl")), "executor_sources": executor_sources()})
    previous = rows(progress) if progress.exists() else []
    if len(previous) > len(selected):
        raise ValueError("Shared progress contains extra trajectories")
    for row, (group, index, source) in zip(previous, selected):
        if (row["group"], row["index"], row["source_id"]) != (group, index, source["id"]):
            raise ValueError("Shared generation resume ordering changed")
        expected_seed = int(hashlib.sha256(f"v3-shared:{c['study']['seed']}:{group}:{source['id']}".encode()).hexdigest()[:8], 16)
        feature = np.asarray(row["phi"], dtype=np.float32)
        if (row["messages"] != source["messages"] or row["seed"] != expected_seed
                or feature.shape != (c["data"]["feature_dimension"],) or not np.isfinite(feature).all()
                or not 1 <= len(row["response_tokens"]) <= c["ppo"]["response_length"]):
            raise ValueError("Shared generation resume content/model dimension changed")
    if len(previous) < len(selected):
        model = load_model(c).requires_grad_(False)
        tokenizer = load_tokenizer(c)
        base = model.base_model
        progress.parent.mkdir(parents=True, exist_ok=True)
        with progress.open("a", encoding="utf-8") as stream:
            for group, index, record in selected[len(previous):]:
                seed = int(hashlib.sha256(f"v3-shared:{c['study']['seed']}:{group}:{record['id']}".encode()).hexdigest()[:8], 16)
                seed_everything(seed)
                prompt = prompt_ids(tokenizer, record["messages"], c["data"]["max_prompt_tokens"])
                query = left_pad([prompt], tokenizer.pad_token_id, next(model.parameters()).device)
                response, _ = stream_generate(model, query, pad=tokenizer.pad_token_id, eos=tokenizer.eos_token_id,
                                             max_new_tokens=c["ppo"]["response_length"], temperature=1.)
                actual = response[0][response[0].ne(tokenizer.pad_token_id)].tolist()
                sequence = prompt + actual
                if sequence[-1] != tokenizer.eos_token_id:
                    sequence.append(tokenizer.eos_token_id)
                if len(sequence) > c["data"]["context_tokens"]:
                    raise ValueError("Shared SFT canonical EOS exceeds context")
                with torch.no_grad():
                    tokens = left_pad([sequence], tokenizer.pad_token_id, query.device)
                    phi = base(**prepare_inputs(tokens, tokenizer.pad_token_id)).last_hidden_state[0, -1].float().cpu().numpy()
                row = {"id": f"{group}-{index:06d}", "source_id": record["id"], "messages": record["messages"],
                       "group": group, "index": index, "seed": seed, "prompt_tokens": prompt,
                       "response_tokens": actual, "has_eos": actual[-1] == tokenizer.eos_token_id, "phi": phi.tolist()}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                previous.append(row)
                if (len(previous) % 16) == 0:
                    print(f"v3 shared SFT {len(previous)}/{len(selected)}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    outputs = []
    for group, records in (("calibration", previous[:count]), ("dev_probe", previous[count:]),
                           ("train_probe", previous[:c["ppo"]["fixed_train_probe_prompts"]])):
        path = run.file(f"shared/{group}.jsonl")
        values = [{k: v for k, v in record.items() if k not in ("phi", "group", "index")} for record in records]
        _save_rows(path, values)
        features = run.file(f"shared/{group}.features.npz")
        _save_features(features, [r["phi"] for r in records], [r["id"] for r in records])
        outputs.extend((path, features))
    return run.publish("shared-sft", outputs, inputs=[run.receipt_path("data"),
                       run.file("data/policy.jsonl"), run.file("data/ppo_dev.jsonl")],
                       metadata={"calibration_responses": count, "dev_responses": len(development),
                                 "new_sft_responses": count + len(development), "api_calls": 0})


def _pack_trajectories(records, *, pad, device, response_length):
    import torch
    from .ppo_trainer import left_pad, response_masks
    prompts = left_pad([r["prompt_tokens"] for r in records], pad, device)
    response = torch.full((len(records), response_length), pad, dtype=torch.long, device=device)
    for i, record in enumerate(records):
        tokens = record["response_tokens"]
        if not 1 <= len(tokens) <= response_length:
            raise ValueError("Stored SFT trajectory is empty or exceeds response cap")
        response[i, :len(tokens)] = torch.as_tensor(tokens, dtype=torch.long, device=device)
    _, value_valid, _ = response_masks(response, pad)
    return torch.cat((prompts, response), dim=1), prompts.shape[1], value_valid


def prefit_critic(critic, reward, trajectories, heldout, *, settings, pad, seed):
    """Fit causal prefix values to Monte Carlo returns, never just EOS labels."""
    import torch
    from .ppo_math import value_statistics
    from .ppo_trainer import prepare_inputs
    p = settings
    count, mini, micro = p["critic_warmup_episodes"], p["critic_warmup_mini_batch_size"], p["per_device_train_batch_size"]
    if len(trajectories) != count or count % mini or mini % micro or not heldout:
        raise ValueError("Critic warmup needs exact whole minibatches and separate heldout trajectories")
    if {r["id"] for r in trajectories} & {r["id"] for r in heldout}:
        raise ValueError("Critic warmup and heldout trajectories overlap")
    device = next(critic.parameters()).device
    roles = {"critic_lora": [v for k, v in critic.named_parameters() if v.requires_grad and "lora_" in k],
             "critic_head": list(critic.score.parameters())}
    optimizer = torch.optim.AdamW([{"params": roles["critic_lora"], "lr": p["critic_learning_rate"]},
                                    {"params": roles["critic_head"], "lr": p["critic_head_learning_rate"]}],
                                   betas=(p["adam_beta1"], p["adam_beta2"]), eps=p["adam_epsilon"], weight_decay=0.)
    with torch.no_grad():
        targets = reward.score([r["prompt_tokens"] for r in trajectories], [r["response_tokens"] for r in trajectories])
        test_targets = reward.score([r["prompt_tokens"] for r in heldout], [r["response_tokens"] for r in heldout])

    def measure():
        predictions, expected = [], []
        with torch.no_grad():
            for start in range(0, len(heldout), micro):
                records = heldout[start:start + micro]
                packed, context, valid = _pack_trajectories(records, pad=pad, device=device, response_length=p["response_length"])
                value = critic(**prepare_inputs(packed, pad))[:, context - 1:-1]
                target = test_targets[start:start + micro, None].expand_as(value)
                predictions.extend(value[valid].cpu().tolist())
                expected.extend(target[valid].cpu().tolist())
        return value_statistics(predictions, expected)

    before, steps, norms = measure(), 0, []
    rng = np.random.default_rng(seed + 65537)
    critic.eval()
    for epoch in range(p["critic_warmup_epochs"]):
        order = rng.permutation(count)
        for start in range(0, count, mini):
            indices = order[start:start + mini]
            denominator = sum(min(len(trajectories[int(i)]["response_tokens"]) + 1, p["response_length"]) for i in indices)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, mini, micro):
                selected = indices[offset:offset + micro]
                records = [trajectories[int(i)] for i in selected]
                packed, context, valid = _pack_trajectories(records, pad=pad, device=device, response_length=p["response_length"])
                predictions = critic(**prepare_inputs(packed, pad))[:, context - 1:-1]
                target = targets[selected, None].expand_as(predictions)
                loss = .5 * (predictions - target).square()[valid].sum() / denominator
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(roles["critic_lora"]) + roles["critic_head"], p["max_grad_norm"], error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            steps += 1
    after = measure()
    expected_steps = count // mini * p["critic_warmup_epochs"]
    if steps != expected_steps or any(int(optimizer.state[v]["step"]) != steps for group in roles.values() for v in group):
        raise RuntimeError("Critic warmup real optimizer steps differ from the stated schedule")
    return {"protocol": "v3_critic_mc_prefit_v1", "episodes": count, "epochs": p["critic_warmup_epochs"],
            "actual_optimizer_steps": steps, "before": before, "after": after,
            "gradient_norm_max_before_clipping": max(norms), "gradient_norm_clipping_executed": True,
            "heldout_trajectories": len(heldout), "passed_performance": bool(after["explained_variance"] is not None
                and after["explained_variance"] > 0 and after["mse"] < after["constant_baseline_mse"])}


def _save_critic(critic, path):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    critic.backbone.save_pretrained(str(path / "adapter"), safe_serialization=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in critic.score.state_dict().items()}, str(path / "value_head.safetensors"))


def evaluate_probe(c, engine, method, run, name):
    """Paired fixed-prompt own-reward/KL probes; evaluation never consumes train RNG."""
    import torch
    from .ppo_trainer import left_pad, prepare_inputs, seed_everything, stream_generate
    stage = f"probe-{method}-{engine.completed_episodes:08d}-{name}"
    path = run.file(f"policy/{method}/probes/{engine.completed_episodes:08d}-{name}.json")
    checkpoint = run.file(f"policy/{method}/checkpoints/episode-{engine.completed_episodes:08d}.pt")
    if not checkpoint.is_file():
        raise ValueError("A probe must be tied to a committed policy checkpoint")
    identity = {"protocol": "v3_fixed_policy_probe_v1", "method": method,
                "episode": engine.completed_episodes, "probe": name,
                "contract_hash": engine.contract_hash, "checkpoint_sha256": digest(checkpoint),
                "shared_probe_sha256": digest(run.file(f"shared/{name}.jsonl")),
                "shared_features_sha256": digest(run.file(f"shared/{name}.features.npz"))}
    if run.completed(stage):
        existing = read_json(path)
        if any(existing.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing policy probe belongs to a different checkpoint/contract")
        return path
    # An unreceipted complete JSON may be the survivor of a crash immediately
    # before publication. Recompute deterministically and let write_json compare
    # it exactly; never overwrite a different existing probe.
    if path.exists() and any(read_json(path).get(k) != v for k, v in identity.items()):
        raise ValueError("Unpublished probe has a different checkpoint/contract")
    records = rows(run.file(f"shared/{name}.jsonl"))
    with np.load(run.file(f"shared/{name}.features.npz"), allow_pickle=False) as feature_file:
        baseline_phi = feature_file["phi"]
    with np.load(run.file(f"reward/{method}.npz"), allow_pickle=False) as reward_file:
        weights = reward_file["w"].astype(np.float32)
    norm = read_json(run.file(f"normalization/{method}.json"))
    baseline = ((baseline_phi.astype(np.float32) @ weights).astype(np.float64) - norm["mean"]) / norm["std"]
    values, value_predictions, mc_targets = [], [], []
    for i, row in enumerate(records):
        # Same source prompt/seed is used at every checkpoint; generated policy
        # draws change only through the actor, not through changing RNG seeds.
        seed_everything(row["seed"])
        query = left_pad([row["prompt_tokens"]], engine.pad, engine.device)
        with torch.no_grad():
            response, old = stream_generate(engine.actor, query, pad=engine.pad, eos=engine.eos,
                                             max_new_tokens=c["ppo"]["response_length"], temperature=1.)
            valid = response.ne(engine.pad)
            packed = torch.cat((query, response), 1)
            with engine.actor.disable_adapter():
                reference = engine._actor_logp(packed, response, query.shape[1])
            log_ratio = float((old - reference)[valid].sum())
            actual = response[0][valid[0]].tolist()
            score = float(engine.reward.score([row["prompt_tokens"]], [actual])[0])
            # Independent Monte Carlo value validation, not a GAE target that
            # already contains the critic's own predictions.
            from .ppo_trainer import response_masks
            _, value_valid, length = response_masks(response, engine.pad)
            prediction = engine.critic(**prepare_inputs(packed, engine.pad))[:, query.shape[1] - 1:-1]
            token_reward = -c["ppo"]["kl_coef"] * (old - reference).masked_fill(~valid, 0.)
            token_reward[0, min(int(length[0]), response.shape[1] - 1)] += score
            returns = token_reward.flip(1).cumsum(1).flip(1)
            value_predictions.extend(prediction[value_valid].cpu().tolist())
            mc_targets.extend(returns[value_valid].cpu().tolist())
        values.append({"id": row["id"], "source_id": row["source_id"], "response_tokens": actual,
                       "normalized_reward": score, "sft_normalized_reward": float(baseline[i]),
                       "paired_reward_delta": score - float(baseline[i]), "sequence_kl_mc": log_ratio,
                       "rlhf_objective": score - c["ppo"]["kl_coef"] * log_ratio,
                       "paired_rlhf_objective_delta": score - c["ppo"]["kl_coef"] * log_ratio - float(baseline[i]),
                       "length": len(actual), "has_eos": actual[-1] == engine.eos})
    delta = np.asarray([r["paired_reward_delta"] for r in values])
    objective_delta = np.asarray([r["paired_rlhf_objective_delta"] for r in values])
    from .ppo_math import value_statistics
    write_json(path, {**identity, "records": values,
                     "mean_paired_reward_delta": float(delta.mean()),
                     "paired_reward_delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None,
                     "mean_paired_rlhf_objective_delta": float(objective_delta.mean()),
                     "paired_rlhf_objective_delta_se": float(objective_delta.std(ddof=1) / np.sqrt(len(objective_delta))) if len(objective_delta) > 1 else None,
                     "critic_monte_carlo": value_statistics(value_predictions, mc_targets),
                     "mean_sequence_kl_mc": float(np.mean([r["sequence_kl_mc"] for r in values]))})
    run.publish(stage, [path], inputs=[checkpoint, run.file(f"policy/{method}/contract.json"),
                run.receipt_path("shared-sft"), run.receipt_path("normalization")],
                metadata=identity)
    return path


def accept_common_recipe(c):
    """Release both formal methods only after the common Empirical pilot learns.

    This uses development data and the actual accepted episode-8192 checkpoint,
    never the final 500 test prompts or a paid Judge.  The lower endpoint of a
    normal-approximation two-sided 95% interval must be positive (z=1.96).
    The exact estimator/threshold is recorded.
    """
    import torch
    from .ppo_trainer import PROTOCOL
    run = Run(c)
    if run.completed("ppo-common-acceptance"):
        return run.require("ppo-common-acceptance")
    episode = 8192
    if episode not in c["ppo"]["checkpoints"] or c["ppo"]["total_episodes"] < episode:
        raise ValueError("Common performance acceptance requires the stated episode-8192 development checkpoint")
    run.require("critic-warmup-empirical")
    dev_path = run.file(f"policy/empirical/probes/{episode:08d}-dev_probe.json")
    train_path = run.file(f"policy/empirical/probes/{episode:08d}-train_probe.json")
    checkpoint = run.file(f"policy/empirical/checkpoints/episode-{episode:08d}.pt")
    for name in ("train_probe", "dev_probe"):
        run.require(f"probe-empirical-{episode:08d}-{name}")
    development, training = read_json(dev_path), read_json(train_path)
    if development.get("episode") != episode or training.get("episode") != episode:
        raise ValueError("Common PPO acceptance probe checkpoint mismatch")
    critic = development["critic_monte_carlo"]
    checks = {
        "positive_training_reward_delta": training["mean_paired_reward_delta"] > 0,
        "positive_development_reward_ci": development["paired_reward_delta_se"] is not None and
            development["mean_paired_reward_delta"] - 1.96 * development["paired_reward_delta_se"] > 0,
        "positive_development_rlhf_objective_ci": development["paired_rlhf_objective_delta_se"] is not None and
            development["mean_paired_rlhf_objective_delta"] - 1.96 * development["paired_rlhf_objective_delta_se"] > 0,
        "critic_beats_constant_mc_baseline": critic["explained_variance"] is not None and
            critic["explained_variance"] > 0 and critic["mse"] < critic["constant_baseline_mse"],
    }
    if not all(checks.values()):
        raise RuntimeError("Common V3 PPO recipe has not passed the heldout performance gate: " + str(checks))
    if not checkpoint.is_file():
        raise ValueError("Common acceptance requires the real episode-8192 policy checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract_path = run.file("policy/empirical/contract.json")
    contract = read_json(contract_path)
    expected_steps = episode // c["ppo"]["rollout_batch_size"] * c["ppo"]["num_ppo_epochs"] * c["ppo"]["num_mini_batches"]
    if (state.get("protocol") != PROTOCOL or state.get("episode") != episode
            or state.get("contract_hash") != contract["contract_hash"]):
        raise ValueError("Development probe does not have the matching real PPO checkpoint")
    for group in state["optimizer"]["param_groups"]:
        for parameter_id in group["params"]:
            if int(state["optimizer"]["state"][parameter_id]["step"]) != expected_steps:
                raise ValueError("Common recipe checkpoint has incorrect actual optimizer counts")
    del state
    path = run.file("policy/common-performance-acceptance.json")
    write_json(path, {"protocol": "v3_empirical_common_performance_gate_v1", "episode": episode,
                     "checks": checks, "passed": True, "z_threshold": 1.96,
                     "preference_budget": run.selection()["budget"],
                     "selection_data": "fixed train and PPO development probes; final test excluded"})
    return run.publish("ppo-common-acceptance", [path], inputs=[dev_path, train_path, checkpoint, contract_path,
                       run.receipt_path(f"probe-empirical-{episode:08d}-train_probe"),
                       run.receipt_path(f"probe-empirical-{episode:08d}-dev_probe"),
                       run.receipt_path("critic-warmup-empirical"), run.receipt_path("normalization")])


def train_policy(c, method, *, stop_after_episodes=None, resume=False):
    from filelock import FileLock
    if method not in ("empirical", "prism"):
        raise ValueError("V3 policy method must be empirical or prism")
    run = Run(c)
    marker = run.file(f"policy/{method}/.execution.lock")
    marker.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(marker), timeout=0):
        return _train_policy(c, method, stop_after_episodes=stop_after_episodes, resume=resume)


def _train_policy(c, method, *, stop_after_episodes=None, resume=False):
    import torch
    from prism.hf import prompt_ids
    from .ppo_trainer import PPOEngine, PROTOCOL, seed_everything
    if method not in ("empirical", "prism"):
        raise ValueError("V3 policy method must be empirical or prism")
    run = Run(c)
    run.selection()
    for stage in ("data", "shared-sft", "reward", "normalization", "training-audit"):
        run.require(stage)
    if method == "prism":
        run.require("ppo-common-acceptance")
    if run.completed(f"policy-{method}"):
        return run.require(f"policy-{method}")
    norm_path, weight_path = run.file(f"normalization/{method}.json"), run.file(f"reward/{method}.npz")
    normalizer = read_json(norm_path)
    with np.load(weight_path, allow_pickle=False) as arrays:
        weights = arrays["w"].copy()
    contract = hash_json({"config": run.fingerprint, "selection": digest(run.file("study/selection.json")),
                          "reward": digest(weight_path), "normalization": digest(norm_path),
                          "training_audit": digest(run.receipt_path("training-audit")),
                          "policy_prompts": digest(run.file("data/policy.jsonl")),
                          "shared_sft": digest(run.receipt_path("shared-sft")), "executor": PROTOCOL,
                          "executor_sources": executor_sources()})
    target = run.file(f"policy/{method}")
    write_json(target / "contract.json", {"protocol": PROTOCOL, "contract_hash": contract, "method": method})
    configure_torch(c)
    seed_everything(c["study"]["seed"])
    tokenizer = load_tokenizer(c)
    actor = make_actor(load_model(c), c["ppo"])
    critic = make_critic(load_model(c, causal=False), weights, normalizer["mean"], normalizer["std"], c["ppo"])
    reward = FrozenNormalizedReward(load_model(c, causal=False), weights, normalizer["mean"], normalizer["std"],
                                    eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id,
                                    context_tokens=c["data"]["context_tokens"])
    if {id(p) for p in critic.parameters()} & {id(p) for p in reward.backbone.parameters()}:
        raise ValueError("Trainable critic must be independent of frozen reward")
    shared = rows(run.file("shared/calibration.jsonl"))
    warmup_path = target / "critic-warmup.json"
    if not resume:
        if warmup_path.exists() or (target / "checkpoints/latest.json").exists():
            raise FileExistsError("Policy work already exists; resume its exact checkpoint")
        count = c["ppo"]["critic_warmup_episodes"]
        heldout = shared[count:count + min(128, len(shared) - count)]
        report = prefit_critic(critic, reward, shared[:count], heldout, settings=c["ppo"],
                              pad=tokenizer.pad_token_id, seed=c["study"]["seed"])
        write_json(warmup_path, report)
        if not report["passed_performance"]:
            raise RuntimeError("Critic prefit did not beat a constant heldout baseline; inspect critic-warmup.json before PPO")
    elif not warmup_path.exists() or not read_json(warmup_path).get("passed_performance"):
        raise ValueError("Resume requires an accepted critic warmup record")
    prompts = [prompt_ids(tokenizer, record["messages"], c["data"]["max_prompt_tokens"])
               for record in rows(run.file("data/policy.jsonl"))]
    engine = PPOEngine(actor, critic, reward, prompts, pad=tokenizer.pad_token_id, eos=tokenizer.eos_token_id,
                       settings=c["ppo"], seed=c["study"]["seed"], contract_hash=contract,
                       checkpoint_dir=target / "checkpoints")
    def probe(instance):
        for name in ("train_probe", "dev_probe"):
            evaluate_probe(c, instance, method, run, name)
        if method == "empirical" and instance.completed_episodes == 8192:
            accept_common_recipe(c)

    if resume:
        run.require(f"critic-warmup-{method}")
        engine.resume(on_checkpoint=probe)
        if method == "empirical" and engine.completed_episodes >= 8192:
            accept_common_recipe(c)
    else:
        initial_checkpoint = engine.save_checkpoint()
        run.publish(f"critic-warmup-{method}", [warmup_path, initial_checkpoint],
                    inputs=[run.receipt_path(stage) for stage in ("shared-sft", "reward", "normalization")],
                    metadata={"method": method, "accepted": True,
                              "actual_optimizer_steps": report["actual_optimizer_steps"]})

    result = engine.run(stop_after_episodes=stop_after_episodes, on_checkpoint=probe)
    result.update(method=method, seed=c["study"]["seed"], budget=run.selection()["budget"],
                  total_episodes=c["ppo"]["total_episodes"],
                  executor=PROTOCOL, critic="independent_sft_lora_plus_scalar_head", normalization=normalizer)
    report_path = target / f"report-{engine.completed_episodes:08d}.json"
    if report_path.exists():
        previous = read_json(report_path)
        if {k: v for k, v in previous.items() if k != "seconds_this_invocation"} != {k: v for k, v in result.items() if k != "seconds_this_invocation"}:
            raise ValueError("Existing PPO report differs from checkpoint state")
        result = previous
    else:
        write_json(report_path, result)
    if not result["completed"]:
        return result
    write_json(target / "report.json", result)
    actor.save_pretrained(str(target / "actor"), safe_serialization=True)
    tokenizer.save_pretrained(str(target / "actor"))
    _save_critic(critic, target / "critic")
    outputs = [p for p in target.rglob("*") if p.is_file() and "checkpoints" not in p.parts and p.name != ".execution.lock"]
    return run.publish(f"policy-{method}", outputs, inputs=[run.receipt_path(stage) for stage in
                       ("study-selection", "data", "shared-sft", "reward", "training-audit", "normalization", f"critic-warmup-{method}", "ppo-common-acceptance")],
                       metadata={"method": method, "episodes": engine.completed_episodes, "executor": PROTOCOL,
                                 "completed": True, "actual_optimizer_steps": result["actual_optimizer_steps"]})
