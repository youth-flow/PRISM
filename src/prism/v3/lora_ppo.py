"""Isolated PPO diagnostic driven by the audited full-pool LoRA Reward.

This module deliberately does not promote the diagnostic Reward into the
formal Empirical/PRISM comparison.  It answers a narrower causal question:
given the strongest already-trained V3 Reward, can the unchanged PPO recipe
improve fixed-policy probes relative to the SFT reference?
"""
from pathlib import Path
import math
import time

import numpy as np

from prism.io import digest, rows
from .artifacts import Run, hash_json, read_json, write_json
from .lora_reward import (PREFIX as REWARD_PREFIX, RECIPE as REWARD_RECIPE,
                          STAGE_PREFIX as REWARD_STAGE_PREFIX, _restore as restore_reward,
                          make_model as make_reward_model)
from .recovery import recover_stage, seal_and_publish, stage_lock
from .reward import save_npz


METHOD = "full-pool-lora"
PREFIX = "diagnostics/full-pool-lora-ppo"
STAGE_PREFIX = "lora-ppo"
PILOT_EPISODE = 8192
EXECUTION_MICRO_BATCH_SIZE = 1
RECIPE = {
    "protocol": "v3_full_pool_lora_reward_distributed_ppo_v2",
    "purpose": "causal_test_of_strong_reward_ppo_vs_sft",
    "method": METHOD,
    "reward_checkpoint": "development_bt_nll_selected_full_pool_schedule",
    "reward_normalization": "fixed_4096_sft_mean_std_ddof0",
    "critic_initialization": "independent_copy_of_reward_adapter_and_normalized_head",
    "actor_initialization": "sft_zero_lora",
    "engineering_episode": 2048,
    "pilot_episode": PILOT_EPISODE,
    "distributed_world_sizes": [4, 8],
    "distributed_strategy": "homogeneous_synchronous_global_batch_gradient_sum",
    "execution_micro_batch_size": EXECUTION_MICRO_BATCH_SIZE,
    "execution_gradient_accumulation_steps": 64,
    "formal_prism_comparison": False,
    "paid_api_calls": 0,
}
PPO_RECIPE = {
    "total_episodes": 65536, "rollout_batch_size": 128, "mini_batch_size": 64,
    "per_device_train_batch_size": 2, "gradient_accumulation_steps": 32,
    "num_mini_batches": 2, "num_ppo_epochs": 4,
    "learning_rate": 1e-5, "critic_learning_rate": 3e-5,
    "critic_head_learning_rate": 1e-4,
    "lora_r": 16, "lora_alpha": 32, "lora_target_modules": "all-linear",
    "lora_dropout": 0., "critic_lora_r": 16, "critic_lora_alpha": 32,
    "critic_warmup_episodes": 2048, "critic_warmup_epochs": 4,
    "critic_warmup_mini_batch_size": 64,
    "warmup_ratio": .05, "minimum_learning_rate_ratio": .1,
    "scheduler": "warmup_cosine_floor", "adam_beta1": .9, "adam_beta2": .999,
    "adam_epsilon": 1e-5, "weight_decay": 0., "gamma": 1., "lam": .95,
    "kl_coef": .05, "cliprange": .2, "cliprange_value": .2, "vf_coef": .1,
    "max_grad_norm": 1., "independent_gradient_clipping": True,
    "response_length": 512, "temperature": 1., "top_p": 1., "top_k": 0,
    "whiten_rewards": False, "whiten_advantages": True,
    "local_rollout_forward_batch_size": 4, "probability_check_rows": 4,
    "max_logp_error": .001, "max_zero_update_approxkl": 1e-7,
    "engineering_checkpoint": 2048, "durable_checkpoint_interval": 2048,
    "checkpoints": [8192, 16384, 32768, 65536], "fixed_train_probe_prompts": 128,
}


def _require_ppo_recipe(c):
    mismatches = {key: (c["ppo"].get(key), value) for key, value in PPO_RECIPE.items()
                  if c["ppo"].get(key) != value}
    if mismatches:
        raise ValueError("Full-pool LoRA PPO must keep the frozen causal-test recipe: " + str(mismatches))
    if c["runtime"].get("dtype") != "float32" or c["runtime"].get("allow_tf32") is not False:
        raise ValueError("Full-pool LoRA PPO requires FP32 with TF32 disabled")


def sources():
    folder = Path(__file__).resolve().parent
    return [Path(__file__), folder / "lora_reward.py", folder / "ppo.py",
            folder / "ppo_trainer.py", folder / "distributed_ppo.py",
            folder / "ppo_math.py"]


def _distributed_execution_settings(settings):
    """Change placement granularity only; preserve the global PPO update."""
    value = dict(settings)
    mini = value["mini_batch_size"]
    micro = EXECUTION_MICRO_BATCH_SIZE
    if mini % micro:
        raise ValueError("Distributed execution microbatch must divide the global minibatch")
    value["per_device_train_batch_size"] = micro
    value["gradient_accumulation_steps"] = mini // micro
    if (value["rollout_batch_size"] != mini * value["num_mini_batches"]
            or value["rollout_batch_size"] != micro * value["gradient_accumulation_steps"]
                                            * value["num_mini_batches"]):
        raise ValueError("Distributed execution changed the global PPO batch")
    return value


def _selected_reward(run):
    """Return the receipt-bound selected checkpoint and its audited identity."""
    run.require(f"{REWARD_STAGE_PREFIX}-train")
    run.require(f"{REWARD_STAGE_PREFIX}-report")
    training_path = run.file(f"{REWARD_PREFIX}/training.json")
    summary_path = run.file(f"{REWARD_PREFIX}/summary.json")
    training, summary = read_json(training_path), read_json(summary_path)
    relative = training.get("selected_checkpoint")
    selected = summary.get("selected", {})
    expected_checkpoint = f"{REWARD_PREFIX}/checkpoints/step-002304.pt"
    if (relative != expected_checkpoint or selected.get("checkpoint") != relative
            or selected.get("step") != training.get("best_step") or selected.get("step") != 2304
            or selected.get("pairs_seen") != 147360
            or training.get("recipe") != REWARD_RECIPE
            or summary.get("recipe") != REWARD_RECIPE):
        raise ValueError("LoRA PPO requires the audited development-NLL-selected Reward checkpoint")
    checkpoint = run.file(relative)
    if not checkpoint.is_file():
        raise ValueError("Selected full-pool LoRA Reward checkpoint is missing")
    return checkpoint, {
        "checkpoint": relative,
        "checkpoint_sha256": digest(checkpoint),
        "training_sha256": digest(training_path),
        "summary_sha256": digest(summary_path),
        "step": selected["step"],
        "pairs_seen": selected["pairs_seen"],
        "development": selected["development"],
        "reward_recipe_sha256": hash_json(REWARD_RECIPE),
    }


def load_selected_reward(c):
    """Load the exact selected adapter/head, then make every tensor immutable."""
    import torch
    from .ppo import load_model
    run = Run(c)
    checkpoint_path, identity = _selected_reward(run)
    model = make_reward_model(load_model(c, causal=False))
    checkpoint = torch.load(checkpoint_path, map_location=model.head.weight.device, weights_only=False)
    if checkpoint.get("step") != identity["step"] or checkpoint.get("seen") != identity["pairs_seen"]:
        raise ValueError("Selected Reward checkpoint cursor differs from the audited report")
    restore_reward(model, checkpoint)
    if hasattr(model.backbone, "gradient_checkpointing_disable"):
        model.backbone.gradient_checkpointing_disable()
    model.requires_grad_(False).eval()
    parameters = list(model.parameters())
    if (not parameters or any(p.requires_grad or p.dtype != torch.float32 or not torch.isfinite(p).all()
                              for p in parameters)):
        raise ValueError("Loaded LoRA Reward is not a finite frozen FP32 model")
    return model, identity


class FrozenLoRANormalizedReward:
    """Canonical EOS scorer using a frozen Reward adapter and scalar head."""
    def __init__(self, model, mean, std, *, eos, pad, context_tokens=4096, batch_size=4):
        import torch
        if (type(mean) not in (int, float) or type(std) not in (int, float)
                or not math.isfinite(mean) or not math.isfinite(std) or std <= 0):
            raise ValueError("Invalid fixed LoRA Reward normalization")
        self.model = model.requires_grad_(False).eval()
        self.device = next(model.parameters()).device
        self.mean, self.std = float(mean), float(std)
        self.eos, self.pad = int(eos), int(pad)
        self.context_tokens, self.batch_size = int(context_tokens), int(batch_size)
        self._torch = torch

    def frozen_parameters(self):
        return self.model.parameters()

    def raw_score(self, prompts, responses):
        if len(prompts) != len(responses) or not prompts:
            raise ValueError("Reward needs matched nonempty prompt/response lists")
        result = []
        with self._torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                sequences = []
                for prompt, response in zip(prompts[start:start + self.batch_size],
                                            responses[start:start + self.batch_size]):
                    if not prompt or not response:
                        raise ValueError("Reward prompt and response must be nonempty")
                    sequence = list(prompt) + list(response)
                    if sequence[-1] != self.eos:
                        sequence.append(self.eos)
                    if len(sequence) > self.context_tokens:
                        raise ValueError("Canonical LoRA Reward sequence exceeds model context")
                    sequences.append(sequence)
                result.append(self.model(sequences, self.pad))
        scores = self._torch.cat(result).float()
        if scores.shape != (len(prompts),) or not self._torch.isfinite(scores).all():
            raise ValueError("LoRA Reward produced invalid scores")
        return scores

    def score(self, prompts, responses):
        return (self.raw_score(prompts, responses) - self.mean) / self.std


def make_lora_critic(backbone, reward_model, mean, std, settings):
    """Copy Reward adapter/head values into a separate trainable value model."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    configurations = list(reward_model.backbone.peft_config.values())
    if len(configurations) != 1:
        raise ValueError("Reward must contain exactly one frozen LoRA adapter")
    reward_adapter = configurations[0]
    reward_targets = set(reward_adapter.target_modules)
    if reward_targets != set(REWARD_RECIPE["target_modules"]):
        raise ValueError("Reward LoRA target modules differ from the audited recipe")
    if (settings["critic_lora_r"] != reward_adapter.r
            or settings["critic_lora_alpha"] != reward_adapter.lora_alpha):
        raise ValueError("Critic LoRA rank/alpha must match the selected Reward initialization")
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Critic needs finite positive Reward normalization")
    backbone.requires_grad_(False)
    adapted = get_peft_model(backbone, LoraConfig(
        r=settings["critic_lora_r"], lora_alpha=settings["critic_lora_alpha"],
        target_modules=sorted(reward_targets), lora_dropout=0.,
        bias="none", task_type=TaskType.FEATURE_EXTRACTION))

    class ValueModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = adapted
            self.config = backbone.config
            dimension = getattr(self.config, "hidden_size", None) or self.config.n_embd
            self.score = torch.nn.Linear(dimension, 1, bias=True, dtype=torch.float32,
                                         device=next(backbone.parameters()).device)

        def forward(self, **inputs):
            output = self.backbone(**inputs)
            return self.score(output.last_hidden_state.float()).squeeze(-1)

    critic = ValueModel().eval()
    reward_adapters = {n: p for n, p in reward_model.named_parameters() if "lora_" in n}
    critic_adapters = {n: p for n, p in critic.named_parameters() if "lora_" in n}
    if not reward_adapters or set(reward_adapters) != set(critic_adapters):
        raise ValueError("Reward and Critic adapter tensor schemas differ")
    with torch.no_grad():
        for name, target in critic_adapters.items():
            source = reward_adapters[name]
            if source.shape != target.shape or not torch.isfinite(source).all():
                raise ValueError("Invalid Reward adapter tensor for Critic initialization")
            target.copy_(source)
        if reward_model.head.bias is not None or reward_model.head.weight.shape != critic.score.weight.shape:
            raise ValueError("Reward scalar head is incompatible with Critic initialization")
        critic.score.weight.copy_(reward_model.head.weight / float(std))
        critic.score.bias.fill_(-float(mean) / float(std))
    reward_parameters = list(reward_model.parameters())
    critic_parameters = list(critic.parameters())
    if ({id(p) for p in reward_parameters} & {id(p) for p in critic_parameters}
            or {p.data_ptr() for p in reward_parameters} & {p.data_ptr() for p in critic_parameters}):
        raise ValueError("Reward and Critic must not share parameter objects or storage")
    if any(p.requires_grad for p in reward_parameters):
        raise ValueError("Reward became trainable while initializing Critic")
    return critic


def _normalizer(run):
    path = run.file(f"{PREFIX}/normalization.json")
    value = read_json(path)
    if (value.get("protocol") != "v3_full_pool_lora_fixed_sft_mean_std_v1"
            or value.get("method") != METHOD or value.get("calibration_responses") != 4096
            or not math.isfinite(value.get("mean", float("nan")))
            or not math.isfinite(value.get("std", float("nan"))) or value["std"] <= 0):
        raise ValueError("Invalid full-pool LoRA Reward normalization artifact")
    return value, path


def _score_records(reward, records, *, normalized):
    scores = (reward.score if normalized else reward.raw_score)(
        [r["prompt_tokens"] for r in records], [r["response_tokens"] for r in records])
    return scores.detach().cpu().numpy().astype(np.float64)


def normalize_reward(c):
    """Fit a fixed scale and cache SFT baselines for all PPO probes."""
    import torch
    from .ppo import load_tokenizer
    _require_ppo_recipe(c)
    run, stage = Run(c), f"{STAGE_PREFIX}-normalization"
    marker = run.file(f"{PREFIX}/normalization.json")
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(marker)
        run.require("shared-sft")
        _, recovery_identity = _selected_reward(run)
        outputs = [run.file(f"{PREFIX}/baselines/{name}.npz")
                   for name in ("calibration", "train_probe", "dev_probe")] + [marker]
        inputs = [run.receipt_path("shared-sft"),
                  run.receipt_path(f"{REWARD_STAGE_PREFIX}-train"),
                  run.receipt_path(f"{REWARD_STAGE_PREFIX}-report"),
                  run.file(recovery_identity["checkpoint"])]
        context = {"recipe": RECIPE, "ppo_recipe": PPO_RECIPE,
                   "reward": recovery_identity}
        def validate_recovered_normalization():
            _normalizer(run)
        if recover_stage(run, stage, outputs, inputs, code_paths=sources(),
                         validator=validate_recovered_normalization, context=context):
            return read_json(marker)
        model, identity = load_selected_reward(c)
        token = load_tokenizer(c)
        raw = FrozenLoRANormalizedReward(model, 0., 1., eos=token.eos_token_id,
                                         pad=token.pad_token_id,
                                         context_tokens=c["data"]["context_tokens"],
                                         batch_size=c["ppo"]["local_rollout_forward_batch_size"])
        calibration = rows(run.file("shared/calibration.jsonl"))
        if (len(calibration) != c["reward"]["calibration_responses"]
                or len({r["id"] for r in calibration}) != len(calibration)):
            raise ValueError("LoRA Reward calibration trajectories changed")
        calibration_scores = _score_records(raw, calibration, normalized=False)
        mean, std = float(calibration_scores.mean(dtype=np.float64)), float(calibration_scores.std(ddof=0, dtype=np.float64))
        if not np.isfinite(calibration_scores).all() or not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-8:
            raise ValueError("LoRA Reward calibration has no usable finite score scale")
        reward = FrozenLoRANormalizedReward(model, mean, std, eos=token.eos_token_id,
                                            pad=token.pad_token_id,
                                            context_tokens=c["data"]["context_tokens"],
                                            batch_size=c["ppo"]["local_rollout_forward_batch_size"])
        outputs = []
        for name, records_ in (("calibration", calibration),
                               ("train_probe", rows(run.file("shared/train_probe.jsonl"))),
                               ("dev_probe", rows(run.file("shared/dev_probe.jsonl")))):
            values = (calibration_scores if name == "calibration"
                      else _score_records(raw, records_, normalized=False))
            normalized_values = (values - mean) / std
            if not np.isfinite(normalized_values).all() or len({r["id"] for r in records_}) != len(records_):
                raise ValueError("Invalid fixed LoRA PPO probe baselines")
            path = run.file(f"{PREFIX}/baselines/{name}.npz")
            save_npz(path, ids=np.asarray([r["id"] for r in records_]), raw_scores=values,
                     normalized_scores=normalized_values)
            outputs.append(path)
        reference = (calibration_scores - mean) / std
        value = {
            "protocol": "v3_full_pool_lora_fixed_sft_mean_std_v1",
            "method": METHOD, "mean": mean, "std": std,
            "calibration_responses": len(calibration),
            "calibration_sha256": digest(run.file("shared/calibration.jsonl")),
            "score_dtype": "float32", "statistics_dtype": "float64", "ddof": 0,
            "reference_normalized_mean": float(reference.mean()),
            "reference_normalized_std": float(reference.std(ddof=0)),
            "reward": identity, "frozen_after_fit": True, "api_calls": 0,
        }
        write_json(marker, value)
        outputs.append(marker)
        seal_and_publish(run, stage, outputs, inputs, metadata={"passed": True, "method": METHOD},
                         code_paths=sources(), context=context)
        del reward, raw, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return value


def _initial_terminal_parity(critic, reward, records, pad):
    import torch
    from .ppo_trainer import left_pad, prepare_inputs
    records = list(records)
    sequences = []
    for row in records:
        sequence = list(row["prompt_tokens"]) + list(row["response_tokens"])
        if sequence[-1] != reward.eos:
            sequence.append(reward.eos)
        sequences.append(sequence)
    tokens = left_pad(sequences, pad, next(critic.parameters()).device)
    with torch.no_grad():
        actual = critic(**prepare_inputs(tokens, pad))[:, -1]
        expected = reward.score([r["prompt_tokens"] for r in records],
                                [r["response_tokens"] for r in records])
    error = float(torch.max(torch.abs(actual - expected)))
    if error > 2e-5:
        raise ValueError(f"Copied Critic does not reproduce normalized LoRA Reward at EOS: {error}")
    return error


def _build_models(c):
    from .ppo import load_model, load_tokenizer, make_actor
    run = Run(c)
    normalizer, _ = _normalizer(run)
    token = load_tokenizer(c)
    reward_model, identity = load_selected_reward(c)
    reward = FrozenLoRANormalizedReward(
        reward_model, normalizer["mean"], normalizer["std"], eos=token.eos_token_id,
        pad=token.pad_token_id, context_tokens=c["data"]["context_tokens"],
        batch_size=c["ppo"]["local_rollout_forward_batch_size"])
    critic = make_lora_critic(load_model(c, causal=False), reward_model,
                              normalizer["mean"], normalizer["std"], c["ppo"])
    actor = make_actor(load_model(c), c["ppo"])
    if ({id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}
            or {id(p) for p in actor.parameters()} & {id(p) for p in reward_model.parameters()}):
        raise ValueError("Actor, Critic and Reward parameter objects must be independent")
    return actor, critic, reward, token, normalizer, identity


def policy_smoke(c):
    """One worst-shape synchronous update with memory and immutability guards."""
    import torch
    from contextlib import nullcontext
    from filelock import FileLock
    from prism.hf import prompt_ids
    from .ppo_trainer import _trainable_state, seed_everything
    from .distributed_ppo import (DistributedPPOEngine, barrier, broadcast_object,
                                  initialize_process_group, rank0)
    _require_ppo_recipe(c)
    rank, world, _ = initialize_process_group()
    run, stage = Run(c), f"{STAGE_PREFIX}-smoke"
    marker = run.file(f"{PREFIX}/smoke.json")
    lock_path = run.file(f"control/locks/{stage}.lock")
    if rank0():
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    barrier()
    lock = FileLock(str(lock_path), timeout=0) if rank0() else nullcontext()
    with lock:
        completed = run.completed(stage) if rank0() else None
        completed = broadcast_object(completed)
        if completed:
            return read_json(marker)
        if rank0():
            run.require(f"{STAGE_PREFIX}-normalization")
        _, recovery_identity = _selected_reward(run)
        folder = run.file(f"{PREFIX}/smoke-checkpoints")
        global_batch = 4 * world
        checkpoint = folder / f"episode-{global_batch:08d}.pt"
        smoke_outputs = [marker, checkpoint, folder / "latest.json"]
        smoke_inputs = [run.receipt_path(f"{STAGE_PREFIX}-normalization")]
        smoke_context = {"recipe": RECIPE, "ppo_recipe": PPO_RECIPE,
                         "reward": recovery_identity,
                         "shape": {"world_size": world, "global_rollout": global_batch,
                                   "local_rollout": 4, "micro": 1,
                                   "response_tokens": 512, "generation_chunk": 4,
                                   "force_full_length": True}}
        def validate_recovered_smoke():
            recovered = read_json(marker)
            if (recovered.get("passed") is not True
                    or recovered.get("forced_response_tokens") != 512
                    or recovered.get("distributed_world_size") != world
                    or recovered.get("local_rollout_batch_size") != 4
                    or recovered.get("train_micro_batch_size") != 1):
                raise ValueError("Recovered LoRA PPO smoke did not pass")
        recovered = (recover_stage(run, stage, smoke_outputs, smoke_inputs,
                                   code_paths=sources(), validator=validate_recovered_smoke,
                                   context=smoke_context) if rank0() else None)
        recovered = broadcast_object(recovered)
        if recovered:
            return read_json(marker)
        barrier()
        seed_everything(c["study"]["seed"])
        actor, critic, reward, token, normalizer, identity = _build_models(c)
        records = rows(run.file("shared/calibration.jsonl"))[:4]
        parity = _initial_terminal_parity(critic, reward, records, token.pad_token_id)
        settings = dict(c["ppo"])
        settings.update(total_episodes=global_batch, rollout_batch_size=global_batch,
                        mini_batch_size=2*world, per_device_train_batch_size=1,
                        gradient_accumulation_steps=2*world, num_mini_batches=2,
                        num_ppo_epochs=1, response_length=512,
                        local_rollout_forward_batch_size=4,
                        engineering_checkpoint=global_batch,
                        durable_checkpoint_interval=global_batch,
                        checkpoints=[global_batch], probability_check_rows=4)
        encoded = [prompt_ids(token, row["messages"], c["data"]["max_prompt_tokens"])
                   for row in rows(run.file("data/policy.jsonl"))]
        if len(encoded) < global_batch:
            raise ValueError("Distributed full-shape smoke lacks policy prompts")
        prompts = [value for _, value in sorted(enumerate(encoded),
                                                key=lambda item: (-len(item[1]), item[0]))[:global_batch]]
        seed_everything(c["study"]["seed"])
        before_actor, before_critic = _trainable_state(actor), _trainable_state(critic)
        reward_versions = {n: p._version for n, p in reward.model.named_parameters()}
        contract = hash_json({"config": run.fingerprint, "reward": identity,
                              "normalization": digest(run.file(f"{PREFIX}/normalization.json")),
                              "recipe": RECIPE, "world_size": world,
                              "sources": {p.name: digest(p) for p in sources()}})
        engine = DistributedPPOEngine(
            actor, critic, reward, prompts, pad=token.pad_token_id, eos=-1,
            settings=settings, seed=c["study"]["seed"], contract_hash=contract,
            checkpoint_dir=folder)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if checkpoint.is_file() or (folder / "latest.json").is_file():
            engine.resume()
            if engine.completed_episodes != global_batch:
                raise ValueError("Recovered LoRA PPO smoke checkpoint has the wrong episode")
        result = engine.run()
        after_actor, after_critic = _trainable_state(actor), _trainable_state(critic)
        if (not any(not torch.equal(before_actor[n], after_actor[n]) for n in before_actor)
                or not any(not torch.equal(before_critic[n], after_critic[n]) for n in before_critic)
                or any(p._version != reward_versions[n] for n, p in reward.model.named_parameters())
                or result["history"][-1]["mean_response_tokens"] != 512.):
            raise ValueError("LoRA PPO smoke did not update actor/critic while preserving Reward")
        peak = torch.tensor(float(torch.cuda.max_memory_allocated()), dtype=torch.float64,
                            device=next(actor.parameters()).device)
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        value = {"protocol": "v3_full_pool_lora_distributed_ppo_smoke_v3", "passed": True,
                 "initial_critic_reward_eos_max_error": parity,
                 "reward_checkpoint": identity, "normalization": normalizer,
                 "reward_parameter_tensors": len(reward_versions),
                 "frozen_reward_unchanged": True, "formal_policy_admitted": False,
                 "distributed_world_size": world, "global_rollout_batch_size": global_batch,
                 "local_rollout_batch_size": 4, "train_micro_batch_size": 1,
                 "generation_forward_batch_size": 4, "forced_response_tokens": 512,
                 "longest_prompt_tokens": max(map(len, prompts)),
                 "shortest_selected_prompt_tokens": min(map(len, prompts)),
                 "result": result, "api_calls": 0,
                 "peak_cuda_bytes": int(peak)}
        if rank0():
            write_json(marker, value)
            seal_and_publish(run, stage, smoke_outputs, smoke_inputs,
                             metadata={"passed": True, "world_size": world},
                             code_paths=sources(), context=smoke_context)
        barrier()
        return value


def _baseline(run, name, records):
    path = run.file(f"{PREFIX}/baselines/{name}.npz")
    with np.load(path, allow_pickle=False) as arrays:
        ids = arrays["ids"].astype(str)
        values = arrays["normalized_scores"].astype(np.float64)
    if ids.tolist() != [r["id"] for r in records] or values.shape != (len(records),) or not np.isfinite(values).all():
        raise ValueError("LoRA PPO fixed SFT probe baseline identities changed")
    return values, path


def _repetition(tokens):
    return 1. - len(set(tokens)) / len(tokens)


def evaluate_probe(c, engine, run, name, identity):
    """Paired LoRA-Reward/KL probe tied to a committed policy checkpoint."""
    import torch
    from .ppo_trainer import (left_pad, prepare_inputs, response_masks,
                              seed_everything, stream_generate)
    episode = engine.completed_episodes
    stage = f"{STAGE_PREFIX}-probe-{episode:08d}-{name}"
    target = run.file(f"policy/{METHOD}/probes/{episode:08d}-{name}.json")
    checkpoint = run.file(f"policy/{METHOD}/checkpoints/episode-{episode:08d}.pt")
    contract_path = run.file(f"policy/{METHOD}/contract.json")
    if not checkpoint.is_file():
        raise ValueError("A LoRA PPO probe must bind a committed policy checkpoint")
    records = rows(run.file(f"shared/{name}.jsonl"))
    baseline, baseline_path = _baseline(run, name, records)
    fixed = {"protocol": "v3_full_pool_lora_fixed_policy_probe_v1", "method": METHOD,
             "episode": episode, "probe": name, "contract_hash": engine.contract_hash,
             "checkpoint_sha256": digest(checkpoint), "shared_probe_sha256": digest(run.file(f"shared/{name}.jsonl")),
             "baseline_sha256": digest(baseline_path), "reward_checkpoint_sha256": identity["checkpoint_sha256"]}
    if run.completed(stage):
        existing = read_json(target)
        if any(existing.get(k) != v for k, v in fixed.items()):
            raise ValueError("Existing LoRA policy probe belongs to another checkpoint")
        return target
    values, predictions, mc_targets = [], [], []
    for i, row in enumerate(records):
        seed_everything(row["seed"])
        query = left_pad([row["prompt_tokens"]], engine.pad, engine.device)
        with torch.no_grad():
            response, old = stream_generate(engine.actor, query, pad=engine.pad, eos=engine.eos,
                                             max_new_tokens=c["ppo"]["response_length"], temperature=1.)
            valid = response.ne(engine.pad)
            packed = torch.cat((query, response), 1)
            with engine.actor.disable_adapter():
                reference = engine._actor_logp(packed, response, query.shape[1])
            ratio = float((old - reference)[valid].sum())
            actual = response[0][valid[0]].tolist()
            score = float(engine.reward.score([row["prompt_tokens"]], [actual])[0])
            _, value_valid, length = response_masks(response, engine.pad)
            prediction = engine.critic(**prepare_inputs(packed, engine.pad))[:, query.shape[1] - 1:-1]
            token_reward = -c["ppo"]["kl_coef"] * (old - reference).masked_fill(~valid, 0.)
            token_reward[0, min(int(length[0]), response.shape[1] - 1)] += score
            returns = token_reward.flip(1).cumsum(1).flip(1)
            predictions.extend(prediction[value_valid].cpu().tolist())
            mc_targets.extend(returns[value_valid].cpu().tolist())
        base_tokens = row["response_tokens"]
        values.append({"id": row["id"], "source_id": row["source_id"], "response_tokens": actual,
                       "normalized_reward": score, "sft_normalized_reward": float(baseline[i]),
                       "paired_reward_delta": score-float(baseline[i]), "sequence_kl_mc": ratio,
                       "rlhf_objective": score-c["ppo"]["kl_coef"]*ratio,
                       "paired_rlhf_objective_delta": score-c["ppo"]["kl_coef"]*ratio-float(baseline[i]),
                       "length": len(actual), "sft_length": len(base_tokens),
                       "repetition_fraction": _repetition(actual),
                       "sft_repetition_fraction": _repetition(base_tokens),
                       "has_eos": actual[-1] == engine.eos,
                       "sft_has_eos": base_tokens[-1] == engine.eos})
    delta = np.asarray([r["paired_reward_delta"] for r in values], dtype=np.float64)
    objective = np.asarray([r["paired_rlhf_objective_delta"] for r in values], dtype=np.float64)
    lengths = np.asarray([r["length"] for r in values], dtype=np.float64)
    length_delta = lengths - np.asarray([r["sft_length"] for r in values], dtype=np.float64)
    correlation = (None if len(values) < 2 or length_delta.std() == 0 or delta.std() == 0
                   else float(np.corrcoef(length_delta, delta)[0, 1]))
    from .ppo_math import value_statistics
    report = {**fixed, "records": values,
              "mean_paired_reward_delta": float(delta.mean()),
              "paired_reward_delta_se": float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta) > 1 else None,
              "mean_paired_rlhf_objective_delta": float(objective.mean()),
              "paired_rlhf_objective_delta_se": float(objective.std(ddof=1)/np.sqrt(len(objective))) if len(objective) > 1 else None,
              "critic_monte_carlo": value_statistics(predictions, mc_targets),
              "mean_sequence_kl_mc": float(np.mean([r["sequence_kl_mc"] for r in values])),
              "mean_response_tokens": float(lengths.mean()),
              "mean_sft_response_tokens": float(np.mean([r["sft_length"] for r in values])),
              "mean_response_length_delta": float(length_delta.mean()),
              "eos_rate": float(np.mean([r["has_eos"] for r in values])),
              "mean_repetition_fraction": float(np.mean([r["repetition_fraction"] for r in values])),
              "reward_delta_response_length_delta_correlation": correlation}
    write_json(target, report)
    run.publish(stage, [target], inputs=[checkpoint, contract_path,
                run.receipt_path("shared-sft"), run.receipt_path(f"{STAGE_PREFIX}-normalization")],
                metadata=fixed)
    return target


def _checkpoint_steps(c, state, episode, contract_hash):
    from .distributed_ppo import PROTOCOL
    expected = episode // c["ppo"]["rollout_batch_size"] * c["ppo"]["num_ppo_epochs"] * c["ppo"]["num_mini_batches"]
    if (state.get("protocol") != PROTOCOL or state.get("episode") != episode
            or state.get("contract_hash") != contract_hash):
        raise ValueError("LoRA PPO checkpoint identity differs from its probe")
    for group in state["optimizer"]["param_groups"]:
        for parameter_id in group["params"]:
            if int(state["optimizer"]["state"][parameter_id]["step"]) != expected:
                raise ValueError("LoRA PPO actual optimizer count differs from schedule")
    return expected


def accept_engineering(c):
    import torch
    run, stage, episode = Run(c), f"{STAGE_PREFIX}-engineering-acceptance", RECIPE["engineering_episode"]
    if run.completed(stage):
        return read_json(run.file(f"{PREFIX}/engineering-acceptance.json"))
    for name in ("train_probe", "dev_probe"):
        run.require(f"{STAGE_PREFIX}-probe-{episode:08d}-{name}")
    run.require(f"{STAGE_PREFIX}-critic-warmup")
    contract_path = run.file(f"policy/{METHOD}/contract.json")
    contract = read_json(contract_path)["contract_hash"]
    checkpoint = run.file(f"policy/{METHOD}/checkpoints/episode-{episode:08d}.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    steps = _checkpoint_steps(c, state, episode, contract)
    warmup = read_json(run.file(f"policy/{METHOD}/critic-warmup.json"))
    checks = {"critic_warmup_passed": warmup.get("passed_performance") is True,
              "critic_warmup_steps": warmup.get("actual_optimizer_steps") == 128,
              "ppo_optimizer_steps": steps == 128,
              "reward_runtime_immutability_guard": True,
              "finite_train_probe": math.isfinite(read_json(run.file(f"policy/{METHOD}/probes/{episode:08d}-train_probe.json"))["mean_paired_reward_delta"]),
              "finite_development_probe": math.isfinite(read_json(run.file(f"policy/{METHOD}/probes/{episode:08d}-dev_probe.json"))["mean_paired_reward_delta"])}
    if not all(checks.values()):
        raise RuntimeError("LoRA PPO engineering gate failed: " + str(checks))
    value = {"protocol": "v3_full_pool_lora_ppo_engineering_gate_v1", "episode": episode,
             "passed": True, "checks": checks, "actual_optimizer_steps": steps}
    path = run.file(f"{PREFIX}/engineering-acceptance.json")
    write_json(path, value)
    run.publish(stage, [path], inputs=[checkpoint, contract_path,
                run.receipt_path(f"{STAGE_PREFIX}-critic-warmup"),
                run.receipt_path(f"{STAGE_PREFIX}-probe-{episode:08d}-train_probe"),
                run.receipt_path(f"{STAGE_PREFIX}-probe-{episode:08d}-dev_probe")])
    return value


def accept_pilot(c):
    import torch
    run, stage, episode = Run(c), f"{STAGE_PREFIX}-pilot-acceptance", PILOT_EPISODE
    if run.completed(stage):
        return read_json(run.file(f"{PREFIX}/pilot-acceptance.json"))
    run.require(f"{STAGE_PREFIX}-engineering-acceptance")
    for name in ("train_probe", "dev_probe"):
        run.require(f"{STAGE_PREFIX}-probe-{episode:08d}-{name}")
    development = read_json(run.file(f"policy/{METHOD}/probes/{episode:08d}-dev_probe.json"))
    training = read_json(run.file(f"policy/{METHOD}/probes/{episode:08d}-train_probe.json"))
    critic = development["critic_monte_carlo"]
    checks = {
        "positive_training_reward_delta": training["mean_paired_reward_delta"] > 0,
        "positive_development_reward_ci": development["paired_reward_delta_se"] is not None and
            development["mean_paired_reward_delta"]-1.96*development["paired_reward_delta_se"] > 0,
        "positive_development_rlhf_objective_ci": development["paired_rlhf_objective_delta_se"] is not None and
            development["mean_paired_rlhf_objective_delta"]-1.96*development["paired_rlhf_objective_delta_se"] > 0,
        "critic_beats_constant_mc_baseline": critic["explained_variance"] is not None and
            critic["explained_variance"] > 0 and critic["mse"] < critic["constant_baseline_mse"],
        "finite_safety_diagnostics": all(math.isfinite(development[k]) for k in
            ("mean_sequence_kl_mc", "mean_response_tokens", "mean_repetition_fraction")),
    }
    if not all(checks.values()):
        raise RuntimeError("Full-pool LoRA PPO pilot failed: " + str(checks))
    contract_path = run.file(f"policy/{METHOD}/contract.json")
    contract = read_json(contract_path)["contract_hash"]
    checkpoint = run.file(f"policy/{METHOD}/checkpoints/episode-{episode:08d}.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    steps = _checkpoint_steps(c, state, episode, contract)
    value = {"protocol": "v3_full_pool_lora_ppo_performance_gate_v1", "episode": episode,
             "passed": True, "checks": checks, "z_threshold": 1.96,
             "actual_optimizer_steps": steps,
             "claim_scope": "PPO optimizes the frozen LoRA Reward on fixed probes",
             "automatic_continuation_allowed": False,
             "independent_sft_benefit_claim_allowed": False,
             "required_before_sft_benefit_claim": "blind independent PPO-vs-SFT evaluation"}
    path = run.file(f"{PREFIX}/pilot-acceptance.json")
    write_json(path, value)
    run.publish(stage, [path], inputs=[checkpoint, contract_path,
                run.receipt_path(f"{STAGE_PREFIX}-engineering-acceptance"),
                run.receipt_path(f"{STAGE_PREFIX}-probe-{episode:08d}-train_probe"),
                run.receipt_path(f"{STAGE_PREFIX}-probe-{episode:08d}-dev_probe")])
    return value


def admit_continuation(c, *, user_confirmed=False):
    """Bind an explicit post-8192 review before spending the remaining horizon."""
    run, stage = Run(c), f"{STAGE_PREFIX}-admission"
    path = run.file("control/lora-ppo-admission.json")
    if run.completed(stage):
        return read_json(path)
    if user_confirmed is not True:
        raise ValueError("LoRA PPO continuation requires explicit user confirmation after reviewing episode 8192")
    run.require(f"{STAGE_PREFIX}-pilot-acceptance")
    episode = PILOT_EPISODE
    checkpoint = run.file(f"policy/{METHOD}/checkpoints/episode-{episode:08d}.pt")
    probes = {name: run.receipt_path(f"{STAGE_PREFIX}-probe-{episode:08d}-{name}")
              for name in ("train_probe", "dev_probe")}
    for name in probes:
        run.require(f"{STAGE_PREFIX}-probe-{episode:08d}-{name}")
    _, reward_identity = _selected_reward(run)
    value = {"protocol": "v3_full_pool_lora_ppo_user_admission_v1",
             "authority": "explicit_user_after_episode_8192_review",
             "user_confirmed": True, "episode": episode,
             "config_sha256": run.fingerprint,
             "checkpoint_sha256": digest(checkpoint),
             "pilot_acceptance_sha256": digest(run.receipt_path(f"{STAGE_PREFIX}-pilot-acceptance")),
             "probe_receipts": {name: digest(receipt) for name, receipt in probes.items()},
             "reward_checkpoint_sha256": reward_identity["checkpoint_sha256"],
             "claim_scope": "continue PPO optimization; no independent SFT-benefit claim"}
    write_json(path, value)
    run.publish(stage, [path], inputs=[checkpoint,
                run.receipt_path(f"{STAGE_PREFIX}-pilot-acceptance"), *probes.values()],
                metadata={"user_confirmed": True, "episode": episode})
    return value


def _validate_transition(c, run, stop_after_episodes, resume):
    allowed = [RECIPE["engineering_episode"], PILOT_EPISODE, *[v for v in c["ppo"]["checkpoints"] if v > PILOT_EPISODE]]
    allowed = sorted(set(allowed))
    if stop_after_episodes not in allowed:
        raise ValueError("LoRA PPO stop point must be an accepted staged checkpoint: " + str(allowed))
    latest = run.file(f"policy/{METHOD}/checkpoints/latest.json")
    checkpoint_dir = latest.parent
    checkpoint_files = sorted(checkpoint_dir.glob("episode-*.pt")) if checkpoint_dir.exists() else []
    if not resume:
        if latest.exists() or checkpoint_files or stop_after_episodes != RECIPE["engineering_episode"]:
            raise ValueError("A new LoRA PPO run must stop first at the 2048 engineering checkpoint")
    else:
        current = read_json(latest)["episode"] if latest.is_file() else None
        if checkpoint_files:
            try:
                newest_file = max(int(path.stem.split("-")[-1]) for path in checkpoint_files)
            except ValueError as exc:
                raise ValueError("Invalid orphan LoRA PPO checkpoint filename") from exc
            current = newest_file if current is None else max(current, newest_file)
        if current is None:
            raise ValueError("LoRA PPO resume needs an existing committed or recoverable checkpoint")
        if stop_after_episodes < current:
            raise ValueError("LoRA PPO resume stop cannot precede the committed checkpoint")
        if current < PILOT_EPISODE and stop_after_episodes > PILOT_EPISODE:
            raise ValueError("LoRA PPO must stop at 8192 for the performance gate before continuing")
        if (current > PILOT_EPISODE
                or (stop_after_episodes > current and current >= PILOT_EPISODE)):
            run.require(f"{STAGE_PREFIX}-admission")


def _ensure_critic_warmup_receipt(run, contract, warmup_path, identity):
    """Publish or recover the validated prefit + episode-0 bootstrap."""
    import torch
    from .distributed_ppo import PROTOCOL
    stage = f"{STAGE_PREFIX}-critic-warmup"
    if run.completed(stage):
        return read_json(warmup_path)
    initial = run.file(f"policy/{METHOD}/checkpoints/episode-00000000.pt")
    if not initial.is_file():
        raise ValueError("Incomplete LoRA PPO Critic bootstrap; preserve the attempt before retrying")
    state = torch.load(initial, map_location="cpu", weights_only=True)
    embedded = state.get("metadata", {}).get("critic_warmup")
    if not isinstance(embedded, dict):
        raise ValueError("LoRA PPO episode-0 checkpoint lacks its Critic warmup report")
    if warmup_path.is_file():
        warmup = read_json(warmup_path)
        if warmup != embedded:
            raise ValueError("Critic warmup JSON differs from the episode-0 checkpoint")
    else:
        warmup = embedded
        write_json(warmup_path, warmup)
    if (warmup.get("passed_performance") is not True
            or warmup.get("actual_optimizer_steps") != 128
            or state.get("protocol") != PROTOCOL or state.get("contract_hash") != contract
            or state.get("episode") != 0 or state.get("round") != 0
            or not state.get("actor") or not state.get("critic")):
        raise ValueError("LoRA PPO Critic bootstrap files failed contract/performance validation")
    run.publish(stage, [warmup_path, initial],
                inputs=[run.receipt_path("shared-sft"),
                        run.receipt_path(f"{STAGE_PREFIX}-normalization"),
                        run.receipt_path(f"{REWARD_STAGE_PREFIX}-report")],
                metadata={"method": METHOD, "accepted": True,
                          "actual_optimizer_steps": warmup["actual_optimizer_steps"],
                          "initial_critic_reward_eos_max_error": warmup["initial_critic_reward_eos_max_error"],
                          "reward_checkpoint_sha256": identity["checkpoint_sha256"]})
    return warmup


def train_policy(c, *, stop_after_episodes, resume=False):
    """Train one synchronous policy, with mandatory 2048 and 8192 stops."""
    import torch
    import traceback
    from contextlib import nullcontext
    from filelock import FileLock
    from prism.hf import prompt_ids
    from .ppo import _save_critic
    from .ppo_trainer import seed_everything
    from .distributed_ppo import (DistributedPPOEngine, PROTOCOL, barrier,
                                  broadcast_object, initialize_process_group,
                                  prefit_critic_distributed, rank0)
    _require_ppo_recipe(c)
    rank, world, _ = initialize_process_group()
    run = Run(c)
    target = run.file(f"policy/{METHOD}")
    target.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target / ".execution.lock"), timeout=0) if rank0() else nullcontext()
    with lock:
        failure = None
        if rank0():
            try:
                for stage in ("data", "shared-sft", f"{REWARD_STAGE_PREFIX}-train",
                              f"{REWARD_STAGE_PREFIX}-report", f"{STAGE_PREFIX}-normalization",
                              f"{STAGE_PREFIX}-smoke"):
                    run.require(stage)
                _validate_transition(c, run, stop_after_episodes, resume)
            except BaseException:
                failure = traceback.format_exc()
        failure = broadcast_object(failure)
        if failure is not None:
            raise RuntimeError("Distributed LoRA PPO admission failed:\n" + failure)
        seed_everything(c["study"]["seed"])
        actor, critic, reward, token, normalizer, identity = _build_models(c)
        parity = _initial_terminal_parity(critic, reward, rows(run.file("shared/calibration.jsonl"))[:4], token.pad_token_id)
        contract = hash_json({"config": run.fingerprint, "method": METHOD,
                              "reward": identity,
                              "reward_receipt": digest(run.receipt_path(f"{REWARD_STAGE_PREFIX}-report")),
                              "normalization": digest(run.file(f"{PREFIX}/normalization.json")),
                              "normalization_receipt": digest(run.receipt_path(f"{STAGE_PREFIX}-normalization")),
                              "policy_prompts": digest(run.file("data/policy.jsonl")),
                              "shared_sft": digest(run.receipt_path("shared-sft")),
                              "executor": PROTOCOL,
                              "distributed_world_size": world,
                              "executor_sources": {p.name: digest(p) for p in sources()},
                              "recipe": RECIPE})
        contract_path = target / "contract.json"
        if rank0():
            write_json(contract_path, {"protocol": RECIPE["protocol"], "contract_hash": contract,
                                       "method": METHOD, "reward": identity,
                                       "distributed_world_size": world,
                                       "execution_micro_batch_size": EXECUTION_MICRO_BATCH_SIZE,
                                       "initial_critic_reward_eos_max_error": parity})
        barrier()
        seed_everything(c["study"]["seed"])
        calibration = rows(run.file("shared/calibration.jsonl"))
        execution_settings = _distributed_execution_settings(c["ppo"])
        warmup_path = target / "critic-warmup.json"
        if not resume:
            count = c["ppo"]["critic_warmup_episodes"]
            heldout = calibration[count:count+min(128, len(calibration)-count)]
            warmup = prefit_critic_distributed(
                critic, reward, calibration[:count], heldout,
                settings=execution_settings, pad=token.pad_token_id,
                seed=c["study"]["seed"])
            warmup.update(initial_critic_reward_eos_max_error=parity,
                          initialization="copied_reward_adapter_and_normalized_head")
            if not warmup["passed_performance"]:
                raise RuntimeError("LoRA-initialized Critic did not pass heldout prefit")
        prompts = [prompt_ids(token, record["messages"], c["data"]["max_prompt_tokens"])
                   for record in rows(run.file("data/policy.jsonl"))]
        engine = DistributedPPOEngine(
            actor, critic, reward, prompts, pad=token.pad_token_id,
            eos=token.eos_token_id, settings=execution_settings,
            seed=c["study"]["seed"],
            contract_hash=contract, checkpoint_dir=target / "checkpoints")

        def probe(instance):
            for name in ("train_probe", "dev_probe"):
                evaluate_probe(c, instance, run, name, identity)
            if instance.completed_episodes == RECIPE["engineering_episode"]:
                accept_engineering(c)
            if instance.completed_episodes == PILOT_EPISODE:
                accept_pilot(c)

        if resume:
            if rank0():
                warmup = _ensure_critic_warmup_receipt(run, contract, warmup_path, identity)
            barrier()
            engine.resume(on_checkpoint=probe)
        else:
            engine.save_checkpoint(metadata={"critic_warmup": warmup})
            if rank0():
                _ensure_critic_warmup_receipt(run, contract, warmup_path, identity)
            barrier()
        if (rank0() and stop_after_episodes > engine.completed_episodes
                and engine.completed_episodes >= PILOT_EPISODE):
            run.require(f"{STAGE_PREFIX}-admission")
        barrier()
        result = engine.run(stop_after_episodes=stop_after_episodes, on_checkpoint=probe)
        result.update(method=METHOD, seed=c["study"]["seed"],
                      reward_training_pairs=identity["pairs_seen"],
                      reward_checkpoint=identity["checkpoint"],
                      total_episodes=c["ppo"]["total_episodes"], executor=PROTOCOL,
                      distributed_world_size=world,
                      execution_micro_batch_size=execution_settings["per_device_train_batch_size"],
                      execution_gradient_accumulation_steps=execution_settings["gradient_accumulation_steps"],
                      critic="independent_reward_initialized_lora_plus_scalar_head",
                      normalization=normalizer, formal_prism_comparison=False)
        if not rank0():
            barrier()
            return result
        report_path = target / f"report-{engine.completed_episodes:08d}.json"
        if report_path.exists():
            previous = read_json(report_path)
            left = {k: v for k, v in previous.items() if k != "seconds_this_invocation"}
            right = {k: v for k, v in result.items() if k != "seconds_this_invocation"}
            if left != right:
                raise ValueError("Existing LoRA PPO report differs from checkpoint state")
            result = previous
        else:
            write_json(report_path, result)
        if not result["completed"]:
            barrier()
            return result
        run.require(f"{STAGE_PREFIX}-pilot-acceptance")
        run.require(f"{STAGE_PREFIX}-admission")
        write_json(target / "report.json", result)
        actor.save_pretrained(str(target / "actor"), safe_serialization=True)
        token.save_pretrained(str(target / "actor"))
        _save_critic(critic, target / "critic")
        outputs = [p for p in target.rglob("*") if p.is_file()
                   and "checkpoints" not in p.parts and p.name != ".execution.lock"]
        final_checkpoint = target / "checkpoints" / f"episode-{engine.completed_episodes:08d}.pt"
        run.publish(f"policy-{METHOD}", outputs,
                    inputs=[final_checkpoint, *[run.receipt_path(stage) for stage in
                            ("data", "shared-sft", f"{REWARD_STAGE_PREFIX}-report",
                             f"{STAGE_PREFIX}-normalization", f"{STAGE_PREFIX}-smoke",
                             f"{STAGE_PREFIX}-critic-warmup",
                             f"{STAGE_PREFIX}-engineering-acceptance",
                             f"{STAGE_PREFIX}-pilot-acceptance",
                             f"{STAGE_PREFIX}-admission")]],
                    metadata={"method": METHOD, "episodes": engine.completed_episodes,
                              "distributed_world_size": world,
                              "executor": PROTOCOL, "completed": True,
                              "actual_optimizer_steps": result["actual_optimizer_steps"]})
        barrier()
        return result
