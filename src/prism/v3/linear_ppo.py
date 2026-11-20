"""Distributed PPO for the two frozen linear V3 Reward models.

The scientific PPO recipe is unchanged from the validated LoRA-Reward run.
Only the Reward implementation, matching Critic initialization and the
user-selected formal horizon differ.  The scheduler deliberately retains the
original 65,536-episode trajectory while the policies are sealed at 16,384.
"""
from contextlib import nullcontext
from pathlib import Path
import math

import numpy as np

from prism.io import digest, rows
from .artifacts import Run, hash_json, read_json, write_json


METHODS = ("empirical", "prism")
FORMAL_HORIZON = 16384
SCHEDULE_EPISODES = 65536
EXECUTION_MICRO_BATCH_SIZE = 1
RECIPE = {
    "protocol": "v3_frozen_linear_reward_distributed_ppo_v1",
    "preference_budget": 14000,
    "design_lambda": .001,
    "reward_l2": .1,
    "formal_horizon": FORMAL_HORIZON,
    "scheduler_total_episodes": SCHEDULE_EPISODES,
    "distributed_world_size": 4,
    "distributed_strategy": "homogeneous_synchronous_global_batch_gradient_sum",
    "reward": "frozen_sft_backbone_plus_method_specific_linear_head",
    "critic": "independent_sft_backbone_plus_lora_and_normalized_linear_head",
    "actor": "sft_plus_zero_initialized_lora",
    "paid_api_calls": 0,
}


def _recipe(c):
    """Return the sealed 14k recipe or the explicitly isolated V5 8k variant."""
    contract = c.get("v5")
    if not isinstance(contract, dict):
        return dict(RECIPE)
    budget = contract.get("preference_budget")
    design_lambda = contract.get("design_lambda")
    reward_l2 = contract.get("reward_l2")
    if budget == 14000 and reward_l2 == .1:
        return dict(RECIPE)
    if (contract.get("study_variant") == "budget_n8000" and budget == 8000
            and math.isclose(design_lambda, .001, rel_tol=0., abs_tol=1e-15)
            and reward_l2 == "shared_development_mean_bt_nll"):
        return {**RECIPE,
                "protocol": "v5_frozen_linear_reward_n8000_distributed_ppo_v1",
                "preference_budget": 8000,
                "reward_l2": reward_l2}
    raise ValueError("Linear PPO rejects an unsealed Reward-budget variant")


def sources():
    folder = Path(__file__).resolve().parent
    return [Path(__file__), folder / "ppo.py", folder / "ppo_trainer.py",
            folder / "distributed_ppo.py", folder / "ppo_math.py"]


def _require_method(method):
    if method not in METHODS:
        raise ValueError("Linear PPO method must be empirical or prism")


def _require_recipe(c):
    from .lora_ppo import PPO_RECIPE
    mismatches = {key: (c["ppo"].get(key), value)
                  for key, value in PPO_RECIPE.items() if c["ppo"].get(key) != value}
    if mismatches:
        raise ValueError("Linear PPO must preserve the validated PPO recipe: " + str(mismatches))
    if (c["ppo"]["total_episodes"] != SCHEDULE_EPISODES
            or c["runtime"].get("dtype") != "float32"
            or c["runtime"].get("allow_tf32") is not False):
        raise ValueError("Linear PPO requires the frozen 65k FP32/TF32-off schedule")


def _execution_settings(settings):
    value = dict(settings)
    mini = value["mini_batch_size"]
    if mini % EXECUTION_MICRO_BATCH_SIZE:
        raise ValueError("Execution microbatch must divide the global minibatch")
    value["per_device_train_batch_size"] = EXECUTION_MICRO_BATCH_SIZE
    value["gradient_accumulation_steps"] = mini // EXECUTION_MICRO_BATCH_SIZE
    if (value["rollout_batch_size"] != mini * value["num_mini_batches"]
            or value["rollout_batch_size"] != value["per_device_train_batch_size"]
            * value["gradient_accumulation_steps"] * value["num_mini_batches"]):
        raise ValueError("Distributed placement changed the global PPO batch")
    return value


def _linear_identity(run, method):
    _require_method(method)
    recipe = _recipe(run.config)
    selection = run.selection()
    reward_report = read_json(run.file("reward/selection.json"))
    selected_l2 = reward_report.get("selected_l2")
    expected_l2 = recipe["reward_l2"]
    l2_valid = (selected_l2 in run.config["reward"]["l2_candidates"]
                and reward_report.get("selection_metric")
                == "equal_method_mean_development_bt_nll") if isinstance(expected_l2, str) else (
                    type(selected_l2) in (int, float)
                    and math.isclose(selected_l2, expected_l2, rel_tol=0., abs_tol=1e-15))
    if (selection.get("budget") != recipe["preference_budget"]
            or not math.isclose(selection.get("design_lambda", -1), recipe["design_lambda"],
                                rel_tol=0., abs_tol=1e-15)
            or reward_report.get("budget") != recipe["preference_budget"]
            or not l2_valid):
        raise ValueError("Linear PPO Reward identity differs from the sealed study")
    reward_path = run.file(f"reward/{method}.npz")
    norm_path = run.file(f"normalization/{method}.json")
    normalizer = read_json(norm_path)
    if (normalizer.get("method") != method or normalizer.get("frozen_after_fit") is not True
            or normalizer.get("ddof") != 0 or normalizer.get("std", 0) <= 0):
        raise ValueError("Method-specific Reward normalization is not the frozen audited artifact")
    with np.load(reward_path, allow_pickle=False) as arrays:
        weights = arrays["w"].astype(np.float32).copy()
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError("Linear Reward head is invalid")
    return weights, normalizer, {
        "method": method,
        "preference_budget": selection["budget"],
        "design_lambda": selection["design_lambda"],
        "reward_l2": reward_report["selected_l2"],
        "selection_sha256": digest(run.file("study/selection.json")),
        "reward_sha256": digest(reward_path),
        "normalization_sha256": digest(norm_path),
        "training_audit_receipt_sha256": digest(run.receipt_path("training-audit")),
    }


def admit(c, *, user_confirmed=False):
    """Freeze the user's 16k decision without changing the 65k LR schedule."""
    _require_recipe(c)
    if user_confirmed is not True:
        raise ValueError("Linear PPO requires explicit user confirmation of the 16k horizon")
    run = Run(c)
    recipe = _recipe(c)
    if run.completed("linear-ppo-admission"):
        return read_json(run.file("control/linear-ppo-admission.json"))
    for stage in ("data", "study-selection", "reward", "training-audit",
                  "shared-sft", "normalization"):
        run.require(stage)
    identities = {method: _linear_identity(run, method)[2] for method in METHODS}
    value = {**recipe, "user_confirmed": True, "methods": identities,
             "unchanged_ppo_fields": [
                 "actor_lora", "critic_lora", "rollout_batch_size", "mini_batch_size",
                 "num_ppo_epochs", "learning_rates", "kl_coef", "clip_ranges",
                 "response_length", "optimizer", "fp32", "scheduler_trajectory"],
             "changed_fields": ["reward_loader", "critic_initialization", "formal_horizon"],
             "api_calls": 0}
    target = run.file("control/linear-ppo-admission.json")
    write_json(target, value)
    return run.publish("linear-ppo-admission", [target],
                       [run.receipt_path(stage) for stage in
                        ("study-selection", "reward", "training-audit", "shared-sft", "normalization")],
                       metadata={"formal_horizon": FORMAL_HORIZON,
                                 "scheduler_total_episodes": SCHEDULE_EPISODES,
                                 "user_confirmed": True})


def _build_models(c, method):
    from .ppo import (FrozenNormalizedReward, load_model, load_tokenizer,
                      make_actor, make_critic)
    run = Run(c)
    weights, normalizer, identity = _linear_identity(run, method)
    token = load_tokenizer(c)
    reward = FrozenNormalizedReward(
        load_model(c, causal=False), weights, normalizer["mean"], normalizer["std"],
        eos=token.eos_token_id, pad=token.pad_token_id,
        context_tokens=c["data"]["context_tokens"],
        batch_size=c["ppo"]["local_rollout_forward_batch_size"])
    critic = make_critic(load_model(c, causal=False), weights, normalizer["mean"],
                         normalizer["std"], c["ppo"])
    actor = make_actor(load_model(c), c["ppo"])
    objects = [set(map(id, model.parameters())) for model in
               (actor, critic, reward.backbone)]
    if objects[0] & objects[1] or objects[0] & objects[2] or objects[1] & objects[2]:
        raise ValueError("Actor, Critic and Reward must not share parameter objects")
    return actor, critic, reward, token, normalizer, identity


def _initial_terminal_parity(critic, reward, records, pad):
    import torch
    from .ppo_trainer import left_pad, prepare_inputs
    sequences = []
    for row in records:
        sequence = list(row["prompt_tokens"]) + list(row["response_tokens"])
        if sequence[-1] != reward.eos:
            sequence.append(reward.eos)
        sequences.append(sequence)
    tokens = left_pad(sequences, pad, next(critic.parameters()).device)
    with torch.no_grad():
        actual = critic(**prepare_inputs(tokens, pad))[:, -1]
        expected = reward.score([row["prompt_tokens"] for row in records],
                                [row["response_tokens"] for row in records])
    error = float(torch.max(torch.abs(actual - expected)))
    if error > 2e-5:
        raise ValueError(f"Linear Critic does not reproduce normalized Reward at EOS: {error}")
    return error


def policy_smoke(c, method):
    """One distributed full-length update for the selected linear Reward."""
    import torch
    from filelock import FileLock
    from prism.hf import prompt_ids
    from .ppo_trainer import _trainable_state, seed_everything
    from .distributed_ppo import (DistributedPPOEngine, barrier, broadcast_object,
                                  initialize_process_group, rank0)
    _require_method(method)
    _require_recipe(c)
    recipe = _recipe(c)
    rank, world, _ = initialize_process_group(allowed_world_sizes=(4,))
    run = Run(c)
    stage = f"linear-ppo-smoke-{method}"
    marker = run.file(f"diagnostics/linear-ppo-smoke/{method}.json")
    folder = run.file(f"diagnostics/linear-ppo-smoke/{method}-checkpoints")
    lock_path = run.file(f"control/locks/{stage}.lock")
    if rank0():
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    barrier()
    lock = FileLock(str(lock_path), timeout=0) if rank0() else nullcontext()
    with lock:
        complete = run.completed(stage) if rank0() else None
        complete = broadcast_object(complete)
        if complete:
            return read_json(marker)
        failure = None
        if rank0():
            try:
                run.require("linear-ppo-admission")
            except BaseException as error:
                failure = repr(error)
        failure = broadcast_object(failure)
        if failure:
            raise RuntimeError("Linear PPO smoke admission failed: " + failure)
        seed_everything(c["study"]["seed"])
        actor, critic, reward, token, normalizer, identity = _build_models(c, method)
        records = rows(run.file("shared/calibration.jsonl"))[:4]
        parity = _initial_terminal_parity(critic, reward, records, token.pad_token_id)
        global_batch = 4 * world
        settings = dict(c["ppo"])
        settings.update(total_episodes=global_batch, rollout_batch_size=global_batch,
                        mini_batch_size=2 * world, per_device_train_batch_size=1,
                        gradient_accumulation_steps=2 * world, num_mini_batches=2,
                        num_ppo_epochs=1, response_length=512,
                        local_rollout_forward_batch_size=4,
                        engineering_checkpoint=global_batch,
                        durable_checkpoint_interval=global_batch,
                        checkpoints=[global_batch], probability_check_rows=4)
        encoded = [prompt_ids(token, row["messages"], c["data"]["max_prompt_tokens"])
                   for row in rows(run.file("data/policy.jsonl"))]
        prompts = [value for _, value in sorted(enumerate(encoded),
                   key=lambda item: (-len(item[1]), item[0]))[:global_batch]]
        if len(prompts) != global_batch:
            raise ValueError("Linear PPO smoke lacks policy prompts")
        before_actor, before_critic = _trainable_state(actor), _trainable_state(critic)
        reward_versions = {name: parameter._version
                           for name, parameter in reward.backbone.named_parameters()}
        reward_weights = reward.weights.detach().clone()
        contract = hash_json({"config": run.fingerprint, "recipe": recipe,
                              "identity": identity, "world_size": world,
                              "sources": {path.name: digest(path) for path in sources()}})
        engine = DistributedPPOEngine(
            actor, critic, reward, prompts, pad=token.pad_token_id, eos=-1,
            settings=settings, seed=c["study"]["seed"], contract_hash=contract,
            checkpoint_dir=folder)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if (folder / "latest.json").is_file():
            engine.resume()
        result = engine.run()
        after_actor, after_critic = _trainable_state(actor), _trainable_state(critic)
        if (not any(not torch.equal(before_actor[name], after_actor[name]) for name in before_actor)
                or not any(not torch.equal(before_critic[name], after_critic[name]) for name in before_critic)
                or any(parameter._version != reward_versions[name]
                       for name, parameter in reward.backbone.named_parameters())
                or not torch.equal(reward.weights, reward_weights)
                or result["history"][-1]["mean_response_tokens"] != 512.):
            raise ValueError("Linear PPO smoke failed update/frozen-Reward checks")
        peak = torch.tensor(float(torch.cuda.max_memory_allocated()), dtype=torch.float64,
                            device=next(actor.parameters()).device)
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        value = {"protocol": "v3_linear_reward_distributed_ppo_smoke_v1",
                 "method": method, "passed": True, "identity": identity,
                 "normalization": normalizer, "initial_critic_reward_eos_max_error": parity,
                 "frozen_reward_unchanged": True, "distributed_world_size": world,
                 "global_rollout_batch_size": global_batch, "train_micro_batch_size": 1,
                 "forced_response_tokens": 512, "result": result,
                 "peak_cuda_bytes": int(peak), "api_calls": 0}
        if rank0():
            write_json(marker, value)
            checkpoint = folder / f"episode-{global_batch:08d}.pt"
            run.publish(stage, [marker, checkpoint, folder / "latest.json"],
                        [run.receipt_path("linear-ppo-admission")],
                        metadata={"method": method, "passed": True, "world_size": world})
        barrier()
        return value


def _ensure_warmup_receipt(run, method, contract, warmup_path):
    import torch
    from .distributed_ppo import PROTOCOL
    stage = f"linear-ppo-critic-warmup-{method}"
    if run.completed(stage):
        return read_json(warmup_path)
    initial = run.file(f"policy/{method}/checkpoints/episode-00000000.pt")
    state = torch.load(initial, map_location="cpu", weights_only=True)
    embedded = state.get("metadata", {}).get("critic_warmup")
    if not isinstance(embedded, dict):
        raise ValueError("Episode-0 checkpoint lacks the Critic warmup report")
    if warmup_path.is_file():
        if read_json(warmup_path) != embedded:
            raise ValueError("Critic warmup JSON differs from episode-0 checkpoint")
    else:
        write_json(warmup_path, embedded)
    if (embedded.get("passed_performance") is not True
            or embedded.get("actual_optimizer_steps") != 128
            or state.get("protocol") != PROTOCOL or state.get("contract_hash") != contract
            or state.get("episode") != 0 or state.get("round") != 0):
        raise ValueError("Linear PPO Critic bootstrap failed its contract")
    run.publish(stage, [warmup_path, initial],
                [run.receipt_path(name) for name in
                 ("shared-sft", "reward", "normalization", "linear-ppo-admission")],
                metadata={"method": method, "accepted": True,
                          "actual_optimizer_steps": embedded["actual_optimizer_steps"]})
    return embedded


def train_policy(c, method, *, stop_after_episodes, resume=False):
    """Run and seal one linear-Reward policy at the user-selected 16k horizon."""
    import torch
    import traceback
    from filelock import FileLock
    from prism.hf import prompt_ids
    from .ppo import _save_critic, evaluate_probe
    from .ppo_trainer import seed_everything
    from .distributed_ppo import (DistributedPPOEngine, PROTOCOL, barrier,
                                  broadcast_object, initialize_process_group,
                                  prefit_critic_distributed, rank0)
    _require_method(method)
    _require_recipe(c)
    recipe = _recipe(c)
    if stop_after_episodes != FORMAL_HORIZON:
        raise ValueError(f"Formal linear PPO must stop at exactly {FORMAL_HORIZON} episodes")
    rank, world, _ = initialize_process_group(allowed_world_sizes=(4,))
    run = Run(c)
    target = run.file(f"policy/{method}")
    target.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target / ".execution.lock"), timeout=0) if rank0() else nullcontext()
    with lock:
        failure = None
        if rank0():
            try:
                for stage in ("data", "shared-sft", "reward", "normalization", "training-audit",
                              "linear-ppo-admission", f"linear-ppo-smoke-{method}"):
                    run.require(stage)
                if run.completed(f"policy-{method}"):
                    failure = "already-complete"
                elif resume is False and (target / "checkpoints/latest.json").is_file():
                    raise FileExistsError("Linear PPO checkpoint exists; resume it explicitly")
            except BaseException:
                failure = traceback.format_exc()
        failure = broadcast_object(failure)
        if failure == "already-complete":
            return run.require(f"policy-{method}") if rank0() else {"already_complete": True}
        if failure is not None:
            raise RuntimeError("Distributed linear PPO admission failed:\n" + failure)
        seed_everything(c["study"]["seed"])
        actor, critic, reward, token, normalizer, identity = _build_models(c, method)
        calibration = rows(run.file("shared/calibration.jsonl"))
        parity = _initial_terminal_parity(critic, reward, calibration[:4], token.pad_token_id)
        execution = _execution_settings(c["ppo"])
        contract = hash_json({
            "config": run.fingerprint, "method": method, "identity": identity,
            "admission": digest(run.receipt_path("linear-ppo-admission")),
            "policy_prompts": digest(run.file("data/policy.jsonl")),
            "shared_sft": digest(run.receipt_path("shared-sft")),
            "executor": PROTOCOL, "distributed_world_size": world,
            "formal_horizon": FORMAL_HORIZON,
            "scheduler_total_episodes": execution["total_episodes"],
            "executor_sources": {path.name: digest(path) for path in sources()},
            "recipe": recipe})
        contract_path = target / "contract.json"
        if rank0():
            write_json(contract_path, {
                "protocol": recipe["protocol"], "contract_hash": contract,
                "method": method, "identity": identity, "distributed_world_size": world,
                "formal_horizon": FORMAL_HORIZON,
                "scheduler_total_episodes": execution["total_episodes"],
                "initial_critic_reward_eos_max_error": parity})
        barrier()
        warmup_path = target / "critic-warmup.json"
        if not resume:
            count = execution["critic_warmup_episodes"]
            heldout = calibration[count:count + min(128, len(calibration) - count)]
            warmup = prefit_critic_distributed(
                critic, reward, calibration[:count], heldout, settings=execution,
                pad=token.pad_token_id, seed=c["study"]["seed"])
            warmup.update(initial_critic_reward_eos_max_error=parity,
                          initialization="normalized_linear_reward_head_plus_zero_critic_lora")
            if not warmup["passed_performance"]:
                raise RuntimeError("Linear Reward Critic did not pass heldout prefit")
        prompts = [prompt_ids(token, row["messages"], c["data"]["max_prompt_tokens"])
                   for row in rows(run.file("data/policy.jsonl"))]
        engine = DistributedPPOEngine(
            actor, critic, reward, prompts, pad=token.pad_token_id, eos=token.eos_token_id,
            settings=execution, seed=c["study"]["seed"], contract_hash=contract,
            checkpoint_dir=target / "checkpoints")

        def probe(instance):
            for name in ("train_probe", "dev_probe"):
                evaluate_probe(c, instance, method, run, name)

        if resume:
            if rank0():
                _ensure_warmup_receipt(run, method, contract, warmup_path)
            barrier()
            engine.resume(on_checkpoint=probe)
        else:
            engine.save_checkpoint(metadata={"critic_warmup": warmup})
            if rank0():
                _ensure_warmup_receipt(run, method, contract, warmup_path)
            barrier()
        result = engine.run(stop_after_episodes=FORMAL_HORIZON, on_checkpoint=probe)
        formal_complete = engine.completed_episodes == FORMAL_HORIZON
        result.update(method=method, seed=c["study"]["seed"],
                      budget=recipe["preference_budget"], executor=PROTOCOL,
                      distributed_world_size=world,
                      formal_horizon=FORMAL_HORIZON,
                      formal_horizon_completed=formal_complete,
                      scheduler_total_episodes=execution["total_episodes"],
                      scheduler_trajectory_completed=result.pop("completed"),
                      completed=formal_complete, total_episodes=FORMAL_HORIZON,
                      execution_micro_batch_size=execution["per_device_train_batch_size"],
                      execution_gradient_accumulation_steps=execution["gradient_accumulation_steps"],
                      critic="independent_sft_lora_plus_normalized_linear_head",
                      reward="frozen_sft_plus_linear_head", normalization=normalizer)
        if not rank0():
            barrier()
            return result
        report_path = target / f"report-{FORMAL_HORIZON:08d}.json"
        write_json(report_path, result)
        if not formal_complete:
            barrier()
            return result
        write_json(target / "report.json", result)
        actor.save_pretrained(str(target / "actor"), safe_serialization=True)
        token.save_pretrained(str(target / "actor"))
        _save_critic(critic, target / "critic")
        outputs = [path for path in target.rglob("*") if path.is_file()
                   and "checkpoints" not in path.parts and path.name != ".execution.lock"]
        final_checkpoint = target / "checkpoints" / f"episode-{FORMAL_HORIZON:08d}.pt"
        run.publish(f"policy-{method}", outputs,
                    [final_checkpoint, *[run.receipt_path(stage) for stage in
                     ("study-selection", "data", "shared-sft", "reward", "training-audit",
                      "normalization", "linear-ppo-admission", f"linear-ppo-smoke-{method}",
                      f"linear-ppo-critic-warmup-{method}")]],
                    metadata={"method": method, "episodes": FORMAL_HORIZON,
                              "scheduler_total_episodes": execution["total_episodes"],
                              "distributed_world_size": world, "executor": PROTOCOL,
                              "completed": True,
                              "actual_optimizer_steps": result["actual_optimizer_steps"]})
        barrier()
        return result


def audit_policy(c, method):
    """Independently validate the sealed 16k checkpoint and optimizer counts."""
    import torch
    from .distributed_ppo import PROTOCOL
    _require_method(method)
    _require_recipe(c)
    run = Run(c)
    stage = f"linear-ppo-audit-{method}"
    if run.completed(stage):
        return read_json(run.file(f"policy/{method}/audit-{FORMAL_HORIZON:08d}.json"))
    run.require(f"policy-{method}")
    report_path = run.file(f"policy/{method}/report.json")
    contract_path = run.file(f"policy/{method}/contract.json")
    checkpoint = run.file(f"policy/{method}/checkpoints/episode-{FORMAL_HORIZON:08d}.pt")
    report, contract = read_json(report_path), read_json(contract_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    rounds = FORMAL_HORIZON // c["ppo"]["rollout_batch_size"]
    steps = rounds * c["ppo"]["num_ppo_epochs"] * c["ppo"]["num_mini_batches"]
    actual = report.get("actual_optimizer_steps", {})
    checks = {
        "formal_horizon": report.get("completed") is True
            and report.get("formal_horizon_completed") is True
            and report.get("completed_episodes") == FORMAL_HORIZON,
        "scheduler_preserved": report.get("scheduler_total_episodes") == SCHEDULE_EPISODES
            and report.get("scheduler_trajectory_completed") is False,
        "checkpoint_contract": state.get("protocol") == PROTOCOL
            and state.get("episode") == FORMAL_HORIZON
            and state.get("round") == rounds
            and state.get("contract_hash") == contract.get("contract_hash"),
        "optimizer_steps": set(actual) == {"actor", "critic_lora", "critic_head"}
            and all(value.get("step_min") == steps and value.get("step_max") == steps
                    and value.get("tensor_count", 0) > 0 for value in actual.values()),
        "probability_checks": len(report.get("probability_checks", [])) == rounds
            and all(row.get("max_abs_logp_error", float("inf")) <= c["ppo"]["max_logp_error"]
                    and row.get("zero_update_approxkl", float("inf"))
                    <= c["ppo"]["max_zero_update_approxkl"]
                    for row in report.get("probability_checks", [])),
        "probes": all(run.completed(f"probe-{method}-{episode:08d}-{name}")
                      for episode in (2048, 8192, FORMAL_HORIZON)
                      for name in ("train_probe", "dev_probe")),
    }
    if not all(checks.values()):
        raise ValueError("Linear PPO independent audit failed: " + str(checks))
    value = {"protocol": "v3_linear_reward_ppo_independent_audit_v1",
             "method": method, "passed": True, "checks": checks,
             "formal_horizon": FORMAL_HORIZON, "rollout_rounds": rounds,
             "actual_optimizer_steps": steps, "checkpoint_sha256": digest(checkpoint),
             "report_sha256": digest(report_path), "api_calls": 0}
    target = run.file(f"policy/{method}/audit-{FORMAL_HORIZON:08d}.json")
    write_json(target, value)
    run.publish(stage, [target], [run.receipt_path(f"policy-{method}"), checkpoint,
                                  report_path, contract_path],
                metadata={"method": method, "passed": True,
                          "formal_horizon": FORMAL_HORIZON})
    return value
