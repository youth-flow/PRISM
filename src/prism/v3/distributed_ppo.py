"""Synchronous data-parallel PPO for one auditable policy trajectory.

Every rank owns a disjoint slice of the *same* global rollout.  Advantage
statistics, token denominators and gradients are reduced globally, so world
size changes placement only: it does not change the configured rollout batch,
minibatch, number of PPO epochs, optimizer steps or learning-rate schedule.
Only homogeneous ranks are supported by the production admission path.
"""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

from prism.io import digest
from prism.numerics import probability_path_metrics
from .artifacts import read_json, write_json
from .ppo_math import learning_rate_multiplier, value_statistics
from .ppo_trainer import (PPOEngine, _restore_trainable, _trainable_state, left_pad,
                          optimizer_steps, prepare_inputs, response_masks,
                          seed_everything, stream_generate, gae)


PROTOCOL = "v3_synchronous_data_parallel_streaming_ppo_v1"


def initialize_process_group(*, allowed_world_sizes=(4, 8)):
    """Initialize the one-node process group before any CUDA model is loaded."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in allowed_world_sizes:
        raise ValueError(f"Distributed LoRA PPO requires world size {allowed_world_sizes}, got {world}")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    if local_rank < 0 or rank < 0 or not torch.cuda.is_available():
        raise ValueError("Distributed LoRA PPO must run under torchrun on CUDA")
    if local_rank >= torch.cuda.device_count():
        raise ValueError("LOCAL_RANK is outside the visible CUDA device set")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    if dist.get_world_size() != world or dist.get_rank() != rank:
        raise RuntimeError("torchrun environment and process group disagree")
    return rank, world, local_rank


def rank0():
    return dist.get_rank() == 0


def barrier():
    dist.barrier()


def broadcast_object(value, source=0):
    payload = [value if dist.get_rank() == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def _global_sum(value):
    result = value.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _global_max_int(value, device):
    result = torch.tensor(int(value), dtype=torch.int64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return int(result)


def _global_whiten(value, mask):
    chosen = value[mask].double()
    statistics = torch.stack((chosen.sum(), chosen.square().sum(),
                              torch.tensor(float(chosen.numel()), dtype=torch.float64,
                                           device=value.device)))
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    total, square, count = statistics
    if count <= 1:
        raise ValueError("Distributed PPO needs at least two valid actions")
    mean = total / count
    variance = ((square - total.square() / count) / (count - 1.)).clamp_min(0.)
    result = (value - mean.float()) * torch.rsqrt(variance.float() + 1e-8)
    return result.masked_fill(~mask, 0.)


def _all_gather_lists(first, second):
    local = (list(map(float, first)), list(map(float, second)))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    left, right = [], []
    for a, b in gathered:
        left.extend(a)
        right.extend(b)
    return left, right


def _distributed_value_statistics(predictions, targets):
    prediction, target = predictions.double(), targets.double()
    error = target - prediction
    values = torch.stack((
        torch.tensor(float(target.numel()), dtype=torch.float64, device=target.device),
        error.square().sum(), (prediction-target).sum(), target.sum(),
        target.square().sum(), error.sum(), error.square().sum()))
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count, squared_error, signed_error, target_sum, target_square, error_sum, error_square = values
    if count <= 0 or not torch.isfinite(values).all():
        raise ValueError("Distributed value statistics are empty or nonfinite")
    target_variance = (target_square/count - (target_sum/count).square()).clamp_min(0.)
    error_variance = (error_square/count - (error_sum/count).square()).clamp_min(0.)
    variance = float(target_variance)
    return {"mse": float(squared_error/count), "bias": float(signed_error/count),
            "target_variance": variance,
            "explained_variance": float(1-error_variance/target_variance)
                if variance > 1e-12 else None,
            "constant_baseline_mse": variance}


def _model_digest(actor, critic):
    checksum = hashlib.sha256()
    for prefix, model in (("actor", actor), ("critic", critic)):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                checksum.update(f"{prefix}:{name}".encode())
                checksum.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def _sync_gradients(parameters):
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradients.append(parameter.grad)
    # LoRA exposes hundreds of small tensors.  One flattened collective avoids
    # hundreds of latency-bound PCIe/NCCL launches per optimizer step.
    flat = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    for gradient, synchronized in zip(
            gradients, torch._utils._unflatten_dense_tensors(flat, gradients)):
        gradient.copy_(synchronized)


class DistributedPPOEngine(PPOEngine):
    """Exact global-batch PPO with manually summed trainable gradients."""

    def __init__(self, *args, **kwargs):
        if not dist.is_initialized():
            raise ValueError("Initialize torch.distributed before constructing PPO")
        super().__init__(*args, **kwargs)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if self.world not in (4, 8):
            raise ValueError("Production distributed PPO supports 4 or 8 homogeneous GPUs")
        if self.batch % self.world or self.mini % self.world:
            raise ValueError("Global rollout and minibatch must be divisible by world size")
        self.local_batch = self.batch // self.world
        self.local_start = self.rank * self.local_batch
        self.local_end = self.local_start + self.local_batch
        # Model construction is seeded identically; rollout RNG is deliberately
        # rank-specific and is checkpointed for exact same-world-size recovery.
        seed_everything(self.seed + 1000003 * self.rank)

    def _pad_to_global_prompt_width(self, query):
        width = _global_max_int(query.shape[1], query.device)
        if query.shape[1] == width:
            return query
        result = torch.full((query.shape[0], width), self.pad, dtype=query.dtype,
                            device=query.device)
        result[:, -query.shape[1]:] = query
        return result

    @torch.no_grad()
    def rollout(self):
        global_indices = self.sampler.take(self.batch)
        owned = global_indices[self.local_start:self.local_end]
        prompts = [self.prompts[i] for i in owned]
        query = self._pad_to_global_prompt_width(left_pad(prompts, self.pad, self.device))
        context = query.shape[1]
        chunk = self.settings["local_rollout_forward_batch_size"]
        response_parts, logp_parts = [], []
        for start in range(0, self.local_batch, chunk):
            response, selected = stream_generate(
                self.actor, query[start:start + chunk], pad=self.pad, eos=self.eos,
                max_new_tokens=self.settings["response_length"], temperature=1.)
            response_parts.append(response)
            logp_parts.append(selected)
        response, old_logp = torch.cat(response_parts), torch.cat(logp_parts)
        query_response = torch.cat((query, response), 1)
        valid, value_valid, lengths = response_masks(response, self.pad)
        references, values = [], []
        for start in range(0, self.local_batch, self.micro):
            end = start + self.micro
            with self.actor.disable_adapter():
                references.append(self._actor_logp(query_response[start:end], response[start:end], context))
            values.append(self.critic(**prepare_inputs(query_response[start:end], self.pad))[:, context - 1:-1])
        reference = torch.cat(references).masked_fill(~valid, 0.)
        value = torch.cat(values).masked_fill(~value_valid, 0.)
        actual = [row[:int(length)].tolist() for row, length in zip(response, lengths)]
        scores = self.reward.score(prompts, actual).to(self.device, dtype=torch.float32)
        if scores.shape != (self.local_batch,) or not torch.isfinite(scores).all():
            raise ValueError("Distributed Reward returned invalid terminal scores")
        old_logp = old_logp.masked_fill(~valid, 0.)
        kl = old_logp - reference
        rewards = -self.settings["kl_coef"] * kl
        endpoint = lengths.clamp_max(response.shape[1] - 1)
        rewards[torch.arange(self.local_batch, device=self.device), endpoint] += scores
        advantages, returns = gae(rewards, value, self.settings["gamma"], self.settings["lam"])
        advantages = _global_whiten(advantages, valid)

        check = None
        if self.rank == 0:
            checks = min(self.settings["probability_check_rows"], self.local_batch)
            before_rng = torch.random.get_rng_state().clone()
            before_cuda = (torch.cuda.get_rng_state(self.device).clone()
                           if self.device.type == "cuda" else None)
            fresh = torch.cat([self._actor_logp(query_response[s:min(s + self.micro, checks)],
                                                response[s:min(s + self.micro, checks)], context)
                               for s in range(0, checks, self.micro)])
            check = probability_path_metrics(
                old_logp[:checks], fresh, valid[:checks],
                max_logp_error=self.settings["max_logp_error"],
                max_approxkl=self.settings["max_zero_update_approxkl"])
            if (not torch.equal(before_rng, torch.random.get_rng_state())
                    or (before_cuda is not None
                        and not torch.equal(before_cuda, torch.cuda.get_rng_state(self.device)))):
                raise RuntimeError("Distributed probability guard consumed RNG")
            check["round"] = self.completed_rounds + 1
        check = broadcast_object(check)
        self.probability_checks.append(check)
        return {"query_response": query_response, "response": response, "context": context,
                "old_logp": old_logp, "value": value, "returns": returns,
                "advantages": advantages, "valid": valid, "value_valid": value_valid,
                "scores": scores, "kl": kl, "reward": rewards,
                "prompt_indices": owned, "lengths": lengths}

    def _owned_minibatch(self, global_indices):
        return np.asarray([int(i) - self.local_start for i in global_indices
                           if self.local_start <= int(i) < self.local_end], dtype=np.int64)

    def update(self, batch):
        p, statistics = self.settings, []
        trainable = [parameter for values in self.groups.values() for parameter in values]
        for _ in range(p["num_ppo_epochs"]):
            order = self.shuffle_rng.permutation(self.batch)
            for start in range(0, self.batch, self.mini):
                indices = self._owned_minibatch(order[start:start + self.mini])
                local_policy = batch["valid"][indices].sum().double()
                local_value = batch["value_valid"][indices].sum().double()
                counts = torch.stack((local_policy, local_value))
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                policy_count, value_count = map(float, counts.tolist())
                if policy_count <= 0 or value_count <= 0:
                    raise RuntimeError("Empty distributed PPO global minibatch")
                self.optimizer.zero_grad(set_to_none=True)
                numerators = torch.zeros(5, dtype=torch.float64, device=self.device)
                for offset in range(0, len(indices), self.micro):
                    ids = indices[offset:offset + self.micro]
                    valid, value_valid = batch["valid"][ids], batch["value_valid"][ids]
                    new = self._actor_logp(batch["query_response"][ids], batch["response"][ids],
                                           batch["context"])
                    delta = new - batch["old_logp"][ids]
                    ratio = delta.exp()
                    unclipped = -batch["advantages"][ids] * ratio
                    clipped = -batch["advantages"][ids] * ratio.clamp(1-p["cliprange"], 1+p["cliprange"])
                    policy_sum = torch.maximum(unclipped, clipped)[valid].sum()
                    (policy_sum / policy_count).backward()
                    numerators[0] += policy_sum.detach().double()
                    numerators[2] += (.5 * delta.detach().square())[valid].sum().double()
                    numerators[3] += (clipped.detach() > unclipped.detach())[valid].sum().double()
                    del new, delta, ratio, unclipped, clipped, policy_sum

                    prediction = self.critic(**prepare_inputs(batch["query_response"][ids], self.pad))[:, batch["context"]-1:-1]
                    old_value, target = batch["value"][ids], batch["returns"][ids]
                    clipped_prediction = old_value + (prediction-old_value).clamp(-p["cliprange_value"], p["cliprange_value"])
                    square = (prediction-target).square()
                    clipped_square = (clipped_prediction-target).square()
                    value_sum = .5 * torch.maximum(square, clipped_square)[value_valid].sum()
                    (p["vf_coef"] * value_sum / value_count).backward()
                    numerators[1] += value_sum.detach().double()
                    numerators[4] += (clipped_square.detach() > square.detach())[value_valid].sum().double()
                    del prediction, clipped_prediction, square, clipped_square, value_sum
                _sync_gradients(trainable)
                if not all(torch.isfinite(parameter.grad).all() for parameter in trainable):
                    raise RuntimeError("Nonfinite distributed PPO gradient")
                actor_norm = torch.nn.utils.clip_grad_norm_(self.groups["actor"], p["max_grad_norm"], error_if_nonfinite=True)
                critic_norm = torch.nn.utils.clip_grad_norm_(self.groups["critic_lora"] + self.groups["critic_head"],
                                                              p["max_grad_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                dist.all_reduce(numerators, op=dist.ReduceOp.SUM)
                statistics.append({"policy_loss": float(numerators[0]/policy_count),
                                   "value_loss": float(numerators[1]/value_count),
                                   "approxkl": float(numerators[2]/policy_count),
                                   "policy_clipfrac": float(numerators[3]/policy_count),
                                   "value_clipfrac": float(numerators[4]/value_count),
                                   "actor_gradient_norm": float(actor_norm),
                                   "critic_gradient_norm": float(critic_norm)})
        expected = (self.completed_rounds + 1) * p["num_ppo_epochs"] * p["num_mini_batches"]
        actual = optimizer_steps(self.optimizer)
        if any(value["step_min"] != expected or value["step_max"] != expected for value in actual.values()):
            raise RuntimeError("Distributed AdamW step counters disagree with schedule")
        if any(parameter._version != version for parameter, version in self._frozen):
            raise RuntimeError("A frozen distributed Actor/Critic/Reward parameter was mutated")
        self.step_checks.append({"round": self.completed_rounds + 1, "expected": expected,
                                 "actual": actual})
        return {name: float(np.mean([row[name] for row in statistics])) for name in statistics[0]}

    def _global_rollout_metrics(self, batch):
        totals = torch.stack((batch["scores"].double().sum(), batch["kl"].double().sum(),
                              batch["reward"].double().sum(), batch["lengths"].double().sum()))
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        value_stats = _distributed_value_statistics(
            batch["value"][batch["value_valid"]],
            batch["returns"][batch["value_valid"]])
        return {"normalized_score": float(totals[0]/self.batch),
                "sequence_kl_mc": float(totals[1]/self.batch),
                "rlhf_objective": float(totals[2]/self.batch),
                "mean_response_tokens": float(totals[3]/self.batch),
                **{"value_"+key: value for key, value in value_stats.items()}}

    def _run_callback(self, callback):
        failure = None
        if self.rank == 0:
            before = self.capture_rng()
            try:
                callback(self)
            except BaseException:
                failure = traceback.format_exc()
            finally:
                self.restore_rng(before)
        failure = broadcast_object(failure)
        if failure is not None:
            raise RuntimeError("Rank-0 PPO checkpoint callback failed:\n" + failure)

    def run(self, *, stop_after_episodes=None, on_checkpoint=None):
        total = self.settings["total_episodes"]
        limit = total if stop_after_episodes is None else stop_after_episodes
        if not self.completed_episodes <= limit <= total or limit % self.batch:
            raise ValueError("Distributed PPO stop must be a whole global rollout")
        total_rounds, started = total // self.batch, time.monotonic()
        while self.completed_episodes < limit:
            multiplier = learning_rate_multiplier(self.completed_rounds+1, total_rounds,
                                                   self.settings["warmup_ratio"],
                                                   self.settings["minimum_learning_rate_ratio"])
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            batch = self.rollout()
            metrics = self.update(batch)
            self.completed_rounds += 1
            self.completed_episodes += self.batch
            metrics.update(episode=self.completed_episodes, rollout_round=self.completed_rounds,
                           learning_rate_multiplier=multiplier,
                           **self._global_rollout_metrics(batch))
            self.history.append(metrics)
            if self.rank == 0:
                print("v3_distributed_ppo " + json.dumps(metrics, allow_nan=False), flush=True)
            diagnostics_due = self.completed_episodes in set(self.settings.get("checkpoints", [])) | {
                self.settings.get("engineering_checkpoint", -1), limit}
            should_checkpoint = diagnostics_due or self.completed_episodes % self.durable_interval == 0
            del batch
            if should_checkpoint:
                self.save_checkpoint(diagnostics_due=diagnostics_due)
                if on_checkpoint is not None and diagnostics_due:
                    self._run_callback(on_checkpoint)
        return {"protocol": PROTOCOL, "completed": self.completed_episodes == total,
                "completed_episodes": self.completed_episodes,
                "rollout_rounds": self.completed_rounds,
                "actual_optimizer_steps": optimizer_steps(self.optimizer),
                "gradient_norm_clipping_executed": True,
                "distributed_world_size": self.world,
                "durable_checkpoint_interval": self.durable_interval,
                "probability_checks": self.probability_checks, "history": self.history,
                "seconds_this_invocation": time.monotonic()-started}

    def save_checkpoint(self, *, diagnostics_due=False, metadata=None):
        if self.checkpoint_dir is None:
            return None
        rank_rng = [None] * self.world
        dist.all_gather_object(rank_rng, self.capture_rng())
        local_digest = _model_digest(self.actor, self.critic)
        model_digests = [None] * self.world
        dist.all_gather_object(model_digests, local_digest)
        if len(set(model_digests)) != 1:
            raise RuntimeError("Distributed Actor/Critic trainable parameters diverged")
        path = self.checkpoint_dir / f"episode-{self.completed_episodes:08d}.pt"
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"Preserve prior distributed PPO checkpoint: {path}")
            temporary = path.with_suffix(f".tmp-{os.getpid()}")
            value = {"protocol": PROTOCOL, "contract_hash": self.contract_hash,
                     "world_size": self.world, "episode": self.completed_episodes,
                     "round": self.completed_rounds, "diagnostics_due": bool(diagnostics_due),
                     "metadata": metadata or {}, "model_sha256": model_digests[0],
                     "actor": _trainable_state(self.actor), "critic": _trainable_state(self.critic),
                     "optimizer": self.optimizer.state_dict(), "sampler": self.sampler.state_dict(),
                     "rng": rank_rng[0], "rank_rng": rank_rng, "history": self.history,
                     "probability_checks": self.probability_checks, "step_checks": self.step_checks}
            try:
                with temporary.open("xb") as stream:
                    torch.save(value, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            write_json(self.checkpoint_dir/"latest.json",
                       {"path": path.name, "sha256": digest(path),
                        "contract_hash": self.contract_hash, "episode": self.completed_episodes,
                        "world_size": self.world, "diagnostics_due": bool(diagnostics_due)},
                       overwrite=True)
        barrier()
        return path

    def _valid_state(self, state, episode):
        expected = state.get("round", -1) * self.settings["num_ppo_epochs"] * self.settings["num_mini_batches"]
        optimizer = state.get("optimizer", {})
        ids = [identifier for group in optimizer.get("param_groups", []) for identifier in group.get("params", [])]
        counts = [int(optimizer.get("state", {}).get(identifier, {}).get("step", 0)) for identifier in ids]
        optimizer_valid = (episode == 0 and state.get("round") == 0 and bool(ids)
                           and not optimizer.get("state")) or (episode > 0 and bool(counts)
                           and all(count == expected for count in counts))
        return (state.get("protocol") == PROTOCOL
                and state.get("contract_hash") == self.contract_hash
                and state.get("world_size") == self.world
                and len(state.get("rank_rng", [])) == self.world
                and state.get("episode") == episode
                and episode == state.get("round", -1) * self.batch
                and optimizer_valid)

    def _select_checkpoint_rank0(self):
        latest_path = self.checkpoint_dir/"latest.json"
        latest = read_json(latest_path) if latest_path.is_file() else None
        current = -1
        if latest is not None:
            path = (self.checkpoint_dir/latest["path"]).resolve()
            if path.parent != self.checkpoint_dir.resolve() or digest(path) != latest["sha256"]:
                raise ValueError("Distributed checkpoint path/hash mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            current = int(latest["episode"])
            if path.name != f"episode-{current:08d}.pt" or not self._valid_state(state, current):
                raise ValueError("Distributed latest checkpoint identity mismatch")
        for candidate in sorted(self.checkpoint_dir.glob("episode-*.pt")):
            episode = int(candidate.stem.split("-")[-1])
            if episode <= current:
                continue
            state = torch.load(candidate, map_location="cpu", weights_only=True)
            if not self._valid_state(state, episode):
                raise ValueError("Incomplete or foreign distributed PPO orphan checkpoint")
            latest = {"path": candidate.name, "sha256": digest(candidate),
                      "contract_hash": self.contract_hash, "episode": episode,
                      "world_size": self.world,
                      "diagnostics_due": bool(state.get("diagnostics_due", True))}
            write_json(latest_path, latest, overwrite=True)
            current = episode
        if latest is None:
            raise ValueError("No committed or recoverable distributed PPO checkpoint")
        return latest

    def resume(self, *, on_checkpoint=None):
        if self.checkpoint_dir is None:
            raise ValueError("No distributed checkpoint directory")
        latest = self._select_checkpoint_rank0() if self.rank == 0 else None
        latest = broadcast_object(latest)
        path = (self.checkpoint_dir/latest["path"]).resolve()
        if path.parent != self.checkpoint_dir.resolve() or digest(path) != latest["sha256"]:
            raise ValueError("Distributed resume checkpoint hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not self._valid_state(state, int(latest["episode"])):
            raise ValueError("Distributed resume state failed its contract")
        _restore_trainable(self.actor, state["actor"])
        _restore_trainable(self.critic, state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.load_state_dict(state["sampler"])
        self.completed_episodes, self.completed_rounds = state["episode"], state["round"]
        self.history = state["history"]
        self.probability_checks, self.step_checks = state["probability_checks"], state["step_checks"]
        self.restore_rng(state["rank_rng"][self.rank])
        if _model_digest(self.actor, self.critic) != state["model_sha256"]:
            raise ValueError("Restored distributed trainable parameter hash mismatch")
        expected = self.completed_rounds*self.settings["num_ppo_epochs"]*self.settings["num_mini_batches"]
        if any(value["step_min"] != expected or value["step_max"] != expected
               for value in optimizer_steps(self.optimizer).values()):
            raise ValueError("Distributed restored optimizer step mismatch")
        if on_checkpoint is not None and self.completed_episodes and state.get("diagnostics_due", True):
            self._run_callback(on_checkpoint)


def prefit_critic_distributed(critic, reward, trajectories, heldout, *, settings, pad, seed):
    """Global-minibatch Critic prefit with identical weights on every rank."""
    from .ppo import _pack_trajectories
    rank, world = dist.get_rank(), dist.get_world_size()
    count, mini, micro = (settings["critic_warmup_episodes"],
                          settings["critic_warmup_mini_batch_size"],
                          settings["per_device_train_batch_size"])
    if len(trajectories) != count or count % world or len(heldout) % world or mini % world:
        raise ValueError("Distributed Critic prefit needs evenly divisible global data")
    local_start, local_end = rank*(count//world), (rank+1)*(count//world)
    local_rows = trajectories[local_start:local_end]
    device = next(critic.parameters()).device
    roles = {"critic_lora": [value for name, value in critic.named_parameters()
                             if value.requires_grad and "lora_" in name],
             "critic_head": list(critic.score.parameters())}
    optimizer = torch.optim.AdamW(
        [{"params": roles["critic_lora"], "lr": settings["critic_learning_rate"]},
         {"params": roles["critic_head"], "lr": settings["critic_head_learning_rate"]}],
        betas=(settings["adam_beta1"], settings["adam_beta2"]),
        eps=settings["adam_epsilon"], weight_decay=0.)
    with torch.no_grad():
        local_targets = reward.score([row["prompt_tokens"] for row in local_rows],
                                     [row["response_tokens"] for row in local_rows])

    def measure():
        width = len(heldout)//world
        records = heldout[rank*width:(rank+1)*width]
        with torch.no_grad():
            targets = reward.score([row["prompt_tokens"] for row in records],
                                   [row["response_tokens"] for row in records])
        predictions, expected = [], []
        with torch.no_grad():
            for start in range(0, len(records), micro):
                part = records[start:start+micro]
                packed, context, valid = _pack_trajectories(
                    part, pad=pad, device=device, response_length=settings["response_length"])
                values = critic(**prepare_inputs(packed, pad))[:, context-1:-1]
                target = targets[start:start+micro, None].expand_as(values)
                predictions.extend(values[valid].cpu().tolist())
                expected.extend(target[valid].cpu().tolist())
        predictions, expected = _all_gather_lists(predictions, expected)
        return value_statistics(predictions, expected)

    before, steps, norms = measure(), 0, []
    rng = np.random.default_rng(seed+65537)
    critic.eval()
    trainable = roles["critic_lora"] + roles["critic_head"]
    for _ in range(settings["critic_warmup_epochs"]):
        order = rng.permutation(count)
        for start in range(0, count, mini):
            global_ids = order[start:start+mini]
            indices = np.asarray([int(i)-local_start for i in global_ids
                                  if local_start <= int(i) < local_end], dtype=np.int64)
            local_denominator = sum(min(len(local_rows[int(i)]["response_tokens"])+1,
                                        settings["response_length"]) for i in indices)
            denominator = torch.tensor(float(local_denominator), dtype=torch.float64, device=device)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(indices), micro):
                selected = indices[offset:offset+micro]
                records = [local_rows[int(i)] for i in selected]
                packed, context, valid = _pack_trajectories(
                    records, pad=pad, device=device, response_length=settings["response_length"])
                predictions = critic(**prepare_inputs(packed, pad))[:, context-1:-1]
                selected_tensor = torch.as_tensor(selected, dtype=torch.long,
                                                  device=local_targets.device)
                target = local_targets[selected_tensor, None].expand_as(predictions)
                (.5*(predictions-target).square()[valid].sum()/denominator.float()).backward()
            _sync_gradients(trainable)
            norm = torch.nn.utils.clip_grad_norm_(trainable, settings["max_grad_norm"], error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            steps += 1
    after = measure()
    expected_steps = count//mini*settings["critic_warmup_epochs"]
    if steps != expected_steps or any(int(optimizer.state[value]["step"]) != steps for value in trainable):
        raise RuntimeError("Distributed Critic warmup optimizer counts changed")
    report = {"protocol": "v3_distributed_critic_mc_prefit_v1", "episodes": count,
              "epochs": settings["critic_warmup_epochs"], "world_size": world,
              "actual_optimizer_steps": steps, "before": before, "after": after,
              "gradient_norm_max_before_clipping": max(norms),
              "gradient_norm_clipping_executed": True, "heldout_trajectories": len(heldout),
              "passed_performance": bool(after["explained_variance"] is not None
                  and after["explained_variance"] > 0
                  and after["mse"] < after["constant_baseline_mse"])}
    return report
