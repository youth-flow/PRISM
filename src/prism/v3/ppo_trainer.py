"""V3 auditable single-GPU PPO, with no retained whole-rollout vocabulary logits.

This is a NEW executor, not the old TRL trainer with a configuration change.
The clipped policy/value objectives, terminal score placement and GAE follow
TRL 0.15.2 (Apache-2.0); model/optimizer/checkpoint handling is independent.
Gradients are token-weighted across the *whole minibatch*, so microbatch size
does not change weighting for unequal response lengths.  Actor and critic have
separate gradient norm clipping and optimizer parameter groups.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from prism.io import digest
from prism.numerics import probability_path_metrics, require_fp32_model, selected_logprobs
from .artifacts import read_json, write_json
from .ppo_math import learning_rate_multiplier, value_statistics


PROTOCOL = "v3_single_card_streaming_ppo_v1"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_inputs(tokens, pad):
    mask = tokens.ne(pad)
    positions = mask.long().cumsum(-1) - mask.long()
    return {"input_ids": tokens.masked_fill(~mask, 0), "attention_mask": mask,
            "position_ids": positions, "use_cache": False, "return_dict": True}


def left_pad(prompts, pad, device):
    if not prompts or any(not p for p in prompts):
        raise ValueError("PPO needs nonempty prompt token sequences")
    width = max(map(len, prompts))
    result = torch.full((len(prompts), width), pad, dtype=torch.long, device=device)
    for i, prompt in enumerate(prompts):
        result[i, -len(prompt):] = torch.as_tensor(prompt, dtype=torch.long, device=device)
    return result


@torch.no_grad()
def stream_generate(actor, prompts, *, pad, eos, max_new_tokens, temperature=1.):
    """Sample unwarped categorical tokens, keeping only [batch,time] selected logp.

    Completed rows are padded after their sampled EOS.  No forced EOS is an
    action.  Only one forward's vocabulary tensor is live; retained state is
    the KV cache, selected tokens and selected probabilities.  This preserves
    each active prefix's sampling law, while ending once every row finishes.
    """
    if temperature <= 0 or max_new_tokens < 1 or pad == eos:
        raise ValueError("Invalid streaming generation protocol")
    actor.eval()
    batch, context = prompts.shape
    response = torch.full((batch, max_new_tokens), pad, dtype=torch.long, device=prompts.device)
    logp = torch.zeros((batch, max_new_tokens), dtype=torch.float32, device=prompts.device)
    mask = prompts.ne(pad)
    position = mask.long().cumsum(-1) - mask.long()
    safe = prompts.masked_fill(~mask, 0)
    active = torch.ones(batch, dtype=torch.bool, device=prompts.device)
    cache = None
    for step in range(max_new_tokens):
        output = actor(input_ids=safe, attention_mask=mask, position_ids=position,
                       past_key_values=cache, use_cache=True, return_dict=True)
        logits = output.logits[:, -1].float() / temperature
        # Separate FP32 log-softmax is used on both cached and complete paths.
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probabilities, 1).squeeze(1)
        selected = selected_logprobs(logits, sampled)
        response[:, step] = torch.where(active, sampled, pad)
        logp[:, step] = torch.where(active, selected, 0.)
        cache = output.past_key_values
        active = active & sampled.ne(eos)
        del output, logits, probabilities, selected
        if not bool(active.any()) or step + 1 == max_new_tokens:
            break
        safe = sampled.masked_fill(~active, 0).unsqueeze(1)
        # An EOS row no longer needs a cache update.  Other rows retain exactly
        # the positions/attention of their unpadded sampled prefix.
        mask = torch.cat((mask, active[:, None]), dim=1)
        position = (mask.long().sum(-1) - 1).clamp_min(0)[:, None]
    return response, logp


def response_masks(response, pad):
    valid = response.ne(pad)
    lengths = valid.long().sum(-1)
    if bool((lengths == 0).any()):
        raise ValueError("Empty generated completion")
    index = torch.arange(response.shape[1], device=response.device)[None]
    # Include one post-EOS value position (if space remains), as TRL does.
    values_valid = index <= lengths[:, None]
    return valid, values_valid, lengths


def gae(rewards, values, gamma, lam):
    carry = torch.zeros(rewards.shape[0], device=rewards.device)
    advantages = torch.zeros_like(values)
    for t in range(rewards.shape[1] - 1, -1, -1):
        following = values[:, t + 1] if t + 1 < rewards.shape[1] else 0.
        carry = rewards[:, t] + gamma * following - values[:, t] + gamma * lam * carry
        advantages[:, t] = carry
    return advantages, advantages + values


def masked_whiten(value, mask):
    chosen = value[mask]
    if chosen.numel() < 2:
        raise ValueError("Need at least two valid actions to whiten advantages")
    result = (value - chosen.mean()) * torch.rsqrt(chosen.var(unbiased=True) + 1e-8)
    return result.masked_fill(~mask, 0.)


class PromptSampler:
    """Explicit permutation/cursor instead of an implicit, loader-bound RNG."""
    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(size).tolist()
        self.cursor = 0

    def take(self, count):
        if not 1 <= count <= self.size:
            raise ValueError("PPO rollout exceeds prompt pool")
        if self.cursor == self.size:
            self.order = self.rng.permutation(self.size).tolist()
            self.cursor = 0
        if self.cursor + count > self.size:
            raise ValueError("Prompt pool must be divisible by the rollout batch")
        result = self.order[self.cursor:self.cursor + count]
        self.cursor += count
        return result

    def state_dict(self):
        return {"size": self.size, "order": self.order, "cursor": self.cursor,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, value):
        if (value["size"] != self.size or sorted(value["order"]) != list(range(self.size))
                or not 0 <= value["cursor"] <= self.size):
            raise ValueError("Checkpoint prompt sampler is invalid")
        self.order, self.cursor = value["order"], value["cursor"]
        self.rng.bit_generator.state = value["rng"]


def partition_parameters(actor, critic):
    groups = {"actor": [], "critic_lora": [], "critic_head": []}
    for name, parameter in actor.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"Unexpected trainable actor parameter: {name}")
            groups["actor"].append(parameter)
    for name, parameter in critic.named_parameters():
        if parameter.requires_grad:
            role = "critic_head" if name.startswith("score.") else "critic_lora"
            if role == "critic_lora" and "lora_" not in name:
                raise ValueError(f"Unexpected trainable critic parameter: {name}")
            groups[role].append(parameter)
    ids = [id(p) for values in groups.values() for p in values]
    if any(not group for group in groups.values()) or len(ids) != len(set(ids)):
        raise ValueError("Actor/critic groups must be nonempty, disjoint and explicit")
    return groups


def optimizer_steps(optimizer):
    result = {}
    for group in optimizer.param_groups:
        counts = [int(optimizer.state.get(p, {}).get("step", 0)) for p in group["params"]]
        result[group["role"]] = {"step_min": min(counts), "step_max": max(counts),
                                 "tensor_count": len(counts)}
    return result


def _trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def _restore_trainable(model, saved):
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(expected) != set(saved):
        raise ValueError("Checkpoint trainable model schema changed")
    with torch.no_grad():
        for name, parameter in expected.items():
            if parameter.shape != saved[name].shape or not torch.isfinite(saved[name]).all():
                raise ValueError("Invalid model tensor in checkpoint")
            parameter.copy_(saved[name].to(parameter.device))


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, (list, tuple)) else value


class PPOEngine:
    def __init__(self, actor, critic, reward, prompts, *, pad, eos, settings, seed,
                 contract_hash, checkpoint_dir=None):
        self.actor, self.critic, self.reward = actor, critic, reward
        self.prompts, self.pad, self.eos = prompts, pad, eos
        self.settings, self.seed, self.contract_hash = dict(settings), seed, contract_hash
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.device = next(actor.parameters()).device
        if self.device != next(critic.parameters()).device:
            raise ValueError("Actor and critic must use the same single device")
        require_fp32_model(actor)
        require_fp32_model(critic)
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("V3 PPO requires disabled TF32")
        p = self.settings
        self.batch = p["rollout_batch_size"]
        self.mini = p["mini_batch_size"]
        self.micro = p["per_device_train_batch_size"]
        self.durable_interval = p.get("durable_checkpoint_interval", 2048)
        if type(self.durable_interval) is not int or self.durable_interval < self.batch or self.durable_interval % self.batch:
            raise ValueError("Durable checkpoint interval must contain complete rollout batches")
        if (self.batch != self.mini * p["num_mini_batches"] or self.mini != self.micro * p["gradient_accumulation_steps"]
                or p["total_episodes"] % self.batch or len(prompts) % self.batch):
            raise ValueError("PPO rollout/minibatch/microbatch or prompt divisibility mismatch")
        if (p["temperature"] != 1. or p.get("top_p", 1.) != 1. or p.get("top_k", 0) != 0
                or p["gamma"] != 1. or p["whiten_rewards"] or not p["whiten_advantages"]):
            raise ValueError("V3 uses T=1, fixed reference score scaling and advantage whitening")
        if not hasattr(actor, "disable_adapter"):
            raise ValueError("Reference policy requires an independently disabled actor adapter")
        self.groups = partition_parameters(actor, critic)
        rates = {"actor": p["learning_rate"], "critic_lora": p["critic_learning_rate"],
                 "critic_head": p["critic_head_learning_rate"]}
        self.optimizer = torch.optim.AdamW(
            [{"params": values, "lr": rates[role], "initial_lr": rates[role], "role": role}
             for role, values in self.groups.items()],
            betas=(p["adam_beta1"], p["adam_beta2"]), eps=p["adam_epsilon"], weight_decay=p["weight_decay"])
        seed_everything(seed)
        self.sampler = PromptSampler(len(prompts), seed)
        self.shuffle_rng = np.random.default_rng(seed + 104729)
        self.completed_episodes, self.completed_rounds, self.history = 0, 0, []
        self.probability_checks, self.step_checks = [], []
        frozen = [p for m in (actor, critic) for p in m.parameters() if not p.requires_grad]
        reward_parameters = getattr(reward, "frozen_parameters", None)
        if reward_parameters is not None:
            reward_frozen = list(reward_parameters())
            if not reward_frozen or any(p.requires_grad for p in reward_frozen):
                raise ValueError("Reward frozen-parameter contract is empty or trainable")
            frozen.extend(reward_frozen)
        self._frozen = [(p, p._version) for p in frozen]
        self.actor.eval()
        self.critic.eval()

    def _actor_logp(self, query_response, response, context):
        output = self.actor(**prepare_inputs(query_response, self.pad))
        logits = output.logits[:, context - 1:-1].float() / self.settings["temperature"]
        return selected_logprobs(logits, response.masked_fill(response.eq(self.pad), 0))

    @torch.no_grad()
    def rollout(self):
        indices = self.sampler.take(self.batch)
        prompts = [self.prompts[i] for i in indices]
        query = left_pad(prompts, self.pad, self.device)
        context = query.shape[1]
        chunk = self.settings["local_rollout_forward_batch_size"]
        response_parts, logp_parts = [], []
        for start in range(0, len(query), chunk):
            responses, selected = stream_generate(self.actor, query[start:start + chunk], pad=self.pad, eos=self.eos,
                                                 max_new_tokens=self.settings["response_length"], temperature=1.)
            response_parts.append(responses)
            logp_parts.append(selected)
        response, old_logp = torch.cat(response_parts), torch.cat(logp_parts)
        query_response = torch.cat((query, response), dim=1)
        valid, value_valid, lengths = response_masks(response, self.pad)
        references, values = [], []
        for start in range(0, self.batch, self.micro):
            end = start + self.micro
            with self.actor.disable_adapter():
                references.append(self._actor_logp(query_response[start:end], response[start:end], context))
            values.append(self.critic(**prepare_inputs(query_response[start:end], self.pad))[:, context - 1:-1])
        reference, value = torch.cat(references).masked_fill(~valid, 0.), torch.cat(values).masked_fill(~value_valid, 0.)
        actual = [r[:int(n)].tolist() for r, n in zip(response, lengths)]
        scores = self.reward.score(prompts, actual).to(self.device, dtype=torch.float32)
        if scores.shape != (self.batch,) or not torch.isfinite(scores).all():
            raise ValueError("Reward scorer must return one finite normalized terminal score per trajectory")
        old_logp = old_logp.masked_fill(~valid, 0.)
        kl = old_logp - reference
        rewards = -self.settings["kl_coef"] * kl
        endpoint = lengths.clamp_max(response.shape[1] - 1)
        rewards[torch.arange(self.batch, device=self.device), endpoint] += scores
        advantages, returns = gae(rewards, value, self.settings["gamma"], self.settings["lam"])
        advantages = masked_whiten(advantages, valid)
        checks = min(self.settings["probability_check_rows"], self.batch)
        before_rng = torch.random.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None
        fresh = torch.cat([self._actor_logp(query_response[s:min(s + self.micro, checks)],
                                          response[s:min(s + self.micro, checks)], context)
                           for s in range(0, checks, self.micro)])
        check = probability_path_metrics(old_logp[:checks], fresh, valid[:checks],
                                         max_logp_error=self.settings["max_logp_error"],
                                         max_approxkl=self.settings["max_zero_update_approxkl"])
        if not torch.equal(before_rng, torch.random.get_rng_state()) or (before_cuda is not None and not torch.equal(before_cuda, torch.cuda.get_rng_state(self.device))):
            raise RuntimeError("Probability guard consumed RNG; model dropout must be disabled")
        check["round"] = self.completed_rounds + 1
        self.probability_checks.append(check)
        return {"query_response": query_response, "response": response, "context": context,
                "old_logp": old_logp, "value": value, "returns": returns, "advantages": advantages,
                "valid": valid, "value_valid": value_valid, "scores": scores, "kl": kl,
                "reward": rewards, "prompt_indices": indices, "lengths": lengths}

    def update(self, batch):
        p = self.settings
        statistics = []
        for _ in range(p["num_ppo_epochs"]):
            order = self.shuffle_rng.permutation(self.batch)
            for start in range(0, self.batch, self.mini):
                indices = order[start:start + self.mini]
                policy_count = batch["valid"][indices].sum()
                value_count = batch["value_valid"][indices].sum()
                self.optimizer.zero_grad(set_to_none=True)
                accum = {"policy_loss": 0., "value_loss": 0., "approxkl": 0., "policy_clipfrac": 0., "value_clipfrac": 0.}
                for offset in range(0, self.mini, self.micro):
                    ids = indices[offset:offset + self.micro]
                    valid, value_valid = batch["valid"][ids], batch["value_valid"][ids]
                    new = self._actor_logp(batch["query_response"][ids], batch["response"][ids], batch["context"])
                    delta = new - batch["old_logp"][ids]
                    ratio = delta.exp()
                    unclipped = -batch["advantages"][ids] * ratio
                    clipped = -batch["advantages"][ids] * ratio.clamp(1 - p["cliprange"], 1 + p["cliprange"])
                    policy_loss = torch.maximum(unclipped, clipped)[valid].sum() / policy_count
                    policy_loss.backward()
                    accum["policy_loss"] += float(policy_loss.detach())
                    accum["approxkl"] += float((.5 * delta.detach().square())[valid].sum() / policy_count)
                    accum["policy_clipfrac"] += float((clipped.detach() > unclipped.detach())[valid].sum() / policy_count)
                    del policy_loss, new, delta, ratio, unclipped, clipped
                    prediction = self.critic(**prepare_inputs(batch["query_response"][ids], self.pad))[:, batch["context"] - 1:-1]
                    old_value, target = batch["value"][ids], batch["returns"][ids]
                    clipped_prediction = old_value + (prediction - old_value).clamp(-p["cliprange_value"], p["cliprange_value"])
                    square, clipped_square = (prediction - target).square(), (clipped_prediction - target).square()
                    value_loss = .5 * torch.maximum(square, clipped_square)[value_valid].sum() / value_count
                    (p["vf_coef"] * value_loss).backward()
                    accum["value_loss"] += float(value_loss.detach())
                    accum["value_clipfrac"] += float((clipped_square.detach() > square.detach())[value_valid].sum() / value_count)
                    del prediction, clipped_prediction, square, clipped_square, value_loss
                actor_norm = torch.nn.utils.clip_grad_norm_(self.groups["actor"], p["max_grad_norm"], error_if_nonfinite=True)
                critic_norm = torch.nn.utils.clip_grad_norm_(self.groups["critic_lora"] + self.groups["critic_head"], p["max_grad_norm"], error_if_nonfinite=True)
                if not all(torch.isfinite(parameter.grad).all() for values in self.groups.values() for parameter in values if parameter.grad is not None):
                    raise RuntimeError("Nonfinite PPO gradient")
                self.optimizer.step()
                accum.update(actor_gradient_norm=float(actor_norm), critic_gradient_norm=float(critic_norm))
                statistics.append(accum)
        expected = (self.completed_rounds + 1) * p["num_ppo_epochs"] * p["num_mini_batches"]
        actual = optimizer_steps(self.optimizer)
        if any(v["step_min"] != expected or v["step_max"] != expected for v in actual.values()):
            raise RuntimeError("Actual actor/critic AdamW step counters disagree with the rollout schedule")
        if any(parameter._version != version for parameter, version in self._frozen):
            raise RuntimeError("A frozen actor/reference/critic/Reward parameter was mutated")
        self.step_checks.append({"round": self.completed_rounds + 1, "expected": expected, "actual": actual})
        return {name: float(np.mean([row[name] for row in statistics])) for name in statistics[0]}

    def run(self, *, stop_after_episodes=None, on_checkpoint=None):
        total = self.settings["total_episodes"]
        limit = total if stop_after_episodes is None else stop_after_episodes
        if not self.completed_episodes <= limit <= total or limit % self.batch:
            raise ValueError("PPO stop point must be an absolute, whole-rollout episode boundary")
        total_rounds = total // self.batch
        started = time.monotonic()
        while self.completed_episodes < limit:
            multiplier = learning_rate_multiplier(self.completed_rounds + 1, total_rounds,
                                                   self.settings["warmup_ratio"], self.settings["minimum_learning_rate_ratio"])
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            batch = self.rollout()
            metrics = self.update(batch)
            self.completed_rounds += 1
            self.completed_episodes += self.batch
            metrics.update(episode=self.completed_episodes, rollout_round=self.completed_rounds,
                           normalized_score=float(batch["scores"].mean()), sequence_kl_mc=float(batch["kl"].sum(-1).mean()),
                           rlhf_objective=float(batch["reward"].sum(-1).mean()), mean_response_tokens=float(batch["lengths"].float().mean()),
                           learning_rate_multiplier=multiplier,
                           **{"value_" + k: v for k, v in value_statistics(batch["value"][batch["value_valid"]].cpu().numpy(), batch["returns"][batch["value_valid"]].cpu().numpy()).items()})
            self.history.append(metrics)
            print("v3_ppo " + json.dumps(metrics, allow_nan=False), flush=True)
            diagnostics_due = self.completed_episodes in set(self.settings.get("checkpoints", [])) | {self.settings.get("engineering_checkpoint", -1), limit}
            should_checkpoint = diagnostics_due or self.completed_episodes % self.durable_interval == 0
            del batch
            if should_checkpoint:
                self.save_checkpoint(diagnostics_due=diagnostics_due)
                if on_checkpoint is not None and diagnostics_due:
                    before = self.capture_rng()
                    try:
                        on_checkpoint(self)
                    finally:
                        self.restore_rng(before)
        return {"protocol": PROTOCOL, "completed": self.completed_episodes == total,
                "completed_episodes": self.completed_episodes, "rollout_rounds": self.completed_rounds,
                "actual_optimizer_steps": optimizer_steps(self.optimizer), "gradient_norm_clipping_executed": True,
                "durable_checkpoint_interval": self.durable_interval,
                "probability_checks": self.probability_checks, "history": self.history,
                "seconds_this_invocation": time.monotonic() - started}

    def capture_rng(self):
        numpy_state = np.random.get_state()
        return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "shuffle": self.shuffle_rng.bit_generator.state}

    def restore_rng(self, state):
        random.setstate(_tuple_tree(state["python"]))
        numpy_state = state["numpy"]
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        torch.random.set_rng_state(state["torch"].cpu())
        if state["cuda"]:
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
        self.shuffle_rng.bit_generator.state = state["shuffle"]

    def save_checkpoint(self, *, diagnostics_due=False, metadata=None):
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"episode-{self.completed_episodes:08d}.pt"
        if path.exists():
            raise FileExistsError(f"Preserve prior PPO checkpoint: {path}")
        temporary = path.with_suffix(f".tmp-{os.getpid()}")
        value = {"protocol": PROTOCOL, "contract_hash": self.contract_hash,
                 "episode": self.completed_episodes, "round": self.completed_rounds,
                 "diagnostics_due": bool(diagnostics_due),
                 "metadata": metadata or {},
                 "actor": _trainable_state(self.actor), "critic": _trainable_state(self.critic),
                 "optimizer": self.optimizer.state_dict(), "sampler": self.sampler.state_dict(), "rng": self.capture_rng(),
                 "history": self.history, "probability_checks": self.probability_checks, "step_checks": self.step_checks}
        try:
            with temporary.open("xb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        write_json(self.checkpoint_dir / "latest.json", {"path": path.name, "sha256": digest(path),
                   "contract_hash": self.contract_hash, "episode": self.completed_episodes,
                   "diagnostics_due": bool(diagnostics_due)}, overwrite=True)
        return path

    def resume(self, *, on_checkpoint=None):
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory")
        latest_path = self.checkpoint_dir / "latest.json"
        latest = read_json(latest_path) if latest_path.is_file() else None
        current = -1
        if latest is not None:
            current_path = (self.checkpoint_dir / latest["path"]).resolve()
            if (current_path.parent != self.checkpoint_dir.resolve()
                    or digest(current_path) != latest["sha256"]):
                raise ValueError("Checkpoint path/hash mismatch")
            try:
                named_episode = int(current_path.stem.split("-")[-1])
            except ValueError as exc:
                raise ValueError("Invalid latest PPO checkpoint filename") from exc
            current_state = torch.load(current_path, map_location="cpu", weights_only=True)
            if (current_path.name != f"episode-{named_episode:08d}.pt"
                    or latest.get("episode") != named_episode
                    or latest.get("contract_hash") != self.contract_hash
                    or current_state.get("protocol") != PROTOCOL
                    or current_state.get("contract_hash") != self.contract_hash
                    or current_state.get("episode") != named_episode
                    or named_episode != current_state.get("round", -1) * self.batch):
                raise ValueError("Latest PPO metadata/checkpoint identity mismatch")
            current = named_episode
            del current_state
        # A checkpoint file is fsynced and atomically renamed before latest.json.
        # Promote a complete newer orphan after a worker loss in that narrow
        # window; never overwrite or delete a differing checkpoint.
        newer = []
        for candidate in self.checkpoint_dir.glob("episode-*.pt"):
            try:
                episode = int(candidate.stem.split("-")[-1])
            except ValueError as exc:
                raise ValueError("Invalid PPO checkpoint filename") from exc
            if episode > current:
                newer.append((episode, candidate))
        for episode, candidate in sorted(newer):
            state = torch.load(candidate, map_location="cpu", weights_only=True)
            expected = state.get("round", -1) * self.settings["num_ppo_epochs"] * self.settings["num_mini_batches"]
            saved_optimizer = state.get("optimizer", {})
            groups = saved_optimizer.get("param_groups", [])
            parameter_ids = [parameter_id for group in groups for parameter_id in group.get("params", [])]
            counts = [int(saved_optimizer.get("state", {}).get(parameter_id, {}).get("step", 0))
                      for parameter_id in parameter_ids]
            # The critic-prefit bootstrap is saved at episode/round zero before
            # PPO AdamW has created per-parameter state.  It is the sole valid
            # checkpoint with empty optimizer state; every trained checkpoint
            # must cover every optimizer parameter at the exact real step.
            optimizer_valid = (episode == 0 and state.get("round") == 0
                               and bool(parameter_ids) and not saved_optimizer.get("state")) or (
                                   episode > 0 and len(counts) == len(parameter_ids)
                                   and bool(counts) and all(count == expected for count in counts))
            if (state.get("protocol") != PROTOCOL or state.get("contract_hash") != self.contract_hash
                    or state.get("episode") != episode or episode != state.get("round", -1) * self.batch
                    or not optimizer_valid):
                raise ValueError("Unindexed PPO checkpoint is incomplete or belongs to another contract")
            latest = {"path": candidate.name, "sha256": digest(candidate),
                      "contract_hash": self.contract_hash, "episode": episode,
                      "diagnostics_due": bool(state.get("diagnostics_due", True))}
            write_json(latest_path, latest, overwrite=True)
            current = episode
        if latest is None:
            raise ValueError("No committed or recoverable PPO checkpoint")
        path = (self.checkpoint_dir / latest["path"]).resolve()
        if path.parent != self.checkpoint_dir.resolve() or digest(path) != latest["sha256"]:
            raise ValueError("Checkpoint path/hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["protocol"] != PROTOCOL or state["contract_hash"] != self.contract_hash or latest["contract_hash"] != self.contract_hash:
            raise ValueError("Cannot resume a different PPO numerical/data/model contract")
        _restore_trainable(self.actor, state["actor"])
        _restore_trainable(self.critic, state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.load_state_dict(state["sampler"])
        self.completed_episodes, self.completed_rounds = state["episode"], state["round"]
        expected = self.completed_rounds * self.settings["num_ppo_epochs"] * self.settings["num_mini_batches"]
        if self.completed_episodes != self.completed_rounds * self.batch or any(v["step_min"] != expected or v["step_max"] != expected for v in optimizer_steps(self.optimizer).values()):
            raise ValueError("Checkpoint episode/optimizer counts are inconsistent")
        self.history, self.probability_checks, self.step_checks = state["history"], state["probability_checks"], state["step_checks"]
        self.restore_rng(state["rng"])
        # The training checkpoint is committed BEFORE its diagnostic callback.
        # A worker exit in that interval must replay/verify diagnostics at this
        # exact actor state, before any later policy update or acceptance gate.
        if on_checkpoint is not None and self.completed_episodes and state.get("diagnostics_due", True):
            before = self.capture_rng()
            try:
                on_checkpoint(self)
            finally:
                self.restore_rng(before)
