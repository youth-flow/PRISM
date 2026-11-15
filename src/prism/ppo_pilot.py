"""Bounded PPO learning-rate pilots on existing rewards, never final-test tuning.

Run on one scheduled GPU. Keep the full learning-rate horizon, stop only after
complete rollout updates, preserve RNG around held-out evaluation, and write a
new diagnostic directory. No API client is imported.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time
import types

import numpy as np
import torch

from .hf import canonical_eos, generate, load_lm, prompt_ids, sequence_stats, tokenizer, train_ppo
from .io import digest, dump, load_config, receipt, rows, verify_receipt, write_rows
from .numerics import configure_numerics


GROUPS = {"A": (3e-6, 3e-6), "B": (3e-6, 3e-5), "C": (1e-5, 3e-5)}
SPLIT_SALT = "prism-ppo-development-20261115-v1"


def signature(row):
    return json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def development_split(prompts):
    ordered = sorted(prompts, key=lambda p: hashlib.sha256((SPLIT_SALT+signature(p)).encode()).hexdigest())
    development, confirmation, training = ordered[:64], ordered[64:128], ordered[128:]
    sets = [set(map(signature, part)) for part in (development, confirmation, training)]
    if len(prompts) != 2000 or sum(map(len, sets)) != 2000 or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Expected 2000 distinct policy prompts with disjoint pilot splits")
    return development, confirmation, training


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite pilot metric")
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1)/np.sqrt(len(a)))}


@torch.no_grad()
def evaluate(model, tok, prompts, config, target, weights):
    """Sample at T=1 and compute chosen-token and full-conditional-vocabulary KL.

    Summing exact conditional KL along sampled prefixes is a lower-variance
    estimate of sequence KL; it is still a Monte Carlo estimate over prefixes.
    It includes only generated response tokens, never the feature-only EOS.
    """
    state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(), torch.cuda.get_rng_state())
    training = model.training
    model.eval()
    records, features = [], []
    try:
        for index, row in enumerate(prompts):
            ids = prompt_ids(tok, row["messages"], config["candidate"]["max_prompt_tokens"])
            seed = int(hashlib.sha256((SPLIT_SALT+signature(row)).encode()).hexdigest()[:8], 16)
            response = generate(model, tok, ids, 512, seed, 1.)
            if hasattr(model, "disable_adapter"):
                tokens = torch.tensor([canonical_eos(ids+response, tok.eos_token_id)], device="cuda")
                output = model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False, return_dict=True)
                actor_logits = output.logits[0, len(ids)-1:len(ids)+len(response)-1].float()
                actor_lp = actor_logits.log_softmax(-1)
                del output, actor_logits
                with model.disable_adapter():
                    output = model(tokens, attention_mask=torch.ones_like(tokens), output_hidden_states=True, use_cache=False, return_dict=True)
                phi = output.hidden_states[-1][0, -1].float().cpu().numpy().copy()
                reference_logits = output.logits[0, len(ids)-1:len(ids)+len(response)-1].float()
                reference_lp = reference_logits.log_softmax(-1)
                del output, reference_logits
                selected = torch.tensor(response, device="cuda").unsqueeze(-1)
                log_ratio = (actor_lp.gather(-1, selected)-reference_lp.gather(-1, selected)).sum().item()
                conditional_kl = sum((actor_lp[start:start+32].exp()*(actor_lp[start:start+32]-reference_lp[start:start+32])).sum(dtype=torch.float64).item()
                                     for start in range(0, len(response), 32))
                del actor_lp, reference_lp, selected, tokens
            else:
                phi, _ = sequence_stats(model, ids, response, tok.eos_token_id)
                log_ratio, conditional_kl = 0., 0.
            features.append(phi)
            records.append({"id": row["id"], "seed": seed, "response_tokens": response,
                            "text": tok.decode(response, skip_special_tokens=True), "length": len(response),
                            "has_eos": response[-1] == tok.eos_token_id,
                            "sampled_log_ratio": log_ratio, "conditional_kl_sum": conditional_kl,
                            "scores": {name: float(phi @ weight.astype(np.float32)) for name, weight in weights.items()}})
            if (index+1) % 16 == 0:
                print(f"development {target.name}: {index+1}/{len(prompts)}", flush=True)
        write_rows(target.with_suffix(".jsonl"), records)
        np.savez_compressed(target.with_suffix(".npz"), phi=np.stack(features))
        result = {key: summary([r[key] for r in records]) for key in ("length", "sampled_log_ratio", "conditional_kl_sum")}
        result["scores"] = {name: summary([r["scores"][name] for r in records]) for name in weights}
        result["eos_fraction"] = float(np.mean([r["has_eos"] for r in records]))
        return result
    finally:
        model.train(training)
        random.setstate(state[0]); np.random.set_state(state[1]); torch.random.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3])
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "train"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--method", choices=["empirical", "prism"], default="empirical")
    parser.add_argument("--group", choices=GROUPS, default="A")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=512)
    args = parser.parse_args()
    if args.episodes != 512 or args.seed != 11:
        raise ValueError("First pilot is fixed at seed 11 and 512 episodes")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    config, missing = load_config(args.config, ready=True)
    assert not missing and config["runtime"].get("numerics") == "fp32_consistent_v1"
    assert config["ppo"]["total_episodes"] == 4096
    assert not config["execution"]["allow_paid_api_calls"]
    configure_numerics(config)
    assert torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name(0)
    dependencies = [args.config, Path(__file__)]
    weights = {}
    for method in ("empirical", "prism"):
        folder = args.study/method/"n1024"/f"seed{args.seed}"
        verify_receipt(folder/"reward.receipt.json")
        dependencies.extend([folder/"reward.receipt.json", folder/"reward.npz"])
        with np.load(folder/"reward.npz", allow_pickle=False) as z:
            weights[method] = z["w"].copy()
    development, confirmation, prompts = development_split(rows(config["choices"]["prompts"]["policy"]))
    job = os.environ["SLURM_JOB_ID"]
    out = args.study/"diagnostics"/f"ppo-pilot-{args.mode}-{args.method}-{args.group}-{job}"
    out.mkdir(parents=True, exist_ok=False)
    dump(out/"split.json", {"salt": SPLIT_SALT, "training": [r["id"] for r in prompts],
                           "development": [r["id"] for r in development], "confirmation": [r["id"] for r in confirmation]})
    config = deepcopy(config)
    actor_lr, critic_lr = GROUPS[args.group]
    config["ppo"].update(learning_rate=actor_lr, critic_learning_rate=critic_lr)
    dump(out/"effective-config.json", config)
    started = time.monotonic()
    report = {"job_id": job, "source_commit": args.source_commit, "method": args.method, "group": args.group,
              "seed": args.seed, "api_calls": 0, "passed": False,
              "reward_scope": "Existing v1 reward weights held fixed; FP32 SFT features. Not a new labeled study.",
              "training_prompts": len(prompts), "development_prompts": len(development), "confirmation_prompts": len(confirmation)}
    if args.mode == "baseline":
        model = load_lm(config).requires_grad_(False)
        tok = tokenizer(config)
        report["development"] = evaluate(model, tok, development, config, out/"development", weights)
        report["confirmation"] = evaluate(model, tok, confirmation, config, out/"confirmation", weights)
    else:
        import prism.ppo_numerics as extension
        native = extension.FP32PPOTrainer
        captured = {}

        class StopPilot(Exception):
            pass

        class PilotTrainer(native):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured["trainer"] = self
                self.outer = 0
                self.real_steps = []
                self.checkpoints = {}
                inner = self.optimizer.optimizer
                original_step = inner.step

                def step(optimizer, *a, **kw):
                    metrics = []
                    for group in optimizer.param_groups:
                        gradients = [p.grad for p in group["params"] if p.grad is not None]
                        norm = float(sum(g.float().square().sum() for g in gradients).sqrt())
                        assert np.isfinite(norm) and norm > 0
                        metrics.append({"role": group["prism_role"], "lr": group["lr"], "grad_norm": norm})
                    self.real_steps.append(metrics)
                    return original_step(*a, **kw)

                step._wrapped_by_lr_sched = True
                inner.step = types.MethodType(step, inner)
                original_schedule = self.lr_scheduler.step

                def schedule(*a, **kw):
                    result = original_schedule(*a, **kw)
                    self.outer += 1
                    if self.outer in (4, 8):
                        torch.cuda.empty_cache()
                        self.checkpoints[str(self.outer*64)] = evaluate(self.policy_model, self.processing_class, development, config,
                                                                       out/f"development-{self.outer*64}", weights)
                        dump(out/"checkpoints.json", self.checkpoints)
                    if self.outer == 8:
                        raise StopPilot()
                    return result

                self.lr_scheduler.step = schedule

            def log(self, logs, *a, **kw):
                # Read only small reductions from rollout tensors; never retain a
                # frame or activation, nor alter the loss/gradients/random stream.
                frame = inspect.currentframe().f_back
                try:
                    if frame.f_code.co_name == "train" and "returns" in frame.f_locals:
                        data = frame.f_locals
                        valid = ~data["padding_mask_p1"]
                        targets = data["returns"][valid].detach().float()
                        predictions = data["values"][valid].detach().float()
                        variance = targets.var(unbiased=False).item()
                        logs["pilot/value_target_mse"] = (predictions-targets).square().mean().item()
                        logs["pilot/value_target_variance"] = variance
                        logs["pilot/value_explained_variance"] = 1-(predictions-targets).var(unbiased=False).item()/variance if variance > 1e-12 else None
                        del data, targets, predictions, valid
                finally:
                    del frame
                return super().log(logs, *a, **kw)

            def train(self):
                try:
                    return super().train()
                except StopPilot:
                    assert self.outer == 8 and len(self.real_steps) == 32

        extension.FP32PPOTrainer = PilotTrainer
        result = train_ppo(config, out/"policy", weights[args.method], prompts, args.seed)
        trainer = captured["trainer"]
        assert result["log_history"][-1]["episode"] == 512 and len(result["probability_checks"]) == 8
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters() if p.requires_grad)
        np.savez_compressed(out/"critic.npz", **{n: p.detach().cpu().numpy() for n, p in trainer.model.value_model.score.named_parameters()})
        report.update(result, actual_episodes=512, actual_rollouts=8, effective_optimizer_steps=len(trainer.real_steps),
                      real_steps=trainer.real_steps, checkpoints=trainer.checkpoints,
                      actor_learning_rate=actor_lr, critic_learning_rate=critic_lr,
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    report.update(passed=True, elapsed_seconds=time.monotonic()-started)
    dump(out/"report.json", report)
    members = [p for p in out.rglob("*") if p.is_file()]
    receipt(out/"receipt.json", dependencies+members)
    print(json.dumps({"output": str(out), "passed": True, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
