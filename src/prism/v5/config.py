"""Validation for the V5 OLMo-1B + HH + frozen-linear-Reward protocol.

V5 deliberately reuses the already audited V3 execution engine.  This module
adds a narrower immutable contract: the formal PPO horizon is 16,384 episodes,
while the learning-rate schedule retains the validated 65,536-episode clock.
"""
from pathlib import Path
import math

from prism.v3.config import load_config as load_v3_config
from prism.v3.lora_ppo import PPO_RECIPE


MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
MODEL_REVISION = "0d85a3d037876ce6ac7d4311d994400fc66ac27f"
DATA_ID = "Anthropic/hh-rlhf"
DATA_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
FORMAL_EPISODES = 16384
SCHEDULE_EPISODES = 65536
PREFERENCE_BUDGET = 14000
DESIGN_LAMBDA = 0.001
REWARD_L2 = 0.1
EIGHT_K_BUDGET = 8000
EIGHT_K_PROFILE = "budget_n8000"
DEVELOPMENT_SELECTED_L2 = "shared_development_mean_bt_nll"
METHODS = ["empirical", "prism"]
EVALUATIONS = ["reward_development", "luna_ab_ba", "ifeval", "sequence_kl"]


def _same(actual, expected):
    if isinstance(expected, float):
        return type(actual) in (int, float) and math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-15)
    return actual == expected


def _require(mapping, expected, prefix):
    mismatches = {
        f"{prefix}.{key}": (mapping.get(key), value)
        for key, value in expected.items()
        if not _same(mapping.get(key), value)
    }
    if mismatches:
        raise ValueError(f"V5 frozen contract mismatch: {mismatches}")


def validate_v5(c):
    contract = c.get("v5")
    if not isinstance(contract, dict):
        raise ValueError("Missing V5 frozen experiment contract")
    profile = contract.get("study_variant", "primary_n14000")
    if profile == "primary_n14000":
        budget, reward_l2 = PREFERENCE_BUDGET, REWARD_L2
    elif profile == EIGHT_K_PROFILE:
        budget, reward_l2 = EIGHT_K_BUDGET, DEVELOPMENT_SELECTED_L2
    else:
        raise ValueError(f"Unknown V5 study variant: {profile}")
    _require(contract, {
        "contract_version": "olmo1b_hh_linear_v5",
        "primary_methods": METHODS,
        "preference_budget": budget,
        "design_lambda": DESIGN_LAMBDA,
        "reward_l2": reward_l2,
        "formal_ppo_episodes": FORMAL_EPISODES,
        "scheduler_total_episodes": SCHEDULE_EPISODES,
        "evaluation_suites": EVALUATIONS,
        "historical_controls_in_registry": True,
    }, "v5")
    registry = contract.get("result_registry")
    if not isinstance(registry, str) or not registry.strip():
        raise ValueError("V5 requires a result registry path")

    _require(c["model"], {"id": MODEL_ID, "revision": MODEL_REVISION}, "model")
    _require(c["data"], {
        "source_id": DATA_ID,
        "revision": DATA_REVISION,
        "feature_dimension": 2048,
        "reward_dev_pairs": 2048,
        "evaluation_prompts": 500,
    }, "data")
    _require(c["study"], {"methods": METHODS}, "study")
    _require(c["reward"], {
        "backbone_trainable": False,
        "feature_dimension": 2048,
        "bias": False,
        "normalization": "fixed_sft_mean_std",
    }, "reward")
    # This is the exact recipe consumed by the audited linear PPO runner, not
    # merely a hand-picked subset of headline hyperparameters.
    _require(c["ppo"], PPO_RECIPE, "ppo")
    _require(c["evaluation"], {
        "model": "gpt-6-luna",
        "reasoning_effort": "medium",
        "answer_orders": ["AB", "BA"],
        "tie_score": 0.5,
    }, "evaluation")
    if c["execution"]["allow_paid_api_calls"] is not False:
        raise ValueError("V5 configuration may not silently authorize paid evaluation")
    return c


def load_config(path):
    c = load_v3_config(path)
    validate_v5(c)
    raw_registry = c["v5"]["result_registry"]
    c["_v5_registry_path"] = str(
        (Path(c["_config_path"]).parent / raw_registry).resolve())
    return c


def plan(c):
    validate_v5(c)
    p = c["ppo"]
    rounds = FORMAL_EPISODES // p["rollout_batch_size"]
    optimizer_steps = rounds * p["num_mini_batches"] * p["num_ppo_epochs"]
    return {
        "protocol": c["v5"]["contract_version"],
        "config_sha256": c["_config_sha256"],
        "sft": {"id": c["model"]["id"], "revision": c["model"]["revision"]},
        "data": {"id": c["data"]["source_id"], "revision": c["data"]["revision"]},
        "reward": {
            "type": "frozen_sft_eos2048_bias_free_linear_bt",
            "preference_budget": c["v5"]["preference_budget"],
            "design_lambda": DESIGN_LAMBDA,
            "l2": c["v5"]["reward_l2"],
        },
        "ppo": {
            "methods": METHODS,
            "formal_episodes_per_method": FORMAL_EPISODES,
            "scheduler_total_episodes": SCHEDULE_EPISODES,
            "rollout_rounds_per_method": rounds,
            "actor_steps_per_method": optimizer_steps,
            "critic_steps_per_method": optimizer_steps,
            "distributed_world_size": 4,
        },
        "evaluation_suites": EVALUATIONS,
        "result_registry": c["_v5_registry_path"],
        "compute_started": False,
        "paid_api_calls_enabled": False,
    }
