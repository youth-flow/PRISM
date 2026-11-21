"""Strict V6 contract: V5 science with only budget-aware Design/sampling."""
from pathlib import Path
from copy import deepcopy
import math

import yaml

from prism.v3.config import load_config as load_v3_config
from prism.v3.lora_ppo import PPO_RECIPE


MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
MODEL_REVISION = "0d85a3d037876ce6ac7d4311d994400fc66ac27f"
DATA_ID = "Anthropic/hh-rlhf"
DATA_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
FORMAL_EPISODES = 16384
SCHEDULE_EPISODES = 65536
DESIGN_LAMBDA = 0.001
METHODS = ["empirical", "prism"]
EVALUATIONS = ["reward_development", "luna_ab_ba", "ifeval", "sequence_kl"]
REWARD_L2 = 0.1
UNCHANGED_SECTIONS = ("execution", "model", "runtime", "study", "data", "features",
                      "reward", "ppo", "evaluation")


def _validate_only_declared_changes(c):
    """Bind V6 to the checked-in V5 contract outside the named changes."""
    raw = c.get("_contract", c)
    parent_path = Path(__file__).resolve().parents[3] / "configs/experiment.v5.yaml"
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    mismatches = [section for section in UNCHANGED_SECTIONS
                  if raw.get(section) != parent.get(section)]
    old_design, new_design = deepcopy(parent["design"]), deepcopy(raw.get("design", {}))
    for key in ("solver", "extra_constraints"):
        old_design.pop(key, None)
        new_design.pop(key, None)
    if old_design != new_design:
        mismatches.append("design(except solver/extra_constraints)")
    old_sampling, new_sampling = (deepcopy(value) for value in
                                  (parent["sampling"], raw.get("sampling", {})))
    old_sampling.pop("protocol", None)
    new_sampling.pop("protocol", None)
    if old_sampling != new_sampling:
        mismatches.append("sampling(except protocol)")
    if (raw.get("schema_version") != parent.get("schema_version")
            or raw.get("protocol") != parent.get("protocol")):
        mismatches.append("schema/protocol")
    if mismatches:
        raise ValueError(f"V6 may only change Design constraints/solver and PRISM sampling: {mismatches}")


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
        raise ValueError(f"V6 frozen contract mismatch: {mismatches}")


def validate_v6(c):
    _validate_only_declared_changes(c)
    contract = c.get("v6")
    if not isinstance(contract, dict):
        raise ValueError("Missing V6 budget-aware experiment contract")
    budget = contract.get("preference_budget")
    if type(budget) is not int or not 1 <= budget <= c["data"]["expected_train_rows"]:
        raise ValueError("V6 preference budget must fit the declared finite pool")
    _require(contract, {
        "contract_version": "olmo1b_hh_linear_v6",
        "parent_contract": "olmo1b_hh_linear_v5",
        "primary_methods": METHODS,
        "design_lambda": DESIGN_LAMBDA,
        "reward_l2": REWARD_L2,
        "formal_ppo_episodes": FORMAL_EPISODES,
        "scheduler_total_episodes": SCHEDULE_EPISODES,
        "evaluation_suites": EVALUATIONS,
    }, "v6")
    parent_registry = contract.get("parent_result_registry")
    if not isinstance(parent_registry, str) or not parent_registry.strip():
        raise ValueError("V6 requires the read-only V5 result-registry reference")
    _require(c["model"], {"id": MODEL_ID, "revision": MODEL_REVISION}, "model")
    _require(c["data"], {
        "source_id": DATA_ID,
        "revision": DATA_REVISION,
        "feature_dimension": 2048,
        "reward_dev_pairs": 2048,
        "evaluation_prompts": 500,
    }, "data")
    _require(c["study"], {"methods": METHODS}, "study")
    _require(c["design"], {
        "solver": "capped-away-fw",
        "probability_floor": 0.0,
        "extra_constraints": "sum_q=1_and_0<=q_i<=1/N",
    }, "design")
    _require(c["sampling"], {
        "protocol": "v6_budget_capped_exact_without_replacement_v1",
        "replacement": False,
        "empirical": "uniform_valid_pair",
    }, "sampling")
    _require(c["reward"], {
        "backbone_trainable": False,
        "feature_dimension": 2048,
        "bias": False,
        "selection": "shared_development_mean_bt_nll",
        "normalization": "fixed_sft_mean_std",
    }, "reward")
    _require(c["ppo"], PPO_RECIPE, "ppo")
    _require(c["evaluation"], {
        "model": "gpt-6-luna",
        "reasoning_effort": "medium",
        "answer_orders": ["AB", "BA"],
        "tie_score": 0.5,
    }, "evaluation")
    if c["execution"]["allow_paid_api_calls"] is not False:
        raise ValueError("V6 configuration may not silently authorize paid evaluation")
    return c


def load_config(path):
    c = load_v3_config(path)
    validate_v6(c)
    raw = c["v6"]["parent_result_registry"]
    c["_v6_parent_registry_path"] = str(
        (Path(c["_config_path"]).parent / raw).resolve())
    return c


def plan(c):
    validate_v6(c)
    p, budget = c["ppo"], c["v6"]["preference_budget"]
    rounds = FORMAL_EPISODES // p["rollout_batch_size"]
    steps = rounds * p["num_mini_batches"] * p["num_ppo_epochs"]
    return {
        "protocol": c["v6"]["contract_version"],
        "parent_protocol": c["v6"]["parent_contract"],
        "config_sha256": c["_config_sha256"],
        "sft": {"id": c["model"]["id"], "revision": c["model"]["revision"]},
        "data": {"id": c["data"]["source_id"], "revision": c["data"]["revision"]},
        "design": {
            "preference_budget": budget,
            "lambda": DESIGN_LAMBDA,
            "objective": "unchanged_trace_inverse_information",
            "solver": "capped_simplex_away_step_frank_wolfe",
            "constraints": ["sum(q)=1", f"0<=q_i<=1/{budget}"],
            "acceptance": "independent_budget_capped_global_fw_gap",
        },
        "sampling": {
            "empirical": "uniform_without_replacement",
            "prism": "fixed_size_dependent_rounding_of_pi=Nq",
            "distinct_pairs_per_method": budget,
        },
        "reward": {
            "type": "frozen_sft_eos2048_bias_free_linear_bt",
            "l2": REWARD_L2,
        },
        "ppo": {
            "methods": METHODS,
            "formal_episodes_per_method": FORMAL_EPISODES,
            "scheduler_total_episodes": SCHEDULE_EPISODES,
            "rollout_rounds_per_method": rounds,
            "actor_steps_per_method": steps,
            "critic_steps_per_method": steps,
            "distributed_world_size": 4,
        },
        "evaluation_suites": EVALUATIONS,
        "parent_result_registry": c["_v6_parent_registry_path"],
        "compute_started": False,
        "paid_api_calls_enabled": False,
    }
