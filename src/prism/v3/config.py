"""Schema-3 validation and portable immutable experiment configuration."""
from copy import deepcopy
import math
from pathlib import Path
import re
from urllib.parse import urlparse

import yaml

from .artifacts import config_hash


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, minimum=0, maximum=None, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= minimum if positive else value < minimum)
            or maximum is not None and value > maximum):
        raise ValueError(f"Invalid finite numeric setting: {name}")


def validate(c):
    if c.get("schema_version") != 3 or c.get("protocol") != "hh_finite_pool_v3":
        raise ValueError("Expected the HH finite-pool V3 protocol")
    for name in ("execution", "model", "runtime", "study", "data", "features", "design", "sampling", "reward", "ppo", "evaluation"):
        if not isinstance(c.get(name), dict):
            raise ValueError(f"Missing V3 configuration section: {name}")
    if not isinstance(c.get("output"), str) or not c["output"].strip():
        raise ValueError("V3 output must be a nonempty path")
    execution, model, runtime, study, data = (c[k] for k in ("execution", "model", "runtime", "study", "data"))
    if execution.get("allow_training_judge") is not False:
        raise ValueError("HH human preferences do not use a training Judge")
    if type(execution.get("allow_paid_api_calls")) is not bool:
        raise ValueError("allow_paid_api_calls must be an explicit boolean")
    if not isinstance(model.get("id"), str) or not model["id"]:
        raise ValueError("A fixed SFT model is required")
    if not isinstance(model.get("revision"), str) or not re.fullmatch("[0-9a-f]{40}", model["revision"]):
        raise ValueError("Pin the SFT model revision")
    if runtime.get("device") not in ("cpu", "cuda") or runtime.get("dtype") != "float32" or runtime.get("numerics") != "fp32_consistent_v1":
        raise ValueError("V3 requires the consistent FP32 numerical protocol")
    if runtime.get("allow_tf32") is not False or type(runtime.get("local_files_only")) is not bool:
        raise ValueError("V3 requires explicit local-files and disabled TF32 settings")
    if study.get("methods") != ["empirical", "prism"]:
        raise ValueError("V3 has exactly Empirical and PRISM in that order")
    _integer(study.get("seed"), "study.seed", 0)
    if study.get("budget") is not None or study.get("selected_design_lambda") is not None:
        raise ValueError("Keep configuration budget unset; freeze the user decision in study/selection.json")
    if study.get("budget_authority") != "user_after_design_report":
        raise ValueError("Only the user chooses the one preference budget after Design")
    if data.get("source_id") != "Anthropic/hh-rlhf" or not re.fullmatch("[0-9a-f]{40}", str(data.get("revision"))):
        raise ValueError("Pin the HH source revision")
    expected_subsets = {"helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"}
    if not isinstance(data.get("subsets"), list) or len(data["subsets"]) != 4 or set(data["subsets"]) != expected_subsets:
        raise ValueError("Use the four HH preference subsets, excluding red-team transcripts")
    for key in ("expected_train_rows", "expected_test_rows", "policy_prompts", "reward_dev_pairs", "ppo_dev_prompts", "evaluation_prompts", "context_tokens", "max_prompt_tokens", "feature_dimension"):
        _integer(data.get(key), "data." + key)
    _integer(data.get("split_seed"), "data.split_seed", 0)
    if data.get("disjoint_design_policy") is not False or data.get("require_disjoint_train_test") is not True or data.get("truncate_human_answers") is not False:
        raise ValueError("Preserve complete human answers and train/test context isolation")
    features, design, reward, ppo, evaluation = (c[k] for k in ("features", "design", "reward", "ppo", "evaluation"))
    if features.get("dtype") != "float32" or features.get("matrix_dtype") != "float64" or features.get("label_blind") is not True:
        raise ValueError("V3 needs label-blind FP32 features and FP64 geometry")
    if features.get("eos_logp") != "included_in_canonical_response":
        raise ValueError("Canonical human response EOS must use the fixed scoring convention")
    _integer(features.get("shards"), "features.shards")
    lambdas = design.get("lambdas")
    if not isinstance(lambdas, list) or len(lambdas) != 3 or len(set(lambdas)) != 3:
        raise ValueError("Exactly three distinct Design ridge candidates are required")
    for value in lambdas:
        _number(value, "design.lambdas", positive=True)
    if design.get("preferred_lambda") not in lambdas:
        raise ValueError("Preferred Design ridge must be one of the diagnosed candidates")
    for key in ("surrogate_ridge", "relative_gap"):
        _number(design.get(key), "design." + key, positive=True)
    for key in ("max_iter", "block_size"):
        _integer(design.get(key), "design." + key)
    v6 = isinstance(c.get("v6"), dict)
    if v6:
        if (design.get("solver") != "capped-away-fw"
                or design.get("probability_floor") != 0.0
                or design.get("extra_constraints") != "sum_q=1_and_0<=q_i<=1/N"):
            raise ValueError("V6 requires the declared budget-capped Design solver and no probability floor")
    elif (design.get("solver") != "joint-lbfgs" or design.get("probability_floor") != 0.0
          or design.get("extra_constraints") != "none"):
        raise ValueError("Keep the original simplex target and solver; do not add probability mass or caps")
    sampling = c["sampling"]
    expected_sampling = ("v6_budget_capped_exact_without_replacement_v1" if v6 else
                         "weighted_without_replacement_gumbel_v1")
    if (sampling.get("protocol") != expected_sampling or sampling.get("replacement") is not False
            or sampling.get("empirical") != "uniform_valid_pair"):
        raise ValueError("Use the protocol-specific fixed finite-pool sampling law")
    if reward.get("backbone_trainable") is not False or reward.get("bias") is not False or reward.get("feature_dimension") != data["feature_dimension"]:
        raise ValueError("Reward is a frozen feature map and a bias-free linear head")
    candidates = reward.get("l2_candidates")
    if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Reward needs distinct positive L2 candidates")
    for value in candidates:
        _number(value, "reward.l2_candidates", positive=True)
    if reward.get("selection") != "shared_development_mean_bt_nll" or reward.get("normalization") != "fixed_sft_mean_std":
        raise ValueError("Reward selection and normalization must use the common development/reference sets")
    for key in ("max_iter", "calibration_responses"):
        _integer(reward.get(key), "reward." + key)
    for key in ("gradient_tolerance", "max_weight_norm"):
        _number(reward.get(key), "reward." + key, positive=True)
    for key in ("total_episodes", "rollout_batch_size", "mini_batch_size", "per_device_train_batch_size", "gradient_accumulation_steps", "num_mini_batches", "num_ppo_epochs", "lora_r", "critic_lora_r", "critic_warmup_episodes", "critic_warmup_epochs", "critic_warmup_mini_batch_size", "response_length", "local_rollout_forward_batch_size", "probability_check_rows", "engineering_checkpoint", "fixed_train_probe_prompts"):
        _integer(ppo.get(key), "ppo." + key)
    computed_batch = ppo["per_device_train_batch_size"] * ppo["gradient_accumulation_steps"] * ppo["num_mini_batches"]
    if computed_batch != ppo["rollout_batch_size"] or ppo["mini_batch_size"] * ppo["num_mini_batches"] != computed_batch:
        raise ValueError("PPO microbatch/accumulation/minibatches must match the statistical batch")
    if ppo["total_episodes"] % computed_batch or data["policy_prompts"] % computed_batch:
        raise ValueError("PPO episode and policy prompt counts must contain complete rollout batches")
    if ppo["critic_warmup_episodes"] % ppo["critic_warmup_mini_batch_size"]:
        raise ValueError("Critic prefit requires complete minibatches")
    if reward["calibration_responses"] <= ppo["critic_warmup_episodes"]:
        raise ValueError("Reserve shared SFT trajectories outside critic prefit for held-out checks")
    for key in ("learning_rate", "critic_learning_rate", "critic_head_learning_rate", "lora_alpha", "critic_lora_alpha", "adam_epsilon", "kl_coef", "cliprange", "cliprange_value", "vf_coef", "max_grad_norm", "max_logp_error", "max_zero_update_approxkl"):
        _number(ppo.get(key), "ppo." + key, positive=True)
    for key in ("gamma", "lam", "warmup_ratio", "minimum_learning_rate_ratio", "adam_beta1", "adam_beta2", "lora_dropout"):
        _number(ppo.get(key), "ppo." + key, maximum=1)
    if ppo["adam_beta1"] == 1 or ppo["adam_beta2"] == 1:
        raise ValueError("Adam beta must be below one")
    _number(ppo.get("weight_decay"), "ppo.weight_decay")
    if ppo.get("scheduler") != "warmup_cosine_floor" or ppo.get("independent_gradient_clipping") is not True or ppo.get("whiten_rewards") is not False or ppo.get("whiten_advantages") is not True:
        raise ValueError("V3 optimizer/numerical protocol differs from the agreed recipe")
    if ppo.get("temperature") != 1.0 or ppo.get("top_p") != 1.0 or ppo.get("top_k") != 0:
        raise ValueError("Use the same untruncated T=1 policy law for generation and scoring")
    if data["max_prompt_tokens"] + ppo["response_length"] + 1 > data["context_tokens"]:
        raise ValueError("Prompt plus response exceeds the experiment context budget")
    checkpoints = ppo.get("checkpoints")
    if not isinstance(checkpoints, list) or checkpoints != sorted(set(checkpoints)) or not checkpoints or checkpoints[-1] != ppo["total_episodes"]:
        raise ValueError("PPO checkpoints must be ordered and include the common final episode")
    for value in checkpoints + [ppo["engineering_checkpoint"]]:
        _integer(value, "ppo.checkpoint")
        if value % computed_batch or value > ppo["total_episodes"]:
            raise ValueError("PPO checkpoints must land on completed rollout boundaries")
    durable = ppo.get("durable_checkpoint_interval", 2048)
    _integer(durable, "ppo.durable_checkpoint_interval")
    if durable % computed_batch:
        raise ValueError("Durable checkpoints must land on completed rollout boundaries")
    if evaluation.get("backend") != "openai" or evaluation.get("api") != "responses" or evaluation.get("model") != "gpt-6-luna" or evaluation.get("reasoning_effort") != "medium":
        raise ValueError("V3 main Judge is Luna medium through Responses")
    url = urlparse(str(evaluation.get("base_url", "")))
    if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use an HTTPS Judge endpoint without embedded credentials")
    if not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", str(evaluation.get("api_key_env"))):
        raise ValueError("Judge credentials must be referenced by environment-variable name")
    if any(k in evaluation for k in ("api_key", "secret", "access_token")):
        raise ValueError("Do not store credentials in configuration")
    if evaluation.get("service_tier") not in ("default", "flex"):
        raise ValueError("Unknown explicit Judge execution tier")
    for key in ("max_output_tokens", "max_attempts", "max_new_tokens"):
        _integer(evaluation.get(key), "evaluation." + key)
    _integer(evaluation.get("bootstrap_samples"), "evaluation.bootstrap_samples", 0)
    _integer(evaluation.get("seed"), "evaluation.seed", 0)
    _number(evaluation.get("timeout_seconds"), "evaluation.timeout_seconds", positive=True)
    if evaluation.get("max_new_tokens") != ppo["response_length"] or evaluation.get("temperature") != 1.0 or evaluation.get("answer_orders") != ["AB", "BA"] or evaluation.get("tie_score") != 0.5:
        raise ValueError("Use matched generation and AB/BA/TIE=0.5 evaluation")
    expected_comparisons = [("prism_vs_empirical", "prism", "empirical", True),
                            ("prism_vs_sft", "prism", "sft", False),
                            ("empirical_vs_sft", "empirical", "sft", False)]
    actual = [(v.get("name"), v.get("left"), v.get("right"), v.get("primary")) for v in evaluation.get("comparisons", []) if isinstance(v, dict)]
    if actual != expected_comparisons:
        raise ValueError("V3 has the three agreed comparisons with PRISM versus Empirical primary")
    if evaluation.get("budget_usd") is not None:
        _number(evaluation["budget_usd"], "evaluation.budget_usd", positive=True)
    if execution["allow_paid_api_calls"] and evaluation.get("budget_usd") is None:
        raise ValueError("Paid evaluation needs an explicit independent V3 USD ceiling")
    return c


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V3 configuration must be a mapping")
    validate(raw)
    c = deepcopy(raw)
    c["output"] = str((path.parent / raw["output"]).resolve())
    c["data"]["raw_dir"] = str((path.parent / raw["data"]["raw_dir"]).resolve())
    if raw["model"]["id"].startswith(".") or Path(raw["model"]["id"]).is_absolute():
        c["model"]["id"] = str((path.parent / raw["model"]["id"]).resolve())
    c.update(_contract=raw, _config_path=str(path))
    c["_config_sha256"] = config_hash(c)
    return c


def as_hf_config(c):
    """Adapt the unchanged frozen-tokenizer/model helpers, never the V2 runner."""
    return {"choices": {"sft": {"model": c["model"]["id"], "revision": c["model"]["revision"]}},
            "runtime": deepcopy(c["runtime"]), "candidate": {"max_prompt_tokens": c["data"]["max_prompt_tokens"]},
            "ppo": deepcopy(c["ppo"]), "evaluation": deepcopy(c["evaluation"])}


def plan(c):
    p = c["ppo"]
    return {"protocol": c["protocol"], "config_sha256": config_hash(c),
            "preference_budget": None, "budget_authority": c["study"]["budget_authority"],
            "methods": c["study"]["methods"], "seed": c["study"]["seed"],
            "design_lambdas": c["design"]["lambdas"], "formal_policy_units": 2,
            "episodes_per_policy": p["total_episodes"], "rollout_batch_size": p["rollout_batch_size"],
            "rollout_rounds": p["total_episodes"] // p["rollout_batch_size"],
            "actor_steps_per_policy": p["total_episodes"] // p["rollout_batch_size"] * p["num_mini_batches"] * p["num_ppo_epochs"],
            "final_valid_judgments": c["data"]["evaluation_prompts"] * len(c["evaluation"]["comparisons"]) * len(c["evaluation"]["answer_orders"]),
            "paid_api_calls_enabled": c["execution"]["allow_paid_api_calls"],
            "evaluation_usd_ceiling": c["evaluation"]["budget_usd"],
            "note": "Plan only; no dataset download, compute job, model, or API is started."}
