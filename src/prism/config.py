"""Schema-2 experiment contracts, with no model loading or API requests.

The schema selects backend adapters; it never treats an API model as an HF
checkpoint.  Structural validation and cheap preparation checks are available
without credentials.  ``ready=True`` additionally validates actual input data
and the selected official benchmark implementation.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse


_REVISION = re.compile(r"[0-9a-f]{40}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMPARISON_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, minimum=0.0, strict=False, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (value <= minimum if strict else value < minimum) or (maximum is not None and value > maximum):
        interval = f"> {minimum}" if strict else f">= {minimum}"
        raise ValueError(f"{name} must be {interval}" + (f" and <= {maximum}" if maximum is not None else ""))


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _revision(value, name, missing):
    if value is None or value == "":
        missing.append(name)
    elif not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError(f"{name} must be a pinned 40-character lowercase HF/Git revision")


def _resolve(value, name, base, missing, *, directory=False):
    if value is None or value == "":
        missing.append(name)
        return value
    _text(value, name)
    resolved = (base / value).resolve()
    if not (resolved.is_dir() if directory else resolved.is_file()):
        missing.append(name)
    return str(resolved)


def _validate_judge(value, role):
    name = f"judges.{role}"
    judge = _mapping(value, name)
    backend = judge.get("backend")
    if backend not in ("deepseek", "openai", "hf"):
        raise ValueError(f"{name}.backend must be deepseek, openai or hf")
    _text(judge.get("model"), f"{name}.model")
    _integer(judge.get("max_output_tokens"), f"{name}.max_output_tokens")
    _integer(judge.get("max_attempts"), f"{name}.max_attempts")
    if any(k in judge for k in ("api_key", "access_token", "secret")):
        raise ValueError(f"{name} must reference credentials via api_key_env only")
    if backend == "hf":
        revision = judge.get("revision")
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise ValueError(f"{name}.revision must be a pinned 40-character lowercase HF revision")
        _number(judge.get("temperature"), f"{name}.temperature", strict=role == "training")
        return
    expected_api = "chat_completions" if backend == "deepseek" else "responses"
    if judge.get("api") != expected_api:
        raise ValueError(f"{name}.api must be {expected_api} for backend {backend}")
    _text(judge.get("base_url"), f"{name}.base_url")
    url = urlparse(judge["base_url"])
    if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError(f"{name}.base_url must be an HTTPS endpoint without embedded credentials or query parameters")
    env = judge.get("api_key_env")
    if not isinstance(env, str) or not _ENV_NAME.fullmatch(env):
        raise ValueError(f"{name}.api_key_env must name an environment variable, not contain a key")
    _number(judge.get("timeout_seconds"), f"{name}.timeout_seconds", strict=True)
    if backend == "deepseek":
        if judge.get("thinking") not in ("enabled", "disabled"):
            raise ValueError(f"{name}.thinking must be enabled or disabled")
        _number(judge.get("temperature"), f"{name}.temperature", strict=role == "training")
    elif judge.get("reasoning_effort") not in ("none", "low", "medium", "high", "xhigh", "max"):
        raise ValueError(f"{name}.reasoning_effort is not supported by the configured Responses adapter")


def _validate_structure(c):
    # Local import keeps io.load_config -> load_v2 free from module-import cycles.
    from .io import METHODS

    if c.get("schema_version") != 2:
        raise ValueError("Expected schema_version 2")
    execution = _mapping(c.get("execution"), "execution")
    _boolean(execution.get("allow_paid_api_calls"), "execution.allow_paid_api_calls")
    _text(c.get("output"), "output")
    methods = c.get("methods")
    if (not isinstance(methods, list) or not methods or any(not isinstance(m, str) or m not in METHODS for m in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError("methods must be a nonempty subset of the supported method registry without duplicates")
    for group, minimum in (("budgets", 1), ("seeds", 0)):
        values = c.get(group)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{group} must be a nonempty list")
        for val in values:
            _integer(val, group, minimum)
        if len(set(values)) != len(values):
            raise ValueError(f"{group} must not contain duplicates")
    sampling = _mapping(c.get("sampling"), "sampling")
    if sampling.get("budget_coupling") not in ("independent", "nested_prefix"):
        raise ValueError("sampling.budget_coupling must be independent or nested_prefix")
    runtime = _mapping(c.get("runtime"), "runtime")
    if runtime.get("device") not in ("cpu", "cuda") or runtime.get("dtype") not in ("float32", "bfloat16"):
        raise ValueError("runtime supports cpu/cuda and float32/bfloat16")
    _boolean(runtime.get("local_files_only"), "runtime.local_files_only")
    data = _mapping(c.get("data"), "data")
    source = _mapping(data.get("source"), "data.source")
    _text(source.get("id"), "data.source.id")
    expected = _mapping(data.get("expected_counts"), "data.expected_counts")
    for role in ("design", "policy", "evaluation", "benchmark"):
        _integer(expected.get(role), f"data.expected_counts.{role}")
    _boolean(data.get("require_disjoint_design_policy"), "data.require_disjoint_design_policy")
    if "model_context_tokens" in data:
        _integer(data["model_context_tokens"], "data.model_context_tokens")
    if "benchmark_source" in data:
        benchmark_source = _mapping(data["benchmark_source"], "data.benchmark_source")
        _text(benchmark_source.get("id"), "data.benchmark_source.id")
    choices = _mapping(c.get("choices"), "choices")
    _mapping(choices.get("prompts"), "choices.prompts")
    _mapping(choices.get("sft"), "choices.sft")
    benchmark_choice = _mapping(choices.get("benchmark"), "choices.benchmark")
    _text(benchmark_choice.get("name"), "choices.benchmark.name")
    if benchmark_choice.get("kind") not in ("ifeval", "exact_match"):
        raise ValueError("choices.benchmark.kind must be ifeval or exact_match")
    judges = _mapping(c.get("judges"), "judges")
    for role in ("training", "evaluation"):
        _validate_judge(judges.get(role), role)
    for section in ("candidate", "design", "reward", "ppo", "evaluation", "benchmark"):
        _mapping(c.get(section), section)
    for section, keys in {
        "candidate": ("count", "max_new_tokens", "max_prompt_tokens"),
        "design": ("fw_max_iter",),
        "reward": ("max_iter",),
        "ppo": ("total_episodes", "response_length", "per_device_train_batch_size", "gradient_accumulation_steps", "num_mini_batches", "num_ppo_epochs", "lora_r"),
        "evaluation": ("max_new_tokens",),
        "benchmark": ("max_new_tokens",),
    }.items():
        for key in keys:
            _integer(c[section].get(key), f"{section}.{key}")
    if c["candidate"]["count"] < 2:
        raise ValueError("candidate.count must be >= 2")
    for section in ("candidate", "evaluation"):
        _integer(c[section].get("seed"), f"{section}.seed", 0)
    for key in ("surrogate_ridge", "matrix_ridge", "fw_relative_gap"):
        _number(c["design"].get(key), f"design.{key}", strict=True)
    for key in ("gradient_tolerance", "max_weight_norm"):
        _number(c["reward"].get(key), f"reward.{key}", strict=True)
    _number(c["reward"].get("l2_regularization", 0.0), "reward.l2_regularization")
    ppo = c["ppo"]
    for key in ("learning_rate", "kl_coef", "cliprange", "lora_alpha"):
        _number(ppo.get(key), f"ppo.{key}", strict=True)
    _number(ppo.get("vf_coef"), "ppo.vf_coef")
    for key in ("cliprange_value", "max_grad_norm"):
        if key in ppo:
            _number(ppo[key], f"ppo.{key}", strict=True)
    for key in ("gamma", "lam"):
        if key in ppo:
            _number(ppo[key], f"ppo.{key}", maximum=1.0)
    if "local_rollout_forward_batch_size" in ppo:
        _integer(ppo["local_rollout_forward_batch_size"], "ppo.local_rollout_forward_batch_size")
    if not (c["candidate"]["max_new_tokens"] == ppo["response_length"] == c["evaluation"]["max_new_tokens"]):
        raise ValueError("Candidate, PPO and same-source KL evaluation response caps must match")
    # The official benchmark has its own cap; an IFEval instruction may require
    # more than the 512-token preference/PPO cap.
    for section in ("candidate", "evaluation"):
        for key, expected_value in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 0)):
            if key in c[section] and (type(c[section][key]) not in (int, float) or c[section][key] != expected_value):
                raise ValueError(f"{section}.{key} must equal protocol value {expected_value}")
    _number(c["benchmark"].get("temperature"), "benchmark.temperature")
    if c["choices"]["benchmark"]["kind"] == "ifeval" and c["benchmark"]["temperature"] != 0:
        raise ValueError("IFEval uses greedy benchmark.temperature=0")
    if "model_context_tokens" in data:
        context = data["model_context_tokens"]
        for cap, name in ((c["candidate"]["max_new_tokens"], "candidate"), (c["benchmark"]["max_new_tokens"], "benchmark")):
            if c["candidate"]["max_prompt_tokens"] + cap + 1 > context:
                raise ValueError(f"{name} prompt/response cap plus feature EOS exceeds data.model_context_tokens")
    batch = ppo["per_device_train_batch_size"]*ppo["gradient_accumulation_steps"]*ppo["num_mini_batches"]
    if ppo["total_episodes"] % batch:
        raise ValueError("ppo.total_episodes must be divisible by the single-GPU PPO rollout batch")
    evaluation = c["evaluation"]
    orders = evaluation.get("answer_orders")
    if (not isinstance(orders, list) or not orders or any(o not in ("AB", "BA") for o in orders)
            or len(set(orders)) != len(orders)):
        raise ValueError("evaluation.answer_orders must be a nonempty unique list of AB/BA")
    comparisons = evaluation.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("evaluation.comparisons must be a list")
    names = []
    for comp in comparisons:
        _mapping(comp, "evaluation.comparisons entry")
        name = comp.get("name")
        if not isinstance(name, str) or not _COMPARISON_NAME.fullmatch(name):
            raise ValueError("comparison name must use letters, digits, underscores or hyphens")
        names.append(name)
        if comp.get("left") not in methods + ["sft"] or comp.get("right") not in methods + ["sft"] or comp["left"] == comp["right"]:
            raise ValueError("comparison participants must be distinct enabled methods or sft")
    if len(set(names)) != len(names):
        raise ValueError("evaluation comparison names must be unique")


def load_v2(path, c, ready=False):
    """Validate and normalize schema 2, returning ``(config, missing_inputs)``.

    ``missing_inputs`` includes not-yet-staged files/directories and unset pinned
    revisions, even when ``ready=False``.  It intentionally excludes API keys:
    checking preparation never authorizes or performs a paid request.
    """
    path = Path(path).resolve()
    c = deepcopy(c)
    _mapping(c, "config")
    _validate_structure(c)
    missing = []
    c["output"] = str((path.parent / c["output"]).resolve())
    for role in ("design", "policy", "evaluation"):
        choices = c["choices"]["prompts"]
        choices[role] = _resolve(choices.get(role), f"choices.prompts.{role}", path.parent, missing)
    sft = c["choices"]["sft"]
    if sft.get("model") is None or sft.get("model") == "":
        missing.append("choices.sft.model")
    else:
        _text(sft["model"], "choices.sft.model")
        if sft["model"].startswith(".") or Path(sft["model"]).is_absolute():
            sft["model"] = str((path.parent / sft["model"]).resolve())
    _revision(sft.get("revision"), "choices.sft.revision", missing)
    for judge in c["judges"].values():
        if judge.get("backend") == "hf" and (judge["model"].startswith(".") or Path(judge["model"]).is_absolute()):
            judge["model"] = str((path.parent / judge["model"]).resolve())
    _revision(c["data"]["source"].get("revision"), "data.source.revision", missing)
    if "manifest_path" in c["data"]:
        c["data"]["manifest_path"] = _resolve(c["data"]["manifest_path"], "data.manifest_path", path.parent, missing)
    if "benchmark_source" in c["data"]:
        _revision(c["data"]["benchmark_source"].get("revision"), "data.benchmark_source.revision", missing)
    benchmark = c["choices"]["benchmark"]
    benchmark["path"] = _resolve(benchmark.get("path"), "choices.benchmark.path", path.parent, missing)
    if benchmark["kind"] == "ifeval":
        benchmark["official_source_path"] = _resolve(benchmark.get("official_source_path"), "choices.benchmark.official_source_path", path.parent, missing, directory=True)
        _revision(benchmark.get("official_source_revision"), "choices.benchmark.official_source_revision", missing)
        if benchmark["official_source_path"] and Path(benchmark["official_source_path"]).is_dir() and not (Path(benchmark["official_source_path"])/"source-manifest.json").is_file():
            missing.append("choices.benchmark.official_source_manifest")
        if "nltk_data_path" in c["benchmark"]:
            c["benchmark"]["nltk_data_path"] = _resolve(c["benchmark"]["nltk_data_path"], "benchmark.nltk_data_path", path.parent, missing, directory=True)
    elif benchmark.get("official_source_path"):
        benchmark["official_source_path"] = str((path.parent / benchmark["official_source_path"]).resolve())
    if ready:
        if missing:
            raise ValueError("Prepare experiment inputs first: " + ", ".join(missing))
        validate_v2_inputs(c)
    return c, missing


def _signature(record):
    # Be conservative about leakage from superficial whitespace/Unicode changes.
    messages = [{"role": m["role"], "content": " ".join(unicodedata.normalize("NFKC", m["content"]).split())} for m in record["messages"]]
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def _validate_normalized_rows(data, name):
    if not isinstance(data, list) or not data:
        raise ValueError(f"{name} has no prompt records")
    ids = []
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)) or isinstance(row.get("id"), bool):
            raise ValueError(f"{name} requires string/integer prompt IDs")
        ids.append(row["id"])
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{name} requires nonempty messages")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str):
                raise ValueError(f"{name} requires text chat messages")
        if messages[-1]["role"] != "user":
            raise ValueError(f"{name} prompts must end at a user message")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} has duplicate prompt IDs")


def validate_v2_inputs(c):
    """Check prepared data and benchmark dependencies without keys or network."""
    from .io import rows
    from .benchmarks import benchmark_rows, validate_benchmark_backend

    prepared = {}
    for role in ("design", "policy", "evaluation"):
        file = Path(c["choices"]["prompts"][role])
        if not file.is_file():
            raise ValueError(f"Prepared {role} prompt file not found: {file}")
        prepared[role] = rows(file)
        _validate_normalized_rows(prepared[role], role)
    if not Path(c["choices"]["benchmark"]["path"]).is_file():
        raise ValueError("Prepared benchmark file not found")
    validate_benchmark_backend(c)
    prepared["benchmark"] = benchmark_rows(c)
    _validate_normalized_rows(prepared["benchmark"], "benchmark")
    for role, data in prepared.items():
        expected = c["data"]["expected_counts"][role]
        if len(data) != expected:
            raise ValueError(f"{role} count mismatch: expected {expected}, found {len(data)}")
    signatures = {role: {_signature(r) for r in data} for role, data in prepared.items()}
    for role, data in prepared.items():
        if len(signatures[role]) != len(data):
            raise ValueError(f"{role} contains duplicate normalized prompt content")
    if c["data"]["require_disjoint_design_policy"] and signatures["design"] & signatures["policy"]:
        raise ValueError("Design/policy prompt overlap detected by normalized conversation content")
    training = signatures["design"] | signatures["policy"]
    if training & (signatures["evaluation"] | signatures["benchmark"]):
        raise ValueError("Training/evaluation or benchmark prompt overlap detected by normalized conversation content")
    return prepared


def plan_summary(c):
    """Pure arithmetic from a validated config; creates no files or processes."""
    methods, budgets, seeds = len(c["methods"]), c["budgets"], len(c["seeds"])
    counts = c["data"]["expected_counts"]
    cells = methods*len(budgets)*seeds
    coupling = c["sampling"]["budget_coupling"]
    labels = methods*seeds*(max(budgets) if coupling == "nested_prefix" else sum(budgets))
    comparisons = counts["evaluation"]*len(c["evaluation"]["comparisons"])*len(budgets)*seeds
    candidates = c["candidate"]["count"]
    ppo = c["ppo"]
    batch = ppo["per_device_train_batch_size"]*ppo["gradient_accumulation_steps"]*ppo["num_mini_batches"]
    return {
        "training_cells": cells,
        "training_valid_labels": labels,
        "evaluation_pair_comparisons": comparisons,
        "evaluation_judge_calls": comparisons*len(c["evaluation"]["answer_orders"]),
        "benchmark_policies_including_sft": cells + 1,
        "benchmark_generations": counts["benchmark"]*(cells + 1),
        "candidate_responses": counts["design"]*candidates,
        "candidate_edges": counts["design"]*candidates*(candidates-1)//2,
        "policy_evaluation_responses": counts["evaluation"]*cells,
        "sft_evaluation_responses": counts["evaluation"],
        "ppo_rollout_batch_size": batch,
        "ppo_rollout_updates_per_cell": ppo["total_episodes"]//batch,
        "ppo_total_episodes": cells*ppo["total_episodes"],
        "budget_coupling": coupling,
    }
