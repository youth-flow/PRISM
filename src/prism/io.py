"""Small artifact contracts. No untrusted pickle or implicit artifact overwrite."""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

METHODS = ("empirical", "a_optimal", "prism")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(path, ready=False):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    if c.get("schema_version") == 2:
        from .config import load_v2
        return load_v2(path, c, ready=ready)
    if c["schema_version"] != 1:
        raise ValueError("Expected schema 1")
    methods = c["methods"]
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("methods must be a nonempty subset of empirical, a_optimal, prism without duplicates")
    if c.get("sampling", {}).get("budget_coupling", "independent") not in ("independent", "nested_prefix"):
        raise ValueError("sampling.budget_coupling must be independent or nested_prefix")
    l2 = c["reward"].get("l2_regularization", 0.0)
    if not isinstance(l2, (int, float)) or not np.isfinite(l2) or l2 < 0:
        raise ValueError("reward.l2_regularization must be finite and nonnegative")
    for group in ("budgets", "seeds"):
        if not c[group] or len(set(c[group])) != len(c[group]) or any(type(v) is not int or v < (1 if group == "budgets" else 0) for v in c[group]):
            raise ValueError(f"Invalid {group}")
    for key in ("surrogate_ridge", "matrix_ridge", "fw_relative_gap"):
        if c["design"][key] <= 0:
            raise ValueError(f"design.{key} must be positive")
    if c["candidate"]["count"] < 2 or c["ppo"]["kl_coef"] <= 0:
        raise ValueError("Need >=2 candidates and positive KL coefficient")
    if c["runtime"]["device"] not in ("cpu", "cuda") or c["runtime"]["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("v1 supports cpu/cuda and float32/bfloat16")
    if c["judge"]["temperature"] <= 0 or c["judge"]["max_attempts"] < 1:
        raise ValueError("Sampling Judge requires positive temperature and >=1 attempts")
    for section, keys in {"candidate": ["count", "max_new_tokens", "max_prompt_tokens"], "design": ["fw_max_iter"], "ppo": ["total_episodes", "per_device_train_batch_size", "gradient_accumulation_steps", "num_mini_batches", "num_ppo_epochs", "response_length", "lora_r"], "evaluation": ["max_new_tokens"]}.items():
        if any(type(c[section][key]) is not int or c[section][key] < 1 for key in keys):
            raise ValueError(f"Invalid positive integer setting in {section}")
    if c["evaluation"]["max_new_tokens"] != c["candidate"]["max_new_tokens"]:
        raise ValueError("Evaluation and training response caps must match for KL comparison")
    if c["ppo"]["response_length"] != c["candidate"]["max_new_tokens"]:
        raise ValueError("Candidate and policy response caps must match")
    batch = c["ppo"]["per_device_train_batch_size"] * c["ppo"]["gradient_accumulation_steps"] * c["ppo"]["num_mini_batches"]
    if c["ppo"]["total_episodes"] % batch:
        raise ValueError("total_episodes must be divisible by PPO rollout batch (single GPU v1)")
    c["output"] = str((path.parent / c["output"]).resolve())
    for key, val in c["choices"]["prompts"].items():
        if val:
            c["choices"]["prompts"][key] = str((path.parent / val).resolve())
    b = c["choices"]["benchmark"]
    if b["path"]:
        b["path"] = str((path.parent / b["path"]).resolve())
    for key in ("sft", "judge"):
        val = c["choices"][key]["model"]
        if val and (val.startswith(".") or Path(val).is_absolute()):
            c["choices"][key]["model"] = str((path.parent / val).resolve())
    missing = [f"{group}.{k}" for group, d in c["choices"].items() for k, v in d.items() if v is None or v == ""]
    if ready and missing:
        raise ValueError("Fill experiment choices first: " + ", ".join(missing))
    if ready:
        import re
        for key in ("sft", "judge"):
            if not re.fullmatch(r"[0-9a-f]{40}", c["choices"][key]["revision"]):
                raise ValueError(f"{key}.revision must be a pinned HF commit")
        validate_prompts(c)
    return c, missing


def prompt_rows(path):
    data = rows(path)
    ids = [r["id"] for r in data]
    if not data or len(set(ids)) != len(ids):
        raise ValueError(f"Empty data or duplicate prompt IDs: {path}")
    for r in data:
        if not r["messages"] or any(m["role"] not in ("system", "user", "assistant") or not isinstance(m["content"], str) for m in r["messages"]):
            raise ValueError("messages must be text chat messages")
        if r["messages"][-1]["role"] != "user":
            raise ValueError("Prompt must end at a user message")
    return data


def validate_prompts(c):
    if c.get("schema_version") == 2:
        from .config import validate_v2_inputs
        return validate_v2_inputs(c)
    ps = {k: prompt_rows(p) for k, p in c["choices"]["prompts"].items()}
    bench = prompt_rows(c["choices"]["benchmark"]["path"])
    if any(not isinstance(r.get("answer"), str) for r in bench):
        raise ValueError("Benchmark v1 requires string answer for deterministic exact-match scoring")
    def signatures(rs):
        return {json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in rs}
    training = signatures(ps["design"]) | signatures(ps["policy"])
    if training & (signatures(ps["evaluation"]) | signatures(bench)):
        raise ValueError("Training/evaluation prompt overlap detected by content")
    return ps


def root(c):
    p = Path(c["output"])
    p.mkdir(parents=True, exist_ok=True)
    contract = {"config": c, "input_sha256": {}}
    for f in list(c["choices"]["prompts"].values()) + [c["choices"]["benchmark"]["path"]]:
        contract["input_sha256"][f] = digest(f)
    extra = [c.get("data", {}).get("manifest_path")]
    source = c["choices"]["benchmark"].get("official_source_path")
    if source:
        extra.append(str(Path(source) / "source-manifest.json"))
    for f in extra:
        if f and Path(f).is_file():
            contract["input_sha256"][f] = digest(f)
    manifest = p / "experiment.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != contract:
            raise ValueError("Config/input changed; use a NEW output directory")
    else:
        dump(manifest, contract)
    return p


def cell(c, method, budget, seed):
    if method not in c["methods"] or budget not in c["budgets"] or seed not in c["seeds"]:
        raise ValueError("Cell not declared in experiment config")
    p = root(c) / method / f"n{budget}" / f"seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fresh(*paths):
    for p in paths:
        if Path(p).exists():
            raise FileExistsError(f"Refusing overwrite: {p}")


def receipt(path, inputs, **extra):
    dump(path, {"input_sha256": {str(p): digest(p) for p in inputs}, **extra})


def verify_receipt(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for p, h in data["input_sha256"].items():
        if digest(p) != h:
            raise ValueError(f"Artifact changed: {p}")
    return data
