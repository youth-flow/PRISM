"""Official fixed-RNG IFEval for the two sealed 16k linear-Reward policies."""
import json
import math
import os

import numpy as np

from prism.io import digest, rows
from .artifacts import Run, config_hash, hash_json, read_json, write_json
from .lora_ifeval import (CAP, PROMPT_LIMIT, _benchmark, _sft_paths,
                          _validate_sft_records, _write_rows, generation_seed)
from .linear_ppo import FORMAL_HORIZON, METHODS


def _require_method(method):
    if method not in METHODS:
        raise ValueError("Linear PPO IFEval method must be empirical or prism")


def _stage(method):
    _require_method(method)
    return f"linear-ppo-ifeval-{method}"


def _validate_row(c, row, expected, index, method):
    tokens = row.get("response_tokens")
    if (row.get("id") != expected["id"] or row.get("index") != index
            or row.get("method") != method or row.get("messages") != expected["messages"]
            or row.get("prompt_sha256") != hash_json(expected["messages"])
            or row.get("generation_seed") != generation_seed(c, index)
            or row.get("config_sha256") != config_hash(c)
            or not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or row.get("response_length") != len(tokens) or len(tokens) > CAP
            or not isinstance(row.get("response"), str)
            or type(row.get("eos")) is not bool or type(row.get("truncated")) is not bool
            or row["truncated"] != (not row["eos"])
            or row["truncated"] and len(tokens) != CAP):
        raise ValueError("Invalid or reordered linear PPO IFEval response")


def _generate_one(c, model, token, record, index, method):
    from prism.hf import generate as hf_generate, prompt_ids
    prompt = prompt_ids(token, record["messages"], PROMPT_LIMIT)
    response = hf_generate(model, token, prompt, CAP, generation_seed(c, index), temperature=0)
    if not response:
        raise ValueError("Empty IFEval response")
    eos = response[-1] == token.eos_token_id
    return {"id": record["id"], "index": index, "method": method,
            "messages": record["messages"],
            "prompt_sha256": hash_json(record["messages"]),
            "generation_seed": generation_seed(c, index),
            "response": token.decode(response, skip_special_tokens=True),
            "response_tokens": response, "response_length": len(response), "eos": eos,
            "truncated": len(response) >= CAP and not eos,
            "config_sha256": config_hash(c)}


def generate(c, method, shard_index, shards, ifeval_input, official_source, nltk_data,
             *, generator=None):
    """Generate a resumable globally indexed IFEval shard."""
    _require_method(method)
    if (type(shards) is not int or not 1 <= shards <= 541 or type(shard_index) is not int
            or not 0 <= shard_index < shards):
        raise ValueError("Invalid linear PPO IFEval shard")
    _, prompts, asset_inputs = _benchmark(ifeval_input, official_source, nltk_data)
    run = Run(c)
    run.require(f"policy-{method}")
    run.require(f"linear-ppo-audit-{method}")
    prefix = _stage(method)
    stage = f"{prefix}-part-{shard_index:05d}-of-{shards:05d}"
    if run.completed(stage):
        return run.require(stage)
    target = run.file(
        f"evaluation/{method}/ifeval/shards/part-{shard_index:05d}-of-{shards:05d}.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(shard_index, len(prompts), shards))
    previous = rows(target) if target.exists() else []
    if len(previous) > len(indices):
        raise ValueError("Unexpected linear PPO IFEval shard prefix")
    for row, index in zip(previous, indices):
        _validate_row(c, row, prompts[index], index, method)
    model = token = None
    if generator is None and len(previous) < len(indices):
        from .evaluation import _load_generator
        model, token = _load_generator(c, method)
    try:
        with target.open("a", encoding="utf-8") as stream:
            for index in indices[len(previous):]:
                row = (generator(c, prompts[index], index) if generator is not None else
                       _generate_one(c, model, token, prompts[index], index, method))
                _validate_row(c, row, prompts[index], index, method)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        del model, token
    return run.publish(stage, [target],
                       [run.receipt_path(f"policy-{method}"),
                        run.receipt_path(f"linear-ppo-audit-{method}"), *asset_inputs],
                       {"method": method, "episode": FORMAL_HORIZON,
                        "shard_index": shard_index, "shards": shards,
                        "indices": indices, "cap": CAP, "temperature": 0})


def merge(c, method, shards, ifeval_input, official_source, nltk_data, sft_directory):
    """Merge, independently fixed-score, and compare with the identical SFT baseline."""
    from prism.scoring_protocol import PROTOCOL, fixed_score
    _require_method(method)
    if type(shards) is not int or not 1 <= shards <= 541:
        raise ValueError("Invalid linear PPO IFEval shard count")
    benchmark, prompts, asset_inputs = _benchmark(ifeval_input, official_source, nltk_data)
    run = Run(c)
    prefix = _stage(method)
    if run.completed(prefix):
        return run.require(prefix)
    run.require("lora-ppo-ifeval-sft")
    combined = {}
    dependencies = [run.receipt_path("lora-ppo-ifeval-sft"),
                    run.receipt_path(f"policy-{method}"),
                    run.receipt_path(f"linear-ppo-audit-{method}"), *asset_inputs]
    for shard in range(shards):
        stage = f"{prefix}-part-{shard:05d}-of-{shards:05d}"
        run.require(stage)
        dependencies.append(run.receipt_path(stage))
        path = run.file(
            f"evaluation/{method}/ifeval/shards/part-{shard:05d}-of-{shards:05d}.jsonl")
        block, expected = rows(path), list(range(shard, len(prompts), shards))
        if [row.get("index") for row in block] != expected:
            raise ValueError("Incomplete or reordered linear PPO IFEval shard")
        for row, index in zip(block, expected):
            _validate_row(c, row, prompts[index], index, method)
            if index in combined:
                raise ValueError("Overlapping linear PPO IFEval shard")
            combined[index] = row
    if set(combined) != set(range(len(prompts))):
        raise ValueError("Incomplete linear PPO IFEval coverage")
    actor_records = [combined[index] for index in range(len(prompts))]
    sft_paths = _sft_paths(sft_directory)
    sft_records = rows(sft_paths["benchmark.responses.jsonl"])
    _validate_sft_records(sft_records, prompts)
    sft_report = read_json(run.file("evaluation/sft/ifeval/import.json"))
    if digest(sft_paths["benchmark.responses.jsonl"]) != sft_report["responses_sha256"]:
        raise ValueError("Imported SFT IFEval responses changed")
    actor_score = fixed_score(benchmark, actor_records)
    sft_score = fixed_score(benchmark, sft_records)
    if (fixed_score(benchmark, actor_records) != actor_score
            or fixed_score(benchmark, sft_records) != sft_score):
        raise RuntimeError("Fixed-RNG IFEval scoring is not reproducible")
    actor_flags = np.asarray([row["strict"]["follow_all_instructions"]
                              for row in actor_score["details"]], dtype=float)
    sft_flags = np.asarray([row["strict"]["follow_all_instructions"]
                            for row in sft_score["details"]], dtype=float)
    delta = actor_flags - sft_flags
    method_offset = METHODS.index(method)
    rng = np.random.default_rng(np.random.SeedSequence(
        [c["evaluation"]["seed"], FORMAL_HORIZON, 541, method_offset]))
    draws = c["evaluation"].get("bootstrap_samples", 2000)
    boot = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1) if draws else None
    paired = {"prompt_level_strict_delta": float(delta.mean()),
              "standard_error": float(delta.std(ddof=1) / math.sqrt(len(delta))),
              "ci95": np.quantile(boot, [.025, .975]).tolist() if draws else None,
              "actor_better": int((delta > 0).sum()), "same": int((delta == 0).sum()),
              "actor_worse": int((delta < 0).sum())}
    response_path = run.file(f"evaluation/{method}/ifeval/responses.jsonl")
    metrics_path = run.file(f"evaluation/{method}/ifeval/metrics.json")
    summary_path = run.file(f"summaries/linear-ppo-{method}-ifeval.json")
    _write_rows(response_path, actor_records)
    write_json(metrics_path, {**actor_score, "scoring_protocol": PROTOCOL})
    summary = {"protocol": "v3_linear_reward_ppo_ifeval_v1", "method": method,
               "episode": FORMAL_HORIZON, "prompts": 541, "instructions": 834,
               "generation": {"cap": CAP, "temperature": 0, "prompt_limit": PROMPT_LIMIT},
               "actor": actor_score["metrics"], "sft": sft_score["metrics"],
               "paired_prompt_strict": paired, "scoring_protocol": PROTOCOL,
               "api_calls": 0}
    write_json(summary_path, summary)
    return run.publish(prefix, [response_path, metrics_path, summary_path], dependencies,
                       {"method": method, "prompts": 541, "instructions": 834,
                        "actor_prompt_strict": actor_score["prompt_level_strict"],
                        "sft_prompt_strict": sft_score["prompt_level_strict"], **paired})
