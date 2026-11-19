"""Official fixed-RNG IFEval for one frozen full-pool LoRA PPO actor."""
import json
import math
import os
from pathlib import Path

import numpy as np

from prism.io import digest, rows, verify_receipt
from .artifacts import Run, config_hash, hash_json, read_json, write_json


METHOD = "full-pool-lora"
CAP = 2048
PROMPT_LIMIT = 2047
SEED_OFFSET = 2_000_000


def _checked_files(root, manifest_name):
    root = Path(root).resolve()
    manifest_path = root/manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = [manifest_path]
    for relative in manifest.get("files", {}):
        path = (root/relative.replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("IFEval manifest contains a missing or escaping file")
        result.append(path)
    return result


def _benchmark(ifeval_input, official_source, nltk_data):
    from prism.benchmarks import (IFEVAL_SOURCE_REVISION, benchmark_rows,
                                  validate_benchmark_backend)
    input_path = Path(ifeval_input).resolve()
    source = Path(official_source).resolve()
    nltk = Path(nltk_data).resolve()
    if not input_path.is_file():
        raise ValueError("Missing immutable IFEval input")
    settings = {
        "choices": {"benchmark": {"name": "IFEval", "kind": "ifeval",
                    "path": str(input_path), "official_source_path": str(source),
                    "official_source_revision": IFEVAL_SOURCE_REVISION}},
        "benchmark": {"max_new_tokens": CAP, "temperature": 0,
                      "max_prompt_tokens": PROMPT_LIMIT, "nltk_data_path": str(nltk)},
    }
    validated = validate_benchmark_backend(settings)
    records = benchmark_rows(settings)
    if validated.get("count") != 541 or len(records) != 541:
        raise ValueError("Official IFEval must retain all 541 prompts")
    dependencies = [input_path, *_checked_files(source, "source-manifest.json"),
                    *_checked_files(nltk, "resource-manifest.json")]
    return settings, records, list(dict.fromkeys(dependencies))


def generation_seed(c, index):
    return int(c["evaluation"]["seed"] + SEED_OFFSET + index)


def _validate_row(c, row, expected, index):
    tokens = row.get("response_tokens")
    if (row.get("id") != expected["id"] or row.get("index") != index
            or row.get("method") != METHOD or row.get("messages") != expected["messages"]
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
        raise ValueError("Invalid or reordered full-pool LoRA IFEval response")


def _generate_one(c, model, token, record, index):
    from prism.hf import generate as hf_generate, prompt_ids
    prompt = prompt_ids(token, record["messages"], PROMPT_LIMIT)
    response = hf_generate(model, token, prompt, CAP, generation_seed(c, index), temperature=0)
    if not response:
        raise ValueError("Empty IFEval response")
    eos = response[-1] == token.eos_token_id
    return {"id": record["id"], "index": index, "method": METHOD,
            "messages": record["messages"], "prompt_sha256": hash_json(record["messages"]),
            "generation_seed": generation_seed(c, index),
            "response": token.decode(response, skip_special_tokens=True),
            "response_tokens": response, "response_length": len(response), "eos": eos,
            "truncated": len(response) >= CAP and not eos, "config_sha256": config_hash(c)}


def generate(c, shard_index, shards, ifeval_input, official_source, nltk_data, *, generator=None):
    """Generate one resumable, globally indexed actor shard."""
    if (type(shards) is not int or not 1 <= shards <= 541 or type(shard_index) is not int
            or not 0 <= shard_index < shards):
        raise ValueError("Invalid LoRA IFEval shard")
    _, prompts, asset_inputs = _benchmark(ifeval_input, official_source, nltk_data)
    run = Run(c)
    run.require("lora-ppo-evaluation-selection")
    stage = f"lora-ppo-ifeval-part-{shard_index:05d}-of-{shards:05d}"
    if run.completed(stage):
        return run.require(stage)
    target = run.file(f"evaluation/{METHOD}/ifeval/shards/part-{shard_index:05d}-of-{shards:05d}.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(shard_index, len(prompts), shards))
    previous = rows(target) if target.exists() else []
    if len(previous) > len(indices):
        raise ValueError("Unexpected LoRA IFEval shard prefix")
    for row, index in zip(previous, indices):
        _validate_row(c, row, prompts[index], index)
    model = token = None
    if generator is None and len(previous) < len(indices):
        from .lora_evaluation import load_selected_actor
        model, token = load_selected_actor(c)
    try:
        with target.open("a", encoding="utf-8") as stream:
            for index in indices[len(previous):]:
                row = (generator(c, prompts[index], index) if generator is not None else
                       _generate_one(c, model, token, prompts[index], index))
                _validate_row(c, row, prompts[index], index)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        del model, token
    return run.publish(stage, [target],
                       [run.receipt_path("lora-ppo-evaluation-selection"), *asset_inputs],
                       {"method": METHOD, "shard_index": shard_index, "shards": shards,
                        "indices": indices, "cap": CAP, "temperature": 0})


def _sft_paths(directory):
    directory = Path(directory).resolve()
    return {name: directory/name for name in
            ("benchmark.responses.jsonl", "benchmark.fixed.metrics.json",
             "benchmark.fixed.receipt.json", "reference.receipt.json")}


def _validate_sft_records(records, prompts):
    if len(records) != len(prompts):
        raise ValueError("SFT IFEval response coverage changed")
    for row, prompt in zip(records, prompts):
        if (row.get("id") != prompt["id"] or row.get("messages") != prompt["messages"]
                or not isinstance(row.get("response"), str) or not row["response"]
                or type(row.get("response_length")) is not int or row["response_length"] < 1):
            raise ValueError("SFT IFEval source is not aligned to the official prompts")


def import_sft(c, ifeval_input, official_source, nltk_data, sft_directory):
    """Bind and independently rescore the already generated identical SFT baseline."""
    from prism.scoring_protocol import PROTOCOL, fixed_score
    benchmark, prompts, asset_inputs = _benchmark(ifeval_input, official_source, nltk_data)
    run = Run(c)
    stage = "lora-ppo-ifeval-sft"
    if run.completed(stage):
        return run.require(stage)
    paths = _sft_paths(sft_directory)
    for name in ("reference.receipt.json", "benchmark.fixed.receipt.json"):
        verify_receipt(paths[name])
    experiment = Path(sft_directory).resolve().parent/"experiment.json"
    if not experiment.is_file():
        raise ValueError("SFT IFEval source lost its experiment contract")
    contract = json.loads(experiment.read_text(encoding="utf-8")).get("config", {})
    sft = contract.get("choices", {}).get("sft", {})
    old_benchmark = contract.get("choices", {}).get("benchmark", {})
    if (sft != {"model": c["model"]["id"], "revision": c["model"]["revision"]}
            or old_benchmark.get("kind") != "ifeval"
            or old_benchmark.get("official_source_revision") != benchmark["choices"]["benchmark"]["official_source_revision"]
            or digest(old_benchmark.get("path", "")) != digest(ifeval_input)
            or contract.get("benchmark", {}).get("max_new_tokens") != CAP
            or contract.get("benchmark", {}).get("temperature") != 0):
        raise ValueError("SFT IFEval source used another model or generation protocol")
    records = rows(paths["benchmark.responses.jsonl"])
    _validate_sft_records(records, prompts)
    stored = read_json(paths["benchmark.fixed.metrics.json"])
    scored = fixed_score(benchmark, records)
    if (fixed_score(benchmark, records) != scored or stored.get("metrics") != scored.get("metrics")
            or stored.get("details") != scored.get("details")
            or stored.get("scoring_protocol", {}).get("protocol") != PROTOCOL):
        raise ValueError("SFT fixed-RNG IFEval independent rescore differs")
    value = {"protocol": "v3_full_pool_lora_sft_ifeval_import_v1",
             "model": sft, "prompts": len(prompts), "instructions": 834,
             "generation": {"cap": CAP, "temperature": 0, "prompt_limit": PROMPT_LIMIT},
             "responses": str(paths["benchmark.responses.jsonl"]),
             "responses_sha256": digest(paths["benchmark.responses.jsonl"]),
             "fixed_metrics_sha256": digest(paths["benchmark.fixed.metrics.json"]),
             "metrics": scored["metrics"], "scoring_protocol": PROTOCOL}
    target = run.file("evaluation/sft/ifeval/import.json")
    write_json(target, value)
    return run.publish(stage, [target], [*asset_inputs, experiment, *paths.values()],
                       {"prompts": 541, "instructions": 834,
                        "prompt_level_strict": scored["prompt_level_strict"]})


def _write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Preserve different merged LoRA IFEval responses")
    else:
        path.write_text(text, encoding="utf-8")


def merge(c, shards, ifeval_input, official_source, nltk_data, sft_directory):
    """Merge actor shards, fixed-score both actors, and publish a paired report."""
    from prism.scoring_protocol import PROTOCOL, fixed_score
    if type(shards) is not int or not 1 <= shards <= 541:
        raise ValueError("Invalid LoRA IFEval shard count")
    benchmark, prompts, asset_inputs = _benchmark(ifeval_input, official_source, nltk_data)
    run = Run(c)
    if run.completed("lora-ppo-ifeval"):
        return run.require("lora-ppo-ifeval")
    run.require("lora-ppo-ifeval-sft")
    combined, dependencies = {}, [run.receipt_path("lora-ppo-ifeval-sft"),
                                  run.receipt_path("lora-ppo-evaluation-selection"), *asset_inputs]
    for shard in range(shards):
        stage = f"lora-ppo-ifeval-part-{shard:05d}-of-{shards:05d}"
        run.require(stage)
        dependencies.append(run.receipt_path(stage))
        path = run.file(f"evaluation/{METHOD}/ifeval/shards/part-{shard:05d}-of-{shards:05d}.jsonl")
        block, expected = rows(path), list(range(shard, len(prompts), shards))
        if [row.get("index") for row in block] != expected:
            raise ValueError("Incomplete or reordered LoRA IFEval shard")
        for row, index in zip(block, expected):
            _validate_row(c, row, prompts[index], index)
            if index in combined:
                raise ValueError("Overlapping LoRA IFEval shard")
            combined[index] = row
    if set(combined) != set(range(len(prompts))):
        raise ValueError("Incomplete LoRA IFEval coverage")
    actor_records = [combined[index] for index in range(len(prompts))]
    sft_paths = _sft_paths(sft_directory)
    sft_records = rows(sft_paths["benchmark.responses.jsonl"])
    _validate_sft_records(sft_records, prompts)
    sft_report = read_json(run.file("evaluation/sft/ifeval/import.json"))
    if digest(sft_paths["benchmark.responses.jsonl"]) != sft_report["responses_sha256"]:
        raise ValueError("Imported SFT IFEval responses changed")
    actor_score, sft_score = fixed_score(benchmark, actor_records), fixed_score(benchmark, sft_records)
    if fixed_score(benchmark, actor_records) != actor_score or fixed_score(benchmark, sft_records) != sft_score:
        raise RuntimeError("Fixed-RNG IFEval scoring is not reproducible")
    actor_flags = np.asarray([row["strict"]["follow_all_instructions"]
                              for row in actor_score["details"]], dtype=float)
    sft_flags = np.asarray([row["strict"]["follow_all_instructions"]
                            for row in sft_score["details"]], dtype=float)
    delta = actor_flags-sft_flags
    rng = np.random.default_rng(np.random.SeedSequence([c["evaluation"]["seed"], 32768, 541]))
    draws = c["evaluation"].get("bootstrap_samples", 2000)
    boot = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1) if draws else None
    paired = {"prompt_level_strict_delta": float(delta.mean()),
              "standard_error": float(delta.std(ddof=1)/math.sqrt(len(delta))),
              "ci95": np.quantile(boot, [.025, .975]).tolist() if draws else None,
              "actor_better": int((delta > 0).sum()), "same": int((delta == 0).sum()),
              "actor_worse": int((delta < 0).sum())}
    response_path = run.file(f"evaluation/{METHOD}/ifeval/responses.jsonl")
    metrics_path = run.file(f"evaluation/{METHOD}/ifeval/metrics.json")
    summary_path = run.file("summaries/lora-ppo-32k-ifeval.json")
    _write_rows(response_path, actor_records)
    write_json(metrics_path, {**actor_score, "scoring_protocol": PROTOCOL})
    selection = read_json(run.file("control/lora-ppo-evaluation-selection.json"))
    summary = {"protocol": "v3_full_pool_lora_ifeval_v1", "selection": selection,
               "prompts": 541, "instructions": 834,
               "generation": {"cap": CAP, "temperature": 0, "prompt_limit": PROMPT_LIMIT},
               "actor": actor_score["metrics"], "sft": sft_score["metrics"],
               "paired_prompt_strict": paired, "scoring_protocol": PROTOCOL, "api_calls": 0}
    write_json(summary_path, summary)
    return run.publish("lora-ppo-ifeval", [response_path, metrics_path, summary_path],
                       dependencies, {"prompts": 541, "instructions": 834,
                                      "actor_prompt_strict": actor_score["prompt_level_strict"],
                                      "sft_prompt_strict": sft_score["prompt_level_strict"], **paired})
