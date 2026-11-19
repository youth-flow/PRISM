"""HH V3 generation, portable Luna packets, and prompt-level scoring.

All imports and export/import/aggregate commands are offline. Paid execution is
local-only, opt-in, and binds a new V3 ledger to exactly one immutable packet.
The mature V2 request parser and USD transport are reused without its manifest.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np

from prism.io import digest, rows
from .artifacts import Run, config_hash, hash_json, read_json, write_json


METHODS = ("sft", "empirical", "prism")
LORA_METHOD = "full-pool-lora"
GENERATION_METHODS = (*METHODS, LORA_METHOD)
JUDGE_KEYS = ("backend", "model", "api", "base_url", "api_key_env", "reasoning_effort",
              "service_tier", "max_output_tokens", "max_attempts", "timeout_seconds")


def _source_name(record):
    source = record.get("source", "unknown")
    return str(source.get("subset", "unknown")) if isinstance(source, dict) else str(source)


def _write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if rows(path) == records:
            return
        raise FileExistsError(f"Preserve different V3 rows: {path}")
    with path.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _prompts(c, run):
    run.require("data")
    records = rows(run.file("data/evaluation.jsonl"))
    if len(records) != c["data"]["evaluation_prompts"] or len({r["id"] for r in records}) != len(records):
        raise ValueError("Final HH evaluation prompt coverage changed")
    return records


def generation_seed(c, index):
    return int(np.random.SeedSequence([c["evaluation"]["seed"], index, 310711]).generate_state(1)[0])


def _model_config(c):
    return {"runtime": dict(c["runtime"]), "choices": {"sft": {
        "model": c["model"]["id"], "revision": c["model"]["revision"]}}}


def _load_generator(c, method, lora_episode=None):
    if lora_episode is not None:
        from .lora_evaluation import load_selected_actor
        return load_selected_actor(c, lora_episode)
    from prism.hf import load_lm, tokenizer
    legacy = _model_config(c)
    token = tokenizer(legacy)
    model = load_lm(legacy)
    if method != "sft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Run(c).file(f"policy/{method}/actor")),
                                          is_trainable=False).eval()
    return model, token


def _logp(model, prompt, response):
    import torch
    from prism.numerics import selected_logprobs
    with torch.no_grad():
        tokens = torch.tensor([prompt + response], device=next(model.parameters()).device)
        output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                       use_cache=False, return_dict=True)
        logits = output.logits[:, len(prompt)-1:len(prompt)+len(response)-1].float()
        target = torch.tensor([response], device=tokens.device)
        return float(selected_logprobs(logits, target).sum())


def _generate_one(c, model, tokenizer, method, record, index):
    from prism.hf import generate, prompt_ids
    prompt = prompt_ids(tokenizer, record["messages"], c["data"]["max_prompt_tokens"])
    seed = generation_seed(c, index)
    response = generate(model, tokenizer, prompt, c["evaluation"]["max_new_tokens"], seed,
                        temperature=c["evaluation"]["temperature"])
    if not response:
        raise ValueError("Empty generated response")
    policy_logp = _logp(model, prompt, response)
    if method == "sft":
        reference_logp = policy_logp
    else:
        with model.disable_adapter():
            reference_logp = _logp(model, prompt, response)
    return {"id": record["id"], "index": index, "method": method,
            "messages": record["messages"], "source": record.get("source", "unknown"),
            "prompt_sha256": hash_json(record["messages"]), "generation_seed": seed,
            "response": tokenizer.decode(response, skip_special_tokens=True),
            "response_tokens": response, "response_length": len(response),
            "eos": response[-1] == tokenizer.eos_token_id,
            "truncated": len(response) >= c["evaluation"]["max_new_tokens"] and response[-1] != tokenizer.eos_token_id,
            "policy_sequence_logp": policy_logp, "sft_sequence_logp": reference_logp,
            "sampled_sequence_log_ratio": policy_logp-reference_logp,
            "config_sha256": config_hash(c)}


def _validate_row(c, row, expected, index, method):
    if (row.get("id") != expected["id"] or row.get("index") != index
            or row.get("method") != method or row.get("messages") != expected["messages"]
            or row.get("prompt_sha256") != hash_json(expected["messages"])
            or row.get("generation_seed") != generation_seed(c, index)
            or row.get("source") != expected.get("source", "unknown")
            or row.get("config_sha256") != config_hash(c)):
        raise ValueError("Generated answer identity/seed/input changed")
    tokens = row.get("response_tokens")
    if (not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens)
            or row.get("response_length") != len(tokens)
            or len(tokens) > c["evaluation"]["max_new_tokens"] or not isinstance(row.get("response"), str)
            or type(row.get("eos")) is not bool or type(row.get("truncated")) is not bool):
        raise ValueError("Invalid generated answer tokens or ending metadata")
    if row["truncated"] != (not row["eos"]) or row["truncated"] and len(tokens) != c["evaluation"]["max_new_tokens"]:
        raise ValueError("Generated EOS/truncation metadata is inconsistent")
    numbers = [row.get(k) for k in ("policy_sequence_logp", "sft_sequence_logp", "sampled_sequence_log_ratio")]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers):
        raise ValueError("Nonfinite generated sequence scores")
    if abs((numbers[0]-numbers[1])-numbers[2]) > 1e-9 or (method == "sft" and numbers[2] != 0):
        raise ValueError("Generated KL differs from actor/reference log probabilities")


def evaluate_generate(c, method, shard_index=0, shards=1, *, generator=None,
                      evaluation_episode=None):
    """Generate a disjoint shard. Only incomplete shard prefixes are resumed."""
    if method not in GENERATION_METHODS or type(shards) is not int or shards < 1 or type(shard_index) is not int or not 0 <= shard_index < shards:
        raise ValueError("Invalid evaluation method or shard")
    lora_episode = None
    if method == LORA_METHOD:
        from .lora_evaluation import DEFAULT_EPISODE, evaluation_method, evaluation_names
        lora_episode = DEFAULT_EPISODE if evaluation_episode is None else evaluation_episode
        method = evaluation_method(lora_episode)
        selection_stage = evaluation_names(lora_episode)["selection_stage"]
    elif evaluation_episode is not None:
        raise ValueError("Only LoRA evaluation generation accepts an evaluation episode")
    run = Run(c)
    prompts = _prompts(c, run)
    if shards > len(prompts):
        raise ValueError("More evaluation shards than prompts")
    dependencies = [run.receipt_path("data")]
    if lora_episode is not None:
        run.require(selection_stage)
        dependencies.append(run.receipt_path(selection_stage))
    elif method != "sft":
        run.require(f"policy-{method}")
        dependencies.append(run.receipt_path(f"policy-{method}"))
    stage = f"eval-{method}-part-{shard_index:05d}-of-{shards:05d}"
    if run.completed(stage):
        return run.require(stage)
    target = run.file(f"evaluation/{method}/shards/part-{shard_index:05d}-of-{shards:05d}.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(shard_index, len(prompts), shards))
    previous = rows(target) if target.exists() else []
    if len(previous) > len(indices):
        raise ValueError("Unexpected evaluation shard prefix length")
    for row, index in zip(previous, indices):
        _validate_row(c, row, prompts[index], index, method)
    model = token = None
    if generator is None and len(previous) < len(indices):
        model, token = _load_generator(c, method, lora_episode)
    try:
        with target.open("a", encoding="utf-8") as stream:
            for index in indices[len(previous):]:
                row = (generator(c, method, prompts[index], index) if generator is not None else
                       _generate_one(c, model, token, method, prompts[index], index))
                _validate_row(c, row, prompts[index], index, method)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        del model, token
    return run.publish(stage, [target], dependencies,
                       {"method": method, "shard_index": shard_index, "shards": shards, "indices": indices})


def merge_evaluation(c, method, shards, *, evaluation_episode=None):
    run = Run(c)
    prompts = _prompts(c, run)
    if method not in GENERATION_METHODS or type(shards) is not int or not 1 <= shards <= len(prompts):
        raise ValueError("Invalid evaluation shard count")
    if method == LORA_METHOD:
        from .lora_evaluation import DEFAULT_EPISODE, evaluation_method
        method = evaluation_method(DEFAULT_EPISODE if evaluation_episode is None else evaluation_episode)
    elif evaluation_episode is not None:
        raise ValueError("Only LoRA evaluation merge accepts an evaluation episode")
    stage = f"eval-{method}"
    if run.completed(stage):
        return run.require(stage)
    combined, dependencies = {}, [run.receipt_path("data")]
    for shard in range(shards):
        name = f"eval-{method}-part-{shard:05d}-of-{shards:05d}"
        run.require(name)
        dependencies.append(run.receipt_path(name))
        path = run.file(f"evaluation/{method}/shards/part-{shard:05d}-of-{shards:05d}.jsonl")
        data = rows(path)
        expected = list(range(shard, len(prompts), shards))
        if [r.get("index") for r in data] != expected:
            raise ValueError("Missing, duplicated or reordered evaluation shard")
        for row, index in zip(data, expected):
            _validate_row(c, row, prompts[index], index, method)
            if index in combined:
                raise ValueError("Evaluation shard overlap")
            combined[index] = row
    if set(combined) != set(range(len(prompts))):
        raise ValueError("Incomplete evaluation generation")
    ordered = [combined[i] for i in range(len(prompts))]
    target = run.file(f"evaluation/{method}/responses.jsonl")
    _write_rows(target, ordered)
    metrics = _generation_metrics(ordered)
    metric_path = run.file(f"evaluation/{method}/generation.metrics.json")
    write_json(metric_path, metrics)
    return run.publish(stage, [target, metric_path], dependencies, {"method": method, **metrics})


def _generation_metrics(records):
    return {"prompts": len(records), "mean_response_tokens": float(np.mean([r["response_length"] for r in records])),
            "eos_rate": float(np.mean([r["eos"] for r in records])),
            "truncation_rate": float(np.mean([r["truncated"] for r in records])),
            "sequence_kl_mc": float(np.mean([r["sampled_sequence_log_ratio"] for r in records]))}


def _settings(c):
    return {key: c["evaluation"][key] for key in JUDGE_KEYS}


def _sources(c, run):
    prompts = _prompts(c, run)
    sources = {}
    for method in METHODS:
        run.require(f"eval-{method}")
        records = rows(run.file(f"evaluation/{method}/responses.jsonl"))
        if len(records) != len(prompts):
            raise ValueError("Final evaluation answer coverage differs")
        for index, (record, prompt) in enumerate(zip(records, prompts)):
            _validate_row(c, record, prompt, index, method)
        sources[method] = records
    return prompts, sources


def export_judge(c):
    from prism.judges import APIJudge, evaluation_call_seed, judge_protocol
    from prism.local_judge import _hash
    run = Run(c)
    selection = run.selection()
    prompts, sources = _sources(c, run)
    settings = _settings(c)
    dependencies = [run.receipt_path("study-selection"), run.receipt_path("data")]
    dependencies += [run.receipt_path(f"eval-{method}") for method in METHODS]
    tasks = []
    for ci, comparison in enumerate(c["evaluation"]["comparisons"]):
        for index, prompt in enumerate(prompts):
            for oi, order in enumerate(c["evaluation"]["answer_orders"]):
                seed_config = {"evaluation": {"seed": c["evaluation"]["seed"]}}
                calls = [{"seed": evaluation_call_seed(seed_config, selection["budget"], c["study"]["seed"], ci, index, oi, a),
                          "flip": order == "BA"} for a in range(settings["max_attempts"])]
                tasks.append({"id": f"{comparison['name']}:{index}:{order}",
                              "messages": prompt["messages"], "left": sources[comparison["left"]][index]["response"],
                              "right": sources[comparison["right"]][index]["response"], "calls": calls})
    packet = {"kind": "prism_local_judge", "version": 1, "stage": "evaluate-judge", "method": "paired-v3",
              "role": "evaluation", "budget": selection["budget"], "seed": c["study"]["seed"],
              "judge_protocol": judge_protocol("evaluation", "openai"), "settings": settings,
              "request_endpoint": APIJudge({"judges": {"evaluation": settings}}, "evaluation").url,
              "config_sha256": run.fingerprint,
              "input_sha256": {run._key(path): digest(path) for path in dependencies},
              "tasks": tasks, "prior_calls": [],
              "v3": {"protocol": "hh_finite_pool_v3", "prompts": len(prompts),
                     "comparisons": c["evaluation"]["comparisons"], "valid_judgments": len(tasks)}}
    packet["packet_id"] = _hash(packet)
    target = run.file("judge/packet.json")
    write_json(target, packet)
    return run.publish("judge-export", [target], dependencies, {"tasks": len(tasks), "packet_id": packet["packet_id"]})


def _bound_packet(c, run):
    from prism.local_judge import _load_packet
    run.require("judge-export")
    packet = _load_packet(run.file("judge/packet.json"))
    if packet.get("v3", {}).get("protocol") != "hh_finite_pool_v3" or packet["config_sha256"] != run.fingerprint or packet["settings"] != _settings(c):
        raise ValueError("Judge packet does not match this V3 study")
    for key, expected in packet["input_sha256"].items():
        if digest(run._resolve(key)) != expected:
            raise ValueError("Judge packet input receipt changed")
    return packet


def _billing_path(results):
    return Path(str(results) + ".billing.json")


def _export_billing(budget, packet, results):
    with budget.transaction() as db:
        records = db.execute("SELECT id,purpose,model,payload_sha,reserved,charged,status,response,request_id FROM dispatch ORDER BY id").fetchall()
    keys = ("id", "purpose", "model", "payload_sha", "reserved_micro_usd", "charged_micro_usd", "status", "response", "request_id")
    entries = [dict(zip(keys, row)) for row in records]
    for entry in entries:
        entry["response"] = json.loads(entry["response"]) if entry["response"] is not None else None
        if not entry["id"].startswith(packet["packet_id"] + "/"):
            raise ValueError("V3 ledger contains another packet")
    value = {"protocol": "v3_usd_dispatch_snapshot_v1", "packet_id": packet["packet_id"],
             "results_sha256": digest(results) if Path(results).exists() else None,
             "ledger_contract": read_json(budget.path.with_name(budget.path.name + ".v3.json")),
             "state": budget.state(), "dispatches": entries}
    write_json(_billing_path(results), value, overwrite=True)
    return value


def _validate_billing(packet, results, billing):
    from prism.api_budget import request_reservation, usage_micro_usd, _flex_no_charge
    from prism.judges import APIJudge, _messages, _usage
    if (billing.get("protocol") != "v3_usd_dispatch_snapshot_v1" or billing.get("packet_id") != packet["packet_id"]
            or billing.get("results_sha256") != digest(results)):
        raise ValueError("V3 billing snapshot/result identity changed")
    entries = billing.get("dispatches", [])
    if not isinstance(entries, list) or len({e["id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate V3 billing dispatch")
    lookup = {entry["id"]: entry for entry in entries}
    tasks = {task["id"]: task for task in packet["tasks"]}
    raw = rows(results)
    known, held = 0, 0
    seen = set()
    judge = APIJudge({"judges": {"evaluation": packet["settings"]}}, "evaluation")
    for record in raw:
        task = tasks[record["task_id"]]
        attempt = record["attempt"]
        identity = f"{packet['packet_id']}/{task['id']}/{attempt}/0"
        result = record["result"]
        if result.get("budget_transport", {}).get("dispatch_id") != identity or identity not in lookup:
            raise ValueError("Judge raw attempt lacks a matching USD dispatch")
        if result.get("label") is not None and lookup[identity]["status"] != "completed":
            raise ValueError("A valid V3 judgment requires a completed original provider response; unknown/no-charge failures cannot provide labels")
        if (result.get("configured_model") != packet["settings"]["model"]
                or result.get("request_endpoint") != packet["request_endpoint"] or result.get("backend") != "openai"):
            raise ValueError("Judge raw result model/endpoint changed")
        seen.add(identity)
    for identity, entry in lookup.items():
        if not identity.startswith(packet["packet_id"] + "/"):
            raise ValueError("V3 billing includes a foreign packet")
        suffix = identity[len(packet["packet_id"])+1:]
        task_id, attempt_text, retry = suffix.rsplit("/", 2)
        if task_id not in tasks or retry != "0" or not attempt_text.isdecimal():
            raise ValueError("V3 billing dispatch identity invalid")
        task = tasks[task_id]
        attempt = int(attempt_text)
        if not 0 <= attempt < len(task["calls"]):
            raise ValueError("V3 billing attempt outside packet")
        call = task["calls"][attempt]
        left, right = (task["right"], task["left"]) if call["flip"] else (task["left"], task["right"])
        payload = judge._payload(_messages(task["messages"], left, right, "evaluation", protocol=packet["judge_protocol"]))
        expected_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if (entry["purpose"] != "evaluation" or entry["model"] != packet["settings"]["model"]
                or entry["payload_sha"] != expected_hash or entry["reserved_micro_usd"] != request_reservation(entry["model"], payload)):
            raise ValueError("V3 billing request payload/reservation changed")
        charge = entry["charged_micro_usd"]
        if charge is None:
            if entry["status"] not in ("reserved", "uncertain"):
                raise ValueError("Unknown V3 charge must retain reservation")
            held += entry["reserved_micro_usd"]
        elif entry["status"] == "completed":
            response = entry["response"]
            if charge != usage_micro_usd(entry["model"], response["usage"], response.get("service_tier")):
                raise ValueError("V3 completed USD settlement differs from raw usage")
            matching = [r["result"] for r in raw if r["result"].get("budget_transport", {}).get("dispatch_id") == identity]
            if not matching or any(r.get("usage") != _usage(response) for r in matching):
                raise ValueError("V3 usage evidence differs from provider response")
            text = "".join(part.get("text", "") for item in response.get("output", [])
                           if item.get("type") == "message" and item.get("role") == "assistant"
                           for part in item.get("content", []) if part.get("type") == "output_text")
            if any(r.get("raw") != text[:16384] or r.get("response_status") != response.get("status")
                   or r.get("service_tier") != response.get("service_tier") for r in matching):
                raise ValueError("V3 judgment differs from the preserved provider response")
            known += charge
        elif entry["status"] == "failed_no_charge" and charge == 0 and _flex_no_charge(payload, entry["response"]):
            pass
        else:
            raise ValueError("V3 zero/known settlement lacks original billing evidence")
    limit = billing["state"]["limits_usd"]
    contract = billing.get("ledger_contract", {})
    if (contract.get("protocol") != "v3_judge_ledger_v1" or contract.get("packet_id") != packet["packet_id"]
            or contract.get("config_sha256") != packet["config_sha256"]
            or type(contract.get("budget_usd")) not in (int, float)
            or not math.isfinite(contract["budget_usd"]) or contract["budget_usd"] <= 0):
        raise ValueError("V3 billing lost its isolated ledger/cap contract")
    if (set(limit) != {"total", "evaluation"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= contract["budget_usd"] for v in limit.values())
            or known + held > round(min(limit.values())*1e6)):
        raise ValueError("V3 billing exceeds isolated USD limits")
    usage = billing["state"].get("usage", {}).get("evaluation", {})
    if (usage.get("dispatches", 0) != len(entries)
            or abs(usage.get("charged_usd_estimate", 0)-known/1e6) > 1e-10
            or abs(usage.get("unresolved_reservations_usd", 0)-held/1e6) > 1e-10):
        raise ValueError("V3 USD totals differ from individual dispatches")
    return {"recorded_judge_calls": len(raw), "usd_dispatches": len(entries), "known_usd": known/1e6,
            "unknown_reserved_usd": held/1e6, "orphan_dispatches": len(set(lookup)-seen),
            "usd_limit": limit["total"], "price_snapshot_is_invoice": False}


def import_judge(c, results):
    from prism.local_judge import _records
    run = Run(c)
    packet = _bound_packet(c, run)
    _, interpreted, accounting = _records(packet, results, with_accounting=True)
    complete = {}
    for task in packet["tasks"]:
        successful = [interpreted[(task["id"], attempt)]["label"] for attempt in range(len(task["calls"]))
                      if (task["id"], attempt) in interpreted and interpreted[(task["id"], attempt)]["label"] is not None]
        if len(successful) != 1:
            raise ValueError("Final V3 evaluation requires one valid result per AB/BA task")
        complete[task["id"]] = successful[0]
    billing_path = _billing_path(results)
    money = _validate_billing(packet, results, read_json(billing_path))
    target = run.file("judge/results.import.json")
    value = {"packet_id": packet["packet_id"], "labels": complete, "accounting": accounting, "billing": money}
    write_json(target, value)
    return run.publish("judge-import", [target], [run.receipt_path("judge-export"), Path(results), billing_path],
                       {"valid_judgments": len(complete), **accounting, **money})


def score_comparison(c, comparison, prompts, labels, *, bootstrap_seed):
    orders = c["evaluation"]["answer_orders"]
    per_prompt, raw = [], []
    for i, prompt in enumerate(prompts):
        scores = [labels[f"{comparison['name']}:{i}:{order}"] for order in orders]
        if any(type(v) not in (int, float) or v not in (0, 0.5, 1) for v in scores):
            raise ValueError("Invalid canonical preference score")
        raw.extend(scores)
        per_prompt.append({"id": prompt["id"], "source": _source_name(prompt),
                           "scores": dict(zip(orders, scores)), "score": float(np.mean(scores))})
    values = np.array([r["score"] for r in per_prompt], dtype=float)
    draws = c["evaluation"]["bootstrap_samples"]
    if draws:
        rng = np.random.default_rng(bootstrap_seed)
        boot = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
    else:
        interval = None
    subsets = {}
    for key in ("helpful", "harmless"):
        selected = [r["score"] for r in per_prompt if str(r["source"]).startswith(key)]
        if selected:
            subsets[key] = {"prompts": len(selected), "win_rate": float(np.mean(selected))}
    for source in sorted({str(r["source"]) for r in per_prompt}):
        selected = [r["score"] for r in per_prompt if str(r["source"]) == source]
        subsets[source] = {"prompts": len(selected), "win_rate": float(np.mean(selected))}
    return {**comparison, "prompts": len(values), "win_rate": float(values.mean()), "ci95": interval,
            "standard_error": float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None,
            "raw_judgment_win_fraction": raw.count(1)/len(raw), "raw_judgment_tie_fraction": raw.count(0.5)/len(raw),
            "raw_judgment_loss_fraction": raw.count(0)/len(raw),
            "position_disagreement_rate": float(np.mean([len(set(r["scores"].values())) > 1 for r in per_prompt])),
            "subsets": subsets, "per_prompt": per_prompt}


def aggregate(c):
    run = Run(c)
    if run.completed("aggregate"):
        return run.require("aggregate")
    run.require("judge-import")
    packet = _bound_packet(c, run)
    imported = read_json(run.file("judge/results.import.json"))
    if imported["packet_id"] != packet["packet_id"]:
        raise ValueError("Imported V3 labels belong to another packet")
    prompts, sources = _sources(c, run)
    comparisons = [score_comparison(c, comp, prompts, imported["labels"],
                                   bootstrap_seed=np.random.SeedSequence([c["evaluation"]["seed"], i, 7781]))
                   for i, comp in enumerate(c["evaluation"]["comparisons"])]
    value = {"protocol": "hh_finite_pool_v3", "selection": run.selection(), "seed": c["study"]["seed"],
             "judge": packet["settings"], "comparisons": comparisons,
             "generation": {method: _generation_metrics(records) for method, records in sources.items()},
             "accounting": imported["accounting"], "billing": imported["billing"],
             "uncertainty_scope": "Prompt bootstrap conditional on one training seed; not across-seed uncertainty"}
    target = run.file("summaries/results.json")
    write_json(target, value)
    csv_path = run.file("summaries/results.csv")
    stream = io.StringIO(newline="")
    fields = ["name", "left", "right", "primary", "prompts", "win_rate", "standard_error", "ci95", "position_disagreement_rate"]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows({k: row.get(k) for k in fields} for row in comparisons)
    if csv_path.exists():
        if csv_path.read_text(encoding="utf-8") != stream.getvalue().replace("\r\n", "\n"):
            raise FileExistsError("Preserve different V3 CSV")
    else:
        with csv_path.open("x", encoding="utf-8", newline="") as target_stream:
            target_stream.write(stream.getvalue())
    return run.publish("aggregate", [target, csv_path], [run.receipt_path("judge-import")],
                       {"completed_comparisons": len(comparisons), "valid_judgments": len(imported["labels"])})


def run_local(packet_path, results, ledger, *, budget_usd=None, workers=8, execute=False, transport=None):
    """Only this explicit local command can dispatch paid V3 judgments."""
    from prism.api_budget import BudgetTransport, USDBudget
    from prism.budgeted_judge import owner_lock
    from prism.local_judge import _load_packet, run_packet
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("V3 OpenAI Judge runs locally, never on cluster compute nodes")
    packet = _load_packet(packet_path)
    if packet.get("v3", {}).get("protocol") != "hh_finite_pool_v3" or packet["role"] != "evaluation":
        raise ValueError("Expected a V3 evaluation-only packet")
    if not execute:
        return {"packet_id": packet["packet_id"], "tasks": len(packet["tasks"]), "http_requests_sent": 0,
                "requires": "--execute, explicit positive USD cap, isolated v3-*.sqlite ledger"}
    if type(budget_usd) not in (int, float) or not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError("V3 evaluation requires an explicit positive USD cap")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("V3 workers must be in 1..16")
    ledger = Path(ledger).resolve()
    if not ledger.name.startswith("v3-") or ledger.suffix != ".sqlite":
        raise ValueError("Use an isolated v3-*.sqlite ledger; V2 ledger reuse is forbidden")
    marker = ledger.with_name(ledger.name + ".v3.json")
    contract = {"protocol": "v3_judge_ledger_v1", "packet_id": packet["packet_id"],
                "config_sha256": packet["config_sha256"], "budget_usd": float(budget_usd)}
    with owner_lock(str(ledger) + ".lock"):
        if ledger.exists() and (not marker.exists() or read_json(marker) != contract):
            raise ValueError("Existing USD ledger is not owned by this V3 packet/cap")
        write_json(marker, contract)
        budget = USDBudget(ledger, total_usd=budget_usd, purposes={"evaluation": budget_usd})
        try:
            result = run_packet(packet_path, results, execute=True,
                                transport=BudgetTransport(budget, "evaluation", transport), workers=workers)
        finally:
            _export_billing(budget, packet, results)
        return {**result, "ledger": budget.state()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run-local"])
    parser.add_argument("--packet", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args()
    print(json.dumps(run_local(a.packet, a.results, a.ledger, budget_usd=a.budget_usd,
                               workers=a.workers, execute=a.execute), ensure_ascii=False))


if __name__ == "__main__":
    main()
