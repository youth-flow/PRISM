"""Shared-reference, role-specific preference and official benchmark evaluation."""
import csv
import gc
import json
from pathlib import Path

import numpy as np

from .benchmarks import benchmark_rows, score_benchmark, validate_benchmark_backend
from .io import cell, dump, fresh, prompt_rows, receipt, root, rows, verify_receipt, write_rows
from .judges import evaluation_call_seed, judge_settings, make_judge


def _reference(c):
    return root(c)/"reference"


def _release(c):
    gc.collect()
    if c["runtime"]["device"] == "cuda":
        import torch
        torch.cuda.empty_cache()


def _benchmark_prompt_limit(c):
    return c["benchmark"].get("max_prompt_tokens", c["data"]["model_context_tokens"]-c["benchmark"]["max_new_tokens"]-1) if "model_context_tokens" in c.get("data", {}) else c["benchmark"].get("max_prompt_tokens", c["candidate"]["max_prompt_tokens"])


def _generate(c, model, tokenizer, inputs, cap, temperature, offset=0, kl=False, prompt_limit=None):
    from .hf import generate, prompt_ids, sequence_stats
    result = []
    for i, record in enumerate(inputs):
        ids = prompt_ids(tokenizer, record["messages"], c["candidate"]["max_prompt_tokens"] if prompt_limit is None else prompt_limit)
        response = generate(model, tokenizer, ids, cap, c["evaluation"]["seed"]+offset+i, temperature=temperature)
        if not response:
            raise ValueError("Evaluation generated an empty token sequence")
        row = {"id": record["id"], "messages": record["messages"], "response": tokenizer.decode(response, skip_special_tokens=True),
               "response_tokens": response, "response_length": len(response),
               "truncated": bool(len(response) >= cap and response[-1] != tokenizer.eos_token_id)}
        if kl:
            _, logp = sequence_stats(model, ids, response, tokenizer.eos_token_id)
            with model.disable_adapter():
                _, ref_logp = sequence_stats(model, ids, response, tokenizer.eos_token_id)
            row["sampled_sequence_log_ratio"] = float(logp-ref_logp)
        result.append(row)
    return result


def _aligned(data, expected, name, *, require_kl=False):
    if [r.get("id") for r in data] != [r["id"] for r in expected]:
        raise ValueError(f"{name} response IDs/order do not match the fixed evaluation prompts")
    for row, prompt in zip(data, expected):
        if row.get("messages") != prompt["messages"] or not isinstance(row.get("response"), str):
            raise ValueError(f"{name} response conversation does not match its input")
        if type(row.get("response_length")) is not int or row["response_length"] < 1:
            raise ValueError(f"{name} requires a positive response_length")
        if require_kl and (type(row.get("sampled_sequence_log_ratio")) not in (int, float) or not np.isfinite(row["sampled_sequence_log_ratio"])):
            raise ValueError(f"{name} requires a finite sampled sequence log ratio")
    return data


def _sources(c, budget, seed):
    """Validate raw generation artifacts as well as the dependency receipt files."""
    expected, ref = prompt_rows(c["choices"]["prompts"]["evaluation"]), _reference(c)
    dependencies = [ref/"reference.receipt.json"]
    verify_receipt(dependencies[0])
    sources = {"sft": _aligned(rows(ref/"evaluation.responses.jsonl"), expected, "sft")}
    for method in c["methods"]:
        out = cell(c, method, budget, seed)
        generated, policy = out/"evaluation.generate.receipt.json", out/"policy.receipt.json"
        if not generated.is_file():
            raise RuntimeError(f"Generate all matched policies before judging: missing {method}/n{budget}/seed{seed}")
        verify_receipt(policy)
        verify_receipt(generated)
        sources[method] = _aligned(rows(out/"evaluation.responses.jsonl"), expected, method, require_kl=True)
        dependencies.extend((generated, policy))
    return expected, sources, dependencies


def evaluate_reference(c):
    """Generate the single shared SFT response set and its official benchmark score."""
    validate_benchmark_backend(c)  # Fail before even importing/loading GPU models.
    out = _reference(c)
    out.mkdir(parents=True, exist_ok=True)
    done = out/"reference.receipt.json"
    if done.exists():
        verify_receipt(done)
        return json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    artifacts = [out/name for name in ("evaluation.responses.jsonl", "benchmark.responses.jsonl", "benchmark.metrics.json", "metrics.json")]
    fresh(*artifacts)
    from .hf import load_lm, tokenizer
    t, model = tokenizer(c), load_lm(c)
    try:
        evaluated = _generate(c, model, t, prompt_rows(c["choices"]["prompts"]["evaluation"]), c["evaluation"]["max_new_tokens"], 1.0, offset=1_000_000)
        benchmark = _generate(c, model, t, benchmark_rows(c), c["benchmark"]["max_new_tokens"], c["benchmark"]["temperature"], offset=2_000_000, prompt_limit=_benchmark_prompt_limit(c))
    finally:
        del model
        _release(c)
    scored = score_benchmark(c, benchmark)
    result = {"method": "sft", "evaluation_prompts": len(evaluated), "sequence_kl_mc": 0.0,
              "mean_response_tokens": float(np.mean([r["response_length"] for r in evaluated])),
              "truncation_rate": float(np.mean([r["truncated"] for r in evaluated])),
              "benchmark_name": c["choices"]["benchmark"]["name"], "benchmark_metric": scored["metric"],
              "benchmark_score": scored["score"], "benchmark_metrics": scored["metrics"]}
    write_rows(artifacts[0], evaluated)
    write_rows(artifacts[1], benchmark)
    dump(artifacts[2], scored)
    dump(artifacts[3], result)
    receipt(done, artifacts+[root(c)/"experiment.json"])
    return result


def evaluate_generate(c, method, budget, seed):
    validate_benchmark_backend(c)
    out, ref = cell(c, method, budget, seed), _reference(c)
    verify_receipt(ref/"reference.receipt.json")
    verify_receipt(out/"policy.receipt.json")
    done = out/"evaluation.generate.receipt.json"
    if done.exists():
        return verify_receipt(done)
    artifacts = [out/name for name in ("evaluation.responses.jsonl", "benchmark.responses.jsonl", "benchmark.metrics.json")]
    fresh(*artifacts)
    from peft import PeftModel
    from .hf import load_lm, tokenizer
    t = tokenizer(c)
    actor = PeftModel.from_pretrained(load_lm(c), str(out/"policy"), is_trainable=False).eval()
    try:
        evaluated = _generate(c, actor, t, prompt_rows(c["choices"]["prompts"]["evaluation"]), c["evaluation"]["max_new_tokens"], 1.0, kl=True)
        benchmark = _generate(c, actor, t, benchmark_rows(c), c["benchmark"]["max_new_tokens"], c["benchmark"]["temperature"], offset=2_000_000, prompt_limit=_benchmark_prompt_limit(c))
    finally:
        del actor
        _release(c)
    write_rows(artifacts[0], evaluated)
    write_rows(artifacts[1], benchmark)
    dump(artifacts[2], score_benchmark(c, benchmark))
    receipt(done, artifacts+[out/"policy.receipt.json", ref/"reference.receipt.json"], method=method, budget=budget, seed=seed)
    return {"evaluation_prompts": len(evaluated), "benchmark_prompts": len(benchmark)}


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None


def _comparison(c, comp, data, complete, budget, seed, index):
    orders = c["evaluation"]["answer_orders"]
    per_prompt = [{"id": row["id"], "order_scores": {order: complete[(comp["name"], i, order)]["label"] for order in orders}}
                  for i, row in enumerate(data)]
    for row in per_prompt:
        row["score"] = float(np.mean(list(row["order_scores"].values())))
    values = [r["score"] for r in per_prompt]
    mean, se = _mean_se(values)
    draws = c["evaluation"].get("bootstrap_samples", 2000)
    interval = None
    if draws:
        rng = np.random.default_rng(np.random.SeedSequence([c["evaluation"]["seed"], budget, seed, index, 443]))
        boot = np.asarray(values)[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
        interval = [float(v) for v in np.quantile(boot, [0.025, 0.975])]
    return {**comp, "win_rate": mean, "standard_error": se, "ci95": interval,
            "position_disagreement_rate": float(np.mean([len(set(r["order_scores"].values())) > 1 for r in per_prompt])) if len(orders) > 1 else None,
            "per_prompt": per_prompt}


def evaluate_judge(c, method, budget, seed, *, judge_factory=None):
    draws = c["evaluation"].get("bootstrap_samples", 2000)
    if type(draws) is not int or draws < 0:
        raise ValueError("evaluation.bootstrap_samples must be a nonnegative integer")
    out = cell(c, method, budget, seed)
    data, sources, dependencies = _sources(c, budget, seed)
    labels = verify_receipt(out/"labels.receipt.json")
    done = out/"evaluation.receipt.json"
    if done.exists():
        verify_receipt(done)
        return json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    owned = [(i, comp) for i, comp in enumerate(c["evaluation"]["comparisons"])
             if (comp["left"] if comp["left"] != "sft" else comp["right"]) == method]
    allowed = {(comp["name"], i, order) for _, comp in owned for i in range(len(data)) for order in c["evaluation"]["answer_orders"]}
    path = out/"evaluation.judge_calls.jsonl"
    previous = rows(path) if path.exists() else []
    complete = {}
    for row in previous:
        key = (row.get("comparison"), row.get("index"), row.get("order"))
        if key not in allowed or row.get("prompt_id") != data[key[1]]["id"] or row.get("flip") != (key[2] == "BA"):
            raise ValueError("Evaluation call log does not match the configured comparison/prompt/order")
        if row.get("label") is not None:
            if type(row["label"]) not in (int, float) or row["label"] not in (0, 0.5, 1) or key in complete:
                raise ValueError("Evaluation log has an invalid or duplicate completed judgment")
            complete[key] = row
    judge, settings = None, judge_settings(c, "evaluation")
    path.touch(exist_ok=True)
    for comparison_index, comp in owned:
        for i, prompt in enumerate(data):
            for order_index, order in enumerate(c["evaluation"]["answer_orders"]):
                key = (comp["name"], i, order)
                if key in complete:
                    continue
                attempts = sum((r["comparison"], r["index"], r["order"]) == key for r in previous)
                for attempt in range(attempts, settings["max_attempts"]):
                    if judge is None:
                        judge = (judge_factory or make_judge)(c, "evaluation")
                    call_seed = evaluation_call_seed(c, budget, seed, comparison_index, i, order_index, attempt)
                    result = judge.call(prompt["messages"], sources[comp["left"]][i]["response"], sources[comp["right"]][i]["response"], call_seed, order == "BA")
                    # Adapter label is already canonical-left even for BA.
                    if result.get("label") is not None and (type(result["label"]) not in (int, float) or result["label"] not in (0, 0.5, 1)):
                        result.update(label=None, error={"type": "invalid_canonical_label"})
                    result = {**result, "comparison": comp["name"], "index": i, "prompt_id": prompt["id"], "order": order, "attempt": attempt}
                    with path.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
                        file.flush()
                    previous.append(result)
                    if result["label"] is not None:
                        complete[key] = result
                        break
                if key not in complete:
                    raise RuntimeError(f"Evaluation Judge failed at {key}; every attempted call is logged")
    comparisons = [_comparison(c, comp, data, complete, budget, seed, i) for i, comp in owned]
    own = sources[method]
    kl, kl_se = _mean_se([r["sampled_sequence_log_ratio"] for r in own])
    scored = json.loads((out/"benchmark.metrics.json").read_text(encoding="utf-8"))
    versus_sft = [r["win_rate"] if r["left"] == method else 1-r["win_rate"] for r in comparisons if {r["left"], r["right"]} == {method, "sft"}]
    result = {"method": method, "budget": budget, "seed": seed, "training_judge_calls": labels["actual_judge_calls"],
              "evaluation_judge_calls": len(previous), "evaluation_prompts": len(data), "win_rate_vs_sft": versus_sft[0] if versus_sft else None,
              "sequence_kl_mc": kl, "sequence_kl_mc_standard_error": kl_se,
              "mean_response_tokens": float(np.mean([r["response_length"] for r in own])),
              "truncation_rate": float(np.mean([r.get("truncated", False) for r in own])),
              "benchmark_name": c["choices"]["benchmark"]["name"], "benchmark_metric": scored["metric"],
              "benchmark_score": scored["score"], "benchmark_metrics": scored["metrics"], "comparisons": comparisons}
    dump(out/"metrics.json", result)
    receipt(done, [out/"metrics.json", path, out/"labels.receipt.json"]+dependencies)
    return result


def _csv(path, records):
    if records:
        fields = list(dict.fromkeys(key for record in records for key in record))
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def aggregate(c):
    p, records, missing = root(c), [], []
    for method in c["methods"]:
        for budget in c["budgets"]:
            for seed in c["seeds"]:
                out = cell(c, method, budget, seed)
                if not (out/"evaluation.receipt.json").exists():
                    missing.append(f"{method}/n{budget}/seed{seed}")
                    continue
                _sources(c, budget, seed)
                verify_receipt(out/"labels.receipt.json")
                verify_receipt(out/"evaluation.receipt.json")
                records.append(json.loads((out/"metrics.json").read_text(encoding="utf-8")))
    scalar = [{**{k: v for k, v in r.items() if not isinstance(v, (dict, list))}, **{f"benchmark_{k}": v for k, v in r["benchmark_metrics"].items()}} for r in records]
    comparisons = [{"owner": r["method"], "budget": r["budget"], "seed": r["seed"], **{k: v for k, v in comp.items() if k not in ("per_prompt", "ci95")},
                    "ci95_lower": comp["ci95"][0] if comp["ci95"] else None, "ci95_upper": comp["ci95"][1] if comp["ci95"] else None}
                   for r in records for comp in r["comparisons"]]
    _csv(p/"results.csv", scalar)
    _csv(p/"comparisons.csv", comparisons)
    groups = []
    for method in c["methods"]:
        for budget in c["budgets"]:
            group = [r for r in records if r["method"] == method and r["budget"] == budget]
            if group:
                result = {"method": method, "budget": budget, "completed_seeds": len(group)}
                for key in ("win_rate_vs_sft", "sequence_kl_mc", "benchmark_score", "mean_response_tokens", "truncation_rate"):
                    values = [r[key] for r in group if r[key] is not None]
                    result[key+"_mean"] = float(np.mean(values)) if values else None
                    result[key+"_seed_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                groups.append(result)
    paths = list((p/"label_streams").glob("*/seed*/judge_calls.jsonl")) if c["sampling"]["budget_coupling"] == "nested_prefix" else [cell(c, m, n, s)/"judge_calls.jsonl" for m in c["methods"] for n in c["budgets"] for s in c["seeds"]]
    streams = [rows(path) for path in paths if path.exists()]
    evaluation_logs = [rows(path) for method in c["methods"] for budget in c["budgets"] for seed in c["seeds"]
                       if (path := cell(c, method, budget, seed)/"evaluation.judge_calls.jsonl").exists()]
    accounting = {"budget_coupling": c["sampling"]["budget_coupling"], "unique_training_calls": sum(map(len, streams)),
                  "unique_valid_observations": sum(len({r["draw_id"] for r in stream if r.get("label") is not None}) for stream in streams),
                  "evaluation_judge_calls": sum(map(len, evaluation_logs))}
    reference = _reference(c)
    sft = None
    if (reference/"reference.receipt.json").exists():
        verify_receipt(reference/"reference.receipt.json")
        sft = json.loads((reference/"metrics.json").read_text(encoding="utf-8"))
    dump(p/"summary.json", {"complete": not missing, "missing_cells": missing, "groups": groups, "sft": sft,
                           "training_label_accounting": accounting,
                           "note": "AB/BA is averaged within each prompt before uncertainty estimation. Training-seed variability is separate; finite-sample MC KL may be negative."})
    return {"completed": len(records), "missing": len(missing), "output": str(p/"summary.json")}
