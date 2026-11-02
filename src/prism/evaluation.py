"""Heldout evaluation, independent of the trained reward head."""
import gc
import json

import numpy as np

from .io import METHODS, cell, dump, fresh, prompt_rows, receipt, root, rows, verify_receipt, write_rows


def exact_match(response, answer):
    # Explicit and modest benchmark contract; no hidden answer extraction heuristic.
    return " ".join(response.strip().casefold().split()) == " ".join(answer.strip().casefold().split())


def evaluate_generate(c, method, budget, seed):
    import torch
    from peft import PeftModel
    from .hf import generate, load_lm, prompt_ids, sequence_stats, tokenizer
    out = cell(c, method, budget, seed)
    verify_receipt(out / "policy.receipt.json")
    fresh(out / "evaluation.responses.jsonl")
    t = tokenizer(c)
    actor = PeftModel.from_pretrained(load_lm(c), str(out / "policy"), is_trainable=False).eval()
    evals = prompt_rows(c["choices"]["prompts"]["evaluation"])
    results = []
    for i, r in enumerate(evals):
        ids = prompt_ids(t, r["messages"], c["candidate"]["max_prompt_tokens"])
        s = c["evaluation"]["seed"] + i
        response = generate(actor, t, ids, c["evaluation"]["max_new_tokens"], s)
        _, logp = sequence_stats(actor, ids, response, t.eos_token_id)
        with actor.disable_adapter():
            _, ref_logp = sequence_stats(actor, ids, response, t.eos_token_id)
            baseline = generate(actor, t, ids, c["evaluation"]["max_new_tokens"], s + 1_000_000)
        results.append({"id": r["id"], "messages": r["messages"], "response": t.decode(response, skip_special_tokens=True), "reference_response": t.decode(baseline, skip_special_tokens=True), "response_tokens": response, "sampled_sequence_log_ratio": logp-ref_logp, "response_length": len(response)})
        print(f"eval generation {i+1}/{len(evals)}", flush=True)
    bench = prompt_rows(c["choices"]["benchmark"]["path"])
    scores = []
    for i, r in enumerate(bench):
        ids = prompt_ids(t, r["messages"], c["candidate"]["max_prompt_tokens"])
        response = generate(actor, t, ids, c["evaluation"]["max_new_tokens"], c["evaluation"]["seed"] + i, temperature=0)
        text = t.decode(response, skip_special_tokens=True)
        scores.append({"id": r["id"], "response": text, "answer": r["answer"], "correct": exact_match(text, r["answer"])})
    write_rows(out / "evaluation.responses.jsonl", results)
    write_rows(out / "benchmark.responses.jsonl", scores)
    receipt(out / "evaluation.generate.receipt.json", [out / "evaluation.responses.jsonl", out / "benchmark.responses.jsonl", out / "policy.receipt.json"])
    del actor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_judge(c, method, budget, seed):
    from .hf import Judge
    out = cell(c, method, budget, seed)
    verify_receipt(out / "evaluation.generate.receipt.json")
    if (out / "evaluation.receipt.json").exists():
        verify_receipt(out / "evaluation.receipt.json")
        return
    data = rows(out / "evaluation.responses.jsonl")
    path = out / "evaluation.judge_calls.jsonl"
    previous = rows(path) if path.exists() else []
    complete = {r["index"]: r for r in previous if r["label"] is not None}
    judge = Judge(c)
    for i, r in enumerate(data):
        if i in complete:
            continue
        attempts = sum(a["index"] == i for a in previous)
        for attempt in range(attempts, c["judge"]["max_attempts"]):
            s = int(np.random.SeedSequence([c["evaluation"]["seed"], i, attempt, 197]).generate_state(1)[0])
            flip = bool(np.random.default_rng(s).integers(2))
            result = {"index": i, "attempt": attempt, **judge.call(r["messages"], r["response"], r["reference_response"], s, flip)}
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            previous.append(result)
            if result["label"] is not None:
                complete[i] = result
                break
        if i not in complete:
            raise RuntimeError(f"Evaluation Judge invalid output: index={i}; inspect logged attempts")
    wins = np.array([complete[i]["label"] for i in range(len(data))])
    ratios = np.array([r["sampled_sequence_log_ratio"] for r in data])
    labels = verify_receipt(out / "labels.receipt.json")
    bench = rows(out / "benchmark.responses.jsonl")
    result = {"method": method, "budget": budget, "seed": seed, "training_judge_calls": labels["actual_judge_calls"], "evaluation_judge_calls": len(previous), "evaluation_prompts": len(data), "win_rate_vs_sft": float(wins.mean()), "win_rate_standard_error": float(wins.std(ddof=1)/np.sqrt(len(wins))) if len(wins)>1 else None, "sequence_kl_mc": float(ratios.mean()), "sequence_kl_mc_standard_error": float(ratios.std(ddof=1)/np.sqrt(len(ratios))) if len(ratios)>1 else None, "mean_response_tokens": float(np.mean([r["response_length"] for r in data])), "benchmark_name": c["choices"]["benchmark"]["name"], "benchmark_metric": "normalized_exact_match", "benchmark_score": float(np.mean([r["correct"] for r in bench]))}
    dump(out / "metrics.json", result)
    receipt(out / "evaluation.receipt.json", [out / "metrics.json", path, out / "evaluation.generate.receipt.json"])
    return result


def aggregate(c):
    import csv
    p = root(c)
    data, missing = [], []
    for method in METHODS:
        for n in c["budgets"]:
            for seed in c["seeds"]:
                out = cell(c, method, n, seed)
                if not (out / "evaluation.receipt.json").exists():
                    missing.append(f"{method}/n{n}/seed{seed}")
                    continue
                verify_receipt(out / "evaluation.receipt.json")
                data.append(json.loads((out / "metrics.json").read_text(encoding="utf-8")))
    if data:
        with (p / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    groups = []
    for method in METHODS:
        for n in c["budgets"]:
            group = [r for r in data if r["method"] == method and r["budget"] == n]
            if not group:
                continue
            g = {"method": method, "budget": n, "completed_seeds": len(group)}
            for metric in ("win_rate_vs_sft", "sequence_kl_mc", "benchmark_score", "training_judge_calls"):
                values = np.array([r[metric] for r in group])
                g[metric + "_mean"] = float(values.mean())
                g[metric + "_seed_std"] = float(values.std(ddof=1)) if len(values)>1 else None
            groups.append(g)
    dump(p / "summary.json", {"complete": not missing, "missing_cells": missing, "groups": groups, "note": "Seed variability and per-prompt uncertainty are separate; MC KL may be negative at finite sample size."})
    return {"completed": len(data), "missing": len(missing), "output": str(p / "summary.json")}
