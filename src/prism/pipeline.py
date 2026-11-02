"""Stage boundaries are immutable; only Judge requests have append-only resume."""
import json
from pathlib import Path

import numpy as np

from .design import edges, empirical, frank_wolfe, proxy
from .io import METHODS, cell, digest, dump, fresh, prompt_rows, receipt, root, rows, verify_receipt, write_rows
from .reward import fit_bt


def rollout(c):
    from .hf import generate, load_lm, prompt_ids, sequence_stats, tokenizer
    p = root(c)
    fresh(p / "pool.npz", p / "pool.jsonl")
    t, m = tokenizer(c), load_lm(c)
    data = prompt_rows(c["choices"]["prompts"]["design"])
    features, logps, records = [], [], []
    for i, r in enumerate(data):
        ids = prompt_ids(t, r["messages"], c["candidate"]["max_prompt_tokens"])
        ff, ll = [], []
        for j in range(c["candidate"]["count"]):
            response = generate(m, t, ids, c["candidate"]["max_new_tokens"], c["candidate"]["seed"] + i * c["candidate"]["count"] + j)
            phi, logp = sequence_stats(m, ids, response, t.eos_token_id)
            ff.append(phi)
            ll.append(logp)
            records.append({"prompt_index": i, "candidate": j, "prompt_id": r["id"], "messages": r["messages"], "prompt_tokens": ids, "response_tokens": response, "response": t.decode(response, skip_special_tokens=True), "forced_feature_eos": response[-1] != t.eos_token_id})
        features.append(ff)
        logps.append(ll)
        print(f"rollout {i+1}/{len(data)}", flush=True)
    np.savez_compressed(p / "pool.npz", phi=np.asarray(features, dtype=np.float32), logp=np.asarray(logps, dtype=np.float64))
    write_rows(p / "pool.jsonl", records)
    receipt(p / "pool.receipt.json", [p / "pool.npz", p / "pool.jsonl", p / "experiment.json"], feature="frozen_sft_final_layer_eos", sequence_logp="all_sampled_response_tokens_including_sampled_eos", feature_only_eos_append=True)


def design(c):
    p = root(c)
    verify_receipt(p / "pool.receipt.json")
    fresh(p / "design.npz")
    with np.load(p / "pool.npz", allow_pickle=False) as a:
        phi, logp = a["phi"].astype(np.float64), a["logp"]
    b, probabilities, K = proxy(phi, logp, c["design"]["surrogate_ridge"])
    index, v = edges(phi)
    q0 = empirical(index)
    distributions, reports = {"empirical": q0}, {}
    for method, target in (("a_optimal", np.eye(v.shape[1])), ("prism", K)):
        print(f"design {method}: {len(v)} edges, dimension {v.shape[1]}", flush=True)
        q, report = frank_wolfe(v, target, q0, c["design"]["matrix_ridge"], c["design"]["fw_max_iter"], c["design"]["fw_relative_gap"])
        distributions[method], reports[method] = q, report
    np.savez_compressed(p / "design.npz", index=index, v=v, b=b, proxy_probabilities=probabilities, K=K, **distributions)
    dump(p / "design.report.json", reports)
    receipt(p / "design.receipt.json", [p / "design.npz", p / "design.report.json", p / "pool.receipt.json"])
    if not all(r["converged"] for r in reports.values()):
        print("WARNING: FW gap tolerance not reached. Inspect design.report.json before collecting labels.", flush=True)


def sample(c, method, budget, seed):
    p, out = root(c), cell(c, method, budget, seed)
    verify_receipt(p / "pool.receipt.json")
    verify_receipt(p / "design.receipt.json")
    reports = json.loads((p / "design.report.json").read_text(encoding="utf-8"))
    if not all(r["converged"] for r in reports.values()):
        raise RuntimeError("FW did not converge for all methods. Resolve the design before spending labels.")
    fresh(out / "queries.jsonl")
    with np.load(p / "design.npz", allow_pickle=False) as a:
        q, index = a[method], a["index"]
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(q), size=budget, replace=True, p=q)
    write_rows(out / "queries.jsonl", [{"draw_id": k, "edge_id": int(e), "prompt_index": int(index[e, 0]), "a": int(index[e, 1]), "b": int(index[e, 2]), "probability": float(q[e])} for k, e in enumerate(draws)])
    receipt(out / "queries.receipt.json", [out / "queries.jsonl", p / "design.receipt.json"], iid_with_replacement=True, budget=budget, unique_edges=len(set(draws.tolist())))


def label(c, method, budget, seed):
    from .hf import Judge
    p, out = root(c), cell(c, method, budget, seed)
    verify_receipt(p / "pool.receipt.json")
    verify_receipt(p / "design.receipt.json")
    verify_receipt(out / "queries.receipt.json")
    if (out / "labels.receipt.json").exists():
        verify_receipt(out / "labels.receipt.json")
        return
    queries = rows(out / "queries.jsonl")
    pool = {(r["prompt_index"], r["candidate"]): r for r in rows(p / "pool.jsonl")}
    calls = out / "judge_calls.jsonl"
    previous = rows(calls) if calls.exists() else []
    completed = {r["draw_id"]: r for r in previous if r["label"] is not None}
    judge = Judge(c)
    for query in queries:
        k = query["draw_id"]
        if k in completed:
            continue
        a, b = pool[(query["prompt_index"], query["a"])], pool[(query["prompt_index"], query["b"])]
        attempts = sum(r["draw_id"] == k for r in previous)
        for attempt in range(attempts, c["judge"]["max_attempts"]):
            # Different methods/budgets have separate calls; never reuse an edge label.
            call_seed = int(np.random.SeedSequence([seed, budget, METHODS.index(method), k, attempt, 991]).generate_state(1)[0])
            flip = bool(np.random.default_rng(call_seed).integers(2))
            result = {**query, "attempt": attempt, **judge.call(a["messages"], a["response"], b["response"], call_seed, flip)}
            with calls.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            previous.append(result)
            if result["label"] is not None:
                completed[k] = result
                break
        if k not in completed:
            raise RuntimeError(f"Judge format failure at draw {k}; attempts logged. No substituted label.")
        if len(completed) % max(1, budget // 20) == 0 or len(completed) == budget:
            print(f"label {len(completed)}/{budget}", flush=True)
    if len(completed) != budget:
        raise RuntimeError("Label count mismatch")
    write_rows(out / "labels.jsonl", [completed[k] for k in range(budget)])
    receipt(out / "labels.receipt.json", [out / "labels.jsonl", calls, out / "queries.receipt.json"], valid_labels=budget, actual_judge_calls=len(previous), invalid_calls=sum(r["label"] is None for r in previous))


def reward(c, method, budget, seed):
    p, out = root(c), cell(c, method, budget, seed)
    verify_receipt(p / "pool.receipt.json")
    verify_receipt(p / "design.receipt.json")
    verify_receipt(out / "queries.receipt.json")
    verify_receipt(out / "labels.receipt.json")
    fresh(out / "reward.npz")
    labeled = rows(out / "labels.jsonl")
    if len(labeled) != budget or [r["draw_id"] for r in labeled] != list(range(budget)):
        raise ValueError("Labels must contain one label per sampled draw in order")
    with np.load(p / "design.npz", allow_pickle=False) as a:
        v = a["v"][[r["edge_id"] for r in labeled]]
    w, report = fit_bt(v, [r["label"] for r in labeled], **c["reward"])
    np.savez(out / "reward.npz", w=w)
    dump(out / "reward.report.json", report)
    receipt(out / "reward.receipt.json", [out / "reward.npz", out / "reward.report.json", out / "labels.receipt.json"])


def policy(c, method, budget, seed):
    from .hf import train_ppo
    out = cell(c, method, budget, seed)
    verify_receipt(out / "labels.receipt.json")
    verify_receipt(out / "reward.receipt.json")
    fresh(out / "policy")
    with np.load(out / "reward.npz", allow_pickle=False) as a:
        w = a["w"]
    result = train_ppo(c, out / "policy", w, prompt_rows(c["choices"]["prompts"]["policy"]), seed)
    dump(out / "policy.report.json", result)
    artifacts = list((out / "policy").glob("*.safetensors")) + [out / "policy" / "adapter_config.json"]
    receipt(out / "policy.receipt.json", artifacts + [out / "reward.receipt.json", out / "policy.report.json"])


def stage_assets(c):
    from huggingface_hub import snapshot_download
    for role in ("sft", "judge"):
        d = c["choices"][role]
        if Path(d["model"]).is_dir():
            print(f"Using supplied local {role} snapshot: {d['model']}")
        else:
            print(snapshot_download(d["model"], revision=d["revision"], ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.gguf"]))


def model_smoke(c):
    """Selected real SFT/Judge preflight; query costs are logged separately."""
    import gc
    import torch
    from trl.trainer.utils import get_reward
    from .hf import FrozenReward, Judge, generate, load_lm, prompt_ids, sequence_stats, tokenizer
    p = root(c)
    fresh(p / "model-smoke.json")
    t, m = tokenizer(c), load_lm(c)
    r = prompt_rows(c["choices"]["prompts"]["design"])[0]
    ids = prompt_ids(t, r["messages"], c["candidate"]["max_prompt_tokens"])
    responses, features, logps = [], [], []
    for j in range(2):
        response = generate(m, t, ids, c["candidate"]["max_new_tokens"], c["candidate"]["seed"]+j)
        f, ell = sequence_stats(m, ids, response, t.eos_token_id)
        responses.append(response)
        features.append(f)
        logps.append(ell)
    w = np.linspace(-0.01, 0.01, len(features[0]))
    adapter = FrozenReward(m.base_model, t.eos_token_id, w).to(c["runtime"]["device"])
    errors = []
    for tokens, f in zip(responses, features):
        query = torch.tensor([ids+tokens], device=c["runtime"]["device"])
        _, reward, _ = get_reward(adapter, query, t.pad_token_id, len(ids))
        errors.append(abs(float(reward[0]) - float(f @ w)))
    if max(errors) > 1e-3:
        raise RuntimeError(f"Real-model EOS reward adapter mismatch: {errors}")
    texts = [t.decode(y, skip_special_tokens=True) for y in responses]
    del adapter, m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    judge = Judge(c)
    attempts = []
    for attempt in range(c["judge"]["max_attempts"]):
        answer = judge.call(r["messages"], texts[0], texts[1], 1234+attempt, bool(attempt % 2))
        attempts.append(answer)
        dump(p / "model-smoke.json", {"eos_adapter_errors": errors, "sequence_logps": logps, "judge_calls": attempts, "passed": answer["label"] is not None})
        if answer["label"] is not None:
            return
    raise RuntimeError("Selected Judge failed strict A/B preflight; inspect model-smoke.json")
