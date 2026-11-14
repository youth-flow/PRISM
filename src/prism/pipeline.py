"""Stage boundaries are immutable; only Judge requests have append-only resume."""
import json
from pathlib import Path

import numpy as np

from .design import edges, empirical, frank_wolfe, proxy
from .io import cell, digest, dump, fresh, prompt_rows, receipt, root, rows, verify_receipt, write_rows
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
        if method not in c["methods"]:
            continue
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
    receipt(out / "queries.receipt.json", [out / "queries.jsonl", p / "design.receipt.json"], iid_with_replacement=True, budget=budget, unique_edges=len(set(draws.tolist())), budget_coupling=c.get("sampling", {}).get("budget_coupling", "independent"))


def _freeze_rows(path, data):
    """Allow recovery after a snapshot was written but its receipt was not."""
    if path.exists():
        if rows(path) != data:
            raise ValueError(f"Immutable label snapshot changed: {path}")
    else:
        write_rows(path, data)


def label(c, method, budget, seed, *, judge_factory=None):
    from .judges import judge_settings, make_judge, training_call_seed
    p, out = root(c), cell(c, method, budget, seed)
    verify_receipt(p / "pool.receipt.json")
    verify_receipt(p / "design.receipt.json")
    verify_receipt(out / "queries.receipt.json")
    if (out / "labels.receipt.json").exists():
        verify_receipt(out / "labels.receipt.json")
        return
    queries = rows(out / "queries.jsonl")
    pool = {(r["prompt_index"], r["candidate"]): r for r in rows(p / "pool.jsonl")}
    coupling = c.get("sampling", {}).get("budget_coupling", "independent")
    calls = (p / "label_streams" / method / f"seed{seed}" / "judge_calls.jsonl"
             if coupling == "nested_prefix" else out / "judge_calls.jsonl")
    calls.parent.mkdir(parents=True, exist_ok=True)
    previous = rows(calls) if calls.exists() else []
    initial_call_count = len(previous)
    expected = {q["draw_id"]: q for q in queries}
    for r in previous:
        if r["draw_id"] in expected and any(r.get(key) != val for key, val in expected[r["draw_id"]].items()):
            raise ValueError("Shared label stream does not match the sampled query prefix")
    completed = {r["draw_id"]: r for r in previous if r["label"] is not None}
    judge = None
    for query in queries:
        k = query["draw_id"]
        if k in completed:
            continue
        a, b = pool[(query["prompt_index"], query["a"])], pool[(query["prompt_index"], query["b"])]
        attempts = sum(r["draw_id"] == k for r in previous)
        for attempt in range(attempts, judge_settings(c, "training")["max_attempts"]):
            # Prefix budgets share the SAME observation. Distinct draw IDs, even
            # for a repeated edge, always receive separate calls.
            call_seed = training_call_seed(seed, method, budget, k, attempt, coupling)
            flip = bool(np.random.default_rng(call_seed).integers(2))
            if judge is None:
                judge = (judge_factory or make_judge)(c, "training")
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
        count = sum(k in completed for k in expected)
        if count % max(1, budget // 20) == 0 or count == budget:
            print(f"label {count}/{budget}", flush=True)
    if any(k not in completed for k in expected):
        raise RuntimeError("Label count mismatch")
    prefix_calls = [r for r in previous if r["draw_id"] in expected]
    snapshot = out / "judge_calls.jsonl"
    if coupling == "nested_prefix":
        _freeze_rows(snapshot, prefix_calls)
    _freeze_rows(out / "labels.jsonl", [completed[k] for k in range(budget)])
    receipt(out / "labels.receipt.json", [out / "labels.jsonl", snapshot, out / "queries.receipt.json"], valid_labels=budget, actual_judge_calls=len(prefix_calls), invalid_calls=sum(r["label"] is None for r in prefix_calls), budget_coupling=coupling, source_call_stream=str(calls), incremental_calls_this_invocation=len(previous)-initial_call_count)


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
    from huggingface_hub import list_repo_files, snapshot_download
    assets = [("sft", c["choices"]["sft"])]
    if c.get("schema_version") == 2:
        assets.extend((f"{role}_judge", settings) for role, settings in c["judges"].items() if settings["backend"] == "hf")
    else:
        assets.append(("judge", c["choices"]["judge"]))
    for role, d in assets:
        if Path(d["model"]).is_dir():
            print(f"Using supplied local {role} snapshot: {d['model']}")
        else:
            files = list_repo_files(d["model"], revision=d["revision"])
            has_safetensors = any(name.endswith(".safetensors") for name in files)
            has_pytorch_bin = any(name.endswith(".bin") for name in files)
            if not has_safetensors and not has_pytorch_bin:
                raise RuntimeError(f"No supported model weights found for pinned {role} snapshot: {d['model']}@{d['revision']}")
            # Prefer safetensors when the repository publishes it, but do not
            # discard the only usable weights from repositories which still
            # publish PyTorch .bin files (as OLMo-2-0425-1B-SFT does).
            ignored = ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.gguf"]
            if has_safetensors:
                ignored.append("*.bin")
            print(snapshot_download(d["model"], revision=d["revision"], ignore_patterns=ignored))
    if c["choices"]["benchmark"].get("kind") == "ifeval":
        from .benchmarks import stage_ifeval_resources, stage_ifeval_sources
        b = c["choices"]["benchmark"]
        stage_ifeval_sources(b["official_source_path"], b["official_source_revision"])
        stage_ifeval_resources(c)


def prepare_data(c):
    from .data import prepare_data as prepare
    return prepare(c)


def model_smoke(c):
    """Selected real SFT/Judge preflight; query costs are logged separately."""
    import gc
    import torch
    from trl.trainer.utils import get_reward
    from .hf import FrozenReward, generate, load_lm, prompt_ids, sequence_stats, tokenizer
    from .judges import judge_settings, make_judge
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
    if c.get("schema_version") == 2:
        dump(p / "model-smoke.json", {"eos_adapter_errors": errors, "sequence_logps": logps, "passed": True, "judge_calls": [], "note": "GPU feature preflight only; no API Judge is invoked by model-smoke."})
        return
    judge = make_judge(c, "training")
    attempts = []
    for attempt in range(judge_settings(c, "training")["max_attempts"]):
        answer = judge.call(r["messages"], texts[0], texts[1], 1234+attempt, bool(attempt % 2))
        attempts.append(answer)
        dump(p / "model-smoke.json", {"eos_adapter_errors": errors, "sequence_logps": logps, "judge_calls": attempts, "passed": answer["label"] is not None})
        if answer["label"] is not None:
            return
    raise RuntimeError("Selected Judge failed strict A/B preflight; inspect model-smoke.json")
