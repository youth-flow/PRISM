"""Portable API work packets; cluster paths and stage receipts stay on cluster.

Export/import are offline. Only run_packet makes requests on the local host,
using the existing API adapter and an explicit execute flag. Imports replay the
original stages, preserving their label orientation, retries and accounting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Event, Lock

import numpy as np

from .io import cell, digest, dump, load_config, root, rows, verify_receipt
from .judges import APIJudge, _parse, evaluation_call_seed, judge_protocol, judge_settings, training_call_seed


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _call_hash(task, call):
    return _hash([task["messages"], task["left"], task["right"], call["seed"], call["flip"]])


def _tasks(c, stage, method, budget, seed):
    p, out = root(c), cell(c, method, budget, seed)
    tasks, dependencies = [], [p / "experiment.json"]
    role = "training" if stage == "label" else "evaluation"
    attempts = range(judge_settings(c, role)["max_attempts"])
    if stage == "label":
        if c["sampling"]["budget_coupling"] != "independent":
            raise ValueError("Local packets currently require independent budgets")
        dependencies += [p / "pool.receipt.json", p / "design.receipt.json", out / "queries.receipt.json"]
        for path in dependencies[1:]:
            verify_receipt(path)
        reports = json.loads((p / "design.report.json").read_text(encoding="utf-8"))
        if not all(r["converged"] for r in reports.values()):
            raise RuntimeError("FW did not converge; no label packet may be exported")
        pool = {(r["prompt_index"], r["candidate"]): r for r in rows(p / "pool.jsonl")}
        queries = rows(out / "queries.jsonl")
        if len(queries) != budget or [q["draw_id"] for q in queries] != list(range(budget)):
            raise ValueError("Query draw IDs must match the declared label budget")
        for q in queries:
            a, b = pool[(q["prompt_index"], q["a"])], pool[(q["prompt_index"], q["b"])]
            calls = []
            for attempt in attempts:
                s = training_call_seed(seed, method, budget, q["draw_id"], attempt, "independent")
                calls.append({"seed": s, "flip": bool(np.random.default_rng(s).integers(2))})
            tasks.append({"id": str(q["draw_id"]), "messages": a["messages"], "left": a["response"], "right": b["response"], "calls": calls})
        log = out / "judge_calls.jsonl"
        def task_id(row):
            return str(row["draw_id"])
    elif stage == "evaluate-judge":
        from .evaluation_v2 import _sources
        data, sources, deps = _sources(c, budget, seed)
        dependencies += deps + [out / "labels.receipt.json"]
        verify_receipt(out / "labels.receipt.json")
        for ci, comp in enumerate(c["evaluation"]["comparisons"]):
            if (comp["left"] if comp["left"] != "sft" else comp["right"]) != method:
                continue
            for i, prompt in enumerate(data):
                for oi, order in enumerate(c["evaluation"]["answer_orders"]):
                    calls = [{"seed": evaluation_call_seed(c, budget, seed, ci, i, oi, a), "flip": order == "BA"} for a in attempts]
                    tasks.append({"id": f"{comp['name']}:{i}:{order}", "messages": prompt["messages"], "left": sources[comp["left"]][i]["response"], "right": sources[comp["right"]][i]["response"], "calls": calls})
        log = out / "evaluation.judge_calls.jsonl"
        def task_id(row):
            return f"{row['comparison']}:{row['index']}:{row['order']}"
    else:
        raise ValueError("Packet stage must be label or evaluate-judge")
    return role, tasks, dependencies, log, task_id


def export_packet(c, stage, method, budget, seed, destination):
    role, tasks, dependencies, log, task_id = _tasks(c, stage, method, budget, seed)
    prior = [{"task_id": task_id(r), "attempt": r["attempt"], "result": r} for r in (rows(log) if log.exists() else [])]
    packet = {"kind": "prism_local_judge", "version": 1, "stage": stage, "method": method, "budget": budget, "seed": seed,
              "judge_protocol": judge_protocol(role),
              "role": role, "settings": judge_settings(c, role), "config_sha256": _hash(c),
              "input_sha256": {str(p): digest(p) for p in dependencies}, "tasks": tasks, "prior_calls": prior}
    packet["packet_id"] = _hash(packet)
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != packet:
            raise FileExistsError("Refusing to replace a different Judge packet")
    else:
        dump(destination, packet)
    return {"packet_id": packet["packet_id"], "tasks": len(tasks), "prior_calls": len(prior)}


def _load_packet(path):
    packet = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in packet.items() if k != "packet_id"}
    if packet.get("kind") != "prism_local_judge" or packet.get("version") != 1 or _hash(body) != packet.get("packet_id"):
        raise ValueError("Judge packet hash/version mismatch")
    if packet["role"] != ("training" if packet["stage"] == "label" else "evaluation"):
        raise ValueError("Judge packet role/stage mismatch")
    legacy = "preference_ab_v1" if packet["role"] == "training" else "preference_ab_tie_v1"
    if packet.get("judge_protocol", legacy) != judge_protocol(packet["role"]):
        raise ValueError("Judge prompt protocol changed; archive the old packet/results before collecting a new batch")
    return packet


def _records(packet, results):
    tasks = {t["id"]: t for t in packet["tasks"]}
    if len(tasks) != len(packet["tasks"]):
        raise ValueError("Duplicate Judge task ID")
    records = list(packet["prior_calls"])
    if Path(results).exists():
        for row in rows(results):
            if row.get("packet_id") != packet["packet_id"]:
                raise ValueError("Judge results belong to another packet")
            records.append(row)
    indexed = {}
    for row in records:
        task, attempt, result = tasks.get(row["task_id"]), row["attempt"], row["result"]
        if task is None or type(attempt) is not int or not 0 <= attempt < len(task["calls"]):
            raise ValueError("Unknown Judge task/attempt")
        key, call = (task["id"], attempt), task["calls"][attempt]
        legacy = "preference_ab_v1" if packet["role"] == "training" else "preference_ab_tie_v1"
        if result.get("judge_protocol", legacy) != packet.get("judge_protocol", legacy):
            raise ValueError("Judge results use a different prompt protocol")
        if key in indexed or result.get("seed") != call["seed"] or result.get("flip") != call["flip"]:
            raise ValueError("Duplicate or misaligned Judge result")
        label = result.get("label")
        if label is not None and (type(label) not in (int, float) or label not in (0, 0.5, 1) or label != _parse(result.get("raw", ""), packet["role"], call["flip"])):
            raise ValueError("Judge label does not match its raw answer/order")
        if packet["role"] == "training" and label == 0.5:
            raise ValueError("Training Judge cannot return a tie")
        earlier = [indexed.get((task["id"], a)) for a in range(attempt)]
        if any(r is None or r["label"] is not None for r in earlier):
            raise ValueError("Judge attempts are not a contiguous failure prefix")
        indexed[key] = result
    return tasks, indexed


def run_packet(packet_path, results, *, execute=False, transport=None, workers=1):
    if not execute:
        raise RuntimeError("Local API execution requires --execute")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Judge workers must be an integer in 1..16")
    packet = _load_packet(packet_path)
    tasks, indexed = _records(packet, results)
    c = {"execution": {"allow_paid_api_calls": True}, "judges": {packet["role"]: packet["settings"]}}
    judge = APIJudge(c, packet["role"], transport=transport)
    results = Path(results)
    results.parent.mkdir(parents=True, exist_ok=True)
    completed, lock, stopped = 0, Lock(), Event()
    def run_one(task):
        nonlocal completed
        if stopped.is_set():
            return
        previous = [indexed[(task["id"], a)] for a in range(len(task["calls"])) if (task["id"], a) in indexed]
        if any(r["label"] is not None for r in previous):
            with lock:
                completed += 1
            return
        for attempt in range(len(previous), len(task["calls"])):
            if stopped.is_set():
                return
            call = task["calls"][attempt]
            try:
                result = judge.call(task["messages"], task["left"], task["right"], call["seed"], call["flip"])
            except BaseException:
                stopped.set()
                raise
            record = {"packet_id": packet["packet_id"], "task_id": task["id"], "attempt": attempt, "result": result}
            with lock:
                with results.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
            if result["label"] is not None:
                with lock:
                    completed += 1
                break
        else:
            stopped.set()
            raise RuntimeError(f"Judge exhausted attempts for task {task['id']}; all attempts preserved")
        with lock:
            if completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks):
                print(f"local-judge {completed}/{len(tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run_one, tasks.values()))
    return {"packet_id": packet["packet_id"], "completed": completed}


def validate_packet(c, packet_path):
    packet = _load_packet(packet_path)
    if _hash(c) != packet["config_sha256"]:
        raise ValueError("Judge packet configuration changed")
    for path, expected in packet["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Judge packet input changed")
    role, tasks, _, _, _ = _tasks(c, packet["stage"], packet["method"], packet["budget"], packet["seed"])
    if tasks != packet["tasks"] or judge_settings(c, role) != packet["settings"]:
        raise ValueError("Judge packet requests do not match the cluster experiment")
    return packet, tasks


def import_results(c, packet_path, results):
    packet, tasks = validate_packet(c, packet_path)
    _, indexed = _records(packet, results)
    by_id = {t["id"]: t for t in tasks}
    recorded = {_call_hash(by_id[tid], by_id[tid]["calls"][a]): r for (tid, a), r in indexed.items()}
    class Replay:
        def call(self, messages, left, right, seed, flip):
            key = _hash([messages, left, right, seed, flip])
            if key not in recorded:
                raise RuntimeError("Local Judge result missing; import made no API request")
            return dict(recorded[key])
    factory = lambda config, role: Replay()
    args = (c, packet["method"], packet["budget"], packet["seed"])
    if packet["stage"] == "label":
        from .pipeline import label
        return label(*args, judge_factory=factory)
    from .evaluation_v2 import evaluate_judge
    return evaluate_judge(*args, judge_factory=factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "run", "import"])
    parser.add_argument("--packet", required=True)
    parser.add_argument("--results")
    parser.add_argument("--config")
    parser.add_argument("--stage", choices=["label", "evaluate-judge"])
    parser.add_argument("--method")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args()
    if a.action == "run":
        if not a.results:
            parser.error("run requires --results")
        result = run_packet(a.packet, a.results, execute=a.execute, workers=a.workers)
    else:
        if not a.config:
            parser.error("export/import requires the cluster --config")
        c, _ = load_config(a.config, ready=True)
        if a.action == "export":
            if any(v is None for v in (a.stage, a.method, a.budget, a.seed)):
                parser.error("export requires --stage, --method, --budget, --seed")
            result = export_packet(c, a.stage, a.method, a.budget, a.seed, a.packet)
        else:
            if not a.results:
                parser.error("import requires --results")
            result = import_results(c, a.packet, a.results)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
