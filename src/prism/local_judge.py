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
import shutil
from threading import Event, Lock

import numpy as np

from .io import cell, digest, dump, load_config, receipt, root, rows, verify_receipt
from .judges import APIJudge, _parse, evaluation_call_seed, judge_protocol, judge_settings, normalize_preference_result, training_call_seed


NETWORK_RETRY_LIMIT = 2


def _network_failure(result):
    error = result.get("error") or {}
    return result.get("label") is None and error.get("retryable") is True and error.get("type") in ("network_error", "http_error")


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
              "judge_protocol": judge_protocol(role, judge_settings(c, role)["backend"]),
              "request_endpoint": APIJudge(c, role).url,
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
    if packet.get("judge_protocol", legacy) != judge_protocol(packet["role"], packet["settings"]["backend"]):
        raise ValueError("Judge prompt protocol changed; archive the old packet/results before collecting a new batch")
    expected_endpoint = APIJudge({"judges": {packet["role"]: packet["settings"]}}, packet["role"]).url
    fallback = None if packet.get("judge_protocol") == "preference_tool_enum_v4" else expected_endpoint
    if packet.get("request_endpoint", fallback) != expected_endpoint:
        raise ValueError("Judge packet execution endpoint changed")
    return packet


def _records(packet, results, *, with_accounting=False):
    tasks = {t["id"]: t for t in packet["tasks"]}
    if len(tasks) != len(packet["tasks"]):
        raise ValueError("Duplicate Judge task ID")
    records = list(packet["prior_calls"])
    if Path(results).exists():
        for row in rows(results):
            if row.get("packet_id") != packet["packet_id"]:
                raise ValueError("Judge results belong to another packet")
            records.append(row)
    indexed, seen, retry_counts = {}, set(), {}
    for row in records:
        task, attempt, result = tasks.get(row["task_id"]), row["attempt"], row["result"]
        if task is None or type(attempt) is not int or not 0 <= attempt < len(task["calls"]):
            raise ValueError("Unknown Judge task/attempt")
        key, call = (task["id"], attempt), task["calls"][attempt]
        retransmission = row.get("transport_retry", 0)
        if type(retransmission) is not int or not 0 <= retransmission <= NETWORK_RETRY_LIMIT:
            raise ValueError("Invalid transport retry index")
        legacy = "preference_ab_v1" if packet["role"] == "training" else "preference_ab_tie_v1"
        if result.get("judge_protocol", legacy) != packet.get("judge_protocol", legacy):
            raise ValueError("Judge results use a different prompt protocol")
        if packet.get("judge_protocol") == "preference_tool_enum_v4":
            calls = result.get("tool_calls") or []
            if (result.get("request_endpoint") != packet.get("request_endpoint")
                    or (result.get("label") is not None and (result.get("finish_reason") != "tool_calls"
                        or len(calls) != 1 or calls[0].get("type") != "function"
                        or calls[0].get("function", {}).get("name") != "record_preference"
                        or calls[0].get("function", {}).get("arguments") != result.get("raw")))):
                raise ValueError("Judge tool channel or endpoint does not match its packet")
        if ((*key, retransmission) in seen or result.get("seed") != call["seed"] or result.get("flip") != call["flip"]):
            raise ValueError("Duplicate or misaligned Judge result")
        if retransmission:
            if (packet.get("judge_protocol") != "preference_tool_enum_v4" or attempt != len(task["calls"])-1
                    or key not in indexed or retry_counts[key] != retransmission-1 or not _network_failure(indexed[key])):
                raise ValueError("Transport retries require the same exhausted final request after a network failure")
        elif key in indexed:
            raise ValueError("Duplicate Judge logical attempt")
        label = result.get("label")
        if label is not None and (type(label) not in (int, float) or label not in (0, 0.5, 1) or label != _parse(result.get("raw", ""), packet["role"], call["flip"], protocol=packet.get("judge_protocol", legacy))):
            raise ValueError("Judge label does not match its raw answer/order")
        if packet["role"] == "training" and label == 0.5:
            raise ValueError("Training Judge cannot return a tie")
        earlier = [indexed.get((task["id"], a)) for a in range(attempt)]
        if any(r is None or r["label"] is not None for r in earlier):
            raise ValueError("Judge attempts are not a contiguous failure prefix")
        indexed[key] = result
        retry_counts[key] = retransmission
        seen.add((*key, retransmission))
    logical_count = len(indexed)
    interpreted = {}
    for task in tasks.values():
        for attempt in range(len(task["calls"])):
            key = (task["id"], attempt)
            if key not in indexed:
                break
            result = normalize_preference_result(indexed[key])
            if retry_counts[key]:
                result = {**result, "transport_retry": retry_counts[key]}
            interpreted[key] = result
            if result["label"] is not None:
                break  # Later historical retries are charged but never select a preferred label.
    if not with_accounting:
        return tasks, interpreted
    normalized = [normalize_preference_result(row["result"]) for row in records]
    accounting = {"actual_judge_calls": len(records),
                  "invalid_calls": sum(r["label"] is None for r in normalized),
                  "source_invalid_calls": sum(row["result"]["label"] is None for row in records),
                  "canonical_judge_calls": len(interpreted),
                  "collapsed_transport_calls": len(records)-logical_count,
                  "superseded_semantic_calls": logical_count-len(interpreted),
                  "max_transport_retry_used": max(retry_counts.values(), default=0)}
    return tasks, interpreted, accounting


def run_packet(packet_path, results, *, execute=False, transport=None, workers=1, network_retries=0):
    if not execute:
        raise RuntimeError("Local API execution requires --execute")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Judge workers must be an integer in 1..16")
    if type(network_retries) is not int or not 0 <= network_retries <= NETWORK_RETRY_LIMIT:
        raise ValueError("Extra network retries must be in 0..2")
    packet = _load_packet(packet_path)
    if network_retries and packet.get("judge_protocol") != "preference_tool_enum_v4":
        raise ValueError("Extra transport retries are restricted to the checked cluster training protocol")
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
            # Three original semantic slots remain fixed. Only an unanswered
            # retryable final request may be retransmitted, with identical seed,
            # order and payload. Every dispatch has its own append-only record.
            attempt = len(task["calls"])-1
            result = previous[-1] if len(previous) == len(task["calls"]) else result
            retry = result.get("transport_retry", 0)
            while _network_failure(result) and retry < network_retries and not stopped.is_set():
                retry += 1
                call = task["calls"][attempt]
                try:
                    result = judge.call(task["messages"], task["left"], task["right"], call["seed"], call["flip"])
                except BaseException:
                    stopped.set()
                    raise
                record = {"packet_id": packet["packet_id"], "task_id": task["id"], "attempt": attempt,
                          "transport_retry": retry, "result": result}
                with lock:
                    with results.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                        stream.flush()
            if result["label"] is None:
                stopped.set()
                raise RuntimeError(f"Judge exhausted attempts for task {task['id']}; all attempts preserved")
            with lock:
                completed += 1
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
    _, indexed, accounting = _records(packet, results, with_accounting=True)
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
        out = cell(c, packet["method"], packet["budget"], packet["seed"])
        marker, calls = out/"labels.receipt.json", out/"judge_calls.jsonl"
        if marker.exists():
            metadata = verify_receipt(marker)
            if packet.get("judge_protocol") == "preference_tool_enum_v4" and metadata.get("decoder_revision") != "native_candidate_alias_v1":
                raise RuntimeError("Preserve and archive the older decoded label snapshot before replaying the current decoder")
            return metadata
        if packet.get("judge_protocol") == "preference_tool_enum_v4" and calls.exists():
            expected = []
            for query in rows(out/"queries.jsonl"):
                complete = False
                for attempt in range(len(by_id[str(query["draw_id"])]["calls"])):
                    result = indexed.get((str(query["draw_id"]), attempt))
                    if result is None:
                        break
                    expected.append({**query, "attempt": attempt, **result})
                    if result["label"] is not None:
                        complete = True
                        break
                if not complete:
                    break
            previous = rows(calls)
            if previous != expected[:len(previous)]:
                archive = out/"label_derivations"/f"partial-{digest(calls)[:16]}-{digest(results)[:16]}"
                archive.mkdir(parents=True, exist_ok=False)
                calls.rename(archive/"judge_calls.jsonl")
                shutil.copyfile(packet_path, archive/"packet.json")
                shutil.copyfile(results, archive/"api.results.jsonl")
                receipt(archive/"receipt.json", [archive/name for name in ("judge_calls.jsonl", "packet.json", "api.results.jsonl")],
                        reason="native_candidate_alias_or_transport_replay", decoder_revision="native_candidate_alias_v1")
        label(*args, judge_factory=factory)
        if packet.get("judge_protocol") == "preference_tool_enum_v4":
            metadata = json.loads(marker.read_text())
            metadata.update(accounting, decoder_revision="native_candidate_alias_v1", raw_api_result_path=str(Path(results).resolve()))
            metadata["input_sha256"].update({str(Path(packet_path).resolve()): digest(packet_path), str(Path(results).resolve()): digest(results)})
            dump(marker, metadata)
        return verify_receipt(marker)
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
