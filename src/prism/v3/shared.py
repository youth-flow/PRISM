"""Independent shared-SFT shards with the serial protocol's exact item seeds."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from prism.io import digest, rows
from . import ppo as model_ops
from .artifacts import Run, read_json, write_json
from .recovery import executor_hashes, recover_stage, seal_and_publish, stage_lock


def _sources():
    folder = Path(__file__).resolve().parent
    return [Path(__file__), folder / "ppo.py", folder / "ppo_trainer.py", folder / "ppo_math.py",
            folder.parent / "hf.py", folder.parent / "numerics.py"]


def _items(c, run):
    policy = rows(run.file("data/policy.jsonl"))
    dev = rows(run.file("data/ppo_dev.jsonl"))
    count = c["reward"]["calibration_responses"]
    if len(policy) < count or count <= c["ppo"]["critic_warmup_episodes"]:
        raise ValueError("Need separate calibration and critic-heldout trajectories")
    return ([("calibration", i, row) for i, row in enumerate(policy[:count])]
            + [("dev_probe", i, row) for i, row in enumerate(dev)])


def _seed(c, group, source):
    return int(hashlib.sha256(f"v3-shared:{c['study']['seed']}:{group}:{source['id']}".encode()).hexdigest()[:8], 16)


def _validate_row(c, row, item, global_index):
    group, index, source = item
    if (row.get("global_index") != global_index or row.get("group") != group or row.get("index") != index
            or row.get("id") != f"{group}-{index:06d}" or row.get("source_id") != source["id"]
            or row.get("messages") != source["messages"] or row.get("seed") != _seed(c, group, source)):
        raise ValueError("Shared SFT shard identity/seed/order changed")
    phi = np.asarray(row["phi"], dtype=np.float32)
    prompt, response = row["prompt_tokens"], row["response_tokens"]
    if (phi.shape != (c["data"]["feature_dimension"],) or not np.isfinite(phi).all()
            or not 1 <= len(prompt) <= c["data"]["max_prompt_tokens"]
            or not 1 <= len(response) <= c["ppo"]["response_length"]
            or any(type(t) is not int or t < 0 for t in [*prompt, *response])
            or len(prompt) + len(response) + int(not row["has_eos"]) > c["data"]["context_tokens"]):
        raise ValueError("Shared SFT shard tokens/feature numerics invalid")


def _generate_row(c, model, tokenizer, item, global_index):
    import torch
    from prism.hf import prompt_ids
    from .ppo_trainer import left_pad, prepare_inputs, seed_everything, stream_generate
    group, index, record = item
    seed = _seed(c, group, record)
    seed_everything(seed)
    prompt = prompt_ids(tokenizer, record["messages"], c["data"]["max_prompt_tokens"])
    query = left_pad([prompt], tokenizer.pad_token_id, next(model.parameters()).device)
    response, _ = stream_generate(model, query, pad=tokenizer.pad_token_id, eos=tokenizer.eos_token_id,
                                 max_new_tokens=c["ppo"]["response_length"], temperature=1.)
    actual = response[0][response[0].ne(tokenizer.pad_token_id)].tolist()
    sequence = prompt + actual
    if sequence[-1] != tokenizer.eos_token_id:
        sequence.append(tokenizer.eos_token_id)
    if len(sequence) > c["data"]["context_tokens"]:
        raise ValueError("Shared SFT canonical EOS exceeds context")
    with torch.no_grad():
        tokens = left_pad([sequence], tokenizer.pad_token_id, query.device)
        phi = model.base_model(**prepare_inputs(tokens, tokenizer.pad_token_id)).last_hidden_state[0, -1].float().cpu().numpy()
    return {"id": f"{group}-{index:06d}", "source_id": record["id"], "messages": record["messages"],
            "group": group, "index": index, "global_index": global_index, "seed": seed,
            "prompt_tokens": prompt, "response_tokens": actual,
            "has_eos": actual[-1] == tokenizer.eos_token_id, "phi": phi.tolist()}


def prepare_shared_part(c, shard_index, shards):
    run = Run(c)
    stage = f"shared-part-{shard_index:05d}-of-{shards:05d}"
    with stage_lock(run, stage):
        run.require("data")
        items = _items(c, run)
        if (type(shards) is not int or not 1 <= shards <= len(items)
                or type(shard_index) is not int or not 0 <= shard_index < shards):
            raise ValueError("Shared SFT shard index outside global coverage")
        if run.completed(stage):
            return run.require(stage)
        indices = list(range(shard_index, len(items), shards))
        directory = run.file(f"shared/parts/{shard_index:05d}-of-{shards:05d}")
        progress, contract_path, marker = (directory / name for name in ("progress.jsonl", "contract.json", "report.json"))
        dependencies = [run.receipt_path("data"), run.file("data/policy.jsonl"), run.file("data/ppo_dev.jsonl")]
        contract = {"protocol": "v3_shared_sft_shard_v1", "config_sha256": run.fingerprint,
                    "shard_index": shard_index, "shards": shards, "global_indices": indices,
                    "inputs": {run._key(p): digest(p) for p in dependencies}, "executor_sources": executor_hashes(_sources())}
        write_json(contract_path, contract)
        previous = rows(progress) if progress.exists() else []
        if len(previous) > len(indices):
            raise ValueError("Shared shard contains extra rows")
        for row, index in zip(previous, indices):
            _validate_row(c, row, items[index], index)
        started = time.monotonic()
        if len(previous) < len(indices):
            import torch
            model = model_ops.load_model(c).requires_grad_(False)
            tokenizer = model_ops.load_tokenizer(c)
            with progress.open("a", encoding="utf-8") as stream:
                for index in indices[len(previous):]:
                    row = _generate_row(c, model, tokenizer, items[index], index)
                    _validate_row(c, row, items[index], index)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    previous.append(row)
                    if len(previous) % 16 == 0:
                        print(f"v3 shared shard={shard_index}/{shards} rows={len(previous)}/{len(indices)} seconds={time.monotonic()-started:.1f}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if marker.exists():
            report = read_json(marker)
            if report.get("global_indices") != indices or report.get("rows") != len(indices):
                raise ValueError("Shared shard completion report changed")
        else:
            report = {"protocol": "v3_shared_sft_shard_v1", "shard_index": shard_index, "shards": shards,
                      "global_indices": indices, "rows": len(previous), "seconds_this_invocation": time.monotonic()-started,
                      "api_calls": 0, "backbone_updates": 0}
            write_json(marker, report)
        return seal_and_publish(run, stage, [progress, contract_path, marker], dependencies,
                                code_paths=_sources(), context={"shards": shards, "shard_index": shard_index}, metadata=report)


def merge_shared(c, shards):
    from filelock import FileLock
    run = Run(c)
    lock = run.file("shared/.execution.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        run.require("data")
        if run.completed("shared-sft"):
            return run.require("shared-sft")
        items = _items(c, run)
        if type(shards) is not int or not 1 <= shards <= len(items):
            raise ValueError("Invalid shared SFT shard count")
        ordered = [None] * len(items)
        dependencies = [run.receipt_path("data"), run.file("data/policy.jsonl"), run.file("data/ppo_dev.jsonl")]
        for shard in range(shards):
            stage = f"shared-part-{shard:05d}-of-{shards:05d}"
            run.require(stage)
            dependencies.append(run.receipt_path(stage))
            prefix = run.file(f"shared/parts/{shard:05d}-of-{shards:05d}")
            data = rows(prefix / "progress.jsonl")
            expected = list(range(shard, len(items), shards))
            if [row.get("global_index") for row in data] != expected:
                raise ValueError("Shared SFT shards do not preserve global coverage/order")
            for index, row in zip(expected, data):
                _validate_row(c, row, items[index], index)
                if ordered[index] is not None:
                    raise ValueError("Shared SFT duplicate global index")
                ordered[index] = row
        if any(row is None for row in ordered):
            raise ValueError("Shared SFT merge has missing rows")
        groups = ("calibration", "dev_probe", "train_probe")
        outputs = [run.file(f"shared/{g}{suffix}") for g in groups for suffix in (".jsonl", ".features.npz")]
        if recover_stage(run, "shared-sft", outputs, dependencies, code_paths=_sources(), context={"shards": shards}):
            return run.require("shared-sft")
        count = c["reward"]["calibration_responses"]
        for group, records in (("calibration", ordered[:count]), ("dev_probe", ordered[count:]),
                               ("train_probe", ordered[:c["ppo"]["fixed_train_probe_prompts"]])):
            values = [{k: v for k, v in record.items() if k not in ("phi", "group", "index", "global_index")} for record in records]
            model_ops._save_rows(run.file(f"shared/{group}.jsonl"), values)
            model_ops._save_features(run.file(f"shared/{group}.features.npz"), [r["phi"] for r in records], [r["id"] for r in records])
        return seal_and_publish(run, "shared-sft", outputs, dependencies, code_paths=_sources(), context={"shards": shards},
                                metadata={"calibration_responses": count, "dev_responses": len(items)-count,
                                          "new_sft_responses": len(items), "shards": shards, "full_coverage": True, "api_calls": 0})


def shared_smoke(c):
    """Actual pinned SFT: compare six serial and six sharded full-cap answers."""
    from copy import deepcopy
    run = Run(c)
    run.require("data")
    if run.completed("shared-smoke"):
        return run.require("shared-smoke")
    policy = rows(run.file("data/policy.jsonl"))[:4]
    dev = rows(run.file("data/ppo_dev.jsonl"))[:2]
    if len(policy) != 4 or len(dev) != 2:
        raise ValueError("Shared equivalence smoke needs six genuine HH contexts")
    diagnostics = run.file("diagnostics/shared-smoke")
    configs = []
    for name in ("serial", "shards"):
        config = {k: deepcopy(v) for k, v in c.items() if not k.startswith("_")}
        config["output"] = str(diagnostics / name)
        config["reward"]["calibration_responses"] = 4
        config["ppo"].update(critic_warmup_episodes=2, fixed_train_probe_prompts=2)
        diagnostic = Run(config)
        for filename, records in (("policy", policy), ("ppo_dev", dev)):
            model_ops._save_rows(diagnostic.file(f"data/{filename}.jsonl"), records)
        diagnostic.publish("data", [diagnostic.file("data/policy.jsonl"), diagnostic.file("data/ppo_dev.jsonl")])
        configs.append(config)
    model_ops.prepare_shared(configs[0])
    for shard in range(2):
        prepare_shared_part(configs[1], shard, 2)
    merge_shared(configs[1], 2)
    max_error = 0.
    for group in ("calibration", "dev_probe", "train_probe"):
        serial, sharded = (Run(config).file(f"shared/{group}.jsonl") for config in configs)
        if serial.read_bytes() != sharded.read_bytes():
            raise ValueError("Actual SFT serial/shard response tokens, seeds or order differ")
        with np.load(serial.with_name(group + ".features.npz")) as a, np.load(sharded.with_name(group + ".features.npz")) as b:
            max_error = max(max_error, float(np.max(np.abs(a["phi"] - b["phi"]))))
            if not np.array_equal(a["phi"], b["phi"]) or not np.array_equal(a["ids"], b["ids"]):
                raise ValueError("Actual SFT serial/shard EOS features differ")
    report = {"protocol": "v3_actual_shared_shard_equivalence_v1", "passed": True,
              "compared_contexts": 6, "actual_generated_responses": 12, "max_feature_error": max_error,
              "response_cap": c["ppo"]["response_length"], "temperature": 1., "formal_ppo_episodes": 0, "api_calls": 0}
    marker = diagnostics / "report.json"
    write_json(marker, report)
    return run.publish("shared-smoke", [marker], [run.receipt_path("data")], metadata=report)
