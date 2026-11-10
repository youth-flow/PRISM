"""Deterministic prompt-only preparation, separate from any model computation.

Only an explicit prepare-data command calls this module. Dataset/tokenizer
dependencies are lazy, and tests supply in-memory sources and a fake tokenizer.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np


def prompt_signature(text):
    """Comparison signature only: never rewrite the prompt sent to a model."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _pinned(source, name):
    if not source.get("id") or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError(f"{name} requires an ID and immutable 40-character revision")


def _destinations(c):
    prompts, bench = c["choices"]["prompts"], c["choices"]["benchmark"]
    paths = {key: Path(prompts[key]) for key in ("design", "policy", "evaluation")}
    paths["benchmark"] = Path(bench["path"])
    paths["manifest"] = Path(c["data"].get("manifest_path") or paths["benchmark"].parent/"data.manifest.json")
    if any(not path.is_absolute() for path in paths.values()):
        raise ValueError("Data output paths must be resolved by the config loader first")
    paths = {key: path.resolve() for key, path in paths.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Data output destinations must be distinct")
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite prepared data: {path}")
    return paths


def _prompt_length(tokenizer, text):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                        tokenize=True, add_generation_prompt=True)
    if not ids:
        raise ValueError("The SFT tokenizer produced an empty prompt")
    return len(ids)


def _benchmark_rows(dataset, expected, tokenizer, context, answer_cap):
    result, seen_ids = [], set()
    for row in dataset:
        required = ("key", "prompt", "instruction_id_list", "kwargs")
        if any(key not in row for key in required):
            raise ValueError("IFEval requires key, prompt, instruction_id_list and kwargs")
        if not isinstance(row["prompt"], str) or not prompt_signature(row["prompt"]):
            raise ValueError("IFEval contains an invalid prompt; benchmark rows cannot be filtered")
        if not isinstance(row["instruction_id_list"], list) or not isinstance(row["kwargs"], list):
            raise ValueError("IFEval instruction_id_list and kwargs must be lists")
        if len(row["instruction_id_list"]) != len(row["kwargs"]):
            raise ValueError("IFEval instruction and kwargs counts differ")
        key = str(row["key"])
        if key in seen_ids:
            raise ValueError("IFEval contains duplicate keys; benchmark rows cannot be deduplicated")
        seen_ids.add(key)
        length = _prompt_length(tokenizer, row["prompt"])
        if length + answer_cap + 1 > context:
            raise ValueError(f"IFEval key {key} exceeds the model context with its benchmark response cap; no rows were removed")
        item = copy.deepcopy(dict(row))
        item.update(id=f"ifeval:{key}", messages=[{"role": "user", "content": row["prompt"]}])
        result.append(item)
    if len(result) != expected:
        raise ValueError(f"Expected the full {expected}-row benchmark, received {len(result)}")
    return result


def _eligible(dataset, source, split, tokenizer, prompt_cap, benchmark_signatures,
              training_signatures=None):
    fields = [source["prompt_field"], source["id_field"]]
    if hasattr(dataset, "select_columns"):
        dataset = dataset.select_columns(fields)
    stats = {key: 0 for key in ("raw", "invalid_prompt", "duplicate_prompt", "benchmark_overlap",
                                "train_test_overlap", "prompt_too_long", "eligible")}
    result, seen, source_ids, all_signatures = [], set(), {}, set()
    for row in dataset:
        stats["raw"] += 1
        # Do not read chosen/rejected responses, ratings, or any existing labels.
        text, source_id = row.get(source["prompt_field"]), row.get(source["id_field"])
        if not isinstance(text, str) or not prompt_signature(text):
            stats["invalid_prompt"] += 1
            continue
        if type(source_id) not in (str, int) or not str(source_id):
            raise ValueError(f"Invalid source prompt ID in split {split}")
        signature = prompt_signature(text)
        all_signatures.add(signature)
        id_text = str(source_id)
        if id_text in source_ids and source_ids[id_text] != signature:
            raise ValueError(f"One source ID names different prompts in split {split}")
        source_ids[id_text] = signature
        if signature in seen:
            stats["duplicate_prompt"] += 1
            continue
        seen.add(signature)
        if signature in benchmark_signatures:
            stats["benchmark_overlap"] += 1
            continue
        if training_signatures is not None and signature in training_signatures:
            stats["train_test_overlap"] += 1
            continue
        length = _prompt_length(tokenizer, text)
        if length > prompt_cap:
            stats["prompt_too_long"] += 1
            continue
        result.append({"id": f"{split}:{id_text}", "source_id": source_id, "source_split": split,
                       "messages": [{"role": "user", "content": text}]})
        stats["eligible"] += 1
    return result, stats, all_signatures


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_transaction(paths, outputs, manifest):
    """Stage every file, publish without overwrite, and publish manifest last.

    Hard-link + unlink implements an atomic no-replace publication on both
    Windows and POSIX, where os.rename has different overwrite semantics.
    The manifest is the completion marker for this multi-file transaction.
    """
    staged, published = {}, []
    try:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            staged[key] = Path(temp)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                if key == "manifest":
                    manifest["outputs"] = {name: {"path": str(paths[name]), "sha256": _sha256(staged[name])}
                                           for name in outputs}
                    json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
                else:
                    for row in outputs[key]:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
        # A racing writer cannot be overwritten. On an ordinary failure, roll
        # back only paths which still refer to the staged files that we own.
        for key, path in paths.items():
            os.link(staged[key], path)
            published.append(key)
    except BaseException:
        for key in reversed(published):
            try:
                if os.path.samefile(paths[key], staged[key]):
                    paths[key].unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def prepare_data(c, load_dataset_fn=None, tokenizer_factory=None):
    """Explicitly materialize pinned, exact-count prompt splits and full IFEval.

    Factories have the signatures of datasets.load_dataset and
    AutoTokenizer.from_pretrained. No language-model weights are loaded here.
    """
    paths = _destinations(c)  # Fail before network access if any output exists.
    settings = c["data"]
    source, benchmark_source = settings["source"], settings["benchmark_source"]
    _pinned(source, "data.source")
    _pinned(benchmark_source, "data.benchmark_source")
    _pinned({"id": c["choices"]["sft"]["model"], "revision": c["choices"]["sft"]["revision"]}, "SFT tokenizer")
    counts = settings["expected_counts"]
    if any(type(counts.get(key)) is not int or counts[key] < 1
           for key in ("design", "policy", "evaluation", "benchmark")):
        raise ValueError("Expected dataset counts must be positive integers")
    context, prompt_cap = settings["model_context_tokens"], c["candidate"]["max_prompt_tokens"]
    response_cap = max(c["candidate"]["max_new_tokens"], c["ppo"]["response_length"], c["evaluation"]["max_new_tokens"])
    if prompt_cap + response_cap + 1 > context:
        raise ValueError("Prompt/response caps plus feature EOS exceed data.model_context_tokens")
    benchmark_cap = c["benchmark"]["max_new_tokens"]
    if load_dataset_fn is None:
        from datasets import load_dataset
        load_dataset_fn = load_dataset
    if tokenizer_factory is None:
        from transformers import AutoTokenizer
        tokenizer_factory = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_factory(c["choices"]["sft"]["model"], revision=c["choices"]["sft"]["revision"],
                                  trust_remote_code=False,
                                  local_files_only=c["runtime"].get("local_files_only", True))
    if not getattr(tokenizer, "chat_template", None) or getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("The selected SFT tokenizer needs a chat template and EOS")
    benchmark_dataset = load_dataset_fn(benchmark_source["id"], revision=benchmark_source["revision"], split=benchmark_source["split"])
    benchmark = _benchmark_rows(benchmark_dataset, counts["benchmark"], tokenizer, context, benchmark_cap)
    benchmark_signatures = {prompt_signature(row["prompt"]) for row in benchmark}
    train_dataset = load_dataset_fn(source["id"], revision=source["revision"], split=source["train_split"])
    train, train_stats, train_signatures = _eligible(train_dataset, source, source["train_split"], tokenizer,
                                                    prompt_cap, benchmark_signatures)
    test_dataset = load_dataset_fn(source["id"], revision=source["revision"], split=source["test_split"])
    test, test_stats, _ = _eligible(test_dataset, source, source["test_split"], tokenizer,
                                   prompt_cap, benchmark_signatures, train_signatures)
    seed = settings["split_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("data.split_seed must be a nonnegative integer")
    # Permute the entire eligible training population, not a sliced source prefix.
    train = [train[i] for i in np.random.default_rng(seed).permutation(len(train))]
    test = [test[i] for i in np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(len(test))]
    disjoint = settings.get("require_disjoint_design_policy", True)
    train_needed = counts["design"] + counts["policy"] if disjoint else max(counts["design"], counts["policy"])
    if len(train) < train_needed or len(test) < counts["evaluation"]:
        raise ValueError(f"Insufficient eligible prompts: train={len(train)} need={train_needed}; test={len(test)} need={counts['evaluation']}")
    outputs = {"design": train[:counts["design"]],
               "policy": train[counts["design"]:train_needed] if disjoint else train[:counts["policy"]],
               "evaluation": test[:counts["evaluation"]], "benchmark": benchmark}
    manifest = {"kind": "prism_prepared_data", "schema_version": 1, "source": copy.deepcopy(source),
                "benchmark_source": copy.deepcopy(benchmark_source),
                "tokenizer": copy.deepcopy(c["choices"]["sft"]),
                "selection": {"split_seed": seed, "algorithm": "numpy Generator PCG64 full permutation",
                              "numpy_version": np.__version__, "signature": "NFKC plus whitespace collapse; original text preserved",
                              "max_prompt_tokens": prompt_cap, "model_context_tokens": context,
                              "benchmark_max_new_tokens": benchmark_cap,
                              "require_disjoint_design_policy": disjoint},
                "counts": {key: len(value) for key, value in outputs.items()},
                "excluded": {"train": train_stats, "test": test_stats},
                "selected_ids": {key: [row["id"] for row in value] for key, value in outputs.items()}}
    _write_transaction(paths, outputs, manifest)
    return {"manifest_path": str(paths["manifest"]), "counts": manifest["counts"],
            "excluded": manifest["excluded"]}
