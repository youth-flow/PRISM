"""Pinned HH input preparation, label-blind pairs and grouped holdouts.

HH transcripts have no role metadata. We choose the last shared, well-formed
Assistant turn boundary, ignoring delimiters inside fenced code. We reject
malformed/ambiguous role histories instead of truncating candidate answers.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

from prism.io import digest, rows, write_rows
from .artifacts import Run, hash_json, read_json, write_json
from .recovery import recover_stage, seal_and_publish, stage_lock


PARSER_PROTOCOL = "hh_common_prefix_turn_boundary_v1"
_TURN = re.compile(r"\n\n(Human|Assistant):")


def _turns(text):
    """Find HH separators outside triple-backtick code fences."""
    result = []
    for match in _TURN.finditer(text):
        if text[:match.start()].count("```") % 2 == 0:
            result.append(match)
    return result


def _history(text):
    markers = _turns(text)
    if not markers or text[:markers[0].start()].strip():
        raise ValueError("missing_initial_human_turn")
    messages = []
    for i, marker in enumerate(markers):
        expected = "Human" if i % 2 == 0 else "Assistant"
        if marker.group(1) != expected:
            raise ValueError("ambiguous_or_nonalternating_turns")
        stop = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        content = text[marker.end():stop]
        if content.startswith(" "):
            content = content[1:]
        if not content.strip():
            raise ValueError("empty_history_turn")
        messages.append({"role": "user" if expected == "Human" else "assistant", "content": content})
    if messages[-1]["role"] != "user":
        raise ValueError("context_does_not_end_in_user")
    return messages


def parse_pair(chosen, rejected):
    """Return shared chat context and complete existing chosen/rejected answers."""
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        raise ValueError("nontext_transcript")
    common = 0
    for a, b in zip(chosen, rejected):
        if a != b:
            break
        common += 1
    candidates = [m for m in _turns(chosen[:common]) if m.group(1) == "Assistant"]
    for boundary in reversed(candidates):
        try:
            messages = _history(chosen[:boundary.start()])
        except ValueError:
            continue
        if _turns(chosen[boundary.end():common]):
            raise ValueError("ambiguous_shared_role_continuation")
        first, second = chosen[boundary.end():], rejected[boundary.end():]
        if first.startswith(" "):
            first = first[1:]
        if second.startswith(" "):
            second = second[1:]
        if not first.strip() or not second.strip():
            raise ValueError("empty_response")
        if first == second:
            raise ValueError("identical_responses")
        return messages, first, second
    raise ValueError("no_unambiguous_common_assistant_boundary")


def canonical_pair(raw, source):
    messages, chosen, rejected = parse_pair(raw.get("chosen"), raw.get("rejected"))
    prompt_id = hash_json(messages)
    answers = sorted((chosen, rejected), key=lambda x: (hashlib.sha256(x.encode()).hexdigest(), x))
    pair_id = hash_json({"prompt_id": prompt_id, "responses": answers})
    pair = {"id": pair_id, "prompt_id": prompt_id, "messages": messages,
            "response_A": answers[0], "response_B": answers[1], "source": source}
    return pair, {"id": pair_id, "label": int(answers[0] == chosen)}


def load_tokenizer(c):
    from prism.hf import tokenizer as fixed_tokenizer
    tokenizer = fixed_tokenizer({"choices": {"sft": {"model": c["model"]["id"],
                                    "revision": c["model"]["revision"]}}, "runtime": c["runtime"]})
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Pinned tokenizer needs a chat template and EOS")
    return tokenizer


def tokenizer_fingerprint(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    try:
        size = len(tokenizer)
    except TypeError:
        size = None
    added = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else None
    return {"class": type(tokenizer).__module__ + ":" + type(tokenizer).__qualname__,
            "chat_template_sha256": hash_json(template), "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None), "length": size,
            "padding_side": getattr(tokenizer, "padding_side", None), "added_vocab_sha256": hash_json(added)}


def _sources():
    folder = Path(__file__).resolve().parent
    return [Path(__file__), folder.parent / "hf.py", folder.parent / "numerics.py"]


def encode_pair(record, tokenizer, context_tokens):
    """Canonical complete responses include EOS in both feature and sequence logp."""
    prompt = tokenizer.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=True)
    if not prompt:
        raise ValueError("empty_token_prompt")
    eos = tokenizer.eos_token_id
    pad = getattr(tokenizer, "pad_token_id", None)
    if eos is None:
        raise ValueError("tokenizer_missing_eos")
    if pad is not None and pad in prompt:
        raise ValueError("reserved_padding_token_in_prompt")
    encoded = []
    for key in ("response_A", "response_B"):
        response = tokenizer.encode(record[key], add_special_tokens=False)
        if not response:
            raise ValueError("empty_token_response")
        if pad is not None and pad in response:
            raise ValueError("reserved_padding_token_in_response")
        if eos in response[:-1]:
            raise ValueError("internal_eos_in_human_response")
        if response[-1] != eos:
            response += [eos]
        if len(prompt) + len(response) > context_tokens:
            raise ValueError("full_response_exceeds_context")
        encoded.append(response)
    return list(prompt), encoded


def download_raw(c):
    """Explicit anonymous public-data download; never calls a model endpoint."""
    raw_dir = Path(c["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / "source.manifest.json"
    if marker.exists():
        value = read_json(marker)
        _manifest_files(c, marker)
        return value
    source = c["data"]["source_id"]
    revision = c["data"]["revision"]
    progress_path = raw_dir / "download.progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {
        "source_id": source, "revision": revision, "files": []}
    if progress.get("source_id") != source or progress.get("revision") != revision:
        raise ValueError("Raw download directory belongs to a different pinned source")
    completed = {r["path"]: r for r in progress.get("files", [])}
    if len(completed) != len(progress.get("files", [])):
        raise ValueError("Duplicate raw download progress entry")
    files = []
    for subset in c["data"]["subsets"]:
        for split in ("train", "test"):
            relative = f"{subset}/{split}.jsonl.gz"
            path = raw_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                previous = completed.get(relative)
                if (previous is None or previous.get("subset") != subset or previous.get("split") != split
                        or digest(path) != previous.get("sha256")):
                    raise FileExistsError(f"Unmanifested or changed raw source must be reviewed: {path}")
                files.append(previous)
                continue
            url = f"https://huggingface.co/datasets/{source}/resolve/{revision}/{relative}"
            temporary = path.with_suffix(path.suffix + ".partial")
            # A .partial file is never published as an input. Retrying replaces
            # only that incomplete transfer, while completed source hashes stay fixed.
            with urlopen(url, timeout=120) as response, temporary.open("wb") as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            temporary.replace(path)
            files.append({"subset": subset, "split": split, "path": relative,
                          "sha256": digest(path), "url": url})
            write_json(progress_path, {"source_id": source, "revision": revision, "files": files}, overwrite=True)
    value = {"protocol": "hh_pinned_sources_v1", "source_id": source, "revision": revision, "files": files}
    write_json(marker, value)
    return value


def _manifest_files(c, source_manifest):
    marker = Path(source_manifest).resolve()
    manifest = read_json(marker)
    if (manifest.get("source_id") != c["data"]["source_id"]
            or manifest.get("revision") != c["data"]["revision"]):
        raise ValueError("HH raw source identity/revision mismatch")
    expected = {(s, split) for s in c["data"]["subsets"] for split in ("train", "test")}
    found, result = set(), []
    for spec in manifest.get("files", []):
        key = (spec.get("subset"), spec.get("split"))
        if key not in expected or key in found:
            raise ValueError("HH raw files need exact unique subset/split coverage")
        path = (marker.parent / spec["path"]).resolve()
        if not path.is_relative_to(marker.parent):
            raise ValueError("Raw source must remain inside its manifest directory")
        if not path.is_file() or digest(path) != spec.get("sha256"):
            raise ValueError(f"HH raw file hash changed: {path}")
        found.add(key)
        result.append((spec, path))
    if found != expected:
        raise ValueError("HH raw manifest missing subset/split files")
    return marker, result


def _representatives(records):
    by_prompt = {}
    for record in sorted(records, key=lambda r: r["id"]):
        by_prompt.setdefault(record["prompt_id"], record)
    return list(by_prompt.values())


def stratified_sample(records, count, seed):
    """Source-proportional sampling of unique contexts, independent of labels."""
    unique = _representatives(records)
    if len(unique) < count:
        raise ValueError(f"Need {count} distinct eligible contexts, found {len(unique)}")
    groups = defaultdict(list)
    for record in unique:
        groups[record["source"]["subset"]].append(record)
    names = sorted(groups)
    exact = np.array([count * len(groups[name]) / len(unique) for name in names])
    quotas = np.floor(exact).astype(int)
    for i in sorted(range(len(names)), key=lambda i: (-(exact[i] - quotas[i]), names[i]))[:count - int(quotas.sum())]:
        quotas[i] += 1
    rng, selected = np.random.default_rng(seed), []
    for i, name in enumerate(names):
        group = sorted(groups[name], key=lambda r: r["prompt_id"])
        order = rng.permutation(len(group))
        selected.extend(group[j] for j in order[:quotas[i]])
    return sorted(selected, key=lambda r: r["prompt_id"])


def _prompt_row(record):
    return {"id": record["prompt_id"], "messages": record["messages"], "source": record["source"]}


def prepare_data(c, source_manifest=None, *, tokenizer=None):
    with stage_lock(Run(c), "data"):
        return _prepare_data(c, source_manifest=source_manifest, tokenizer=tokenizer)


def _prepare_data(c, source_manifest=None, *, tokenizer=None):
    run = Run(c)
    if run.completed("data"):
        return read_json(run.file("data/manifest.json"))
    marker = Path(source_manifest or Path(c["data"]["raw_dir"]) / "source.manifest.json")
    marker, sources = _manifest_files(c, marker)
    dependencies = [marker] + [p for _, p in sources]
    output_names = ("pool", "labels", "policy", "ppo_dev", "evaluation", "reward_dev", "rejected", "provenance")
    expected_outputs = [run.file(f"data/{name}.jsonl") for name in output_names] + [run.file("data/manifest.json")]
    tokenizer_context = {"tokenizer_backend": "pinned_hf" if tokenizer is None else
                         type(tokenizer).__module__ + ":" + type(tokenizer).__qualname__}
    if recover_stage(run, "data", expected_outputs, dependencies, code_paths=_sources(),
                     context=tokenizer_context):
        return read_json(run.file("data/manifest.json"))
    tokenizer = load_tokenizer(c) if tokenizer is None else tokenizer
    pairs, labels, provenance, conflicts = {}, {}, defaultdict(list), set()
    rejected, raw_counts, accepted_counts = [], Counter(), Counter()
    for spec, path in sources:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                raw_counts[spec["split"]] += 1
                source = {"subset": spec["subset"], "split": spec["split"], "line": line,
                          "raw_sha256": hashlib.sha256(text.encode()).hexdigest()}
                try:
                    raw = json.loads(text)
                    if not isinstance(raw, dict):
                        raise ValueError("nonmapping_record")
                    pair, label = canonical_pair(raw, source)
                    prompt, _ = encode_pair(pair, tokenizer, c["data"]["context_tokens"])
                    pair["prompt_tokens"] = len(prompt)
                except (ValueError, TypeError, KeyError) as exc:
                    rejected.append({"source": source, "reason": str(exc)})
                    continue
                identity = (spec["split"], pair["id"])
                provenance[identity].append(source)
                if identity in pairs and labels[identity]["label"] != label["label"]:
                    conflicts.add(identity)
                pairs.setdefault(identity, pair)
                labels.setdefault(identity, label)
                accepted_counts[spec["split"]] += 1
    for split, key in (("train", "expected_train_rows"), ("test", "expected_test_rows")):
        expected = c["data"].get(key)
        if expected is not None and raw_counts[split] != expected:
            raise ValueError(f"HH {split} raw row count {raw_counts[split]} != {expected}")
    train, test = [], []
    for identity, pair in sorted(pairs.items()):
        if identity in conflicts:
            rejected.append({"id": pair["id"], "reason": "conflicting_human_labels", "sources": provenance[identity]})
            continue
        pair["source_count"] = len(provenance[identity])
        (train if identity[0] == "train" else test).append(pair)
    train_prompts = {r["prompt_id"] for r in train}
    overlap = [r for r in test if r["prompt_id"] in train_prompts]
    test = [r for r in test if r["prompt_id"] not in train_prompts]
    eligible = lambda rs: [r for r in rs if r["prompt_tokens"] <= c["data"]["max_prompt_tokens"]]
    split_seed = c["data"]["split_seed"]
    evaluation = stratified_sample(eligible(test), c["data"]["evaluation_prompts"], split_seed)
    used = {r["prompt_id"] for r in evaluation}
    dev = stratified_sample(eligible([r for r in test if r["prompt_id"] not in used]),
                            c["data"]["ppo_dev_prompts"], split_seed + 1)
    used.update(r["prompt_id"] for r in dev)
    reward_dev = stratified_sample([r for r in test if r["prompt_id"] not in used],
                                   c["data"]["reward_dev_pairs"], split_seed + 2)
    policy = stratified_sample(eligible(train), c["data"]["policy_prompts"], split_seed + 3)
    outputs = []
    for name, values in (("pool", train), ("labels", [labels[("train", r["id"])] for r in train]),
                         ("policy", list(map(_prompt_row, policy))), ("ppo_dev", list(map(_prompt_row, dev))),
                         ("evaluation", list(map(_prompt_row, evaluation))),
                         ("reward_dev", [{**r, "label": labels[("test", r["id"])]["label"]} for r in reward_dev]),
                         ("rejected", rejected),
                         ("provenance", [{"split": key[0], "id": key[1], "sources": value}
                                         for key, value in sorted(provenance.items())])):
        path = run.file(f"data/{name}.jsonl")
        write_rows(path, values)
        outputs.append(path)
    report = {"protocol": PARSER_PROTOCOL, "source_revision": c["data"]["revision"],
              "raw_counts": dict(raw_counts), "parse_token_accepted_rows": dict(accepted_counts),
              "valid_pool_pairs": len(train), "valid_test_pairs": len(test),
              "train_unique_contexts": len(train_prompts), "excluded_test_overlap_pairs": len(overlap),
              "duplicate_rows_collapsed": sum(len(v) - 1 for v in provenance.values()),
              "conflicting_pair_groups": len(conflicts),
              "rejected_by_reason": dict(Counter(r["reason"] for r in rejected)),
              "policy_prompts": len(policy), "reward_dev_pairs": len(reward_dev),
              "ppo_dev_prompts": len(dev), "evaluation_prompts": len(evaluation),
              "pool_order": "canonical_pair_id", "training_labels_separate": True,
              "eos_logp": "included_in_canonical_response", "context_tokens": c["data"]["context_tokens"],
              "tokenizer": c["model"], "tokenizer_semantics": tokenizer_fingerprint(tokenizer),
              "subsets": c["data"]["subsets"]}
    report["source_counts"] = {name: dict(Counter(r["source"]["subset"] for r in values))
                               for name, values in (("pool", train), ("policy", policy),
                                   ("reward_dev", reward_dev), ("ppo_dev", dev), ("evaluation", evaluation))}
    write_json(run.file("data/manifest.json"), report)
    outputs.append(run.file("data/manifest.json"))
    seal_and_publish(run, "data", outputs, dependencies, metadata=report,
                     code_paths=_sources(), context=tokenizer_context)
    return report
