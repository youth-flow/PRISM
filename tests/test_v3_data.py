import gzip
import json
from pathlib import Path

import pytest

from prism.io import digest, rows
from prism.v3.artifacts import Run, write_json
from prism.v3.data import canonical_pair, parse_pair, prepare_data


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1] + [2] * sum(len(m["content"]) for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 251 + 1 for char in text]


def transcript(prompt, answer):
    return f"\n\nHuman: {prompt}\n\nAssistant: {answer}"


def raw_pair(prompt, a="good", b="bad"):
    return {"chosen": transcript(prompt, a), "rejected": transcript(prompt, b)}


def tiny_config(tmp_path):
    return {"schema_version": 3, "output": str(tmp_path / "run"),
            "study": {"seed": 11}, "model": {"id": "test/model", "revision": "a" * 40},
            "runtime": {"device": "cpu", "dtype": "float32", "numerics": "fp32_consistent_v1", "local_files_only": True},
            "data": {"source_id": "Anthropic/hh-rlhf", "revision": "b" * 40,
                     "raw_dir": str(tmp_path / "raw"), "subsets": ["helpful-base", "harmless-base"],
                     "policy_prompts": 5, "reward_dev_pairs": 3, "ppo_dev_prompts": 2,
                     "evaluation_prompts": 2, "context_tokens": 4096, "max_prompt_tokens": 2048,
                     "feature_dimension": 3, "split_seed": 11},
            "design": {"surrogate_ridge": .01, "lambdas": [.001, .01, .1],
                       "max_iter": 500, "relative_gap": 1e-4, "block_size": 4}}


def write_sources(c, additions=None):
    additions = additions or {}
    root = Path(c["data"]["raw_dir"])
    files, counts = [], {"train": 0, "test": 0}
    for subset in c["data"]["subsets"]:
        for split in ("train", "test"):
            path = root / subset / f"{split}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            records = [raw_pair(f"{subset}-{split}-{i}", f"answerA {i}", f"answerB {i}") for i in range(9)]
            records += additions.get((subset, split), [])
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            counts[split] += len(records)
            files.append({"subset": subset, "split": split, "path": path.relative_to(root).as_posix(), "sha256": digest(path)})
    c["data"].update(expected_train_rows=counts["train"], expected_test_rows=counts["test"])
    marker = root / "source.manifest.json"
    write_json(marker, {"source_id": c["data"]["source_id"], "revision": c["data"]["revision"], "files": files})
    return marker


def test_shared_boundary_preserves_answer_quoted_role_markers_and_multiturn():
    history = "\n\nHuman: first\n\nAssistant: prior\n\nHuman: final"
    a = history + "\n\nAssistant: A quotes\n\nHuman: pretend\n\nAssistant: still A"
    b = history + "\n\nAssistant: B ordinary"
    messages, first, second = parse_pair(a, b)
    assert messages == [{"role": "user", "content": "first"}, {"role": "assistant", "content": "prior"},
                        {"role": "user", "content": "final"}]
    assert first == "A quotes\n\nHuman: pretend\n\nAssistant: still A"
    assert second == "B ordinary"
    a = transcript("code", "```\n\nHuman: x\n\nAssistant: quoted```\nA")
    b = transcript("code", "```\n\nHuman: x\n\nAssistant: quoted```\nB")
    assert parse_pair(a, b)[0] == [{"role": "user", "content": "code"}]


def test_ambiguous_context_divergence_and_identical_answers_rejected():
    a = "\n\nHuman: first\n\nAssistant: same\n\nHuman: differing A\n\nAssistant: yes"
    b = "\n\nHuman: first\n\nAssistant: same\n\nHuman: differing B\n\nAssistant: no"
    with pytest.raises(ValueError, match="ambiguous"):
        parse_pair(a, b)
    with pytest.raises(ValueError, match="identical"):
        parse_pair(transcript("x", "same"), transcript("x", "same"))
    with pytest.raises(ValueError):
        parse_pair("not a transcript", "neither")
    with pytest.raises(ValueError, match="ambiguous"):
        parse_pair(transcript("x", "same\n\nAssistant: A"), transcript("x", "same\n\nAssistant: B"))


def test_canonical_pair_is_label_blind_and_maps_reversed_human_label():
    first = raw_pair("question", "alpha", "beta")
    second = {"chosen": first["rejected"], "rejected": first["chosen"]}
    a, label_a = canonical_pair(first, {"subset": "helpful-base"})
    b, label_b = canonical_pair(second, {"subset": "helpful-base"})
    assert a == b
    assert label_a["label"] == 1 - label_b["label"]
    assert "label" not in a and "chosen" not in a


def test_preparation_deduplicates_conflicts_and_splits_contexts_without_leakage(tmp_path):
    c = tiny_config(tmp_path)
    consistent, conflicting = raw_pair("duplicate"), raw_pair("conflict")
    reversed_conflict = {"chosen": conflicting["rejected"], "rejected": conflicting["chosen"]}
    additions = {("helpful-base", "train"): [consistent, consistent, conflicting, reversed_conflict,
                raw_pair("x" * 2100), raw_pair("oversize", "z" * 4096, "small")],
                 ("helpful-base", "test"): [consistent]}
    manifest = write_sources(c, additions)
    result = prepare_data(c, manifest, tokenizer=FakeTokenizer())
    assert result["duplicate_rows_collapsed"] == 2
    assert result["conflicting_pair_groups"] == 1
    assert result["excluded_test_overlap_pairs"] == 1
    assert result["rejected_by_reason"]["full_response_exceeds_context"] == 1
    run = Run(c)
    pool, labels = rows(run.file("data/pool.jsonl")), rows(run.file("data/labels.jsonl"))
    assert len(pool) == 20 and len(labels) == 20
    assert [r["id"] for r in pool] == [r["id"] for r in labels]
    assert all("label" not in row for row in pool)
    names = ("policy", "ppo_dev", "evaluation", "reward_dev")
    selected = {name: rows(run.file(f"data/{name}.jsonl")) for name in names}
    signatures = {name: {json.dumps(r["messages"], sort_keys=True) for r in value} for name, value in selected.items()}
    for name in ("ppo_dev", "evaluation", "reward_dev"):
        assert not signatures[name] & {json.dumps(r["messages"], sort_keys=True) for r in pool}
        for other in names:
            if other != name:
                assert not signatures[name] & signatures[other]
    assert all(len(row["messages"][-1]["content"]) < 2048 for row in selected["policy"])
    assert result == prepare_data(c, manifest, tokenizer=FakeTokenizer())
    run.require("data")


def test_source_hash_and_raw_count_fail_before_data_publication(tmp_path):
    c = tiny_config(tmp_path)
    manifest = write_sources(c)
    path = Path(c["data"]["raw_dir"]) / "helpful-base/train.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="hash changed"):
        prepare_data(c, manifest, tokenizer=FakeTokenizer())
    assert not Run(c).completed("data")


def test_insufficient_eligible_holdout_is_explicit(tmp_path):
    c = tiny_config(tmp_path)
    manifest = write_sources(c)
    c["data"]["evaluation_prompts"] = 100
    with pytest.raises(ValueError, match="distinct eligible"):
        prepare_data(c, manifest, tokenizer=FakeTokenizer())
    assert not Run(c).completed("data")


def test_public_download_resumes_completed_hashes_after_interrupted_transfer(tmp_path, monkeypatch):
    from io import BytesIO
    from prism.v3 import data
    c = tiny_config(tmp_path)
    calls = []
    content = gzip.compress((json.dumps(raw_pair("public fixture")) + "\n").encode())
    def first_transfer(url, timeout):
        calls.append(url)
        if len(calls) == 2:
            raise OSError("fixture connection interrupted")
        return BytesIO(content)
    monkeypatch.setattr(data, "urlopen", first_transfer)
    with pytest.raises(OSError, match="interrupted"):
        data.download_raw(c)
    completed = Path(c["data"]["raw_dir"]) / "helpful-base/train.jsonl.gz"
    original = digest(completed)
    resumed = []
    def remaining_transfer(url, timeout):
        resumed.append(url)
        return BytesIO(content)
    monkeypatch.setattr(data, "urlopen", remaining_transfer)
    result = data.download_raw(c)
    assert len(result["files"]) == 4 and len(resumed) == 3
    assert not any(url.endswith("helpful-base/train.jsonl.gz") for url in resumed)
    assert digest(completed) == original
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no repeat GET")))
    assert data.download_raw(c) == result
