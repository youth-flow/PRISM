"""Preparation uses only in-memory fixture datasets and tokenizer metadata."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from prism import data


class FixtureTokenizer:
    chat_template = "fixture only"
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return list(range(len(messages[0]["content"].split()) + 2))


class ExistingResponseMustNotBeRead:
    def __deepcopy__(self, memo):
        raise AssertionError("Existing preference responses must not be copied")


def fixtures(tmp_path):
    folder = tmp_path/"prepared"
    config = {
        "choices": {"prompts": {key: str(folder/f"{key}.jsonl") for key in ("design", "policy", "evaluation")},
                    "benchmark": {"path": str(folder/"ifeval.jsonl")},
                    "sft": {"model": "fixture/sft", "revision": "3"*40}},
        "runtime": {"local_files_only": True},
        "candidate": {"max_prompt_tokens": 20, "max_new_tokens": 4},
        "ppo": {"response_length": 4}, "evaluation": {"max_new_tokens": 4},
        "benchmark": {"max_new_tokens": 50},
        "data": {"source": {"id": "fixture/h4", "revision": "1"*40, "train_split": "train_gen",
                            "test_split": "test_gen", "prompt_field": "prompt", "id_field": "prompt_id"},
                 "benchmark_source": {"id": "fixture/ifeval", "revision": "2"*40, "split": "train"},
                 "expected_counts": {"design": 3, "policy": 3, "evaluation": 2, "benchmark": 2},
                 "model_context_tokens": 1000, "split_seed": 17, "require_disjoint_design_policy": True}}
    original = "  TRAIN\u3000row\n0  "
    train = [{"prompt_id": i, "prompt": original if i == 0 else f"TRAIN row {i}",
              "chosen": ExistingResponseMustNotBeRead(), "rejected": ExistingResponseMustNotBeRead()}
             for i in range(6)]
    train.extend([{"prompt_id": 10, "prompt": "TRAIN row 0"},  # NFKC/whitespace duplicate.
                  {"prompt_id": 11, "prompt": "Ｂｅｎｃｈ  instruction"},
                  {"prompt_id": 12, "prompt": "long "*30}, {"prompt_id": 13, "prompt": " \n "}])
    test = [{"prompt_id": i, "prompt": f"TEST row {i}"} for i in range(4)]
    test.extend([{"prompt_id": 10, "prompt": " TRAIN  row\n1 "},
                 {"prompt_id": 11, "prompt": "Bench instruction"},
                 {"prompt_id": 12, "prompt": " TEST row 0 "}])
    benchmark = [{"key": 1000, "prompt": "Bench instruction", "instruction_id_list": ["fixture:check"],
                  "kwargs": [{"word": None, "number": 2, "optional": None}]},
                 {"key": 1001, "prompt": "Another benchmark", "instruction_id_list": ["fixture:other"],
                  "kwargs": [{"target": None}]}]
    populations = {("fixture/h4", "train_gen"): train, ("fixture/h4", "test_gen"): test,
                   ("fixture/ifeval", "train"): benchmark}
    requests, tokenizer_requests = [], []

    def loader(name, *, revision, split):
        requests.append((name, revision, split))
        return populations[(name, split)]

    def tokenizer(name, **kwargs):
        tokenizer_requests.append((name, kwargs))
        return FixtureTokenizer()

    return config, populations, loader, tokenizer, requests, tokenizer_requests


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def signatures(rows):
    return {data.prompt_signature(row["messages"][0]["content"]) for row in rows}


def test_exact_counts_disjointness_original_text_and_pinned_provenance(tmp_path):
    c, populations, loader, tokenizer, requests, tokenizer_requests = fixtures(tmp_path)
    original_benchmark = copy.deepcopy(populations[("fixture/ifeval", "train")])
    report = data.prepare_data(c, loader, tokenizer)
    outputs = {key: read_rows(path) for key, path in c["choices"]["prompts"].items()}
    bench = read_rows(c["choices"]["benchmark"]["path"])
    assert report["counts"] == {"design": 3, "policy": 3, "evaluation": 2, "benchmark": 2}
    assert not signatures(outputs["design"]) & signatures(outputs["policy"])
    assert not (signatures(outputs["design"]) | signatures(outputs["policy"])) & signatures(outputs["evaluation"])
    benchmark_signatures = {data.prompt_signature(row["prompt"]) for row in bench}
    assert all(not signatures(rows) & benchmark_signatures for rows in outputs.values())
    selected = outputs["design"] + outputs["policy"]
    assert next(row for row in selected if row["source_id"] == 0)["messages"][0]["content"] == "  TRAIN\u3000row\n0  "
    assert all(set(row) == {"id", "source_id", "source_split", "messages"} for row in selected)
    for expected, actual in zip(original_benchmark, bench):
        assert all(actual[key] == value for key, value in expected.items())
    assert populations[("fixture/ifeval", "train")] == original_benchmark
    assert "id" not in populations[("fixture/h4", "train_gen")][0]
    assert tokenizer_requests == [("fixture/sft", {"revision": "3"*40, "trust_remote_code": False, "local_files_only": True})]
    assert requests == [("fixture/ifeval", "2"*40, "train"), ("fixture/h4", "1"*40, "train_gen"), ("fixture/h4", "1"*40, "test_gen")]
    assert report["excluded"]["train"] == {"raw": 10, "invalid_prompt": 1, "duplicate_prompt": 1,
                                          "benchmark_overlap": 1, "train_test_overlap": 0,
                                          "prompt_too_long": 1, "eligible": 6}
    assert report["excluded"]["test"]["train_test_overlap"] == 1
    manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == "1"*40
    for key, info in manifest["outputs"].items():
        assert hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"]
        assert len(manifest["selected_ids"][key]) == report["counts"][key]


def test_same_seed_same_populations_give_same_splits(tmp_path):
    first = fixtures(tmp_path/"first")
    second = fixtures(tmp_path/"second")
    data.prepare_data(first[0], first[2], first[3])
    data.prepare_data(second[0], second[2], second[3])
    for split in ("design", "policy", "evaluation"):
        assert Path(first[0]["choices"]["prompts"][split]).read_bytes() == Path(second[0]["choices"]["prompts"][split]).read_bytes()


def test_existing_destination_fails_before_any_loader(tmp_path):
    c, _, loader, tokenizer, requests, tokenizer_requests = fixtures(tmp_path)
    existing = Path(c["choices"]["prompts"]["design"])
    existing.parent.mkdir(parents=True)
    existing.write_text("DO NOT CHANGE", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        data.prepare_data(c, loader, tokenizer)
    assert existing.read_text() == "DO NOT CHANGE"
    assert not requests and not tokenizer_requests


@pytest.mark.parametrize("failure", ["too_few_train", "wrong_benchmark_count", "benchmark_context", "unpinned"])
def test_preparation_failure_never_publishes_partial_data(tmp_path, failure):
    c, populations, loader, tokenizer, *_ = fixtures(tmp_path)
    if failure == "too_few_train":
        c["data"]["expected_counts"]["policy"] = 20
    elif failure == "wrong_benchmark_count":
        populations[("fixture/ifeval", "train")].pop()
    elif failure == "benchmark_context":
        populations[("fixture/ifeval", "train")][0]["prompt"] = "benchmark "*960
    else:
        c["data"]["source"]["revision"] = "main"
    with pytest.raises(ValueError):
        data.prepare_data(c, loader, tokenizer)
    folder = Path(c["choices"]["benchmark"]["path"]).parent
    assert not folder.exists() or not list(folder.iterdir())


def test_failed_publication_preserves_competing_writer_and_rolls_back_owned_files(tmp_path, monkeypatch):
    c, _, loader, tokenizer, *_ = fixtures(tmp_path)
    real_link = data.os.link
    competitor = Path(c["choices"]["prompts"]["policy"])
    counter = 0

    def racing_link(source, destination):
        nonlocal counter
        counter += 1
        if counter == 2:
            competitor.write_text("COMPETING WRITER", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(data.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        data.prepare_data(c, loader, tokenizer)
    assert competitor.read_text() == "COMPETING WRITER"
    folder = competitor.parent
    assert list(folder.iterdir()) == [competitor]


def test_shared_design_policy_pool_is_explicitly_supported(tmp_path):
    c, populations, loader, tokenizer, *_ = fixtures(tmp_path)
    c["data"]["require_disjoint_design_policy"] = False
    populations[("fixture/h4", "train_gen")] = populations[("fixture/h4", "train_gen")][:3]
    data.prepare_data(c, loader, tokenizer)
    assert read_rows(c["choices"]["prompts"]["design"]) == read_rows(c["choices"]["prompts"]["policy"])


def test_selection_permutates_whole_population_and_excludes_unselected_train_overlap(tmp_path):
    c, _, loader, tokenizer, *_ = fixtures(tmp_path)
    c["data"]["split_seed"] = 0
    c["data"]["expected_counts"].update(design=1, policy=1)
    data.prepare_data(c, loader, tokenizer)
    selected = read_rows(c["choices"]["prompts"]["design"]) + read_rows(c["choices"]["prompts"]["policy"])
    assert any(row["source_id"] >= 2 for row in selected)  # Not a shuffled first-two source prefix.
    eval_signatures = signatures(read_rows(c["choices"]["prompts"]["evaluation"]))
    assert "TRAIN row 1" not in eval_signatures


def test_same_source_id_cannot_silently_name_two_prompts(tmp_path):
    c, populations, loader, tokenizer, *_ = fixtures(tmp_path)
    populations[("fixture/h4", "train_gen")].append({"prompt_id": 0, "prompt": "Different question"})
    with pytest.raises(ValueError, match="One source ID names different prompts"):
        data.prepare_data(c, loader, tokenizer)
