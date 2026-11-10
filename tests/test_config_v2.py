"""Configuration contracts without external models, keys, network or paid APIs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from prism.config import load_v2, plan_summary


def fixture_v2(tmp_path):
    data = tmp_path/"data"
    data.mkdir()
    counts = {"design": 3, "policy": 2, "evaluation": 2, "benchmark": 2}
    for role, count in counts.items():
        records = [{"id": f"{role}-{i}", "messages": [{"role": "user", "content": f"{role} prompt {i}"}], "answer": "test"} for i in range(count)]
        (data/f"{role}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    c = {
        "schema_version": 2,
        "execution": {"allow_paid_api_calls": False},
        "output": "runs/test",
        "choices": {
            "prompts": {k: f"data/{k}.jsonl" for k in ("design", "policy", "evaluation")},
            "sft": {"model": "test-only-model", "revision": "a"*40},
            "benchmark": {"name": "tiny", "kind": "exact_match", "path": "data/benchmark.jsonl"},
        },
        "data": {"source": {"id": "test-only-dataset", "revision": "b"*40}, "expected_counts": counts, "require_disjoint_design_policy": True},
        "methods": ["empirical", "prism"], "budgets": [8], "seeds": [11, 23],
        "sampling": {"budget_coupling": "independent"},
        "judges": {
            "training": {"backend": "deepseek", "model": "deepseek-flash", "api": "chat_completions", "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY", "thinking": "disabled", "temperature": 1, "max_output_tokens": 64, "max_attempts": 3, "timeout_seconds": 60},
            "evaluation": {"backend": "openai", "model": "gpt-6-luna", "api": "responses", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "reasoning_effort": "low", "max_output_tokens": 1024, "max_attempts": 3, "timeout_seconds": 60},
        },
        "runtime": {"device": "cpu", "dtype": "float32", "local_files_only": True},
        "candidate": {"count": 4, "max_new_tokens": 16, "max_prompt_tokens": 32, "seed": 9},
        "design": {"surrogate_ridge": 0.01, "matrix_ridge": 0.01, "fw_max_iter": 50, "fw_relative_gap": 1e-4},
        "reward": {"l2_regularization": 0.01, "max_iter": 100, "gradient_tolerance": 1e-6, "max_weight_norm": 1000},
        "ppo": {"total_episodes": 16, "response_length": 16, "learning_rate": 3e-6, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 2, "num_mini_batches": 2, "num_ppo_epochs": 2, "kl_coef": 0.05, "cliprange": 0.2, "vf_coef": 0.1, "lora_r": 8, "lora_alpha": 16, "lora_target_modules": "all-linear"},
        "evaluation": {"max_new_tokens": 16, "seed": 4, "answer_orders": ["AB", "BA"], "comparisons": [{"name": "prism_vs_empirical", "left": "prism", "right": "empirical"}, {"name": "prism_vs_sft", "left": "prism", "right": "sft"}, {"name": "empirical_vs_sft", "left": "empirical", "right": "sft"}]},
        "benchmark": {"max_new_tokens": 128, "temperature": 0},
    }
    return tmp_path/"experiment.yaml", c


def fake_benchmark_backend(monkeypatch):
    """Isolate input-contract tests from official checker dependency installation."""
    module = ModuleType("prism.benchmarks")
    calls = []
    def validate(c):
        calls.append("validate")
    def records(c):
        calls.append("rows")
        return [json.loads(line) for line in Path(c["choices"]["benchmark"]["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    module.validate_benchmark_backend, module.benchmark_rows = validate, records
    monkeypatch.setitem(sys.modules, "prism.benchmarks", module)
    return calls


def test_paths_inputs_and_readiness_need_no_api_keys(tmp_path, monkeypatch):
    path, original = fixture_v2(tmp_path)
    calls = fake_benchmark_backend(monkeypatch)
    for env in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    c, missing = load_v2(path, original, ready=False)
    assert not missing and not calls
    assert c["output"] == str(tmp_path/"runs/test")
    assert c["choices"]["prompts"]["design"] == str(tmp_path/"data/design.jsonl")
    assert original["output"] == "runs/test"  # Caller data are not mutated.
    ready, missing = load_v2(path, original, ready=True)
    assert not missing and calls == ["validate", "rows"]
    assert ready["execution"]["allow_paid_api_calls"] is False
    assert not (tmp_path/"runs").exists()


def test_missing_files_and_official_source_are_visible_without_importing_checker(tmp_path, monkeypatch):
    path, c = fixture_v2(tmp_path)
    calls = fake_benchmark_backend(monkeypatch)
    c["choices"]["prompts"]["design"] = "data/not-staged.jsonl"
    c["choices"]["benchmark"].update(kind="ifeval", official_source_path="official-checker", official_source_revision="c"*40)
    c["data"]["manifest_path"] = "data/data.manifest.json"
    c["benchmark"]["nltk_data_path"] = "assets/nltk_data"
    _, missing = load_v2(path, c)
    assert "choices.prompts.design" in missing
    assert "choices.benchmark.official_source_path" in missing
    assert "data.manifest_path" in missing
    assert "benchmark.nltk_data_path" in missing
    assert not calls
    with pytest.raises(ValueError, match="Prepare experiment inputs"):
        load_v2(path, c, ready=True)
    assert not calls
    (tmp_path/"official-checker").mkdir()
    _, missing = load_v2(path, c)
    assert "choices.benchmark.official_source_manifest" in missing


def test_ready_checks_counts_and_normalized_content_overlap(tmp_path, monkeypatch):
    path, c = fixture_v2(tmp_path)
    fake_benchmark_backend(monkeypatch)
    c["data"]["expected_counts"]["design"] += 1
    with pytest.raises(ValueError, match="design count mismatch"):
        load_v2(path, c, ready=True)
    c["data"]["expected_counts"]["design"] -= 1
    policy_path = tmp_path/"data/policy.jsonl"
    policy = [json.loads(line) for line in policy_path.read_text().splitlines()]
    policy[0]["messages"][0]["content"] = "  ｄｅｓｉｇｎ\t prompt 0\n"
    policy_path.write_text("".join(json.dumps(r)+"\n" for r in policy))
    with pytest.raises(ValueError, match="Design/policy prompt overlap"):
        load_v2(path, c, ready=True)
    c["data"]["require_disjoint_design_policy"] = False
    load_v2(path, c, ready=True)
    evaluation_path = tmp_path/"data/evaluation.jsonl"
    evaluations = [json.loads(line) for line in evaluation_path.read_text().splitlines()]
    evaluations[0]["messages"][0]["content"] = "design prompt 0"
    evaluation_path.write_text("".join(json.dumps(r)+"\n" for r in evaluations))
    with pytest.raises(ValueError, match="Training/evaluation"):
        load_v2(path, c, ready=True)


@pytest.mark.parametrize("section,key,value", [
    (None, "budgets", [True]), (None, "seeds", [False]), (None, "methods", ["prism", "prism"]),
    ("ppo", "total_episodes", 15), ("ppo", "learning_rate", float("nan")),
    ("design", "matrix_ridge", float("inf")), ("reward", "l2_regularization", -0.1),
    ("candidate", "count", True), ("candidate", "temperature", 0.7),
    ("evaluation", "max_new_tokens", 8), ("evaluation", "answer_orders", ["AB", "AB"]),
    ("runtime", "device", "auto"), ("runtime", "local_files_only", "true"),
])
def test_structural_rejections(tmp_path, section, key, value):
    path, c = fixture_v2(tmp_path)
    (c if section is None else c[section])[key] = value
    with pytest.raises(ValueError):
        load_v2(path, c)


def test_api_fields_are_provider_aware_and_not_hf_revisions(tmp_path):
    path, c = fixture_v2(tmp_path)
    load_v2(path, c)  # No 40-character HF revision is required for either API model.
    c["judges"]["evaluation"]["api"] = "chat_completions"
    with pytest.raises(ValueError, match="must be responses"):
        load_v2(path, c)
    c["judges"]["evaluation"]["api"] = "responses"
    c["judges"]["training"]["base_url"] = "https://user:password@example.test/v1"
    with pytest.raises(ValueError, match="embedded credentials"):
        load_v2(path, c)


def test_ready_rejects_duplicate_prompt_content_even_with_distinct_ids(tmp_path, monkeypatch):
    path, c = fixture_v2(tmp_path)
    fake_benchmark_backend(monkeypatch)
    file = tmp_path/"data/evaluation.jsonl"
    records = [json.loads(line) for line in file.read_text().splitlines()]
    records[1]["messages"] = records[0]["messages"]
    file.write_text("".join(json.dumps(r)+"\n" for r in records))
    with pytest.raises(ValueError, match="duplicate normalized prompt content"):
        load_v2(path, c, ready=True)


@pytest.mark.parametrize("role", ["training", "evaluation"])
def test_api_validation_matches_adapters_https_requirement(tmp_path, role):
    path, c = fixture_v2(tmp_path)
    c["judges"][role]["base_url"] = "http://api.example.test/v1"
    with pytest.raises(ValueError, match="HTTPS endpoint"):
        load_v2(path, c)


def test_hf_judges_need_pinned_revision_but_no_api_fields(tmp_path):
    path, c = fixture_v2(tmp_path)
    for role, temperature in (("training", 1.0), ("evaluation", 0.0)):
        c["judges"][role] = {"backend": "hf", "model": f"./models/{role}", "revision": "c"*40, "max_output_tokens": 32, "max_attempts": 3, "temperature": temperature}
    normalized, missing = load_v2(path, c)
    assert not missing
    for role in ("training", "evaluation"):
        assert normalized["judges"][role]["model"] == str(tmp_path/"models"/role)
        assert "base_url" not in normalized["judges"][role]
        assert "api_key_env" not in normalized["judges"][role]
    c["judges"]["training"]["temperature"] = 0
    with pytest.raises(ValueError, match="temperature"):
        load_v2(path, c)
    c["judges"]["training"]["temperature"] = 1
    c["judges"]["evaluation"]["revision"] = "main"
    with pytest.raises(ValueError, match="pinned 40-character"):
        load_v2(path, c)


def test_comparisons_cannot_reference_disabled_methods(tmp_path):
    path, c = fixture_v2(tmp_path)
    c["evaluation"]["comparisons"][0]["right"] = "a_optimal"
    with pytest.raises(ValueError, match="distinct enabled methods"):
        load_v2(path, c)


def test_independent_benchmark_cap_and_context_limits(tmp_path):
    path, c = fixture_v2(tmp_path)
    c["data"]["model_context_tokens"] = 256
    load_v2(path, c)  # Benchmark cap 128 is independent of same-source cap 16.
    c["benchmark"]["max_new_tokens"] = 256
    with pytest.raises(ValueError, match="exceeds data.model_context_tokens"):
        load_v2(path, c)


def test_plan_summary_final_counts_and_arbitrary_smaller_plan(tmp_path):
    path, c = fixture_v2(tmp_path)
    c["data"]["expected_counts"] = {"design": 3000, "policy": 2000, "evaluation": 500, "benchmark": 541}
    c["budgets"] = [1024]
    c["ppo"].update(total_episodes=4096, per_device_train_batch_size=2, gradient_accumulation_steps=16, num_mini_batches=2)
    normalized, _ = load_v2(path, c)
    summary = plan_summary(normalized)
    assert {key: summary[key] for key in ("training_cells", "training_valid_labels", "evaluation_pair_comparisons", "evaluation_judge_calls", "benchmark_generations")} == {"training_cells": 4, "training_valid_labels": 4096, "evaluation_pair_comparisons": 3000, "evaluation_judge_calls": 6000, "benchmark_generations": 2705}
    assert summary["candidate_responses"] == 12000 and summary["candidate_edges"] == 18000
    assert summary["ppo_rollout_batch_size"] == 64 and summary["ppo_rollout_updates_per_cell"] == 64
    small = deepcopy(c)
    small["methods"], small["budgets"], small["seeds"] = ["prism"], [2, 7], [0, 2, 4]
    small["data"]["expected_counts"] = {"design": 3, "policy": 2, "evaluation": 2, "benchmark": 2}
    small["candidate"]["count"] = 3
    small["evaluation"].update(answer_orders=["BA"], comparisons=[{"name": "one", "left": "prism", "right": "sft"}])
    small, _ = load_v2(path, small)
    summary = plan_summary(small)
    assert summary["training_cells"] == 6 and summary["training_valid_labels"] == 27
    assert summary["evaluation_pair_comparisons"] == summary["evaluation_judge_calls"] == 12
    assert summary["benchmark_generations"] == 14
    assert summary["candidate_responses"] == summary["candidate_edges"] == 9
    small["sampling"]["budget_coupling"] = "nested_prefix"
    assert plan_summary(small)["training_valid_labels"] == 21
    assert not (tmp_path/"runs").exists()
