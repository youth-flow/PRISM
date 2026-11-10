"""Saved-output evaluation and mocked generation: no models or paid requests."""
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from prism import evaluation_v2 as evaluation
from prism.benchmarks import score_benchmark
from prism.config import load_v2
from prism.io import cell, dump, receipt, root, rows, verify_receipt, write_rows
from test_config_v2 import fixture_v2


def config(tmp_path, comparisons=None):
    path, c = fixture_v2(tmp_path)
    if comparisons is not None:
        c["evaluation"]["comparisons"] = comparisons
    c["evaluation"]["bootstrap_samples"] = 200
    return load_v2(path, c)[0]


def save_generations(c):
    p, ref = root(c), root(c)/"reference"
    ref.mkdir()
    prompts = rows(c["choices"]["prompts"]["evaluation"])
    benchmarks = rows(c["choices"]["benchmark"]["path"])
    def responses(method):
        return [{"id": r["id"], "messages": r["messages"], "response": f"{method} answer {i}",
                 "response_tokens": [4, 1], "response_length": 2, "truncated": method == "prism" and i == 0,
                 "sampled_sequence_log_ratio": (-0.2+i*0.1) if method == "prism" else 0.4+i*0.2}
                for i, r in enumerate(prompts)]
    def benchmark(path, method):
        records = [{"id": r["id"], "response": "test" if method == "prism" or i == 0 else "wrong"} for i, r in enumerate(benchmarks)]
        write_rows(path/"benchmark.responses.jsonl", records)
        scored = score_benchmark(c, records)
        dump(path/"benchmark.metrics.json", scored)
        return scored
    write_rows(ref/"evaluation.responses.jsonl", responses("sft"))
    scored = benchmark(ref, "sft")
    dump(ref/"metrics.json", {"method": "sft", "sequence_kl_mc": 0.0, "benchmark_name": "tiny", "benchmark_metric": scored["metric"], "benchmark_score": scored["score"], "benchmark_metrics": scored["metrics"]})
    receipt(ref/"reference.receipt.json", [ref/name for name in ("evaluation.responses.jsonl", "benchmark.responses.jsonl", "benchmark.metrics.json", "metrics.json")]+[p/"experiment.json"])
    for method in c["methods"]:
        for budget in c["budgets"]:
            for seed in c["seeds"]:
                out = cell(c, method, budget, seed)
                write_rows(out/"judge_calls.jsonl", [{"draw_id": i, "label": i % 2} for i in range(budget)])
                write_rows(out/"labels.jsonl", rows(out/"judge_calls.jsonl"))
                receipt(out/"labels.receipt.json", [out/"labels.jsonl", out/"judge_calls.jsonl"], actual_judge_calls=budget)
                (out/"policy.stub").write_text("saved-test-policy")
                receipt(out/"policy.receipt.json", [out/"policy.stub", out/"labels.receipt.json"])
                write_rows(out/"evaluation.responses.jsonl", responses(method))
                benchmark(out, method)
                receipt(out/"evaluation.generate.receipt.json", [out/name for name in ("evaluation.responses.jsonl", "benchmark.responses.jsonl", "benchmark.metrics.json", "policy.receipt.json")]+[ref/"reference.receipt.json"])


class CountingJudge:
    calls = []
    fail_first = True
    position_only = False

    def call(self, messages, left, right, seed, flip):
        self.calls.append((left, right, flip))
        if self.fail_first and len(self.calls) == 1:
            return {"label": None, "raw": "invalid", "flip": flip, "seed": seed, "usage": {"input_tokens": 7}, "request_id": "failed-but-billed", "error": {"type": "invalid_format"}}
        a, b = (right, left) if flip else (left, right)
        rank = {"sft": 0, "empirical": 1, "prism": 2}
        raw = "A" if self.position_only else "TIE" if messages[0]["content"].endswith("1") else "A" if rank[a.split()[0]] > rank[b.split()[0]] else "B"
        shown_score = {"A": 1, "B": 0, "TIE": 0.5}[raw]
        return {"label": 1-shown_score if flip else shown_score, "raw": raw, "flip": flip, "seed": seed,
                "usage": {"input_tokens": 5, "output_tokens": 1}, "request_id": str(len(self.calls)), "error": None}


def install_judge(monkeypatch, *, fail_first=True, position_only=False):
    judge = CountingJudge()
    judge.calls, judge.fail_first, judge.position_only = [], fail_first, position_only
    monkeypatch.setattr(evaluation, "make_judge", lambda c, role: judge)
    return judge


def test_ab_ba_tie_retry_resume_and_aggregate(tmp_path, monkeypatch):
    c = config(tmp_path)
    save_generations(c)
    judge = install_judge(monkeypatch)
    for seed in c["seeds"]:
        for method in ("prism", "empirical"):
            metrics = evaluation.evaluate_judge(c, method, 8, seed)
            assert metrics["win_rate_vs_sft"] == 0.75
            for comparison in metrics["comparisons"]:
                assert comparison["win_rate"] == 0.75
                assert comparison["standard_error"] == 0.25  # Two prompts, not four independent order judgments.
                assert comparison["position_disagreement_rate"] == 0
            if method == "prism":
                assert metrics["sequence_kl_mc"] < 0  # Finite MC estimates must not be clamped.
                assert metrics["truncation_rate"] == 0.5
            before = len(judge.calls)
            assert evaluation.evaluate_judge(c, method, 8, seed) == metrics
            assert len(judge.calls) == before
    assert len(judge.calls) == 2*3*2*2+1
    log = rows(cell(c, "prism", 8, 11)/"evaluation.judge_calls.jsonl")
    assert log[0]["usage"]["input_tokens"] == 7 and log[0]["request_id"] == "failed-but-billed"
    assert any(r["order"] == "BA" and r["raw"] == "B" and r["label"] == 1 for r in log)
    result = evaluation.aggregate(c)
    assert result["completed"] == 4 and result["missing"] == 0
    summary = json.loads((root(c)/"summary.json").read_text())
    assert len(summary["groups"]) == 2
    assert summary["sft"]["benchmark_score"] == 0.5
    assert summary["training_label_accounting"]["unique_training_calls"] == 32
    assert summary["training_label_accounting"]["evaluation_judge_calls"] == 25
    with (root(c)/"comparisons.csv").open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 6
    with (root(c)/"results.csv").open(newline="") as file:
        scalar = list(csv.DictReader(file))
    assert len(scalar) == 4 and "comparisons" not in scalar[0]


def test_position_bias_is_averaged_and_reverse_sft_comparison_is_supported(tmp_path, monkeypatch):
    comparisons = [{"name": "sft_vs_prism", "left": "sft", "right": "prism"}]
    c = config(tmp_path, comparisons)
    save_generations(c)
    judge = install_judge(monkeypatch, fail_first=False, position_only=True)
    result = evaluation.evaluate_judge(c, "prism", 8, 11)
    assert result["win_rate_vs_sft"] == 0.5
    assert result["comparisons"][0]["position_disagreement_rate"] == 1
    other = evaluation.evaluate_judge(c, "empirical", 8, 11)
    assert other["win_rate_vs_sft"] is None and other["comparisons"] == []
    assert len(judge.calls) == 4
    partial = evaluation.aggregate(c)
    assert partial["missing"] == 2
    groups = json.loads((root(c)/"summary.json").read_text())["groups"]
    assert next(r for r in groups if r["method"] == "empirical")["win_rate_vs_sft_mean"] is None


def test_interrupted_evaluation_resumes_by_comparison_prompt_and_order(tmp_path, monkeypatch):
    c = config(tmp_path)
    save_generations(c)
    first = CountingJudge()
    first.calls, first.fail_first = [], False
    class Interrupted:
        def call(self, *args):
            if len(first.calls) == 1:
                raise RuntimeError("simulated interruption before next request")
            return first.call(*args)
    monkeypatch.setattr(evaluation, "make_judge", lambda c, role: Interrupted())
    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluation.evaluate_judge(c, "prism", 8, 11)
    path = cell(c, "prism", 8, 11)/"evaluation.judge_calls.jsonl"
    completed = rows(path)[0]
    evaluation.aggregate(c)
    summary = json.loads((root(c)/"summary.json").read_text())
    assert summary["training_label_accounting"]["evaluation_judge_calls"] == 1
    resumed = install_judge(monkeypatch, fail_first=False)
    result = evaluation.evaluate_judge(c, "prism", 8, 11)
    assert len(resumed.calls) == 7  # Eight required keys, one already completed.
    assert result["evaluation_judge_calls"] == 8 and rows(path)[0] == completed


def test_exhausted_retry_log_does_not_spend_again_on_resume(tmp_path, monkeypatch):
    c = config(tmp_path)
    save_generations(c)
    class Invalid:
        calls = 0
        def call(self, messages, left, right, seed, flip):
            self.calls += 1
            return {"label": None, "flip": flip, "seed": seed, "raw": "bad", "usage": {"input_tokens": 3}, "error": {"type": "format"}}
    judge = Invalid()
    monkeypatch.setattr(evaluation, "make_judge", lambda c, role: judge)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="every attempted call is logged"):
            evaluation.evaluate_judge(c, "prism", 8, 11)
    assert judge.calls == 3
    assert len(rows(cell(c, "prism", 8, 11)/"evaluation.judge_calls.jsonl")) == 3


def test_counterpart_generation_and_id_order_are_checked_before_paid_calls(tmp_path, monkeypatch):
    c = config(tmp_path)
    save_generations(c)
    judge = install_judge(monkeypatch)
    counterpart = cell(c, "empirical", 8, 11)
    generated = counterpart/"evaluation.generate.receipt.json"
    saved = generated.read_text()
    generated.unlink()
    with pytest.raises(RuntimeError, match="Generate all matched policies"):
        evaluation.evaluate_judge(c, "prism", 8, 11)
    assert not judge.calls
    generated.write_text(saved)
    data = rows(counterpart/"evaluation.responses.jsonl")
    (counterpart/"evaluation.responses.jsonl").unlink()
    write_rows(counterpart/"evaluation.responses.jsonl", list(reversed(data)))
    receipt(generated, [counterpart/name for name in ("evaluation.responses.jsonl", "benchmark.responses.jsonl", "benchmark.metrics.json", "policy.receipt.json")]+[root(c)/"reference/reference.receipt.json"])
    with pytest.raises(ValueError, match="IDs/order"):
        evaluation.evaluate_judge(c, "prism", 8, 11)
    assert not judge.calls


def test_completed_metrics_depend_on_counterpart_receipt(tmp_path, monkeypatch):
    c = config(tmp_path)
    save_generations(c)
    judge = install_judge(monkeypatch, fail_first=False)
    evaluation.evaluate_judge(c, "prism", 8, 11)
    path = cell(c, "empirical", 8, 11)/"evaluation.generate.receipt.json"
    metadata = json.loads(path.read_text())
    metadata["new_metadata"] = "changed-after-evaluation"
    dump(path, metadata)
    before = len(judge.calls)
    with pytest.raises(ValueError, match="Artifact changed"):
        evaluation.evaluate_judge(c, "prism", 8, 11)
    assert len(judge.calls) == before


def test_shared_reference_and_separate_benchmark_cap_with_mocked_generation(tmp_path, monkeypatch):
    c = config(tmp_path)
    c["data"]["model_context_tokens"] = 256
    preflights, generated, loaded, prompt_limits = [], [], [], []
    monkeypatch.setattr(evaluation, "validate_benchmark_backend", lambda c: preflights.append(True))
    class Model:
        active = False
        def eval(self):
            return self
        @contextmanager
        def disable_adapter(self):
            active, self.active = self.active, False
            try:
                yield
            finally:
                self.active = active
    class Tokenizer:
        eos_token_id = 1
        def decode(self, tokens, skip_special_tokens):
            return "test"
    def load(c):
        assert preflights  # Checker preflight always occurs before model loading.
        loaded.append(True)
        return Model()
    def generate(model, tokenizer, ids, cap, seed, temperature):
        generated.append((model.active, cap, temperature))
        return [4, 1]
    class Peft:
        @staticmethod
        def from_pretrained(model, path, is_trainable):
            model.active = True
            return model
    module = ModuleType("peft")
    module.PeftModel = Peft
    monkeypatch.setitem(sys.modules, "peft", module)
    import prism.hf
    monkeypatch.setattr(prism.hf, "load_lm", load)
    monkeypatch.setattr(prism.hf, "tokenizer", lambda c: Tokenizer())
    def prompt_ids(tokenizer, messages, cap):
        prompt_limits.append(cap)
        return [3]
    monkeypatch.setattr(prism.hf, "prompt_ids", prompt_ids)
    monkeypatch.setattr(prism.hf, "generate", generate)
    monkeypatch.setattr(prism.hf, "sequence_stats", lambda model, *args: (np.zeros(1), 2.0 if model.active else 1.0))
    first = evaluation.evaluate_reference(c)
    assert first["benchmark_score"] == 1
    assert evaluation.evaluate_reference(c) == first and len(loaded) == 1
    out = cell(c, "prism", 8, 11)
    (out/"policy.stub").write_text("fake")
    receipt(out/"policy.receipt.json", [out/"policy.stub"])
    evaluation.evaluate_generate(c, "prism", 8, 11)
    assert all(r["sampled_sequence_log_ratio"] == 1 for r in rows(out/"evaluation.responses.jsonl"))
    assert generated.count((False, 16, 1.0)) == 2  # Shared SFT generated only once.
    assert generated.count((True, 16, 1.0)) == 2
    assert generated.count((True, 128, 0)) == 2  # Independent benchmark cap.
    assert prompt_limits.count(127) == 4  # Both benchmark sets can use the context-safe limit.
    assert prompt_limits.count(32) == 4
    before = len(generated)
    evaluation.evaluate_generate(c, "prism", 8, 11)
    assert len(generated) == before
    verify_receipt(root(c)/"reference/reference.receipt.json")
