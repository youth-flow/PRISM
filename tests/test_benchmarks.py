"""Adapter contracts plus an opt-in integration test of the genuine checker."""
import json
import os
from pathlib import Path

import pytest

from prism.benchmarks import (IFEVAL_SOURCE_REVISION, benchmark_rows, score_benchmark,
                              stage_ifeval_sources, validate_benchmark_backend)
from prism.io import write_rows


def test_exact_match_adapter_preserves_legacy_scoring_and_checks_response_identity(tmp_path):
    path = tmp_path/"benchmark.jsonl"
    write_rows(path, [{"id": "a", "messages": [{"role": "user", "content": "Question"}], "answer": "HELLO world"}])
    c = {"choices": {"benchmark": {"path": str(path), "name": "unit-test"}}}
    assert validate_benchmark_backend(c) == {"kind": "exact_match", "count": 1}
    result = score_benchmark(c, [{"id": "a", "response": " hello   WORLD\n"}])
    assert result["metric"] == "normalized_exact_match" and result["score"] == 1
    assert score_benchmark(c, [{"id": "a", "response": "wrong"}])["score"] == 0
    with pytest.raises(ValueError, match="exactly match"):
        score_benchmark(c, [{"id": "missing", "response": "hello"}])
    with pytest.raises(ValueError, match="exactly match"):
        score_benchmark(c, [{"id": "a", "response": "hello"}]*2)


def ifeval_config(tmp_path):
    path = tmp_path/"ifeval.jsonl"
    write_rows(path, [{"key": 1, "prompt": "Respond in lowercase.", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{"unused_hf_padding": None}]}])
    return {"choices": {"benchmark": {"kind": "ifeval", "path": str(path), "official_source_path": str(tmp_path/"official"), "official_source_revision": IFEVAL_SOURCE_REVISION}}}


def test_ifeval_normalization_preserves_original_data_and_fails_before_generation(tmp_path):
    c = ifeval_config(tmp_path)
    records = benchmark_rows(c)
    assert records[0]["id"] == "1"
    assert records[0]["messages"] == [{"role": "user", "content": records[0]["prompt"]}]
    assert records[0]["kwargs"] == [{"unused_hf_padding": None}]
    with pytest.raises(RuntimeError, match="not staged"):
        validate_benchmark_backend(c)
    with pytest.raises(ValueError, match="exactly match"):
        score_benchmark(c, [])


def test_ifeval_rejects_changed_prompt_and_mismatched_instruction_arguments(tmp_path):
    c = ifeval_config(tmp_path)
    path = Path(c["choices"]["benchmark"]["path"])
    row = benchmark_rows(c)[0]
    row["messages"][0]["content"] = "A modified benchmark prompt"
    path.write_text(json.dumps(row)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="original prompt"):
        benchmark_rows(c)
    del row["messages"]
    row["kwargs"] = []
    path.write_text(json.dumps(row)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matched instruction_id_list"):
        benchmark_rows(c)


def test_explicit_staging_is_small_immutable_and_checks_corruption(tmp_path, monkeypatch):
    # Transport-only unit fixture; these contents are NEVER used as an official checker.
    import prism.benchmarks as module
    requested = []
    def download(revision, relative):
        requested.append((revision, relative))
        return ("TRANSPORT_FIXTURE_ONLY " + relative).encode()
    monkeypatch.setattr(module, "_download_official", download)
    revision, destination = "a"*40, tmp_path/"official"
    manifest = stage_ifeval_sources(destination, revision)
    assert len(requested) == 7
    assert manifest["revision"] == revision
    assert (destination/"LICENSE").exists()
    stage_ifeval_sources(destination, revision)
    assert len(requested) == 7  # No implicit refetch or silent overwrite.
    (destination/"instruction_following_eval/instructions.py").write_text("changed")
    with pytest.raises(ValueError, match="source hash changed"):
        stage_ifeval_sources(destination, revision)


def test_genuine_official_ifeval_checker_when_explicitly_staged(tmp_path):
    source = os.environ.get("PRISM_TEST_IFEVAL_SOURCE")
    if not source:
        pytest.skip("Set PRISM_TEST_IFEVAL_SOURCE to an explicitly staged genuine official snapshot")
    for module in ("absl", "langdetect", "nltk", "immutabledict"):
        pytest.importorskip(module)
    c = ifeval_config(tmp_path)
    c["choices"]["benchmark"]["official_source_path"] = source
    # Choose deterministic official constraints: lowercase also invokes language
    # detection, which is unsuitable for a tiny fixed-answer regression fixture.
    Path(c["choices"]["benchmark"]["path"]).write_text(
        json.dumps({"key": 1, "prompt": "Respond without a comma.",
                    "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{"unused_hf_padding": None}]})+"\n",
        encoding="utf-8",
    )
    resource_path = os.environ.get("PRISM_TEST_NLTK_DATA") or os.environ.get("NLTK_DATA")
    if resource_path:
        c["benchmark"] = {"nltk_data_path": resource_path}
    import nltk
    original_resource_paths = list(nltk.data.path)
    try:
        validate_benchmark_backend(c)
    except RuntimeError as exc:
        if "NLTK resources are missing" in str(exc):
            pytest.skip(str(exc))
        raise
    passed = score_benchmark(c, [{"id": "1", "response": "without a comma"}])
    failed = score_benchmark(c, [{"id": "1", "response": "hello, world"}])
    assert passed["metric"] == "prompt_level_strict" and passed["score"] == 1
    assert failed["score"] == 0
    assert set(passed["metrics"]) == {"prompt_level_strict", "instruction_level_strict", "prompt_level_loose", "instruction_level_loose"}
    # Distinguish prompt- and instruction-level denominators, and exercise the
    # official loose checker's first/last-line removal rather than faking it.
    Path(c["choices"]["benchmark"]["path"]).write_text(
        json.dumps({"key": 1, "prompt": "Use no commas and wrap the answer in quotation marks.",
                    "instruction_id_list": ["punctuation:no_comma", "startend:quotation"], "kwargs": [{}, {}]}) + "\n" +
        json.dumps({"key": 2, "prompt": "Respond without a comma.",
                    "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}) + "\n",
        encoding="utf-8",
    )
    result = score_benchmark(c, [{"id": "1", "response": 'comment\n"hello"\ncomment'}, {"id": "2", "response": "hello"}])
    assert result["prompt_level_strict"] == 0.5
    assert result["instruction_level_strict"] == pytest.approx(2/3)
    assert result["prompt_level_loose"] == 1.0
    assert result["instruction_level_loose"] == 1.0
    assert nltk.data.path == original_resource_paths  # No global environment/path mutation.
