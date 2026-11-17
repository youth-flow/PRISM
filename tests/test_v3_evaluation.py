"""Real packets, USD reservations and receipts with offline fake HTTP only."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.error import URLError

import numpy as np
import pytest
import yaml

from prism.io import digest, rows
from prism.v3.artifacts import Run, config_hash, read_json, write_json
from prism.v3.evaluation import (METHODS, aggregate, evaluate_generate, export_judge,
                                 generation_seed, import_judge, merge_evaluation,
                                 run_local, score_comparison)


def fixture(tmp_path, count=3, service_tier="default"):
    c = yaml.safe_load((Path(__file__).parents[1]/"configs/experiment.v3.yaml").read_text(encoding="utf-8"))
    c["output"] = str(tmp_path/"run")
    c["data"]["evaluation_prompts"] = count
    c["evaluation"]["bootstrap_samples"] = 200
    c["evaluation"]["service_tier"] = service_tier
    run = Run(c)
    prompts = [{"id": f"p{i}", "messages": [{"role": "user", "content": f"Question {i}"}],
                "source": {"subset": "harmless-base" if i % 2 else "helpful-base", "split": "test", "line": i}}
               for i in range(count)]
    target = run.file("data/evaluation.jsonl")
    target.parent.mkdir(parents=True)
    target.write_text("".join(json.dumps(r)+"\n" for r in prompts), encoding="utf-8")
    run.publish("data", [target])
    selection = run.file("study/selection.json")
    write_json(selection, {"budget_authority": "user_after_design_report", "budget": 17, "design_lambda": .01})
    run.publish("study-selection", [selection])
    for method in METHODS[1:]:
        artifact = run.file(f"policy/{method}/actor/fixture.json")
        write_json(artifact, {"method": method})
        run.publish(f"policy-{method}", [artifact])
    return c, run, prompts


def fake_generator(c, method, record, index):
    logp = -3.0
    ref = logp if method == "sft" else -2.8
    from prism.v3.artifacts import hash_json
    return {"id": record["id"], "index": index, "method": method, "messages": record["messages"],
            "source": record["source"], "prompt_sha256": hash_json(record["messages"]),
            "generation_seed": generation_seed(c, index), "response": f"{method}: answer {index}",
            "response_tokens": [10 + index, 2], "response_length": 2, "eos": True, "truncated": False,
            "policy_sequence_logp": logp, "sft_sequence_logp": ref,
            "sampled_sequence_log_ratio": logp-ref, "config_sha256": config_hash(c)}


def generate_all(c, shards=2):
    for method in METHODS:
        for shard in range(shards):
            evaluate_generate(c, method, shard, shards, generator=fake_generator)
        merge_evaluation(c, method, shards)


def test_shards_cover_order_and_preserve_negative_mc_kl(tmp_path):
    c, run, prompts = fixture(tmp_path)
    generate_all(c)
    for method in METHODS:
        data = rows(run.file(f"evaluation/{method}/responses.jsonl"))
        assert [r["id"] for r in data] == [r["id"] for r in prompts]
        assert [r["generation_seed"] for r in data] == [generation_seed(c, i) for i in range(len(prompts))]
        assert all(r["sampled_sequence_log_ratio"] < 0 for r in data) if method != "sft" else all(r["sampled_sequence_log_ratio"] == 0 for r in data)
    # A shard receipt makes repeated generation a read-only operation.
    evaluate_generate(c, "prism", 0, 2, generator=lambda *args: pytest.fail("completed shard regenerated"))
    assert read_json(run.file("evaluation/prism/generation.metrics.json"))["sequence_kl_mc"] < 0


def test_shard_prefix_resume_does_not_repeat_answers(tmp_path):
    c, run, _ = fixture(tmp_path, count=4)
    sent = []
    def crash(c, method, record, index):
        if index == 2:
            raise RuntimeError("interrupted")
        sent.append(index)
        return fake_generator(c, method, record, index)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_generate(c, "sft", generator=crash)
    assert sent == [0, 1]
    resumed = []
    def resume(c, method, record, index):
        resumed.append(index)
        return fake_generator(c, method, record, index)
    evaluate_generate(c, "sft", generator=resume)
    assert resumed == [2, 3]
    merge_evaluation(c, "sft", 1)
    assert len(rows(run.file("evaluation/sft/responses.jsonl"))) == 4


def test_tie_half_is_distinct_from_position_disagreement(tmp_path):
    c, _, prompts = fixture(tmp_path)
    comp = c["evaluation"]["comparisons"][0]
    tie = {f"{comp['name']}:{i}:{order}": .5 for i in range(len(prompts)) for order in ("AB", "BA")}
    out = score_comparison(c, comp, prompts, tie, bootstrap_seed=1)
    assert out["win_rate"] == .5 and out["ci95"] == [.5, .5]
    assert out["raw_judgment_tie_fraction"] == 1 and out["position_disagreement_rate"] == 0
    opposed = {key: (1 if key.endswith(":AB") else 0) for key in tie}
    opposite = score_comparison(c, comp, prompts, opposed, bootstrap_seed=1)
    assert opposite["win_rate"] == .5 and opposite["raw_judgment_tie_fraction"] == 0
    assert opposite["position_disagreement_rate"] == 1
    assert set(out["subsets"]) == {"helpful", "harmless", "helpful-base", "harmless-base"}


def fake_http(sent, text="TIE"):
    def response(url, payload, key, timeout):
        assert url == "https://api.openai.com/v1/responses" and key == "OFFLINE_TEST_KEY"
        sent.append(payload)
        return {"status": "completed", "service_tier": payload["service_tier"],
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 9, "output_tokens": 1}}, f"fake-{len(sent)}"
    return response


def test_full_offline_packet_ledger_import_aggregate_and_replay(tmp_path, monkeypatch):
    c, run, prompts = fixture(tmp_path)
    generate_all(c)
    export_judge(c)
    packet = run.file("judge/packet.json")
    payload = read_json(packet)
    assert len(payload["tasks"]) == len(prompts)*3*2
    assert payload["settings"]["model"] == "gpt-6-luna"
    assert c["execution"]["allow_paid_api_calls"] is False
    results, ledger = tmp_path/"results.jsonl", tmp_path/"v3-eval.sqlite"
    sent = []
    inspect = run_local(packet, results, ledger)
    assert inspect["http_requests_sent"] == 0 and not ledger.exists()
    monkeypatch.setenv("OPENAI_API_KEY", "OFFLINE_TEST_KEY")
    run_local(packet, results, ledger, budget_usd=1, execute=True, transport=fake_http(sent), workers=3)
    assert len(sent) == 18
    run_local(packet, results, ledger, budget_usd=1, execute=True, transport=lambda *a: pytest.fail("duplicate dispatch"))
    imported = import_judge(c, results)
    assert imported["metadata"]["valid_judgments"] == 18
    assert imported["metadata"]["unknown_reserved_usd"] == 0
    aggregate(c)
    aggregate(c)  # completed summaries are immutable and idempotent
    summary = read_json(run.file("summaries/results.json"))
    assert all(r["win_rate"] == .5 and r["raw_judgment_tie_fraction"] == 1 for r in summary["comparisons"])
    assert summary["generation"]["prism"]["sequence_kl_mc"] < 0
    assert summary["billing"]["usd_dispatches"] == 18
    assert len(run.file("summaries/results.csv").read_text().splitlines()) == 4
    # Frozen input edits cannot silently acquire their old evaluation results.
    run.file("evaluation/prism/responses.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        aggregate(c)


def test_no_old_ledger_or_cap_reset_and_no_cluster_api(tmp_path, monkeypatch):
    from prism.api_budget import USDBudget
    c, run, _ = fixture(tmp_path)
    generate_all(c)
    export_judge(c)
    packet, results = run.file("judge/packet.json"), tmp_path/"results.jsonl"
    with pytest.raises(ValueError, match="positive USD"):
        run_local(packet, results, tmp_path/"v3-new.sqlite", execute=True)
    with pytest.raises(ValueError, match="V2 ledger reuse"):
        run_local(packet, results, tmp_path/"v2-api-budget.sqlite", budget_usd=1, execute=True)
    other = tmp_path/"v3-other.sqlite"
    USDBudget(other, total_usd=1, purposes={"training": 1})
    original = digest(other)
    with pytest.raises(ValueError, match="not owned"):
        run_local(packet, results, other, budget_usd=1, execute=True)
    assert digest(other) == original
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="locally"):
        run_local(packet, results, tmp_path/"v3-new.sqlite", budget_usd=1, execute=True)


def test_unknown_network_cost_retained_and_cannot_import_as_tie(tmp_path, monkeypatch):
    c, run, _ = fixture(tmp_path, count=1)
    generate_all(c, shards=1)
    export_judge(c)
    monkeypatch.setenv("OPENAI_API_KEY", "OFFLINE_TEST_KEY")
    results = tmp_path/"results.jsonl"
    def outage(*args):
        raise URLError("offline simulated unanswered request")
    with pytest.raises(RuntimeError, match="exhausted"):
        run_local(run.file("judge/packet.json"), results, tmp_path/"v3-cost.sqlite", budget_usd=1,
                  workers=1, execute=True, transport=outage)
    billing = read_json(str(results)+".billing.json")
    assert len(billing["dispatches"]) == 3
    assert all(r["charged_micro_usd"] is None and r["status"] == "uncertain" for r in billing["dispatches"])
    assert billing["state"]["usage"]["evaluation"]["unresolved_reservations_usd"] > 0
    with pytest.raises(ValueError, match="one valid"):
        import_judge(c, results)


def test_raw_judgment_cannot_be_changed_while_reusing_real_usage(tmp_path, monkeypatch):
    c, run, _ = fixture(tmp_path, count=1)
    generate_all(c, shards=1)
    export_judge(c)
    results = tmp_path/"results.jsonl"
    monkeypatch.setenv("OPENAI_API_KEY", "OFFLINE_TEST_KEY")
    run_local(run.file("judge/packet.json"), results, tmp_path/"v3-eval.sqlite", budget_usd=1,
              execute=True, workers=1, transport=fake_http([]))
    records = rows(results)
    records[0]["result"].update(raw="A", label=1)
    results.write_text("".join(json.dumps(r)+"\n" for r in records))
    billing_path = Path(str(results)+".billing.json")
    billing = read_json(billing_path)
    billing["results_sha256"] = digest(results)
    write_json(billing_path, billing, overwrite=True)
    with pytest.raises(ValueError, match="preserved provider"):
        import_judge(c, results)


def test_bootstrap_is_prompt_level_not_order_level(tmp_path):
    c, _, prompts = fixture(tmp_path, count=2)
    comp = c["evaluation"]["comparisons"][0]
    labels = {f"{comp['name']}:{i}:{order}": float(i) for i in (0, 1) for order in ("AB", "BA")}
    scores = score_comparison(c, comp, prompts, labels, bootstrap_seed=123)
    assert scores["standard_error"] == pytest.approx(.5)
    assert scores["position_disagreement_rate"] == 0
    again = score_comparison(c, comp, prompts, labels, bootstrap_seed=123)
    assert scores == again


@pytest.mark.parametrize("status", ["uncertain", "failed_no_charge"])
def test_unknown_or_documented_no_charge_dispatch_cannot_supply_valid_judgment(tmp_path, monkeypatch, status):
    c, run, _ = fixture(tmp_path, count=1, service_tier="flex")
    generate_all(c, shards=1)
    export_judge(c)
    results = tmp_path/"results.jsonl"
    monkeypatch.setenv("OPENAI_API_KEY", "OFFLINE_TEST_KEY")
    run_local(run.file("judge/packet.json"), results, tmp_path/"v3-eval.sqlite", budget_usd=1,
              execute=True, workers=1, transport=fake_http([]))
    billing_path = Path(str(results)+".billing.json")
    billing = read_json(billing_path)
    # The raw valid TIE is unchanged. Removing its completed-provider evidence
    # must not leave it usable merely because a reservation/no-charge row exists.
    entry = billing["dispatches"][0]
    entry["status"] = status
    entry["charged_micro_usd"] = None if status == "uncertain" else 0
    entry["response"] = None if status == "uncertain" else {
        "kind": "prism_received_http_error", "status": 429, "body_complete": True,
        "body_base64": base64.b64encode(b'{"error":{"code":"resource_unavailable"}}').decode()}
    usage = billing["state"]["usage"]["evaluation"]
    usage["charged_usd_estimate"] = sum(e["charged_micro_usd"] or 0 for e in billing["dispatches"])/1e6
    usage["unresolved_reservations_usd"] = sum(e["reserved_micro_usd"] for e in billing["dispatches"] if e["charged_micro_usd"] is None)/1e6
    write_json(billing_path, billing, overwrite=True)
    with pytest.raises(ValueError, match="completed original provider response"):
        import_judge(c, results)


def _shell_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


@pytest.fixture
def platform(tmp_path):
    bash = shutil.which("bash")
    if os.name == "nt":
        bash = str(Path(shutil.which("git")).parents[1]/"bin/bash.exe")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash unavailable")
    storage = tmp_path/"storage"
    repo = storage/"PRISM"
    repo.mkdir(parents=True)
    shutil.copytree(Path(__file__).parents[1]/"scripts/yancheng", repo/"scripts/yancheng")
    for command in (["git", "init", "-b", "v3"], ["git", "add", "."],
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    config = storage/"v3 frozen.yaml"
    config.write_text("test: true\n")
    env = {**os.environ, "PRISM_STORAGE": _shell_path(storage), "PRISM_REPO": _shell_path(repo),
           "PRISM_ACCOUNT": "account", "PRISM_GPU_PARTITION": "q_a100", "PRISM_CPU_PARTITION": "q_amd_share",
           "PRISM_TEST_MODE": "1", "PRISM_ALLOW_SUBMIT": "0"}
    for name, relative in {"IMAGE": "images/prism.sif", "HF_CACHE": "hf", "WORK": "work", "LOGS": "logs",
                           "APPTAINER_CACHE": "cache", "APPTAINER_TMP": "tmp"}.items():
        env["PRISM_"+name] = _shell_path(storage/relative)
    def run(*args, overrides=None):
        return subprocess.run([bash, _shell_path(repo/"scripts/yancheng/submit_v3.sh"), *args],
                              env={**env, **(overrides or {})}, capture_output=True, text=True, encoding="utf-8")
    return run, config, repo, storage


def test_v3_submit_resources_snapshot_paths_and_no_side_effects(platform):
    run, config, repo, storage = platform
    for stage in ("geometry", "design"):
        result = run("--dry-run", stage, "--config", _shell_path(config))
        assert result.returncode == 0, result.stderr
        assert "--cpus-per-task=16" in result.stdout and "--mem=24G" in result.stdout
        assert "--gres" not in result.stdout
    gpu = run("--dry-run", "evaluate-generate", "--config", _shell_path(config), "--method", "sft")
    assert gpu.returncode == 0, gpu.stderr
    assert "--cpus-per-task=2" in gpu.stdout and "--mem=16G" in gpu.stdout and "--gres=gpu:1" in gpu.stdout
    assert "/work/releases/" in gpu.stdout and "run_v3.sh" in gpu.stdout
    policy = run("--dry-run", "policy", "--config", _shell_path(config), "--method", "prism")
    assert policy.returncode == 0 and "--cpus-per-task=8" in policy.stdout and "--mem=64G" in policy.stdout
    assert not (storage/"work").exists() and not (storage/"logs").exists()
    # A user's untracked research file is never part of git archive.
    (repo/"unrelated.pdf").write_bytes(b"user document")
    assert run("--dry-run", "design", "--config", _shell_path(config)).returncode == 0


def test_v3_submit_blocks_api_arrays_excess_memory_and_actual_submit(platform):
    run, config, _, _ = platform
    assert run("--dry-run", "evaluate-judge", "--config", _shell_path(config)).returncode != 0
    bad_memory = run("--dry-run", "design", "--config", _shell_path(config), overrides={"PRISM_MEMORY": "64G"})
    assert bad_memory.returncode != 0 and "memory/core" in bad_memory.stderr
    matched = run("--dry-run", "design", "--config", _shell_path(config), overrides={"PRISM_MEMORY": "72G", "PRISM_CPUS": "48"})
    assert matched.returncode == 0, matched.stderr
    # The GPU partition's host memory is not subjected to the CPU queue limit.
    assert run("--dry-run", "policy", "--config", _shell_path(config)).returncode == 0
    array = run("--dry-run", "extract-features", "--config", _shell_path(config), "--shards", "32",
                overrides={"PRISM_ARRAY": "0-31%8"})
    assert array.returncode == 0 and "--array=0-31%8" in array.stdout
    excess = run("--dry-run", "extract-features", "--config", _shell_path(config), "--shards", "64",
                 overrides={"PRISM_ARRAY": "0-50%8"})
    assert excess.returncode != 0
    blocked = run("policy", "--config", _shell_path(config), overrides={"PRISM_ALLOW_SUBMIT": "1"})
    assert blocked.returncode != 0 and "Test mode" in blocked.stderr
