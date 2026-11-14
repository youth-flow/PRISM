"""Exercise the shipped config and script through every schema-2 stage.

Only dataset transport and Judge HTTP are simulated. OLMo2 inference, feature
extraction, design, BT fitting, TRL PPO, adapter reload and official IFEval run.
The official checker/resources must have been explicitly staged beforehand.
"""
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import yaml

from prism.io import load_config, root, rows, verify_receipt
from prism.workflow import stage_plan
from test_prism_model_contract import tiny_olmo2


def test_shipped_config_drives_every_stage_from_another_directory(tmp_path, monkeypatch):
    source = os.environ.get("PRISM_TEST_IFEVAL_SOURCE")
    resources = os.environ.get("PRISM_TEST_NLTK_DATA")
    if not source or not resources:
        pytest.skip("Explicitly stage genuine IFEval source and NLTK data for this integration test")
    for dependency in ("absl", "langdetect", "nltk", "immutabledict"):
        pytest.importorskip(dependency)
    repo = Path(__file__).resolve().parents[1]
    model_path = tmp_path/"tiny-olmo2"
    tiny_olmo2(model_path)
    # Derive from the actual experiment, so new config fields and interfaces
    # participate in the same execution rather than a parallel test-only setup.
    c = yaml.safe_load((repo/"configs/experiment.yaml").read_text(encoding="utf-8"))
    c["output"] = "runs/cli-smoke"
    c["choices"]["sft"] = {"model": str(model_path), "revision": "0"*40}
    c["choices"]["prompts"] = {role: f"prepared/{role}.jsonl" for role in ("design", "policy", "evaluation")}
    c["choices"]["benchmark"].update(path="prepared/ifeval.jsonl", official_source_path=str(Path(source).resolve()))
    c["data"].update(manifest_path="prepared/data.manifest.json", model_context_tokens=128,
                     expected_counts={"design": 3, "policy": 2, "evaluation": 2, "benchmark": 2})
    c["budgets"], c["seeds"] = [16], [7, 9]
    c["runtime"] = {"device": "cpu", "dtype": "float32", "local_files_only": True}
    c["candidate"].update(max_prompt_tokens=20, max_new_tokens=4)
    c["design"].update(fw_max_iter=2000)
    c["ppo"].update(total_episodes=2, response_length=4, learning_rate=0.001,
                    per_device_train_batch_size=2, gradient_accumulation_steps=1,
                    num_mini_batches=1, num_ppo_epochs=1, lora_r=2, lora_alpha=4)
    c["evaluation"].update(max_new_tokens=4, bootstrap_samples=20)
    c["benchmark"].update(max_new_tokens=8, nltk_data_path=str(Path(resources).resolve()))
    # This temporary test config enables only the mocked transport. The actual
    # experiment's paid-call switch and all user credentials remain untouched.
    c["execution"]["allow_paid_api_calls"] = True
    for role in ("training", "evaluation"):
        name = f"PRISM_TEST_{role.upper()}_KEY"
        c["judges"][role]["api_key_env"] = name
        monkeypatch.setenv(name, "TEST_ONLY_NOT_A_REAL_KEY")
    config = tmp_path/"experiment.yaml"
    config.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")

    datasets_seen, calls = [], {"training": [], "evaluation": []}
    populations = {
        "train_gen": [{"prompt_id": i, "prompt": f"hello token{i+6}",
                       "chosen": "UNUSED_EXISTING_ANSWER", "rejected": "UNUSED_EXISTING_LABEL"} for i in range(7)],
        "test_gen": [{"prompt_id": i+100, "prompt": f"world token{i+14}"} for i in range(3)],
        "train": [{"key": 1, "prompt": "Respond without a comma.",
                   "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{"unused_hf_padding": None}]},
                  {"key": 2, "prompt": "Wrap your answer in quotation marks.",
                   "instruction_id_list": ["startend:quotation"], "kwargs": [{}]}],
    }

    def load_dataset(name, *, revision, split):
        datasets_seen.append((name, revision, split))
        assert name in (c["data"]["source"]["id"], c["data"]["benchmark_source"]["id"])
        return populations[split]

    def no_download(*args, **kwargs):
        raise AssertionError("This integration test must use only local staged assets")

    def transport(url, payload, api_key, timeout):
        assert api_key == "TEST_ONLY_NOT_A_REAL_KEY"
        role = "evaluation" if url.endswith("/responses") else "training"
        calls[role].append(payload)
        if role == "training":
            return {"id": "simulated-training", "model": payload["model"],
                    "choices": [{"message": {"content": '{"preferred_response":"response_A"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1}}, "mock-request"
        return {"id": "simulated-evaluation", "model": payload["model"], "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "TIE"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 1}}, "mock-request"

    import datasets
    import huggingface_hub
    import prism.benchmarks
    import prism.judges
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_download)
    monkeypatch.setattr(prism.benchmarks, "_download_official", no_download)
    monkeypatch.setattr(prism.judges, "request_json", transport)
    elsewhere = tmp_path/"unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    def run(stage, **cell):
        argv = [str(repo/"scripts/run.py"), stage, "--config", str(config)]
        for key, value in cell.items():
            argv.extend((f"--{key}", str(value)))
        monkeypatch.setattr(sys, "argv", argv)
        runpy.run_path(str(repo/"scripts/run.py"), run_name="__main__")

    run("validate")
    run("plan")
    normalized, _ = load_config(config, ready=False)
    jobs = stage_plan(normalized)
    assert len(jobs) == 31
    for job in jobs:
        run(job["stage"], **{key: job[key] for key in ("method", "budget", "seed") if key in job})
    prepared, missing = load_config(config, ready=True)
    assert not missing
    assert len(datasets_seen) == 3
    assert len(calls["training"]) == 64  # 2 methods x 2 seeds x 16 fresh labels.
    assert len(calls["evaluation"]) == 24  # 2 seeds x 3 comparisons x 2 prompts x AB/BA.
    output = root(prepared)
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] and not summary["missing_cells"]
    assert sum(group["completed_seeds"] for group in summary["groups"]) == 4
    assert summary["training_label_accounting"]["unique_training_calls"] == 64
    assert summary["training_label_accounting"]["evaluation_judge_calls"] == 24
    for method in c["methods"]:
        for seed in c["seeds"]:
            cell_dir = output/method/"n16"/f"seed{seed}"
            verify_receipt(cell_dir/"policy.receipt.json")
            verify_receipt(cell_dir/"evaluation.receipt.json")
            metrics = json.loads((cell_dir/"metrics.json").read_text(encoding="utf-8"))
            assert metrics["training_judge_calls"] == 16
            assert metrics["win_rate_vs_sft"] == 0.5
            assert np.isfinite(metrics["sequence_kl_mc"])
            assert set(metrics["benchmark_metrics"]) == {
                "prompt_level_strict", "instruction_level_strict", "prompt_level_loose", "instruction_level_loose"}
            assert len(rows(cell_dir/"labels.jsonl")) == 16
            assert (cell_dir/"policy/adapter_model.safetensors").is_file()
    assert (output/"results.csv").is_file() and (output/"comparisons.csv").is_file()
    assert len(rows(output/"reference/benchmark.responses.jsonl")) == 2
    # Restarting resumable API/evaluation stages must reuse successful calls.
    before = {role: len(records) for role, records in calls.items()}
    for job in jobs:
        if job["stage"] in ("label", "evaluate-reference", "evaluate-generate", "evaluate-judge", "aggregate"):
            run(job["stage"], **{key: job[key] for key in ("method", "budget", "seed") if key in job})
    assert {role: len(records) for role, records in calls.items()} == before
    # Config edits require a separate output directory, preventing mixed runs.
    changed = yaml.safe_load(config.read_text(encoding="utf-8"))
    changed["ppo"]["learning_rate"] *= 2
    config.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Config/input changed; use a NEW output directory"):
        run("rollout")
