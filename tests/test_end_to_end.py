"""Full artifact flow with a real tiny LM/PPO and an explicitly simulated Judge."""
import json

import numpy as np

from prism import evaluation, pipeline
from prism.io import root, verify_receipt
from test_contracts import fixture_config
from test_hf_integration import tiny_model


def test_real_model_stages_and_evaluation_contract(tmp_path, monkeypatch):
    model_path = tmp_path / "tiny"
    tiny_model(model_path)
    c = fixture_config(tmp_path)
    c["choices"]["sft"]["model"] = str(model_path)
    c["runtime"] = {"device": "cpu", "dtype": "float32", "local_files_only": True}
    c["candidate"]["max_new_tokens"] = c["evaluation"]["max_new_tokens"] = 4
    c["ppo"].update(total_episodes=2, response_length=4, per_device_train_batch_size=2, gradient_accumulation_steps=1, num_mini_batches=1, num_ppo_epochs=1, lora_r=2, lora_alpha=4)
    class SimulatedJudge:
        def __init__(self, c): pass
        def call(self, messages, a, b, seed, flip):
            return {"label": int(np.random.default_rng(seed).integers(2)), "raw": "SIMULATED_TEST_ONLY", "flip": flip, "seed": seed}
    import prism.hf
    monkeypatch.setattr(prism.hf, "Judge", SimulatedJudge)
    pipeline.model_smoke(c)
    pipeline.rollout(c)
    pipeline.design(c)
    pipeline.sample(c, "empirical", 1000, 7)
    pipeline.label(c, "empirical", 1000, 7)
    pipeline.reward(c, "empirical", 1000, 7)
    pipeline.policy(c, "empirical", 1000, 7)
    evaluation.evaluate_generate(c, "empirical", 1000, 7)
    metrics = evaluation.evaluate_judge(c, "empirical", 1000, 7)
    assert metrics["training_judge_calls"] == 1000
    assert 0 <= metrics["win_rate_vs_sft"] <= 1
    assert np.isfinite(metrics["sequence_kl_mc"])
    report = evaluation.aggregate(c)
    assert report["completed"] == 1 and report["missing"] == 2
    p = root(c)
    verify_receipt(p / "empirical/n1000/seed7/evaluation.receipt.json")
    assert not json.loads((p/"summary.json").read_text())["complete"]
