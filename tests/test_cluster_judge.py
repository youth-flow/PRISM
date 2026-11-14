import pytest

from prism.cluster_judge import run_training
from prism.io import rows, verify_receipt
from test_evaluation_v2 import config
from test_local_judge import fake_transport, training_fixture


def test_cluster_deepseek_reuses_original_training_receipts_and_calls(tmp_path, monkeypatch):
    c = config(tmp_path)
    out = training_fixture(c)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "LOCAL_TEST_KEY")
    calls = []
    def transport(*args):
        calls.append(True)
        return fake_transport(*args)
    with pytest.raises(RuntimeError, match="explicit execution"):
        run_training(c, "prism", 8, 11, transport=transport)
    run_training(c, "prism", 8, 11, workers=3, execute=True, transport=transport)
    assert len(calls) == 8
    assert len(rows(out / "labels.jsonl")) == 8
    assert verify_receipt(out / "labels.receipt.json")["actual_judge_calls"] == 8
    run_training(c, "prism", 8, 11, execute=True, transport=transport)
    assert len(calls) == 8
    assert c["execution"]["allow_paid_api_calls"] is False


def test_cluster_runner_rejects_openai_backend(tmp_path):
    c = config(tmp_path)
    c["judges"]["training"]["backend"] = "openai"
    with pytest.raises(ValueError, match="restricted to the DeepSeek"):
        run_training(c, "prism", 8, 11, execute=True)
