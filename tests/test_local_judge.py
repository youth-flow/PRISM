"""Cross-host packets use fake API transports, real stage receipts and scoring."""
import json

import pytest

from prism import pipeline
from prism.io import cell, dump, receipt, root, rows, verify_receipt, write_rows
from prism.local_judge import export_packet, import_results, run_packet
from test_evaluation_v2 import config, save_generations


def fake_transport(url, payload, key, timeout):
    assert key == "LOCAL_TEST_KEY"
    if url.endswith("/responses"):
        return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "TIE"}]}], "usage": {"input_tokens": 9, "output_tokens": 1}}, "test-eval"
    return {"choices": [{"message": {"content": "A"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 1}}, "test-train"


def test_evaluation_packet_round_trip_resume_and_input_binding(tmp_path, monkeypatch):
    c = config(tmp_path)
    save_generations(c)
    packet, result = tmp_path / "packet.json", tmp_path / "result.jsonl"
    exported = export_packet(c, "evaluate-judge", "prism", 8, 11, packet)
    assert exported["tasks"] == 8  # Two comparisons x two prompts x AB/BA.
    monkeypatch.setenv("OPENAI_API_KEY", "LOCAL_TEST_KEY")
    calls = []
    def transport(*args):
        calls.append(True)
        return fake_transport(*args)
    with pytest.raises(RuntimeError, match="--execute"):
        run_packet(packet, result, transport=transport)
    run_packet(packet, result, execute=True, transport=transport, workers=3)
    assert len(calls) == 8
    run_packet(packet, result, execute=True, transport=transport)
    assert len(calls) == 8
    metrics = import_results(c, packet, result)
    assert metrics["win_rate_vs_sft"] == 0.5
    assert metrics["evaluation_judge_calls"] == 8
    assert import_results(c, packet, result) == metrics
    assert c["execution"]["allow_paid_api_calls"] is False
    assert "LOCAL_TEST_KEY" not in packet.read_text(encoding="utf-8") + result.read_text(encoding="utf-8")
    target = root(c) / "reference" / "evaluation.responses.jsonl"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="Artifact changed"):
        import_results(c, packet, result)


def training_fixture(c):
    p, out = root(c), cell(c, "prism", 8, 11)
    pool = [{"prompt_index": 0, "candidate": i, "messages": [{"role": "user", "content": "question"}], "response": f"answer {i}"} for i in range(2)]
    write_rows(p / "pool.jsonl", pool)
    receipt(p / "pool.receipt.json", [p / "pool.jsonl", p / "experiment.json"])
    dump(p / "design.report.json", {"prism": {"converged": True}})
    receipt(p / "design.receipt.json", [p / "design.report.json", p / "pool.receipt.json"])
    queries = [{"draw_id": i, "edge_id": 0, "prompt_index": 0, "a": 0, "b": 1, "probability": 1.0} for i in range(8)]
    write_rows(out / "queries.jsonl", queries)
    receipt(out / "queries.receipt.json", [out / "queries.jsonl", p / "design.receipt.json"])
    return out


def test_training_fresh_repeated_draws_retry_and_import_resume(tmp_path, monkeypatch):
    c = config(tmp_path)
    out = training_fixture(c)
    packet, result = tmp_path / "packet.json", tmp_path / "result.jsonl"
    export_packet(c, "label", "prism", 8, 11, packet)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "LOCAL_TEST_KEY")
    calls = []
    def transport(*args):
        calls.append(True)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "bad"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9}}, "failed-test"
        if len(calls) == 4:
            raise RuntimeError("interrupted locally")
        return fake_transport(*args)
    with pytest.raises(RuntimeError, match="interrupted locally"):
        run_packet(packet, result, execute=True, transport=transport)
    assert len(rows(result)) == 3
    # Importing partial work preserves successful/failed calls on the cluster.
    with pytest.raises(RuntimeError, match="Local Judge result missing"):
        import_results(c, packet, result)
    assert len(rows(out / "judge_calls.jsonl")) == 3
    run_packet(packet, result, execute=True, transport=fake_transport)
    import_results(c, packet, result)
    assert len(rows(out / "labels.jsonl")) == 8
    assert len(rows(out / "judge_calls.jsonl")) == 9
    assert verify_receipt(out / "labels.receipt.json")["actual_judge_calls"] == 9
    # Resume from a packet exported after partial imported work also skips it.
    next_packet = tmp_path / "next.json"
    export_packet(c, "label", "prism", 8, 11, next_packet)
    assert run_packet(next_packet, tmp_path / "no_new_calls.jsonl", execute=True, transport=lambda *args: pytest.fail("no repeat spending"))["completed"] == 8


def test_wrong_packet_or_altered_order_is_rejected_before_import(tmp_path, monkeypatch):
    c = config(tmp_path)
    training_fixture(c)
    packet, result = tmp_path / "packet.json", tmp_path / "result.jsonl"
    export_packet(c, "label", "prism", 8, 11, packet)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "LOCAL_TEST_KEY")
    run_packet(packet, result, execute=True, transport=fake_transport)
    saved = result.read_text(encoding="utf-8")
    data = rows(result)
    data[0]["result"]["flip"] = not data[0]["result"]["flip"]
    result.write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")
    with pytest.raises(ValueError, match="misaligned"):
        import_results(c, packet, result)
    result.write_text(saved, encoding="utf-8")
    data = json.loads(packet.read_text(encoding="utf-8"))
    data["tasks"][0]["left"] = "altered answer"
    packet.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash/version"):
        run_packet(packet, result, execute=True, transport=lambda *args: pytest.fail("no network"))
