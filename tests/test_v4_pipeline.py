"""Frozen cohort transfer, full-pool fitting and fully offline V4 Judge accounting."""
from copy import deepcopy

import pytest

from prism.io import rows
from prism.v3.artifacts import Run, read_json, write_json
from prism.v4.acceptance import prepare_reference_data, cohort_audit


def test_retokenize_locked_cohorts_and_reject_reference_tampering(tmp_path):
    from test_v3_data import tiny_config, write_sources, FakeTokenizer
    from prism.v3.data import prepare_data
    c3 = tiny_config(tmp_path / "v3")
    c3["protocol"] = "hh_finite_pool_v3"
    write_sources(c3)
    prepare_data(c3, tokenizer=FakeTokenizer())
    c4 = deepcopy(c3)
    c4.update(schema_version=4, protocol="hh_finite_pool_v4", output=str(tmp_path / "v4"))
    c4["model"] = {"id": "test/olmo3", "revision": "c" * 40}
    class NewTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [3, 4] + super().apply_chat_template(messages, **kwargs)
    report = prepare_reference_data(c4, c3["output"], tokenizer=NewTokenizer())
    assert report["canonical_cohorts_preserved"] and not report["token_ids_reused"]
    assert report["tokenizer"] == c4["model"]
    audit = cohort_audit(c4, c3["output"])
    assert audit["passed"] and audit["counts"]["reward_dev"] == 3
    assert rows(Run(c4).file("data/labels.jsonl")) == rows(Run(c3).file("data/labels.jsonl"))
    assert rows(Run(c4).file("data/pool.jsonl"))[0]["prompt_tokens"] == rows(Run(c3).file("data/pool.jsonl"))[0]["prompt_tokens"] + 2
    path = Run(c3).file("data/pool.jsonl")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|Reference cohort"):
        cohort_audit(c4, c3["output"])


def test_v4_full_frozen_head_does_not_require_geometry_or_design(tmp_path):
    from v3_helpers import configuration, numerical_run
    from prism.v3.full_pool_reward import fit_full_pool_candidate, summarize_full_pool
    c = configuration(tmp_path)
    c.pop("_contract")
    c.update(protocol="hh_finite_pool_v4", schema_version=4)
    run = numerical_run(c)
    path = run.file("diagnostics/cohort-audit.json")
    write_json(path, {"passed": True})
    run.publish("cohort-audit", [path])
    # Deliberately remove the synthetic geometry receipt: fitting only needs phi.
    run.receipt_path("geometry").unlink()
    for penalty in c["reward"]["l2_candidates"]:
        result = fit_full_pool_candidate(c, penalty)
        assert result["training_pairs"] == 6
        assert result["independent_fit_check"]["passed"]
    summary = summarize_full_pool(c)
    assert summary["independent_audit_passed"] and not summary["PPO_started"]
    assert not run.receipt_path("sampling").exists()


def test_v4_judge_offline_transport_and_separate_ledger(tmp_path, monkeypatch):
    from test_v3_evaluation import fixture, generate_all, fake_http
    from prism.v3.evaluation import export_judge, run_local, import_judge, aggregate
    # Recreate the fixture's contracts under V4 without sending network requests.
    c3, old, _ = fixture(tmp_path / "old", count=2)
    c4 = deepcopy(c3)
    c4.update(schema_version=4, protocol="hh_finite_pool_v4", output=str(tmp_path / "new"))
    run = Run(c4)
    for stage in ("data", "study-selection", "policy-empirical", "policy-prism"):
        receipt = old.require(stage)
        outputs = []
        for key in receipt["outputs"]:
            relative = key[4:]
            dest = run.file(relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(old.file(relative).read_bytes())
            outputs.append(dest)
        run.publish(stage, outputs)
    generate_all(c4, shards=1)
    export_judge(c4)
    packet = read_json(run.file("judge/packet.json"))
    assert packet["v4"]["protocol"] == "hh_finite_pool_v4" and "v3" not in packet
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "OFFLINE_TEST_KEY")
    with pytest.raises(ValueError, match="isolated v4"):
        run_local(run.file("judge/packet.json"), tmp_path / "results.jsonl", tmp_path / "v3-eval.sqlite",
                  budget_usd=1, execute=True, transport=fake_http([]))
    results = tmp_path / "results.jsonl"
    run_local(run.file("judge/packet.json"), results, tmp_path / "v4-eval.sqlite", budget_usd=1,
              execute=True, workers=1, transport=fake_http([]))
    import_judge(c4, results)
    aggregate(c4)
    assert read_json(run.file("summaries/results.json"))["protocol"] == "hh_finite_pool_v4"
    assert read_json(tmp_path / "v4-eval.sqlite.v4.json")["protocol"] == "v4_judge_ledger_v1"
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    with pytest.raises(RuntimeError, match="locally"):
        run_local(run.file("judge/packet.json"), results, tmp_path / "v4-eval.sqlite")
