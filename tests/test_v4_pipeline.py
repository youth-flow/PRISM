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
    assert summary["formal_reward_reference"] == {} and "N14000_reference" not in summary
    assert not run.receipt_path("sampling").exists()


def test_linear_reward_resumes_same_one_pass_and_scores_heldout(tmp_path, monkeypatch):
    import numpy as np
    from prism.io import write_rows
    from prism.v3.reward import save_npz
    from prism.v4 import linear_reward as curve
    from v3_helpers import configuration, numerical_run

    c = configuration(tmp_path)
    c.pop("_contract")
    c.update(protocol="hh_finite_pool_v4", schema_version=4)
    run = numerical_run(c)
    manifest = run.file("features/manifest.json")
    write_json(manifest, {"pool_rows": 6})
    audit = run.file("diagnostics/cohort-audit.json")
    write_json(audit, {"passed": True})
    run.publish("cohort-audit", [audit])
    run.receipt_path("geometry").unlink()
    monkeypatch.setattr(curve, "PARTIAL_BUDGETS", (2, 3, 5))
    monkeypatch.setattr(curve, "BATCH", 2)

    advance = curve._advance
    seen = []

    def interrupt_at_second_update(features, labels, ids, arrays, step, options):
        if step == 2:
            raise RuntimeError("interrupted")
        seen.extend(ids)
        return advance(features, labels, ids, arrays, step, options)

    monkeypatch.setattr(curve, "_advance", interrupt_at_second_update)
    with pytest.raises(RuntimeError, match="interrupted"):
        curve.train(c)
    checkpoint = run.require("linear-reward-2")

    def record_update(features, labels, ids, arrays, step, options):
        seen.extend(ids)
        return advance(features, labels, ids, arrays, step, options)

    monkeypatch.setattr(curve, "_advance", record_update)
    reports = curve.train(c)
    expected = np.random.default_rng(c["study"]["seed"]).permutation(6)
    assert seen == expected.tolist()
    assert run.require("linear-reward-2") == checkpoint
    assert [r["training_pairs"] for r in reports] == [2, 3, 5, 6]
    assert [r["optimizer_steps"] for r in reports] == [1, 2, 3, 4]

    test = run.file(f"{curve.PREFIX}/test.jsonl")
    write_rows(test, [{"id": "heldout", "label": 1}])
    test_report = run.file(f"{curve.PREFIX}/test-data.json")
    write_json(test_report, {"test_pairs": 1})
    run.publish("linear-test-data", [test, test_report])
    features = run.file(f"{curve.PREFIX}/test-features.npz")
    save_npz(features, ids=np.array(["heldout"]), indices=np.array([0]), phi=np.ones((1, 2, 3), dtype=np.float32))
    run.publish("linear-test-features", [features], [run.receipt_path("linear-test-data")])
    report = curve.summarize(c)
    assert report["independent_audit_passed"]
    assert all(r["test"]["bt_nll"] == pytest.approx(np.log(2)) for r in report["checkpoints"])
    assert [r["training"]["pairs"] for r in report["checkpoints"]] == [2, 3, 5, 6]


def test_linear_reward_holdout_excludes_contexts_and_conflicted_labels(tmp_path):
    import gzip
    import json
    from prism.v3.data import canonical_pair
    from prism.v4.linear_reward import _test_rows

    def raw(prompt, chosen="answer one", rejected="answer two"):
        context = f"\n\nHuman: {prompt}\n\nAssistant: "
        return {"chosen": context + chosen, "rejected": context + rejected}

    blocked = canonical_pair(raw("blocked"), {})[0]["prompt_id"]
    records = [raw("blocked", "other", "answers"), raw("held out"), raw("held out"),
               raw("conflict"), raw("conflict", "answer two", "answer one")]
    source = tmp_path / "test.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    groups = [({"split": "test", "subset": "helpful-base"}, source)]
    heldout, counts = _test_rows({"data": {"expected_test_rows": len(records)}}, groups, {blocked})
    assert [r["messages"][0]["content"] for r in heldout] == ["held out"]
    assert counts["excluded_context_overlap_rows"] == 1
    assert counts["conflicting_pairs"] == 1


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
