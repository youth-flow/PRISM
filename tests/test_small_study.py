"""Small-study regressions: stable head fitting and paid-observation accounting."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit

from prism import evaluation, pipeline
from prism.io import cell, load_config, receipt, root, rows, verify_receipt, write_rows
from prism.reward import fit_bt
from test_contracts import fixture_config


def test_fixed_l2_handles_separation_and_nullspace_without_changing_mean_loss():
    # One separable observation, three parameters: unregularized finite MLE does
    # not exist. The two unobserved coordinates must remain at the unique L2 optimum.
    v, labels, lam = np.array([[1., 0., 0.]]), [1], 0.2
    w, report = fit_bt(v, labels, l2_regularization=lam)
    expected = brentq(lambda z: lam*z - expit(-z), 0, 10)
    np.testing.assert_allclose(w, [expected, 0, 0], atol=2e-6)
    np.testing.assert_allclose(report["loss"], np.logaddexp(0, -w[0]) + lam*(w @ w)/2)
    repeated, _ = fit_bt(np.repeat(v, 17, axis=0), [1]*17, l2_regularization=lam)
    np.testing.assert_allclose(repeated, w, atol=2e-6)
    assert report["finite_unique_penalized_minimizer_exists"]
    assert not report["finite_mle_certified"]
    for invalid in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="l2_regularization"):
            fit_bt(v, labels, l2_regularization=invalid)


def test_small_runtime_has_exactly_four_declared_cells():
    path = Path(__file__).parents[1]/"configs/experiment.yaml"
    c, missing = load_config(path)
    assert c["methods"] == ["empirical", "prism"]
    assert c["seeds"] == [11, 23]
    assert c["budgets"] == [1024]
    assert len(c["methods"])*len(c["budgets"])*len(c["seeds"]) == 4
    batch = c["ppo"]["per_device_train_batch_size"]*c["ppo"]["gradient_accumulation_steps"]*c["ppo"]["num_mini_batches"]
    assert batch == 64 and c["ppo"]["total_episodes"]//batch == 64
    assert c["reward"]["l2_regularization"] == 0.01
    assert missing  # API Judges and the official IFEval adapter are not wired yet.


def test_nested_labels_reuse_observations_not_edges_and_keep_old_receipts(tmp_path, monkeypatch):
    c = fixture_config(tmp_path)
    c["methods"], c["budgets"], c["seeds"] = ["empirical", "prism"], [50, 100, 200], [7, 9]
    c["sampling"] = {"budget_coupling": "nested_prefix"}
    p = root(c)
    rng = np.random.default_rng(4)
    np.savez(p/"pool.npz", phi=rng.normal(size=(8, 4, 3)), logp=rng.normal(size=(8, 4)))
    write_rows(p/"pool.jsonl", [{"prompt_index": i, "candidate": j, "messages": [{"role": "user", "content": str(i)}], "response": str(j)} for i in range(8) for j in range(4)])
    receipt(p/"pool.receipt.json", [p/"pool.npz", p/"pool.jsonl", p/"experiment.json"])
    pipeline.design(c)
    with np.load(p/"design.npz") as design:
        assert "prism" in design and "a_optimal" not in design
    with pytest.raises(ValueError, match="Cell not declared"):
        cell(c, "a_optimal", 50, 7)

    class CountingJudge:
        calls = 0
        def __init__(self, c):
            pass
        def call(self, messages, a, b, seed, flip):
            CountingJudge.calls += 1
            # A failed format consumes one paid call but produces no observation.
            return {"label": None if CountingJudge.calls == 1 else int(seed % 2), "raw": "TEST_ONLY", "flip": flip, "seed": seed}

    import prism.hf
    monkeypatch.setattr(prism.hf, "Judge", CountingJudge)
    for method in c["methods"]:
        for seed in c["seeds"]:
            for budget in (100, 50, 200):  # Both extending and requesting an older prefix.
                pipeline.sample(c, method, budget, seed)
                pipeline.label(c, method, budget, seed)
            big, small = cell(c, method, 200, seed), cell(c, method, 50, seed)
            assert rows(small/"queries.jsonl") == rows(big/"queries.jsonl")[:50]
            assert rows(small/"labels.jsonl") == rows(big/"labels.jsonl")[:50]
            assert len({r["edge_id"] for r in rows(big/"labels.jsonl")}) < 200
            verify_receipt(small/"labels.receipt.json")  # Master append did not invalidate it.
            previous = CountingJudge.calls
            pipeline.label(c, method, 50, seed)
            assert CountingJudge.calls == previous
    assert CountingJudge.calls == 2*2*200 + 1
    evaluation.aggregate(c)
    accounting = json.loads((p/"summary.json").read_text())["training_label_accounting"]
    assert accounting["unique_training_calls"] == 801
    assert accounting["unique_valid_observations"] == 800
