import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from prism import pipeline
from prism.io import dump, load_config, receipt, root, rows, verify_receipt, write_rows


def fixture_config(tmp_path):
    c = yaml.safe_load((Path(__file__).parents[1]/"configs/experiment.yaml").read_text(encoding="utf-8"))
    for key in c["choices"]["prompts"]:
        path = tmp_path / f"{key}.jsonl"
        write_rows(path, [{"id": f"{key}-{i}", "messages": [{"role": "user", "content": f"{key} question {i}"}]} for i in range(8)])
        c["choices"]["prompts"][key] = str(path)
    path = tmp_path / "bench.jsonl"
    write_rows(path, [{"id": "b", "messages": [{"role": "user", "content": "benchmark question"}], "answer": "a"}])
    c["choices"]["benchmark"] = {"path": str(path), "name": "synthetic-test-only"}
    for key in ("sft", "judge"):
        c["choices"][key] = {"model": "test-only-local", "revision": "0"*40}
    c["output"] = str(tmp_path / "run")
    c["budgets"], c["seeds"] = [1000], [7]
    c["design"]["fw_relative_gap"] = 0.01
    c["design"]["fw_max_iter"] = 2000
    return c


def test_missing_choices_and_overlap(tmp_path):
    source = Path(__file__).parents[1]/"configs/experiment.yaml"
    _, missing = load_config(source)
    assert len(missing) == 9
    with pytest.raises(ValueError, match="Fill experiment"):
        load_config(source, ready=True)
    c = fixture_config(tmp_path)
    c["choices"]["prompts"]["evaluation"] = c["choices"]["prompts"]["design"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_config(path, ready=True)


def test_immutable_inputs(tmp_path):
    c = fixture_config(tmp_path)
    root(c)
    changed = copy.deepcopy(c)
    changed["ppo"]["kl_coef"] = 0.2
    with pytest.raises(ValueError, match="changed"):
        root(changed)
    path = tmp_path / "a.txt"
    path.write_text("one")
    receipt(tmp_path / "receipt.json", [path])
    path.write_text("two")
    with pytest.raises(ValueError, match="changed"):
        verify_receipt(tmp_path / "receipt.json")


def test_synthetic_design_sampling_training_pipeline(tmp_path, monkeypatch):
    # Synthetic data are confined to tests; production stages have no fake label switch.
    c = fixture_config(tmp_path)
    p = root(c)
    rng = np.random.default_rng(5)
    phi = rng.normal(size=(8, 4, 3))
    np.savez(p/"pool.npz", phi=phi, logp=rng.normal(size=(8, 4)))
    write_rows(p/"pool.jsonl", [{"prompt_index": i, "candidate": j, "messages": [{"role": "user", "content": str(i)}], "response": str(j)} for i in range(8) for j in range(4)])
    receipt(p/"pool.receipt.json", [p/"pool.npz", p/"pool.jsonl", p/"experiment.json"])
    pipeline.design(c)
    class TestJudge:
        calls = 0
        def __init__(self, c): pass
        def call(self, messages, a, b, seed, flip):
            TestJudge.calls += 1
            return {"label": int(np.random.default_rng(seed).integers(2)), "raw": "TEST_ONLY", "flip": flip, "seed": seed}
    import prism.hf
    monkeypatch.setattr(prism.hf, "Judge", TestJudge)
    for method in c["methods"]:
        pipeline.sample(c, method, 1000, 7)
        out = p/method/"n1000"/"seed7"
        qs = rows(out/"queries.jsonl")
        assert len(qs) == 1000 and len(set(r["edge_id"] for r in qs)) < 1000
        pipeline.label(c, method, 1000, 7)
        count = TestJudge.calls
        pipeline.label(c, method, 1000, 7)
        assert TestJudge.calls == count
        pipeline.reward(c, method, 1000, 7)
        assert (out/"reward.npz").exists()
    assert TestJudge.calls == 3000  # Each repeated edge gets a new call.
