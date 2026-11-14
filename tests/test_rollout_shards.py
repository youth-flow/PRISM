from copy import deepcopy
import sys
from types import ModuleType

import numpy as np
import pytest

from prism import pipeline
from prism.io import root, rows, verify_receipt
from test_evaluation_v2 import config


def install_backend(monkeypatch):
    class Tokenizer:
        eos_token_id = 1
        def decode(self, tokens, skip_special_tokens):
            return " ".join(map(str, tokens))
    backend = ModuleType("prism.hf")
    backend.tokenizer = lambda c: Tokenizer()
    backend.load_lm = lambda c: object()
    backend.prompt_ids = lambda t, messages, cap: [len(messages[0]["content"]), 2]
    backend.generate = lambda model, t, ids, cap, seed: [int(seed % 13 + 3), 1]
    backend.sequence_stats = lambda model, ids, response, eos: (np.array([ids[0], response[0], 1.0], dtype=np.float32), float(-response[0]))
    monkeypatch.setitem(sys.modules, "prism.hf", backend)


def test_shards_preserve_global_seeds_features_and_canonical_record_order(tmp_path, monkeypatch):
    c = config(tmp_path)
    install_backend(monkeypatch)
    pipeline.rollout(c)
    baseline = root(c)
    split = deepcopy(c)
    split["output"] += "-sharded"
    # Execute out of order to catch accidental completion-order merging.
    for i in (2, 0, 1):
        pipeline.rollout(split, shard_index=i, shards=3)
    pipeline.merge_rollout(split, shards=3)
    merged = root(split)
    with np.load(baseline / "pool.npz", allow_pickle=False) as a, np.load(merged / "pool.npz", allow_pickle=False) as b:
        np.testing.assert_array_equal(a["phi"], b["phi"])
        np.testing.assert_array_equal(a["logp"], b["logp"])
    assert rows(baseline / "pool.jsonl") == rows(merged / "pool.jsonl")
    assert verify_receipt(merged / "pool.receipt.json")["rollout_shards"] == 3


def test_merge_refuses_missing_or_corrupt_shards_without_publishing_pool(tmp_path, monkeypatch):
    c = config(tmp_path)
    install_backend(monkeypatch)
    pipeline.rollout(c, shard_index=0, shards=2)
    with pytest.raises(FileNotFoundError):
        pipeline.merge_rollout(c, shards=2)
    assert not (root(c) / "pool.npz").exists()
    pipeline.rollout(c, shard_index=1, shards=2)
    path = root(c) / "rollout_shards/part-01-of-02/pool.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact changed"):
        pipeline.merge_rollout(c, shards=2)
    assert not (root(c) / "pool.receipt.json").exists()


def test_real_tiny_olmo_shards_match_unsharded_generation(tmp_path):
    pytest.importorskip("torch")
    from test_prism_model_contract import tiny_olmo2
    model = tmp_path / "tiny-olmo2"
    tiny_olmo2(model)
    c = config(tmp_path)
    c["choices"]["sft"] = {"model": str(model), "revision": "0" * 40}
    pipeline.rollout(c)
    split = deepcopy(c)
    split["output"] += "-real-sharded"
    for i in (1, 0):
        pipeline.rollout(split, shard_index=i, shards=2)
    pipeline.merge_rollout(split, shards=2)
    assert rows(root(c) / "pool.jsonl") == rows(root(split) / "pool.jsonl")
    with np.load(root(c) / "pool.npz", allow_pickle=False) as a, np.load(root(split) / "pool.npz", allow_pickle=False) as b:
        np.testing.assert_array_equal(a["phi"], b["phi"])
        np.testing.assert_array_equal(a["logp"], b["logp"])


def test_32_small_shards_merge_with_uneven_last_groups(tmp_path, monkeypatch):
    from prism.io import write_rows
    c = config(tmp_path)
    path = tmp_path / "more-design.jsonl"
    write_rows(path, [{"id": str(i), "messages": [{"role": "user", "content": f"prompt {i}"}]} for i in range(40)])
    c["choices"]["prompts"]["design"] = str(path)
    install_backend(monkeypatch)
    for shard in reversed(range(32)):
        pipeline.rollout(c, shard_index=shard, shards=32)
    result = pipeline.merge_rollout(c, shards=32)
    assert result == {"prompts": 40, "responses": 160, "shards": 32}
    records = rows(root(c) / "pool.jsonl")
    assert [(r["prompt_index"], r["candidate"]) for r in records] == [(i, j) for i in range(40) for j in range(4)]
    verify_receipt(root(c) / "pool.receipt.json")
