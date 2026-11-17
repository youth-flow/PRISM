import numpy as np
import pytest

from prism.io import rows
from prism.v3.artifacts import Run
from prism.v3.data import encode_pair, prepare_data
from prism.v3.features import HFPairScorer, extract_features, merge_features
from test_v3_data import FakeTokenizer, tiny_config, write_sources


def deterministic_scorer(record):
    values = [sum(map(ord, record[key])) for key in ("response_A", "response_B")]
    return np.asarray([[v % 13, v % 17, v % 19] for v in values], dtype=np.float32), -np.asarray(values, dtype=float) / 10


def test_feature_shards_merge_exactly_in_canonical_order(tmp_path):
    c = tiny_config(tmp_path)
    source = write_sources(c)
    prepare_data(c, source, tokenizer=FakeTokenizer())
    for index in range(3):
        report = extract_features(c, index, 3, scorer=deterministic_scorer)
        assert report["generated_responses"] == report["backbone_updates"] == 0
    result = merge_features(c, 3)
    assert result["full_coverage"]
    run = Run(c)
    for role, filename in (("pool", "pool.npz"), ("reward_dev", "reward_dev.npz")):
        records = rows(run.file(f"data/{role}.jsonl"))
        with np.load(run.file(f"features/{filename}"), allow_pickle=False) as actual:
            assert actual["ids"].tolist() == [r["id"] for r in records]
            np.testing.assert_array_equal(actual["phi"], [deterministic_scorer(r)[0] for r in records])
            np.testing.assert_array_equal(actual["logp"], [deterministic_scorer(r)[1] for r in records])
    assert merge_features(c, 3) == result
    assert extract_features(c, 0, 3, scorer=lambda r: (_ for _ in ()).throw(AssertionError()))["shard_index"] == 0


def test_feature_merge_detects_missing_or_modified_shards(tmp_path):
    c = tiny_config(tmp_path)
    prepare_data(c, write_sources(c), tokenizer=FakeTokenizer())
    extract_features(c, 0, 2, scorer=deterministic_scorer)
    with pytest.raises(FileNotFoundError):
        merge_features(c, 2)
    run = Run(c)
    shard = run.file("features/part-00000-of-00002.npz")
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        merge_features(c, 2)


def test_nonfinite_or_wrong_dimension_features_never_publish(tmp_path):
    c = tiny_config(tmp_path)
    prepare_data(c, write_sources(c), tokenizer=FakeTokenizer())
    with pytest.raises(ValueError, match="Invalid human-answer"):
        extract_features(c, scorer=lambda r: (np.full((2, 3), np.nan), [-1., -2.]))
    assert not Run(c).completed("features-part-00000-of-00001")


def test_existing_human_answer_eos_feature_and_logp_match_independent_tiny_olmo(tmp_path, monkeypatch):
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version("transformers")
    except PackageNotFoundError:
        pytest.skip("Pinned transformers runtime is not installed")
    if installed != "4.52.3":
        pytest.skip("Tiny OLMo check requires the pinned transformers==4.52.3 runtime")
    transformers = pytest.importorskip("transformers", minversion="4.52.3")
    torch = pytest.importorskip("torch")
    from prism import hf
    from prism.v3.data import canonical_pair
    from test_v3_data import raw_pair
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = transformers.Olmo2ForCausalLM(transformers.Olmo2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
        eos_token_id=1, bos_token_id=0, attention_dropout=0.)).float().eval()
    class TinyTokenizer(FakeTokenizer):
        eos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return [ord(char) % 29 + 2 for char in text]
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(hf, "load_lm", lambda c: model)
    monkeypatch.setattr(hf, "tokenizer", lambda c: tokenizer)
    c = tiny_config(tmp_path)
    c["data"]["feature_dimension"] = 16
    pair, _ = canonical_pair(raw_pair("hello", "long human text", "other answer"), {"subset": "test"})
    before = {k: v.clone() for k, v in model.state_dict().items()}
    scorer = HFPairScorer(c)
    phi, logp = scorer(pair)
    prompt, responses = encode_pair(pair, tokenizer, 4096)
    with torch.no_grad():
        for i, response in enumerate(responses):
            tokens = torch.tensor([prompt + response])
            output = model(tokens, attention_mask=torch.ones_like(tokens), output_hidden_states=True, use_cache=False)
            np.testing.assert_array_equal(phi[i], output.hidden_states[-1][0, -1].numpy())
            distribution = output.logits[0].float().log_softmax(-1)
            reference = sum(distribution[len(prompt) - 1 + j, token].item() for j, token in enumerate(response))
            assert abs(logp[i] - reference) < 1e-5
    np.testing.assert_array_equal(scorer(pair)[0], phi)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
