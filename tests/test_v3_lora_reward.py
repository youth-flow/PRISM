"""Actual tiny OLMo2/PEFT Reward updates, with no network or pretrained weights."""
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Olmo2Model"):
    pytest.skip("Pinned Transformers needed for OLMo2", allow_module_level=True)

from prism.v3.lora_reward import (RECIPE, PackedPairs, _pack, _restore, _save_checkpoint,
                                  frozen_hash, logit_metrics, make_model, optimizer, predict,
                                  trainable_state, update)
from prism.v3.ppo_trainer import seed_everything


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    seed_everything(11)
    c = transformers.Olmo2Config(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                                 bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_dropout=0.)
    c._attn_implementation = "eager"
    backbone = transformers.Olmo2Model(c).float().eval()
    weights = np.linspace(-.2, .2, 16, dtype=np.float32)
    settings = {**RECIPE, "rank": 2, "alpha": 4, "adapter_learning_rate": .001, "head_learning_rate": .002}
    model = make_model(backbone, weights, settings)
    return model, settings


class Data:
    labels = np.array([1, 0, 1])

    def pair(self, index):
        return [([1, 4, 5, 6, 7] if index == 2 else [1, 4, 5]) + [2], [1, 4, 9, 2]]


def test_zero_adapter_matches_unpadded_full_EOS_and_only_adapters_head_change():
    model, recipe = fixture()
    data = Data()
    values = predict(model, data, [0, 2], 37)
    assert values.shape == (2, 2)
    with torch.no_grad(), model.backbone.disable_adapter():
        for i, row in enumerate([0, 2]):
            for j, sequence in enumerate(data.pair(row)):
                output = model.backbone(input_ids=torch.tensor([sequence]), use_cache=False)
                expected = model.head(output.last_hidden_state[:, -1]).item()
                assert values[i, j] == pytest.approx(expected, abs=2e-7)
    before, old = frozen_hash(model), trainable_state(model)
    opt = optimizer(model, recipe)
    update(model, opt, data, [0, 1], 37, recipe)
    update(model, opt, data, [2], 37, recipe)
    new = trainable_state(model)
    assert frozen_hash(model) == before
    assert not torch.equal(new["head.weight"], old["head.weight"])
    assert any("lora_B" in n and not torch.equal(old[n], new[n]) for n in old)
    assert all("lora_" in n or n == "head.weight" for n in old)


def test_gradient_accumulation_matches_pair_mean_including_short_last_batch():
    model, recipe = fixture()
    other = deepcopy(model)
    # Disable clipping to compare the exact objective's gradients after accumulation.
    recipe = {**recipe, "gradient_norm": 1e6}
    opt = optimizer(model, recipe)
    update(model, opt, Data(), [0, 1, 2], 37, recipe)
    other.train()
    losses = []
    for i in range(3):
        score = other(Data().pair(i), 37)
        z = score[0]-score[1]
        losses.append(torch.nn.functional.softplus(z)-Data.labels[i]*z)
    exact = torch.stack(losses).mean() + .5*recipe["head_l2"]*other.head.weight.square().sum()
    exact.backward()
    for (n, p), (m, q) in zip(model.named_parameters(), other.named_parameters()):
        assert n == m
        if p.requires_grad:
            assert torch.allclose(p.grad, q.grad, atol=2e-7, rtol=1e-5)


def test_checkpoint_resume_reproduces_next_actual_adam_step(tmp_path):
    model, recipe = fixture()
    opt = optimizer(model, recipe)
    update(model, opt, Data(), [0, 1], 37, recipe)
    p = tmp_path/"checkpoint.pt"
    _save_checkpoint(p, model, opt, 1, 2, (1, .6), [], {"test": True})
    checkpoint = torch.load(p, weights_only=False)
    replica, _ = fixture()
    _restore(replica, checkpoint)
    opt2 = optimizer(replica, recipe)
    opt2.load_state_dict(checkpoint["optimizer"])
    update(model, opt, Data(), [2], 37, recipe)
    update(replica, opt2, Data(), [2], 37, recipe)
    for name, value in trainable_state(model).items():
        assert torch.equal(value, trainable_state(replica)[name])
    with pytest.raises(FileExistsError):
        _save_checkpoint(p, model, opt, 2, 3, (1, .6), [], {})


def test_label_orientation_and_tie_credit():
    m = logit_metrics([2., -2., 0.], [1, 0, 1])
    assert m["accuracy"] == pytest.approx(2.5/3)
    assert m["bt_nll"] == pytest.approx((2*np.logaddexp(0, -2)+np.log(2))/3)


def test_pack_preserves_complete_answers_and_rejects_offset_corruption(tmp_path):
    class Tokenizer:
        eos_token_id, pad_token_id = 2, 37

        def apply_chat_template(self, *a, **k):
            return [1, 4]

        def encode(self, text, **k):
            return [5, 6] if text == "A" else [7]

    records = [{"id": "first", "messages": [], "response_A": "A", "response_B": "B"}]
    report = _pack(tmp_path/"pairs", records, [1], Tokenizer(), 64)
    data = PackedPairs(tmp_path/"pairs")
    assert data.pair(0) == [[1, 4, 5, 6, 2], [1, 4, 7, 2]]
    assert report["tokens"] == 9 and report["max_sequence_tokens"] == 5
    with pytest.raises(FileExistsError):
        _pack(tmp_path/"pairs", records, [1], Tokenizer(), 64)
    np.savez(tmp_path/"pairs.npz", offsets=np.array([[0, 10, 9]]), ids=np.array(["first"]), labels=np.array([1]))
    with pytest.raises(ValueError, match="packed"):
        PackedPairs(tmp_path/"pairs")
