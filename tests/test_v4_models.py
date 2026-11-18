"""Real tiny OLMo3 + PEFT: sharing, numerical gradients and PPO recovery."""
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Olmo3ForCausalLM"):
    pytest.skip("V4 needs its isolated Transformers 4.57.1 runtime", allow_module_level=True)
pytest.importorskip("peft")
from transformers import Olmo3Config, Olmo3ForCausalLM
from prism.v4.models import (frozen_view, policy_models, response_logps, sequence_stats,
                             selected_projection, storage_report)
from prism.v3.ppo_trainer import (PPOEngine, prepare_inputs, seed_everything, _trainable_state,
                                  partition_parameters)
from prism.hf import sequence_stats as dense_stats


@pytest.fixture(autouse=True)
def threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_num_threads(original)


def tiny():
    seed_everything(19)
    c = Olmo3Config(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                   sliding_window=16, layer_types=["sliding_attention", "full_attention"],
                   eos_token_id=2, pad_token_id=0, attention_dropout=0.)
    c._attn_implementation = "eager"
    return Olmo3ForCausalLM(c).float().eval()


def settings():
    from test_v3_ppo import settings as original
    p = original()
    p.update(executor="shared_frozen_storage_v4", logp_chunk_tokens=2, gradient_checkpointing=True)
    return p


def bundle():
    return policy_models(tiny(), np.linspace(-.5, .5, 16, dtype=np.float32), .3, 1.2, settings(),
                         eos=2, pad=37, context_tokens=128)


def test_chunked_projection_values_and_gradients_match_dense():
    lm = tiny()
    h = torch.randn(2, 7, 16, requires_grad=True)
    head = lm.get_output_embeddings().requires_grad_(False)
    target = torch.randint(0, 37, (2, 7))
    reference = head(h).log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    grad = torch.autograd.grad(reference.sum(), h)[0]
    actual = selected_projection(h, head, target, chunk_size=2)
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch.autograd.grad(actual.sum(), h)[0], grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("response", [[4, 2], [4, 5]])
def test_eos_and_sampled_logp_convention(response):
    lm = tiny()
    actual = sequence_stats(lm, [1, 3], response, 2, chunk_size=1)
    expected = dense_stats(lm, [1, 3], response, 2)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == pytest.approx(expected[1], abs=1e-6)


def test_shared_parameters_distinct_modules_and_trainables():
    actor, critic, reward = bundle()
    groups = partition_parameters(actor, critic)
    ids = [{id(p) for p in group} for group in groups.values()]
    assert all(not x & y for i, x in enumerate(ids) for y in ids[i + 1:])
    shared = {id(p) for p in reward.backbone.parameters()}
    assert shared <= {id(p) for p in actor.parameters()}
    assert shared <= {id(p) for p in critic.parameters()}
    assert all(not p.requires_grad for p in reward.backbone.parameters())
    assert actor.get_base_model().model is not reward.backbone
    report = storage_report(actor, critic, reward.backbone)
    assert report["unique_storage_bytes"] < report["logical_parameter_bytes"] * .7
    with pytest.raises(ValueError, match="Freeze"):
        frozen_view(tiny())


def engine(models, path=None):
    return PPOEngine(*models, [[1, 3], [1, 4], [1, 5], [1, 6]], pad=37, eos=2, settings=settings(),
                     seed=11, contract_hash="v4-tiny-verified", checkpoint_dir=path)


def test_shared_ppo_matches_three_independent_bases_and_restores(tmp_path):
    shared = bundle()
    # Use three conventional model copies as a numerical reference.
    from prism.v3.ppo import make_actor, make_critic, FrozenNormalizedReward
    lm = tiny().requires_grad_(False)
    independent = (make_actor(lm, settings()),
                   make_critic(deepcopy(tiny().model), np.linspace(-.5, .5, 16, dtype=np.float32), .3, 1.2, settings()),
                   FrozenNormalizedReward(deepcopy(tiny().model), np.linspace(-.5, .5, 16, dtype=np.float32),
                                          .3, 1.2, eos=2, pad=37, context_tokens=128))
    # All adapters receive the identical initialization, irrespective of RNG consumption.
    from prism.v3.ppo_trainer import _restore_trainable
    for source, destination in zip(shared[:2], independent[:2]):
        _restore_trainable(destination, _trainable_state(source))
    before_base = {k: v.clone() for k, v in shared[2].backbone.state_dict().items()}
    before_score = shared[2].score([[1, 3]], [[4, 2]]).clone()
    a, b = engine(shared, tmp_path / "shared"), engine(independent, tmp_path / "separate")
    batch_a = a.rollout()
    seed_everything(11)
    batch_b = b.rollout()
    assert torch.equal(batch_a["response"], batch_b["response"])
    for name in ("old_logp", "value", "returns", "advantages", "scores"):
        assert torch.allclose(batch_a[name], batch_b[name], atol=2e-6, rtol=2e-6), name
    a.update(batch_a)
    b.update(batch_b)
    for x, y in zip(shared[:2], independent[:2]):
        for k, v in _trainable_state(x).items():
            assert torch.allclose(v, _trainable_state(y)[k], atol=3e-6, rtol=3e-6), k
    assert torch.equal(before_score, shared[2].score([[1, 3]], [[4, 2]]))
    for k, v in shared[2].backbone.state_dict().items():
        assert torch.equal(before_base[k], v), k
    # Exercise the actual engine's durable save/resume and next-round equivalence.
    restored_models = bundle()
    first = engine(restored_models, tmp_path / "recovery")
    first.run(stop_after_episodes=4)
    resumed = engine(restored_models, tmp_path / "recovery")
    resumed.resume()
    assert resumed.completed_episodes == 4
    result = resumed.run()
    assert result["completed"]
    assert all(v["step_min"] == 12 for v in result["actual_optimizer_steps"].values())


def test_reference_path_and_critic_cannot_change_reward():
    actor, critic, reward = bundle()
    tokens = torch.tensor([[1, 3, 4, 2]])
    reference = reward.score([[1, 3]], [[4, 2]]).clone()
    with actor.disable_adapter():
        baseline = response_logps(actor, prepare_inputs(tokens, 37), tokens[:, 2:], 2)
    for name, p in actor.named_parameters():
        if p.requires_grad:
            with torch.no_grad():
                p.add_(.2)
    for name, p in critic.named_parameters():
        if p.requires_grad:
            with torch.no_grad():
                p.add_(.3)
    with actor.disable_adapter():
        actual = response_logps(actor, prepare_inputs(tokens, 37), tokens[:, 2:], 2)
    assert torch.equal(actual, baseline)
    assert torch.equal(reference, reward.score([[1, 3]], [[4, 2]]))
