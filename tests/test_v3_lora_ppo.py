"""Real tiny OLMo2/PEFT coverage for the isolated full-pool LoRA PPO path."""
import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Olmo2Model"):
    pytest.skip("Pinned Transformers needed for OLMo2", allow_module_level=True)

from transformers import Olmo2Config, Olmo2ForCausalLM, Olmo2Model

from prism.v3.lora_ppo import FrozenLoRANormalizedReward, make_lora_critic
from prism.v3.lora_reward import RECIPE as REWARD_RECIPE, make_model as make_reward_model
from prism.v3.ppo import make_actor
from prism.v3.ppo_trainer import PPOEngine, left_pad, prepare_inputs, seed_everything


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(original)


def configuration():
    value = Olmo2Config(vocab_size=37, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
                        pad_token_id=0, attention_dropout=0., tie_word_embeddings=False)
    value._attn_implementation = "eager"
    return value


def settings():
    return {"total_episodes": 4, "rollout_batch_size": 2, "mini_batch_size": 2,
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 2,
            "num_mini_batches": 1, "num_ppo_epochs": 1, "learning_rate": .002,
            "critic_learning_rate": .003, "critic_head_learning_rate": .004,
            "lora_r": 2, "lora_alpha": 4, "lora_target_modules": "all-linear", "lora_dropout": 0.,
            "critic_lora_r": 2, "critic_lora_alpha": 4,
            "warmup_ratio": .05, "minimum_learning_rate_ratio": .1,
            "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-5, "weight_decay": 0.,
            "gamma": 1., "lam": .95, "kl_coef": .05, "cliprange": .2, "cliprange_value": .2,
            "vf_coef": .1, "max_grad_norm": 1., "response_length": 3, "temperature": 1.,
            "top_p": 1., "top_k": 0, "whiten_rewards": False, "whiten_advantages": True,
            "local_rollout_forward_batch_size": 1, "probability_check_rows": 2,
            "max_logp_error": 1e-3, "max_zero_update_approxkl": 1e-7,
            "engineering_checkpoint": 2, "checkpoints": [2, 4],
            "durable_checkpoint_interval": 2,
            "critic_warmup_episodes": 2, "critic_warmup_epochs": 1,
            "critic_warmup_mini_batch_size": 2}


def models():
    seed_everything(11)
    config = configuration()
    recipe = {**REWARD_RECIPE, "rank": 2, "alpha": 4}
    reward_model = make_reward_model(Olmo2Model(config).float().eval(), recipe=recipe)
    with torch.no_grad():
        reward_model.head.weight.copy_(torch.linspace(-.2, .2, 16)[None])
        # A nonzero adapter is essential: copying only the scalar head must not
        # accidentally pass the Critic/Reward parity test.
        for name, parameter in reward_model.named_parameters():
            if "lora_A" in name:
                parameter.copy_(torch.linspace(-.03, .03, parameter.numel()).reshape_as(parameter))
            elif "lora_B" in name:
                parameter.copy_(torch.linspace(.02, -.02, parameter.numel()).reshape_as(parameter))
    mean, std = .17, 1.3
    reward = FrozenLoRANormalizedReward(reward_model, mean, std, eos=2, pad=37,
                                        context_tokens=64, batch_size=2)
    critic = make_lora_critic(Olmo2Model(config).float().eval(), reward_model, mean, std, settings())
    actor = make_actor(Olmo2ForCausalLM(config).float().eval(), settings())
    return actor, critic, reward, reward_model


def trainable(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def test_lora_reward_natural_and_feature_only_eos_match_direct_scores():
    _, _, reward, model = models()
    prompts, responses = [[1, 3], [1, 4]], [[7, 2], [8, 9]]
    actual = reward.score(prompts, responses)
    sequences = [prompts[0] + responses[0], prompts[1] + responses[1] + [2]]
    with torch.no_grad():
        expected = (model(sequences, 37) - reward.mean) / reward.std
    assert torch.allclose(actual, expected, atol=2e-7)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(ValueError, match="context"):
        FrozenLoRANormalizedReward(model, reward.mean, reward.std, eos=2, pad=37,
                                   context_tokens=3).score([[1, 3]], [[7, 2]])


def test_fixed_sft_normalization_is_float64_ddof_zero_and_unit_scaled():
    _, _, _, model = models()
    prompts = [[1, 3], [1, 4], [1, 5], [1, 6]]
    responses = [[7, 2], [8, 2], [9, 2], [10, 2]]
    with torch.no_grad():
        raw = model([p + r for p, r in zip(prompts, responses)], 37).double().cpu().numpy()
    mean, std = float(raw.mean(dtype=np.float64)), float(raw.std(ddof=0, dtype=np.float64))
    subject = FrozenLoRANormalizedReward(model, mean, std, eos=2, pad=37,
                                         context_tokens=64, batch_size=2)
    normalized = subject.score(prompts, responses).double().cpu().numpy()
    assert normalized.mean() == pytest.approx(0., abs=2e-6)
    assert normalized.std(ddof=0) == pytest.approx(1., abs=2e-6)
    with pytest.raises(ValueError, match="normalization"):
        FrozenLoRANormalizedReward(model, mean, 0., eos=2, pad=37)


def test_critic_copies_reward_adapter_and_normalized_head_without_sharing_storage():
    _, critic, reward, reward_model = models()
    sequences = [[1, 3, 7, 2], [1, 4, 8, 9, 2]]
    tokens = left_pad(sequences, 37, "cpu")
    with torch.no_grad():
        expected = reward.score([[1, 3], [1, 4]], [[7, 2], [8, 9]])
        values = critic(**prepare_inputs(tokens, 37))[:, -1]
    assert torch.allclose(values, expected, atol=2e-6)

    reward_adapters = {name: parameter for name, parameter in reward_model.named_parameters()
                       if "lora_" in name}
    critic_adapters = {name: parameter for name, parameter in critic.named_parameters()
                       if "lora_" in name}
    assert reward_adapters.keys() == critic_adapters.keys()
    for name in reward_adapters:
        assert torch.equal(reward_adapters[name], critic_adapters[name])
        assert reward_adapters[name] is not critic_adapters[name]
        assert reward_adapters[name].data_ptr() != critic_adapters[name].data_ptr()
    assert torch.allclose(critic.score.weight, reward_model.head.weight / reward.std)
    assert torch.allclose(critic.score.bias, torch.tensor([-reward.mean / reward.std]))


def test_ppo_updates_actor_and_critic_but_keeps_entire_lora_reward_bitwise_frozen():
    actor, critic, reward, reward_model = models()
    before_reward = {name: parameter.detach().clone() for name, parameter in reward_model.named_parameters()}
    versions = {name: parameter._version for name, parameter in reward_model.named_parameters()}
    before_actor, before_critic = trainable(actor), trainable(critic)
    subject = PPOEngine(actor, critic, reward, [[1, 5], [1, 6]], pad=37, eos=2,
                        settings=settings(), seed=11, contract_hash="lora-reward-fixture")
    result = subject.run()
    assert result["completed"] and result["completed_episodes"] == 4
    assert any(not torch.equal(value, trainable(actor)[name]) for name, value in before_actor.items())
    assert any(not torch.equal(value, trainable(critic)[name]) for name, value in before_critic.items())
    for name, parameter in reward_model.named_parameters():
        assert torch.equal(parameter, before_reward[name])
        assert parameter._version == versions[name]


def test_critic_copy_rejects_incompatible_rank_instead_of_partially_loading():
    _, _, _, reward_model = models()
    bad = settings()
    bad["critic_lora_r"] = 4
    with pytest.raises(ValueError, match="rank|adapter|LoRA"):
        make_lora_critic(Olmo2Model(configuration()).float(), reward_model, .1, 1.2, bad)
