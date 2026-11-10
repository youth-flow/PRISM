"""OLMo2 architecture contracts using random tiny weights, never a real checkpoint."""
import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from scipy.special import expit

transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from prism.hf import FrozenReward, sequence_stats, train_ppo
from prism.reward import fit_bt


def tiny_olmo2(path=None):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    config = Olmo2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                        max_position_embeddings=128, eos_token_id=1, bos_token_id=0,
                        attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = Olmo2ForCausalLM(config).eval()
    if path is not None:
        model.save_pretrained(path)
        vocab = {"<unk>": 0, "<eos>": 1, "user": 2, "assistant": 3,
                 "hello": 4, "world": 5, **{f"token{i}": i for i in range(6, 32)}}
        raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>")
        tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ' ' + m['content'] + ' '}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(path)
    return model


def test_olmo2_eos_reward_and_independently_summed_response_logp():
    from trl.trainer.utils import get_reward
    model = tiny_olmo2()
    weight = np.linspace(-0.4, 0.5, 32)
    adapter = FrozenReward(model.model, 1, weight)
    # Prompt padding differs by row. Cases cover sampled EOS, length truncation
    # without EOS, and an early EOS followed by padding within the common width.
    queries = torch.tensor([[32, 32, 2, 4, 5, 6, 1, 32],
                            [2, 4, 3, 7, 8, 9, 10, 11],
                            [32, 32, 32, 2, 4, 1, 32, 32]])
    _, rewards, _ = get_reward(adapter, queries, 32, 4)
    cases = [([2, 4], [5, 6, 1]), ([2, 4, 3, 7], [8, 9, 10, 11]), ([2], [4, 1])]
    for row, (prompt, response) in enumerate(cases):
        feature, logp = sequence_stats(model, prompt, response, 1)
        np.testing.assert_allclose(rewards[row].item(), feature @ weight, atol=2e-6)
        # Independent forward over exactly the generated sequence: this excludes
        # an artificial feature-only EOS even in the length-truncated case.
        tokens = torch.tensor([prompt + response])
        logits = model(tokens).logits[0].float().log_softmax(-1)
        manual = sum(logits[len(prompt)+i-1, token].item() for i, token in enumerate(response))
        np.testing.assert_allclose(logp, manual, atol=2e-6)
    assert not any(parameter.requires_grad for parameter in adapter.parameters())
    adapter.train()
    assert not adapter.backbone.model.training


def test_rank_deficient_2048_dimensional_bt_matches_scalar_stationarity_root():
    # The actual planned n/d dimensions, but a maximally rank-deficient and
    # separable design. Positive L2 has a finite optimum in span(v); nullspace
    # coordinates cannot resolve labels and must not acquire arbitrary weight.
    vector = np.random.default_rng(19).normal(size=2048)
    design = np.repeat(vector[None, :], 1024, axis=0)
    regularization = 0.01
    weight, report = fit_bt(design, np.ones(1024), l2_regularization=regularization)
    norm_squared = vector @ vector
    margin = brentq(lambda z: regularization*z/norm_squared - expit(-z), 0, 30)
    expected = margin/norm_squared*vector
    np.testing.assert_allclose(weight, expected, atol=2e-6, rtol=0)
    assert report["finite_unique_penalized_minimizer_exists"]
    assert report["gradient_inf"] < 1e-6


def test_real_tiny_olmo2_ppo_update_keeps_reward_frozen_and_reloads_adapter(tmp_path, monkeypatch):
    from peft import PeftModel
    import prism.hf as hf
    model_path = tmp_path/"tiny-olmo2"
    base = tiny_olmo2(model_path)
    captured = {}
    def capture_reward(backbone, eos, weight):
        reward = FrozenReward(backbone, eos, weight)
        captured["reward"] = reward
        captured["before"] = {key: value.detach().cpu().clone() for key, value in reward.state_dict().items()}
        return reward
    monkeypatch.setattr(hf, "FrozenReward", capture_reward)
    c = {"choices": {"sft": {"model": str(model_path), "revision": "0"*40}},
         "runtime": {"device": "cpu", "dtype": "float32", "local_files_only": True},
         "candidate": {"max_prompt_tokens": 20},
         "ppo": {"total_episodes": 2, "response_length": 4, "learning_rate": 0.001,
                 "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                 "num_mini_batches": 1, "num_ppo_epochs": 1, "kl_coef": 0.05,
                 "cliprange": 0.2, "vf_coef": 0.1, "lora_r": 2, "lora_alpha": 4,
                 "lora_target_modules": "all-linear"}}
    prompts = [{"messages": [{"role": "user", "content": "hello world"}]}]*4
    result = train_ppo(c, tmp_path/"policy", np.linspace(-0.1, 0.1, 32), prompts, 5)
    assert result["total_episodes"] == 2 and result["rollout_batch_size"] == 2
    assert result["log_history"]
    reward = captured["reward"]
    for key, value in reward.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), captured["before"][key], rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in reward.parameters())
    reloaded = PeftModel.from_pretrained(base, tmp_path/"policy")
    assert any(parameter.detach().abs().sum() > 0 for name, parameter in reloaded.named_parameters() if "lora_B" in name)
