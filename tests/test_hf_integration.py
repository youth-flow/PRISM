import copy
import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from prism.hf import FrozenReward, parse_judgment, sequence_stats, train_ppo


def tiny_model(path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=128, n_embd=16, n_layer=1, n_head=2, eos_token_id=1, bos_token_id=0, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    model.save_pretrained(path)
    vocab = {"<unk>": 0, "<eos>": 1, "user": 2, "assistant": 3, "hello": 4, "world": 5, "A": 6, "B": 7, **{f"token{i}": i for i in range(8, 16)}}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    t = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>")
    t.chat_template = "{% for m in messages %}{{m['role'] + ' ' + m['content'] + ' '}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    t.save_pretrained(path)
    return model.eval()


def test_eos_adapter_and_full_sequence_logp(tmp_path):
    from trl.trainer.utils import get_reward
    m = tiny_model(tmp_path/"tiny")
    w = np.linspace(-1, 1, 16)
    adapter = FrozenReward(m.transformer, 1, w)
    # Includes left padding, sampled EOS, and max-length truncation without EOS.
    query_responses = torch.tensor([[16, 2, 4, 5, 1, 16], [2, 4, 3, 6, 7, 8]])
    _, rewards, _ = get_reward(adapter, query_responses, 16, 3)
    for i, (prompt, response) in enumerate([([2, 4], [5, 1]), ([2, 4, 3], [6, 7, 8])]):
        phi, ell = sequence_stats(m, prompt, response, 1)
        np.testing.assert_allclose(rewards[i].item(), phi @ w, atol=2e-6)
        ids = torch.tensor([prompt+response])
        logits = m(ids).logits[0].float().log_softmax(-1)
        manual = sum(logits[len(prompt)+j-1, token].item() for j, token in enumerate(response))
        np.testing.assert_allclose(ell, manual, atol=2e-6)
    assert not any(p.requires_grad for p in adapter.parameters())
    adapter.train()
    assert not adapter.backbone.model.training


def test_strict_judge_parser():
    assert parse_judgment(" A\n") == 1
    assert parse_judgment("B") == 0
    with pytest.raises(ValueError):
        parse_judgment("Answer A because...")


def test_real_trl_ppo_update_and_adapter_reload(tmp_path):
    from peft import PeftModel
    model_path = tmp_path/"tiny"
    base = tiny_model(model_path)
    c = {"choices": {"sft": {"model": str(model_path), "revision": "0"*40}}, "runtime": {"device": "cpu", "dtype": "float32", "local_files_only": True}, "candidate": {"max_prompt_tokens": 20}, "ppo": {"total_episodes": 2, "response_length": 4, "learning_rate": 0.001, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1, "num_mini_batches": 1, "num_ppo_epochs": 1, "kl_coef": 0.05, "cliprange": 0.2, "vf_coef": 0.1, "lora_r": 2, "lora_alpha": 4, "lora_target_modules": "all-linear"}}
    prompts = [{"messages": [{"role": "user", "content": "hello world"}]}]*4
    result = train_ppo(c, tmp_path/"policy", np.linspace(-0.1, 0.1, 16), prompts, 5)
    assert result["total_episodes"] == 2
    assert (tmp_path/"policy"/"adapter_model.safetensors").exists()
    reloaded = PeftModel.from_pretrained(base, tmp_path/"policy")
    assert any(p.detach().abs().sum() > 0 for name, p in reloaded.named_parameters() if "lora_B" in name)
    assert result["log_history"]
