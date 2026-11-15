"""Probability law, numerical guard and pinned PPO extension contracts."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import textwrap

import pytest
import torch

from prism.numerics import configure_numerics, probability_path_metrics, selected_logprobs


def test_fp32_selected_probabilities_and_gradients_match_cross_entropy():
    torch.manual_seed(12)
    logits = torch.randn(2, 7, 13, dtype=torch.bfloat16, requires_grad=True)
    selected = torch.randint(13, (2, 7))
    actual = selected_logprobs(logits, selected)
    reference = -torch.nn.functional.cross_entropy(logits.float().reshape(-1,13), selected.flatten(), reduction="none").reshape(2,7)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=0)
    first = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    second = torch.autograd.grad(reference.sum(), logits)[0]
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_unchanged_policy_guard_rejects_real_disagreement_and_ignores_padding():
    old = torch.zeros(2,5)
    fresh = old.clone(); fresh[:,:3] = 1e-6; fresh[:,3:] = 100
    valid = torch.tensor([[True,True,True,False,False]]*2)
    result = probability_path_metrics(old, fresh, valid)
    assert result["valid_tokens"] == 6 and result["zero_update_approxkl"] < 1e-11
    fresh[:,0] = .03
    with pytest.raises(RuntimeError,match="paths disagree"):
        probability_path_metrics(old, fresh, valid)
    with pytest.raises(ValueError,match="FP32"):
        probability_path_metrics(old.bfloat16(), fresh, valid)


def test_protocol_is_explicit_and_legacy_settings_are_not_changed():
    before = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        assert configure_numerics({"runtime":{"dtype":"bfloat16"}}) is None
        assert torch.backends.cuda.matmul.allow_tf32
        with pytest.raises(ValueError):
            configure_numerics({"runtime":{"dtype":"bfloat16","numerics":"fp32_consistent_v1"}})
        result = configure_numerics({"runtime":{"dtype":"float32","numerics":"fp32_consistent_v1"}})
        assert result["tf32"] is False and not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


def test_new_config_preserves_research_parameters_and_original_file():
    import hashlib
    import yaml
    folder = Path(__file__).resolve().parents[1]/"configs"
    original = folder/"experiment.yaml"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == "98f6b8102fb6ab53d949b9b72b2769a8f89867386e5e676dda3f083fc933816e"
    old = yaml.safe_load(original.read_text(encoding="utf-8"))
    new = yaml.safe_load((folder/"experiment.fp32.yaml").read_text(encoding="utf-8"))
    new = deepcopy(new)
    assert new["runtime"].pop("numerics") == "fp32_consistent_v1"
    assert new["runtime"]["dtype"] == "float32"
    new["runtime"]["dtype"] = old["runtime"]["dtype"]
    assert new.pop("output") != old["output"]
    assert new == {k:v for k,v in old.items() if k != "output"}


def test_fp32_ppo_preserves_upstream_optimization_ast():
    pytest.importorskip("trl")
    from trl import PPOTrainer
    from prism.ppo_numerics import FP32PPOTrainer

    class UndoNumericalChanges(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "selected_logprobs": node.id = "selective_log_softmax"
            return node
        def visit_Delete(self, node):
            return None if ast.unparse(node) == "del logitss" else self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) == "self.check_probability_paths": return None
            return self.generic_visit(node)
        def visit_Call(self, node):
            name = ast.unparse(node.func)
            if name == "masked_mean":
                expr = ast.unparse(node.args[0])
                if expr in ("logprobs_diff ** 2", "ratio", "entropy"):
                    return ast.Call(func=ast.Attribute(value=node.args[0],attr="mean",ctx=ast.Load()),args=[],keywords=[])
            return self.generic_visit(node)

    original = ast.parse(textwrap.dedent(inspect.getsource(PPOTrainer.train)))
    new = ast.parse(textwrap.dedent(inspect.getsource(FP32PPOTrainer.train)))
    new = UndoNumericalChanges().visit(new)
    assert ast.dump(original,include_attributes=False) == ast.dump(new,include_attributes=False)


@pytest.mark.parametrize("critic_rate", [None, .003])
def test_fp32_protocol_real_olmo_ppo_passes_guards_and_updates_lora(tmp_path, critic_rate):
    pytest.importorskip("trl")
    from test_prism_model_contract import tiny_olmo2
    from prism.hf import train_ppo
    from peft import PeftModel
    import numpy as np
    path = tmp_path/"tiny"
    base = tiny_olmo2(path)
    config = {"choices":{"sft":{"model":str(path),"revision":"0"*40}},
              "runtime":{"device":"cpu","dtype":"float32","local_files_only":True,"numerics":"fp32_consistent_v1"},
              "candidate":{"max_prompt_tokens":20},
              "ppo":{"total_episodes":4,"response_length":4,"learning_rate":.001,
                     "per_device_train_batch_size":2,"gradient_accumulation_steps":1,
                     "num_mini_batches":1,"num_ppo_epochs":1,"kl_coef":.05,"cliprange":.2,
                     "vf_coef":.1,"lora_r":2,"lora_alpha":4,"lora_target_modules":"all-linear"}}
    prompts = [{"messages":[{"role":"user","content":"hello world"}]}]*4
    if critic_rate is not None:
        config["ppo"]["critic_learning_rate"] = critic_rate
    result = train_ppo(config,tmp_path/"policy",np.linspace(-.1,.1,32),prompts,5)
    assert len(result["probability_checks"]) == 2
    assert all(row["zero_update_approxkl"] < 1e-7 for row in result["probability_checks"])
    assert result["numerics"]["protocol"] == "fp32_consistent_v1"
    assert result["gradient_norm_clipping_executed"] is False
    if critic_rate is not None:
        assert {g["role"] for g in result["optimizer_groups"]} == {"actor", "critic"}
        assert all(g["initial_lr"] == (critic_rate if g["role"] == "critic" else .001)
                   for g in result["optimizer_groups"])
    actor = PeftModel.from_pretrained(base,tmp_path/"policy")
    assert any(p.detach().abs().sum()>0 for n,p in actor.named_parameters() if "lora_B" in n)
