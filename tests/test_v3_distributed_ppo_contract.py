"""CPU/Gloo contract test for synchronous multi-rank PPO and recovery."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import os
import socket

import pytest

torch = pytest.importorskip("torch")

from prism.v3.distributed_ppo import DistributedPPOEngine


VOCAB, HIDDEN, PAD, EOS = 31, 8, 30, 2


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(101)
        table = torch.randn(VOCAB, VOCAB, generator=generator, dtype=torch.float32) * .05
        table[:, PAD] = -20.
        self.base = torch.nn.Parameter(table, requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.randn(VOCAB, 2, generator=generator) * .02)
        self.lora_B = torch.nn.Parameter(torch.zeros(2, VOCAB))
        self._adapter_enabled = True

    @contextmanager
    def disable_adapter(self):
        previous, self._adapter_enabled = self._adapter_enabled, False
        try:
            yield
        finally:
            self._adapter_enabled = previous

    def forward(self, input_ids, **kwargs):
        del kwargs
        logits = self.base[input_ids]
        if self._adapter_enabled:
            logits = logits + self.lora_A[input_ids] @ self.lora_B
        return SimpleNamespace(logits=logits, past_key_values=())


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(202)
        self.base = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN, generator=generator) * .05,
                                       requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.randn(VOCAB, 2, generator=generator) * .02)
        self.lora_B = torch.nn.Parameter(torch.zeros(2, HIDDEN))
        self.score = torch.nn.Linear(HIDDEN, 1, dtype=torch.float32)

    def forward(self, input_ids, **kwargs):
        del kwargs
        hidden = self.base[input_ids] + self.lora_A[input_ids] @ self.lora_B
        return self.score(hidden).squeeze(-1)


class Reward:
    def __init__(self):
        self.scale = torch.nn.Parameter(torch.tensor(.01), requires_grad=False)

    def frozen_parameters(self):
        return [self.scale]

    def score(self, prompts, responses):
        return torch.stack([self.scale * torch.tensor(float(sum(prompt)+sum(response)))
                            for prompt, response in zip(prompts, responses)])


def settings():
    return {"total_episodes": 16, "rollout_batch_size": 8, "mini_batch_size": 4,
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
            "num_mini_batches": 2, "num_ppo_epochs": 1,
            "learning_rate": .002, "critic_learning_rate": .003,
            "critic_head_learning_rate": .004,
            "warmup_ratio": .05, "minimum_learning_rate_ratio": .1,
            "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-5,
            "weight_decay": 0., "gamma": 1., "lam": .95, "kl_coef": .05,
            "cliprange": .2, "cliprange_value": .2, "vf_coef": .1,
            "max_grad_norm": 1., "response_length": 3, "temperature": 1.,
            "top_p": 1., "top_k": 0, "whiten_rewards": False,
            "whiten_advantages": True, "local_rollout_forward_batch_size": 2,
            "probability_check_rows": 2, "max_logp_error": 1e-6,
            "max_zero_update_approxkl": 1e-10, "engineering_checkpoint": 8,
            "durable_checkpoint_interval": 8, "checkpoints": [8, 16]}


def trainable(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def make_engine(path=None):
    prompts = [[1, 3+(index % 20), 4+(index % 15)] for index in range(16)]
    return DistributedPPOEngine(Actor(), Critic(), Reward(), prompts, pad=PAD, eos=EOS,
                                settings=settings(), seed=11,
                                contract_hash="gloo-distributed-fixture",
                                checkpoint_dir=path)


def _worker(rank, world, init_method, folder):
    import torch.distributed as distributed
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", rank=rank, world_size=world,
                                   init_method=init_method)
    try:
        continuous = make_engine()
        continuous.run()
        checkpoint = Path(folder)/"checkpoint"
        interrupted = make_engine(checkpoint)
        interrupted.run(stop_after_episodes=8)
        resumed = make_engine(checkpoint)
        resumed.resume()
        result = resumed.run()
        assert result["completed"] and resumed.history == continuous.history
        for name, value in trainable(continuous.actor).items():
            assert torch.equal(value, trainable(resumed.actor)[name])
        for name, value in trainable(continuous.critic).items():
            assert torch.equal(value, trainable(resumed.critic)[name])
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(os.name == "nt", reason="Windows Gloo has no supported loopback device")
def test_four_rank_global_batch_and_resume_are_exact(tmp_path):
    import torch.multiprocessing as multiprocessing
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    multiprocessing.spawn(_worker,
                          args=(4, f"tcp://127.0.0.1:{port}", str(tmp_path)),
                          nprocs=4, join=True)
