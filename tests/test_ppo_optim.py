import pytest
import torch
from torch import nn

from prism.ppo_optim import optimizer_step_report, separate_critic_learning_rate


def test_step_report_reads_actual_adamw_state_counters_for_both_roles():
    actor=nn.Parameter(torch.ones(2));critic=nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([{'params':[actor],'prism_role':'actor'},{'params':[critic],'prism_role':'critic'}],lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad();(actor.sum()+critic.sum()).backward();optimizer.step()
    report=optimizer_step_report(optimizer)
    assert report=={role:{'tensor_count':1,'step_min':5,'step_max':5} for role in ('actor','critic')}


def test_production_prompt_boundaries_preserve_256_real_adamw_updates():
    from accelerate import Accelerator
    from torch.utils.data import DataLoader,TensorDataset
    from prism.ppo_optim import configure_rollout_accumulation
    def measured(fixed):
        model=nn.Linear(1,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        acc=Accelerator(cpu=True,gradient_accumulation_steps=16)
        acc.gradient_state.plugin_kwargs['sync_with_dataloader']=True
        if fixed:configure_rollout_accumulation(acc)
        loader=DataLoader(TensorDataset(torch.ones(2000,1)),batch_size=64,drop_last=True)
        model,optimizer,loader=acc.prepare(model,optimizer,loader)
        def repeated():
            while True:yield from loader
        iterator=iter(repeated());counts=[];last=0
        try:
            for update in range(64):
                batch=next(iterator)[0]
                for _ in range(64):
                    with acc.accumulate(model):
                        acc.backward(model(batch[:2]).square().mean())
                        optimizer.step();optimizer.zero_grad()
                actual=max(int(v['step'].item()) for v in optimizer.state.values())
                counts.append(actual-last);last=actual
        finally:iterator.close()
        return counts
    old=measured(False)
    assert sum(old)==376 and [(i+1,n) for i,n in enumerate(old) if n!=4]==[(31,64),(62,64)]
    assert measured(True)==[4]*64


def test_separate_rates_preserve_decay_coverage_and_scheduler_ratio():
    model = nn.Module()
    model.policy = nn.Module()
    model.policy.lora_A = nn.Linear(3, 2, bias=False)
    model.value_model = nn.Module()
    model.value_model.score = nn.Linear(3, 1)
    model.frozen = nn.Linear(3, 3).requires_grad_(False)
    actor = list(model.policy.parameters())
    weight, bias = list(model.value_model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": actor + [weight], "weight_decay": .1},
        {"params": [bias], "weight_decay": 0.},
    ], lr=3e-6)
    report = separate_critic_learning_rate(optimizer, model, 3e-5)
    assert sum(g["parameters"] for g in report) == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert [g["weight_decay"] for g in report] == [.1, .1, 0.]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1-step/64)
    for _ in range(8):
        optimizer.zero_grad()
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step()
        scheduler.step()
    for group in optimizer.param_groups:
        initial = 3e-6 if group["prism_role"] == "actor" else 3e-5
        assert group["lr"] == pytest.approx(initial*56/64)
    with pytest.raises(ValueError, match="before the first"):
        separate_critic_learning_rate(optimizer, model, 3e-5)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True])
def test_separate_rates_reject_invalid_rate(rate):
    with pytest.raises(ValueError, match="finite and positive"):
        separate_critic_learning_rate(None, None, rate)
