"""CPU-only contract checks for the isolated LoRA-Reward PPO path."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prism.v3.artifacts import Run, write_json
from prism.v3.lora_ppo import (FrozenLoRANormalizedReward,
                               _ensure_critic_warmup_receipt, _selected_reward,
                               _validate_transition)
from prism.v3.lora_reward import RECIPE as REWARD_RECIPE
from prism.v3.ppo_trainer import PROTOCOL


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(self, sequences, pad):
        del pad
        return torch.stack([self.scale * torch.tensor(float(sum(row))) for row in sequences])


def config(tmp_path):
    return {"output": str(tmp_path / "run"),
            "ppo": {"checkpoints": [8192, 16384, 32768, 65536]}}


def test_frozen_lora_scorer_appends_only_missing_eos_and_normalizes():
    model = ScalarModel()
    reward = FrozenLoRANormalizedReward(model, 1., 2., eos=2, pad=37,
                                        context_tokens=16, batch_size=1)
    actual = reward.score([[1], [1]], [[3, 2], [3]])
    # Both canonical sequences are [1, 3, 2]; the natural EOS is not doubled.
    assert torch.allclose(actual, torch.tensor([.25, .25]))
    assert all(not parameter.requires_grad for parameter in reward.frozen_parameters())
    with pytest.raises(ValueError, match="context"):
        FrozenLoRANormalizedReward(model, 0., 1., eos=2, pad=37,
                                   context_tokens=2).score([[1]], [[3]])
    with pytest.raises(ValueError, match="normalization"):
        FrozenLoRANormalizedReward(model, 0., 0., eos=2, pad=37)


def test_selected_reward_is_bound_to_audited_step_pairs_and_bytes(tmp_path):
    run = Run(config(tmp_path))
    folder = run.file("diagnostics/full-pool-lora-reward-fresh-head")
    checkpoint = folder / "checkpoints/step-002304.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable checkpoint fixture")
    training = folder / "training.json"
    summary = folder / "summary.json"
    relative = "diagnostics/full-pool-lora-reward-fresh-head/checkpoints/step-002304.pt"
    write_json(training, {"selected_checkpoint": relative, "best_step": 2304,
                          "recipe": REWARD_RECIPE})
    write_json(summary, {"selected": {"checkpoint": relative, "step": 2304,
                                      "pairs_seen": 147360,
                                      "development": {"accuracy": .6962890625,
                                                      "bt_nll": .5665896117732541}},
                         "recipe": REWARD_RECIPE})
    run.publish("lora-reward-fresh-train", [training, checkpoint])
    run.publish("lora-reward-fresh-report", [summary],
                [run.receipt_path("lora-reward-fresh-train")])
    path, identity = _selected_reward(run)
    assert path == checkpoint
    assert identity["step"] == 2304 and identity["pairs_seen"] == 147360
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        _selected_reward(run)


def test_selected_reward_rejects_a_receipt_bound_nonselected_step(tmp_path):
    run = Run(config(tmp_path))
    folder = run.file("diagnostics/full-pool-lora-reward-fresh-head")
    checkpoint = folder / "checkpoints/step-002303.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"valid but nonselected checkpoint fixture")
    training = folder / "training.json"
    summary = folder / "summary.json"
    relative = "diagnostics/full-pool-lora-reward-fresh-head/checkpoints/step-002303.pt"
    write_json(training, {"selected_checkpoint": relative, "best_step": 2303,
                          "recipe": REWARD_RECIPE})
    write_json(summary, {"selected": {"checkpoint": relative, "step": 2303,
                                      "pairs_seen": 147296,
                                      "development": {"accuracy": .69, "bt_nll": .57}},
                         "recipe": REWARD_RECIPE})
    run.publish("lora-reward-fresh-train", [training, checkpoint])
    run.publish("lora-reward-fresh-report", [summary],
                [run.receipt_path("lora-reward-fresh-train")])
    with pytest.raises(ValueError, match="development-NLL-selected"):
        _selected_reward(run)


def test_critic_warmup_receipt_recovers_from_files_committed_before_receipt(tmp_path):
    run = Run(config(tmp_path))
    dependencies = []
    for stage in ("shared-sft", "lora-ppo-normalization", "lora-reward-fresh-report"):
        output = run.file(f"fixture/{stage}.json")
        write_json(output, {"stage": stage})
        run.publish(stage, [output])
        dependencies.append(run.receipt_path(stage))
    warmup_path = run.file("policy/full-pool-lora/critic-warmup.json")
    warmup = {"passed_performance": True,
              "actual_optimizer_steps": 128,
              "initial_critic_reward_eos_max_error": 0.0}
    initial = run.file("policy/full-pool-lora/checkpoints/episode-00000000.pt")
    initial.parent.mkdir(parents=True)
    torch.save({"protocol": PROTOCOL, "contract_hash": "contract",
                "episode": 0, "round": 0,
                "metadata": {"critic_warmup": warmup},
                "actor": {"adapter": torch.tensor([1.0])},
                "critic": {"adapter": torch.tensor([2.0])}}, initial)
    identity = {"checkpoint_sha256": "reward-checkpoint-sha"}

    assert not warmup_path.exists()
    recovered = _ensure_critic_warmup_receipt(
        run, "contract", warmup_path, identity)

    assert recovered == warmup
    assert warmup_path.is_file()
    receipt = run.require("lora-ppo-critic-warmup")
    assert set(receipt["inputs"]) == {run._key(path) for path in dependencies}
    assert receipt["metadata"]["reward_checkpoint_sha256"] == "reward-checkpoint-sha"
    # A second invocation validates and reuses the durable receipt.
    assert _ensure_critic_warmup_receipt(run, "contract", warmup_path, identity) == recovered


def test_lora_policy_transition_must_stop_at_engineering_then_pilot(tmp_path):
    c = config(tmp_path)
    run = Run(c)
    _validate_transition(c, run, 2048, False)
    with pytest.raises(ValueError, match="2048"):
        _validate_transition(c, run, 8192, False)
    latest = run.file("policy/full-pool-lora/checkpoints/latest.json")
    write_json(latest, {"episode": 2048})
    _validate_transition(c, run, 8192, True)
    with pytest.raises(ValueError, match="8192"):
        _validate_transition(c, run, 16384, True)
    write_json(latest, {"episode": 8192}, overwrite=True)
    # Replaying a checkpoint committed just before its callback must be legal.
    _validate_transition(c, run, 8192, True)
    pilot = run.file("diagnostics/full-pool-lora-ppo/pilot-acceptance.json")
    write_json(pilot, {"passed": True})
    run.publish("lora-ppo-pilot-acceptance", [pilot])
    with pytest.raises(FileNotFoundError):
        _validate_transition(c, run, 16384, True)
    admission = run.file("control/lora-ppo-admission.json")
    write_json(admission, {"user_confirmed": True})
    run.publish("lora-ppo-admission", [admission],
                [run.receipt_path("lora-ppo-pilot-acceptance")])
    _validate_transition(c, run, 16384, True)
    with pytest.raises(ValueError, match="precede"):
        _validate_transition(c, run, 2048, True)
    # A final checkpoint can be reopened solely to replay diagnostics/finalize.
    write_json(latest, {"episode": 65536}, overwrite=True)
    _validate_transition(c, run, 65536, True)
