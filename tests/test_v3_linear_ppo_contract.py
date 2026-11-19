"""CPU-only contracts for the matched Empirical/PRISM distributed PPO path."""
from pathlib import Path

import pytest

from prism.v3.config import load_config
from prism.v3.linear_ppo import (EXECUTION_MICRO_BATCH_SIZE, FORMAL_HORIZON,
                                 RECIPE, SCHEDULE_EPISODES, _execution_settings,
                                 _require_recipe, admit)


def test_linear_ppo_changes_placement_but_not_global_update():
    frozen = {"rollout_batch_size": 128, "mini_batch_size": 64,
              "num_mini_batches": 2, "per_device_train_batch_size": 2,
              "gradient_accumulation_steps": 32}
    actual = _execution_settings(frozen)
    assert frozen["per_device_train_batch_size"] == 2
    assert actual["per_device_train_batch_size"] == EXECUTION_MICRO_BATCH_SIZE == 1
    assert actual["gradient_accumulation_steps"] == 64
    assert (actual["per_device_train_batch_size"]
            * actual["gradient_accumulation_steps"]
            * actual["num_mini_batches"] == actual["rollout_batch_size"])


def test_shipped_recipe_keeps_65k_scheduler_and_seals_at_16k():
    path = Path(__file__).resolve().parents[1] / "configs/experiment.v3.yaml"
    c = load_config(path)
    _require_recipe(c)
    assert c["ppo"]["total_episodes"] == SCHEDULE_EPISODES == 65536
    assert FORMAL_HORIZON == 16384
    assert RECIPE["preference_budget"] == 14000
    assert RECIPE["reward_l2"] == pytest.approx(.1)


def test_linear_ppo_admission_requires_explicit_user_authority():
    path = Path(__file__).resolve().parents[1] / "configs/experiment.v3.yaml"
    c = load_config(path)
    with pytest.raises(ValueError, match="explicit user confirmation"):
        admit(c, user_confirmed=False)
