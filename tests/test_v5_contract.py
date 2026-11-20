import copy
import json
from pathlib import Path

import pytest

from prism.v5 import cli
from prism.v5.config import (FORMAL_EPISODES, PREFERENCE_BUDGET,
                             SCHEDULE_EPISODES, load_config, plan, validate_v5)
from prism.v5.registry import load_registry, validate_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment.v5.yaml"
REGISTRY = ROOT / "results" / "v5" / "evaluation-registry.json"


def test_v5_frozen_linear_contract():
    c = load_config(CONFIG)
    value = plan(c)
    assert value["reward"] == {
        "type": "frozen_sft_eos2048_bias_free_linear_bt",
        "preference_budget": PREFERENCE_BUDGET,
        "design_lambda": pytest.approx(0.001),
        "l2": pytest.approx(0.1),
    }
    assert value["ppo"]["formal_episodes_per_method"] == FORMAL_EPISODES
    assert value["ppo"]["scheduler_total_episodes"] == SCHEDULE_EPISODES
    assert value["ppo"]["actor_steps_per_method"] == 1024
    assert value["ppo"]["critic_steps_per_method"] == 1024
    assert value["ppo"]["distributed_world_size"] == 4
    assert value["paid_api_calls_enabled"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "revision", "0" * 40),
        ("data", "revision", "1" * 40),
        ("reward", "backbone_trainable", True),
        ("ppo", "learning_rate", 3e-5),
        ("ppo", "total_episodes", 16384),
        ("v5", "formal_ppo_episodes", 65536),
    ],
)
def test_v5_rejects_protocol_drift(section, key, value):
    c = load_config(CONFIG)
    c = copy.deepcopy(c)
    c[section][key] = value
    with pytest.raises(ValueError, match="V5 frozen contract mismatch"):
        validate_v5(c)


def test_registry_contains_primary_and_separated_controls():
    value = load_registry(REGISTRY)
    assert value["counts"] == {"reward_runs": 14, "ppo_runs": 5}
    primary_rewards = [run for run in value["reward_runs"]
                       if run["classification"] == "v5_primary"]
    primary_policies = [run for run in value["ppo_runs"]
                        if run["classification"] == "v5_primary"]
    assert {run["method"] for run in primary_rewards} == {"empirical", "prism"}
    assert {run["method"] for run in primary_policies} == {"empirical", "prism"}
    assert all(run["architecture"] == "frozen_linear" for run in primary_rewards)
    assert all(run["reward_architecture"] == "frozen_linear" for run in primary_policies)
    assert all(run["episodes"] == 16384 for run in primary_policies)
    candidates = [run for run in value["reward_runs"]
                  if run["classification"] == "v5_reward_candidate"]
    assert len(candidates) == 4
    assert {(run["method"], run["l2"]) for run in candidates} == {
        ("empirical", 0.001), ("prism", 0.001),
        ("empirical", 0.01), ("prism", 0.01),
    }
    assert any(run["architecture"] == "lora_plus_linear"
               and run["classification"] == "out_of_scope_control"
               for run in value["reward_runs"])
    assert any(run["episodes"] == 65536
               and run["classification"] == "out_of_scope_control"
               for run in value["ppo_runs"])


def test_registry_seals_key_reported_results():
    value = load_registry(REGISTRY)
    policies = {run["id"]: run for run in value["ppo_runs"]}
    prism = policies["ppo-linear-prism-n14000-e16384"]["metrics"]
    empirical = policies["ppo-linear-empirical-n14000-e16384"]["metrics"]
    assert prism["luna_vs_empirical"]["win_rate"] == pytest.approx(0.592)
    assert prism["luna_vs_sft"]["win_rate"] == pytest.approx(0.7135)
    assert empirical["luna_vs_sft"]["win_rate"] == pytest.approx(0.669)
    assert prism["generation"]["sequence_kl_mc"] == pytest.approx(10.552648082017898)
    assert empirical["generation"]["sequence_kl_mc"] == pytest.approx(12.773201878488063)
    assert prism["ifeval"]["actor_prompt_strict"] == pytest.approx(0.4048059149722736)
    assert empirical["ifeval"]["actor_prompt_strict"] == pytest.approx(0.4195933456561922)
    rewards = {run["id"]: run for run in value["reward_runs"]}
    assert rewards["reward-linear-prism-n14000-l2-0.01"]["metrics"][
        "development_bt_nll"] == pytest.approx(0.6880422639599899)


def test_registry_rejects_promoting_lora_control_to_primary():
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    run = next(run for run in value["ppo_runs"] if run["method"] == "full_pool_lora")
    run["classification"] = "v5_primary"
    with pytest.raises(ValueError, match="14k frozen-linear"):
        validate_registry(value)


def test_cli_validates_and_rejects_lora_execution(capsys):
    cli.main(["validate", "--config", str(CONFIG)])
    output = json.loads(capsys.readouterr().out)
    assert output["structural_validation"] == "passed"
    assert output["registry"]["counts"] == {"reward_runs": 14, "ppo_runs": 5}
    with pytest.raises(SystemExit, match="rejects non-linear"):
        cli.main(["lora-ppo-policy", "--config", str(CONFIG)])


def test_v5_delegates_only_after_contract_validation(capsys):
    cli.main(["resource", "--for-stage", "linear-ppo-policy", "--config", str(CONFIG)])
    assert capsys.readouterr().out.strip() == "gpu"


@pytest.mark.parametrize(
    "argv",
    [
        ["freeze-study", "--budget", "8000", "--design-lambda", "0.001"],
        ["freeze-study", "--budget", "14000", "--design-lambda", "0.01"],
        ["linear-ppo-policy", "--method", "prism", "--stop-after-episodes", "8192"],
        ["evaluate-generate", "--method", "prism", "--evaluation-episode", "65536"],
        ["evaluate-generate", "--method", "full-pool-lora"],
    ],
)
def test_v5_execution_guard_rejects_non_primary_recipe(argv):
    with pytest.raises(SystemExit):
        cli.main([*argv, "--config", str(CONFIG)])
