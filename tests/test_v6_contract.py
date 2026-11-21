import copy
from pathlib import Path

import pytest
import yaml

from prism.v6 import cli
from prism.v6.config import load_config, plan
from prism.v3.linear_ppo import _recipe


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "configs/experiment.v5.yaml"
V6 = ROOT / "configs/experiment.v6.yaml"


def _write(tmp_path, raw):
    path = tmp_path / "v6.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_shipped_v6_changes_only_design_constraints_sampling_and_identity():
    old = yaml.safe_load(V5.read_text(encoding="utf-8"))
    new = yaml.safe_load(V6.read_text(encoding="utf-8"))
    for section in ("execution", "model", "runtime", "study", "data", "features",
                    "reward", "ppo", "evaluation"):
        assert new[section] == old[section], section
    old_design = copy.deepcopy(old["design"])
    new_design = copy.deepcopy(new["design"])
    for key in ("solver", "extra_constraints"):
        old_design.pop(key)
        new_design.pop(key)
    assert new_design == old_design
    old_sampling = copy.deepcopy(old["sampling"])
    new_sampling = copy.deepcopy(new["sampling"])
    old_sampling.pop("protocol")
    new_sampling.pop("protocol")
    assert new_sampling == old_sampling
    assert new["schema_version"] == old["schema_version"]
    assert new["protocol"] == old["protocol"]
    assert new["v6"]["preference_budget"] == 20000


def test_v6_plan_is_budget_capped_and_does_not_start_compute_or_api():
    config = load_config(V6)
    value = plan(config)
    assert value["design"] == {
        "preference_budget": 20000,
        "lambda": .001,
        "objective": "unchanged_trace_inverse_information",
        "solver": "capped_simplex_away_step_frank_wolfe",
        "constraints": ["sum(q)=1", "0<=q_i<=1/20000"],
        "acceptance": "independent_budget_capped_global_fw_gap",
    }
    assert value["sampling"]["prism"] == "fixed_size_dependent_rounding_of_pi=Nq"
    assert value["ppo"]["formal_episodes_per_method"] == 16384
    assert value["ppo"]["scheduler_total_episodes"] == 65536
    assert value["compute_started"] is False
    assert value["paid_api_calls_enabled"] is False
    recipe = _recipe(config)
    assert recipe["protocol"] == "v6_budget_capped_linear_reward_n20000_distributed_ppo_v1"
    assert recipe["preference_budget"] == 20000
    assert recipe["formal_horizon"] == 16384


@pytest.mark.parametrize("mutation", [
    lambda c: c["design"].update(solver="joint-lbfgs"),
    lambda c: c["design"].update(extra_constraints="none"),
    lambda c: c["sampling"].update(protocol="weighted_without_replacement_gumbel_v1"),
    lambda c: c["reward"].update(backbone_trainable=True),
    lambda c: c["reward"].update(max_iter=1999),
    lambda c: c["ppo"].update(kl_coef=.1),
    lambda c: c["v6"].update(design_lambda=.01),
])
def test_v6_rejects_contract_mutations(tmp_path, mutation):
    raw = yaml.safe_load(V6.read_text(encoding="utf-8"))
    mutation(raw)
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, raw))


def test_v6_cli_guards_budget_lambda_and_formal_horizon(monkeypatch):
    forwarded = []
    monkeypatch.setattr(cli, "v3_main", lambda argv: forwarded.append(argv) or "ok")
    config = str(V6)
    assert cli.main(["freeze-study", "--budget", "20000", "--design-lambda", ".001",
                     "--config", config]) == "ok"
    assert forwarded[-1][0] == "freeze-study"
    assert cli.main(["linear-ppo-policy", "--method", "prism",
                     "--stop-after-episodes", "16384", "--config", config]) == "ok"
    with pytest.raises(SystemExit, match="--budget 20000"):
        cli.main(["freeze-study", "--budget", "14000", "--design-lambda", ".001",
                  "--config", config])
    with pytest.raises(SystemExit, match="exactly 16384"):
        cli.main(["linear-ppo-policy", "--method", "prism",
                  "--stop-after-episodes", "32768", "--config", config])
    with pytest.raises(SystemExit, match="rejects LoRA"):
        cli.main(["linear-ppo-policy", "--method", "full-pool-lora",
                  "--stop-after-episodes", "16384", "--config", config])
