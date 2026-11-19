from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from prism.v3.artifacts import Run, read_json, write_json
from prism.v3.cli import main
from prism.v3.config import load_config, plan, validate
from v3_helpers import configuration


def test_shipped_v3_is_single_budget_closed_and_exact_update_count():
    c = load_config(Path(__file__).resolve().parents[1] / "configs/experiment.v3.yaml")
    p = plan(c)
    assert p["preference_budget"] is None
    assert p["formal_policy_units"] == 2
    assert p["actor_steps_per_policy"] == 4096
    assert p["final_valid_judgments"] == 3000
    assert p["evaluation_usd_ceiling"] is None
    assert p["paid_api_calls_enabled"] is False


@pytest.mark.parametrize("section,key,value", [
    ("study", "budget", 65536), ("study", "seed", True),
    ("execution", "allow_training_judge", True), ("execution", "allow_paid_api_calls", True),
    ("runtime", "dtype", "bfloat16"), ("runtime", "allow_tf32", True),
    ("sampling", "replacement", True), ("design", "probability_floor", 1e-12),
    ("design", "extra_constraints", "cap"), ("design", "lambdas", [0.01, 0.01, 0.1]),
    ("reward", "backbone_trainable", True), ("reward", "l2_candidates", [float("nan")]),
    ("ppo", "rollout_batch_size", 999), ("ppo", "adam_beta1", 1),
    ("evaluation", "tie_score", 0), ("evaluation", "max_new_tokens", 1),
    ("evaluation", "api_key", "TEST_NOT_REAL"),
])
def test_invalid_protocol_is_rejected(tmp_path, section, key, value):
    c = deepcopy(configuration(tmp_path)["_contract"])
    c[section][key] = value
    with pytest.raises(ValueError):
        validate(c)


def test_plan_validate_and_resource_do_not_create_run(tmp_path, capsys):
    c = configuration(tmp_path)
    for command in (["plan"], ["validate"], ["resource", "--for-stage", "policy"],
                    ["resource", "--for-stage", "lora-ppo-normalize"],
                    ["resource", "--for-stage", "lora-ppo-smoke"],
                    ["resource", "--for-stage", "lora-ppo-policy"],
                    ["resource", "--for-stage", "lora-ppo-ifeval-generate"],
                    ["resource", "--for-stage", "lora-ppo-admit"], ["status"]):
        main(command + ["--config", c["_config_path"]])
        assert not Path(c["output"]).exists()
        capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["sample", "--config", c["_config_path"]])
    assert not Path(c["output"]).exists()


def test_lora_ppo_policy_requires_explicit_stop_checkpoint(tmp_path):
    c = configuration(tmp_path)
    with pytest.raises(SystemExit):
        main(["lora-ppo-policy", "--execute", "--config", c["_config_path"]])
    assert not Path(c["output"]).exists()


def test_lora_ppo_admission_requires_explicit_post_pilot_authority(tmp_path):
    c = configuration(tmp_path)
    with pytest.raises(ValueError, match="explicit user confirmation"):
        main(["lora-ppo-admit", "--execute", "--config", c["_config_path"]])


def test_receipt_checks_transitive_dependency_and_path_escape(tmp_path):
    c = configuration(tmp_path)
    run = Run(c)
    first = run.file("input.json")
    write_json(first, {"value": 1})
    run.publish("first", [first])
    second = run.file("output.json")
    write_json(second, {"value": 2})
    run.publish("second", [second], [run.receipt_path("first")])
    assert run.completed("second")
    first.write_text('{"value":3}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        run.require("second")
    with pytest.raises(ValueError, match="inside"):
        run.file("../escape.json")
    c2 = deepcopy(c)
    c2["_contract"]["study"]["seed"] = 23
    with pytest.raises(ValueError, match="different experiment"):
        Run(c2)


def test_raw_source_receipts_survive_verified_run_relocation(tmp_path):
    import shutil
    c = configuration(tmp_path)
    raw = deepcopy(c["_contract"])
    raw["output"], raw["data"]["raw_dir"] = "run", "raw"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    config = first / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    c1 = load_config(config)
    run = Run(c1)
    source = first / "raw/source.json"
    write_json(source, {"public": "pinned raw input"})
    target = run.file("prepared.json")
    write_json(target, {"prepared": True})
    receipt = run.publish("prepared", [target], [source])
    assert "raw:source.json" in receipt["inputs"]
    shutil.copytree(first, second)
    moved = Run(load_config(second / "experiment.yaml"))
    assert moved.require("prepared") == receipt
    with pytest.raises(ValueError, match="escapes"):
        moved._resolve("raw:../outside.json")


def test_changed_manifest_body_cannot_keep_a_valid_fingerprint(tmp_path):
    c = configuration(tmp_path)
    run = Run(c)
    path = run.file("experiment.json")
    value = read_json(path)
    value["config"]["study"]["seed"] = 23
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="different experiment"):
        Run(c)
