from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from prism.v3.artifacts import Run
from prism.v4.config import load_config, validate, MODEL_REVISION
from prism.v4.cli import main

CONFIG = Path(__file__).parents[1] / "configs/experiment.v4.yaml"


def test_new_contract_and_plan_do_not_allocate(tmp_path, capsys):
    c = load_config(CONFIG)
    assert c["model"]["revision"] == MODEL_REVISION
    assert c["data"]["feature_dimension"] == c["reward"]["feature_dimension"] == 4096
    assert c["study"]["budget"] is None
    main(["validate", "--config", str(CONFIG)])
    assert '"structural_validation": "passed"' in capsys.readouterr().out
    main(["resource", "--for-stage", "policy-smoke", "--config", str(CONFIG)])
    assert capsys.readouterr().out.strip() == "gpu"
    with pytest.raises(ValueError, match="formal PPO is paused"):
        main(["policy", "--execute", "--method", "empirical", "--config", str(CONFIG)])


@pytest.mark.parametrize("path,value", [(("model", "id"), "allenai/Olmo-3-7B-Instruct"),
    (("data", "feature_dimension"), 2048), (("runtime_packages", "transformers"), "4.52.3"),
    (("ppo", "executor"), "three_copies"), (("admission", "policy"), True)])
def test_reject_invalid_v4(path, value):
    c = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    c[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate(c)


def test_v3_outputs_cannot_be_opened_as_v4(tmp_path):
    c = load_config(CONFIG)
    c.pop("_contract")
    c["output"] = str(tmp_path)
    run = Run(c)
    assert run.path.joinpath("experiment.json").is_file()
    old = deepcopy(c)
    old.update(protocol="hh_finite_pool_v3", schema_version=3)
    with pytest.raises(ValueError, match="different experiment"):
        Run(old)
