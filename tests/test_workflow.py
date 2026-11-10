import json
from pathlib import Path
import subprocess
import sys

import pytest

from prism.config import load_v2, plan_summary
from prism.io import load_config
from prism.workflow import stage_plan, stage_resource
from test_config_v2 import fixture_v2


def test_current_config_plan_and_thin_script_work_from_another_directory(tmp_path):
    repo = Path(__file__).parents[1]
    c, missing = load_config(repo/"configs/experiment.yaml")
    counts = plan_summary(c)
    assert counts["training_cells"] == 4
    assert counts["training_valid_labels"] == 4096
    assert counts["evaluation_judge_calls"] == 6000
    assert counts["benchmark_generations"] == 2705
    jobs = stage_plan(c)
    assert [j["stage"] for j in jobs[:3]] == ["stage-assets", "prepare-data", "model-smoke"]
    assert len([j for j in jobs if j["stage"] == "policy"]) == 4
    assert len([j for j in jobs if j["stage"] == "evaluate-reference"]) == 1
    assert max(i for i,j in enumerate(jobs) if j["stage"] == "evaluate-generate") < min(i for i,j in enumerate(jobs) if j["stage"] == "evaluate-judge")
    assert stage_resource(c, "label") == stage_resource(c, "evaluate-judge") == "cpu"
    assert c["execution"]["allow_paid_api_calls"] is False
    result = subprocess.run([sys.executable, str(repo/"scripts/run.py"), "plan", "--json"], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"] == counts
    names = subprocess.run([sys.executable, str(repo/"scripts/run.py"), "api-env-names", "--judge-role", "training"], cwd=tmp_path, text=True, capture_output=True)
    assert names.returncode == 0 and names.stdout.strip() == "DEEPSEEK_API_KEY"
    assert not (tmp_path/"runs").exists()
    assert missing  # No generated data or selected-model assets are invented.


@pytest.mark.parametrize("preparation_metadata", [False, True])
def test_prepared_exact_match_plan_reuses_inputs_without_ifeval_preparation(tmp_path, preparation_metadata):
    path, raw = fixture_v2(tmp_path)
    if preparation_metadata:
        raw["data"]["source"].update(train_split="train_gen", test_split="test_gen", prompt_field="prompt", id_field="prompt_id")
        raw["data"].update(benchmark_source={"id": "fixture/benchmark", "revision": "c"*40, "split": "train"},
                           model_context_tokens=256, split_seed=0)
    c, missing = load_v2(path, raw, ready=True)
    assert not missing
    inputs = [Path(value) for value in c["choices"]["prompts"].values()]
    inputs.append(Path(c["choices"]["benchmark"]["path"]))
    before = {file: file.read_bytes() for file in inputs}
    jobs = stage_plan(c)
    assert "prepare-data" not in [job["stage"] for job in jobs]
    assert [job["stage"] for job in jobs[:2]] == ["stage-assets", "model-smoke"]
    assert sum(job["stage"] == "evaluate-reference" for job in jobs) == 1
    assert before == {file: file.read_bytes() for file in inputs}
    assert not Path(c["output"]).exists()


@pytest.mark.parametrize("section,key", [
    ("source", "prompt_field"), ("source", "test_split"),
    ("benchmark_source", "split"), (None, "model_context_tokens"), (None, "split_seed"),
])
def test_ifeval_plan_only_prepares_when_adapter_source_metadata_is_complete(section, key):
    c, _ = load_config(Path(__file__).parents[1]/"configs/experiment.yaml")
    del (c["data"] if section is None else c["data"][section])[key]
    assert "prepare-data" not in [job["stage"] for job in stage_plan(c)]
