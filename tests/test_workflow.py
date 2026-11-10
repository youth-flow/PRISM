import json
from pathlib import Path
import subprocess
import sys

from prism.config import plan_summary
from prism.io import load_config
from prism.workflow import stage_plan, stage_resource


def test_current_config_plan_and_thin_script_work_from_another_directory(tmp_path):
    repo = Path(__file__).parents[1]
    c, missing = load_config(repo/"configs/experiment.yaml")
    counts = plan_summary(c)
    assert counts["training_cells"] == 4
    assert counts["training_valid_labels"] == 4096
    assert counts["evaluation_judge_calls"] == 6000
    assert counts["benchmark_generations"] == 2705
    jobs = stage_plan(c)
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
