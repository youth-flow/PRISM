import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_yancheng import platform, shell_path


def call(platform, *args, **overrides):
    _, cfg, repo, env = platform
    bash = str(Path(shutil.which("git")).parents[1] / "bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    return subprocess.run([bash, shell_path(repo / "scripts/yancheng/submit_v4.sh"), "--dry-run", *args,
                           "--config", shell_path(cfg)], env={**env, **overrides}, capture_output=True,
                          text=True, encoding="utf-8")


@pytest.mark.parametrize("stage,cpus,memory,gpu", [("extract-features", 4, "64G", True),
    ("policy-smoke", 8, "64G", True), ("design", 32, "48G", False),
    ("full-pool-reward", 32, "48G", False), ("runtime-bootstrap", 4, "6G", False)])
def test_v4_resource_profiles(platform, stage, cpus, memory, gpu):
    result = call(platform, stage)
    assert result.returncode == 0, result.stderr
    assert f"--cpus-per-task={cpus}" in result.stdout and f"--mem={memory}" in result.stdout
    assert ("--gres=gpu:1" in result.stdout) == gpu
    assert "run_v4.sh" in result.stdout and "prism-v4-" in result.stdout
    assert "--exclusive" not in result.stdout


def test_v4_gates_and_array_coverage(platform):
    result = call(platform, "policy")
    assert result.returncode != 0 and "admission is closed" in result.stderr
    result = call(platform, "lora-reward-train")
    assert result.returncode != 0
    good = call(platform, "extract-features", "--shards", "32", PRISM_ARRAY="0-31%7")
    assert good.returncode == 0 and "--array=0-31%7" in good.stdout
    bad = call(platform, "extract-features", "--shards", "32", PRISM_ARRAY="0-32%7")
    assert bad.returncode != 0
    limit = call(platform, "extract-features", "--shards", "64", PRISM_ARRAY="0-50%7")
    assert limit.returncode != 0 and "50" in limit.stderr


def test_v4_shell_syntax_and_runtime_seals(platform):
    _, _, repo, _ = platform
    bash = str(Path(shutil.which("git")).parents[1] / "bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    for name in ("run_v4.sh", "submit_v4.sh", "env.v4.example.sh"):
        result = subprocess.run([bash, "-n", shell_path(repo / "scripts/yancheng" / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    body = (repo / "scripts/yancheng/run_v4.sh").read_text(encoding="utf-8")
    assert 'sha256sum --check --status runtime.sha256' in body
    assert 'unset OPENAI_API_KEY' in body
    assert 'smoke_gpu_v4.py' in body
