"""Exercise submission construction and fail-closed guards without a scheduler."""
import os
from pathlib import Path
import shutil
import shlex
import subprocess

import pytest


def shell_path(path, resolve=True):
    s = (Path(path).resolve() if resolve else Path(path).absolute()).as_posix()
    return f"/{s[0].lower()}{s[2:]}" if os.name == "nt" else s


@pytest.fixture
def platform(tmp_path):
    bash = shutil.which("bash")
    if os.name == "nt":
        bash = str(Path(shutil.which("git")).parents[1]/"bin/bash.exe")
    if not bash or not Path(bash).exists():
        pytest.skip("Bash unavailable")
    storage = tmp_path/"storage"
    repo = storage/"PRISM"
    repo.mkdir(parents=True)
    shutil.copytree(Path(__file__).parents[1]/"scripts/yancheng", repo/"scripts/yancheng")
    for command in (["git", "init", "-b", "main"], ["git", "add", "."], ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    cfg = storage/"frozen config.yaml"
    cfg.write_text("test: true\n")
    env = {**os.environ, "PRISM_STORAGE": shell_path(storage), "PRISM_REPO": shell_path(repo), "PRISM_ACCOUNT": "sc001", "PRISM_GPU_PARTITION": "q_intel_gpu_nvidia_A100_share", "PRISM_CPU_PARTITION": "q_intel_gpu_nvidia_A100_share", "PRISM_TEST_MODE": "1", "PRISM_ALLOW_SUBMIT": "0"}
    for key, relative in {"IMAGE": "images/prism.sif", "HF_CACHE": "hf", "WORK": "work", "LOGS": "logs", "APPTAINER_CACHE": "acache", "APPTAINER_TMP": "atmp"}.items():
        env["PRISM_"+key] = shell_path(storage/relative)
    def run(*args, overrides=None):
        return subprocess.run([bash, shell_path(repo/"scripts/yancheng/submit.sh"), *args], env={**env, **(overrides or {})}, capture_output=True, text=True)
    return run, cfg, repo, env


def test_gpu_cpu_routing_and_dependency(platform):
    run, cfg, repo, _ = platform
    gpu = run("--dry-run", "policy", "--config", shell_path(cfg))
    assert gpu.returncode == 0, gpu.stderr
    assert "--gres=gpu:1" in gpu.stdout and "--account=sc001" in gpu.stdout
    assert "--exclusive" not in gpu.stdout
    cpu = run("--dry-run", "design", "--config", shell_path(cfg), overrides={"PRISM_DEPENDENCY": "123"})
    assert cpu.returncode == 0, cpu.stderr
    assert "--gres" not in cpu.stdout and "afterok:123" in cpu.stdout
    assert not (Path(repo).parent/"logs").exists()  # dry-run has no setup side effects
    api = run("--dry-run", "--resource", "cpu", "label", "--config", shell_path(cfg))
    assert api.returncode == 0, api.stderr
    assert "--gres" not in api.stdout
    local_judge = run("--dry-run", "--resource", "gpu", "evaluate-judge", "--config", shell_path(cfg))
    assert local_judge.returncode == 0 and "--gres=gpu:1" in local_judge.stdout


def test_no_actual_submission_or_storage_escape(platform):
    run, cfg, repo, _ = platform
    blocked = run("smoke", overrides={"PRISM_ALLOW_SUBMIT": "1"})
    assert blocked.returncode != 0 and "Test mode" in blocked.stderr
    outside = Path(cfg).parent.parent/"outside.yaml"
    outside.write_text("test: true")
    assert run("--dry-run", "policy", "--config", shell_path(outside)).returncode != 0
    assert run("--dry-run", "smoke", overrides={"PRISM_GPU_GRES": "gpu:8"}).returncode != 0
    assert run("--dry-run", "smoke", overrides={"PRISM_DEPENDENCY": "1; touch owned"}).returncode != 0
    (repo/"uncommitted.txt").write_text("dirty")
    assert run("--dry-run", "smoke").returncode != 0


def test_symlink_escape(platform, tmp_path):
    run, _, repo, _ = platform
    link = Path(repo).parent/"escape"
    target = tmp_path/"outside"
    target.mkdir()
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink privilege unavailable")
    result = run("--dry-run", "smoke", overrides={"PRISM_WORK": shell_path(link, resolve=False)})
    assert result.returncode != 0


def test_container_api_environment_forwarding_uses_names_without_printing_values(platform):
    _, _, repo, env = platform
    bash = str(Path(shutil.which("git")).parents[1]/"bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    source = shlex.quote(shell_path(repo/"scripts/yancheng/common.sh"))
    script = f"source {source}\nprism_forward_api_envs TEST_JUDGE_KEY\n[[ $APPTAINERENV_TEST_JUDGE_KEY == fixture-only-key ]]\nprintf passed"
    result = subprocess.run([bash, "-c", script], env={**env, "TEST_JUDGE_KEY": "fixture-only-key"}, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == "passed"
    assert "fixture-only-key" not in result.stdout + result.stderr
    missing = subprocess.run([bash, "-c", f"source {source}\nprism_forward_api_envs PRISM_MISSING_TEST_KEY"], env=env, capture_output=True, text=True)
    assert missing.returncode != 0 and "Missing API environment variable" in missing.stderr
