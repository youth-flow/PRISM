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
    env = {**os.environ, "PRISM_STORAGE": shell_path(storage), "PRISM_REPO": shell_path(repo), "PRISM_ACCOUNT": "sc001", "PRISM_GPU_PARTITION": "q_intel_gpu_nvidia_A100_share", "PRISM_CPU_PARTITION": "q_cpu_verified", "PRISM_TEST_MODE": "1", "PRISM_ALLOW_SUBMIT": "0"}
    for key, relative in {"IMAGE": "images/prism.sif", "HF_CACHE": "hf", "WORK": "work", "LOGS": "logs", "APPTAINER_CACHE": "acache", "APPTAINER_TMP": "atmp"}.items():
        env["PRISM_"+key] = shell_path(storage/relative)
    def run(*args, overrides=None):
        return subprocess.run([bash, shell_path(repo/"scripts/yancheng/submit.sh"), *args], env={**env, **(overrides or {})}, capture_output=True, text=True, encoding="utf-8")
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
    assert "--partition=q_cpu_verified" in cpu.stdout
    joined = run("--dry-run", "design", "--config", shell_path(cfg), overrides={"PRISM_DEPENDENCY": "123:456"})
    assert joined.returncode == 0 and "afterok:123:456" in joined.stdout
    assert not (Path(repo).parent/"logs").exists()  # dry-run has no setup side effects
    api = run("--dry-run", "--resource", "cpu", "label", "--config", shell_path(cfg))
    assert api.returncode == 0, api.stderr
    assert "--gres" not in api.stdout
    local_judge = run("--dry-run", "--resource", "gpu", "evaluate-judge", "--config", shell_path(cfg))
    assert local_judge.returncode == 0 and "--gres=gpu:1" in local_judge.stdout


def test_stage_specific_resource_profiles_and_cpu_smoke(platform):
    run, cfg, _, _ = platform
    smoke = run("--dry-run", "cpu-smoke")
    assert smoke.returncode == 0, smoke.stderr
    assert "--partition=q_cpu_verified" in smoke.stdout
    assert "--cpus-per-task=1" in smoke.stdout and "--mem=1500M" in smoke.stdout
    design = run("--dry-run", "design", "--config", shell_path(cfg))
    assert design.returncode == 0, design.stderr
    assert "--cpus-per-task=16" in design.stdout and "--mem=24G" in design.stdout
    rollout = run("--dry-run", "rollout", "--config", shell_path(cfg))
    assert rollout.returncode == 0, rollout.stderr
    assert "--cpus-per-task=8" in rollout.stdout and "--mem=64G" in rollout.stdout
    assert "--time=24:00:00" in rollout.stdout


def test_opportunistic_shard_array_keeps_one_gpu_per_task(platform):
    run, cfg, _, _ = platform
    result = run("--dry-run", "rollout-shard", "--config", shell_path(cfg), "--shards", "8", overrides={"PRISM_ARRAY": "0-7%8"})
    assert result.returncode == 0, result.stderr
    assert "--array=0-7%8" in result.stdout and "--gres=gpu:1" in result.stdout
    balanced = run("--dry-run", "rollout-shard", "--config", shell_path(cfg), "--shards", "32", overrides={"PRISM_ARRAY": "0-31%8"})
    assert balanced.returncode == 0 and "--array=0-31%8" in balanced.stdout
    mismatch = run("--dry-run", "rollout-shard", "--config", shell_path(cfg), "--shards", "4", overrides={"PRISM_ARRAY": "0-7%8"})
    assert mismatch.returncode != 0
    training = run("--dry-run", "cluster-label", "--config", shell_path(cfg))
    assert training.returncode == 0 and "--gres" not in training.stdout
    assert "--cpus-per-task=2" in training.stdout


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


def test_unverified_or_gpu_cpu_partition_is_rejected(platform):
    run, cfg, _, _ = platform
    placeholder = run("--dry-run", "design", "--config", shell_path(cfg), overrides={"PRISM_CPU_PARTITION": "FILL_FROM_VERIFIED_CPU_PARTITION"})
    assert placeholder.returncode != 0 and "CPU partition is not verified" in placeholder.stderr
    gpu_partition = run("--dry-run", "design", "--config", shell_path(cfg), overrides={"PRISM_CPU_PARTITION": "q_intel_gpu_nvidia_A100_share"})
    assert gpu_partition.returncode != 0 and "GPU partition requires GRES" in gpu_partition.stderr


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


def test_submit_execution_changes_to_canonical_repo_path(platform):
    _, _, repo, env = platform
    bash = str(Path(shutil.which("git")).parents[1]/"bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    source = shlex.quote(shell_path(repo/"scripts/yancheng/common.sh"))
    expected = shell_path(repo)
    script = f"source {source}\nprism_exec_from_repo pwd"
    result = subprocess.run([bash, "-c", script], env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_container_test_uses_a_validated_unique_temporary_directory():
    definition = (Path(__file__).parents[1]/"containers/prism.def").read_text(encoding="utf-8")
    assert 'mktemp -d /tmp/prism-tests.XXXXXX' in definition
    assert 'case "$test_root" in' in definition
    assert 'trap \'rm -rf -- "$test_root"\' EXIT' in definition
    assert '--basetemp "$test_root/pytest"' in definition
    assert '--basetemp /tmp/prism-tests\n' not in definition


def test_singularity_prefix_forwards_proxy_and_key_without_disclosing_values(platform):
    _, _, repo, env = platform
    bash = str(Path(shutil.which("git")).parents[1]/"bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    source = shlex.quote(shell_path(repo/"scripts/yancheng/common.sh"))
    script = f"set -e\nsource {source}\nprism_forward_api_envs TEST_JUDGE_KEY\nprism_forward_proxy_envs\n[[ $SINGULARITYENV_TEST_JUDGE_KEY == fixture-only-key && $SINGULARITYENV_https_proxy == http://proxy.invalid:3128 ]]\nprintf passed"
    result = subprocess.run([bash, "-c", script], env={**env, "PRISM_CONTAINER_RUNTIME": "singularity", "TEST_JUDGE_KEY": "fixture-only-key", "https_proxy": "http://proxy.invalid:3128"}, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == "passed"
    assert "fixture-only-key" not in result.stdout + result.stderr
