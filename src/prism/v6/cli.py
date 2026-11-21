"""Narrow V6 CLI around the audited V3/V5 linear-Reward engine."""
import argparse
import json
from pathlib import Path

from prism.v3.cli import main as v3_main

from .config import load_config, plan


DEFAULT_CONFIG = str(Path(__file__).resolve().parents[3] / "configs/experiment.v6.yaml")
LOCAL_STAGES = {"validate", "plan"}
EXECUTION_STAGES = {
    "import-upstream",
    "resource", "status", "download-data", "prepare-data", "extract-features",
    "merge-features", "geometry", "geometry-audit", "design", "design-report",
    "freeze-study", "sample", "reward", "audit-training", "prepare-shared",
    "merge-shared", "shared-smoke", "normalize-reward", "model-smoke",
    "linear-ppo-admit", "linear-ppo-smoke", "linear-ppo-policy",
    "linear-ppo-audit", "evaluate-generate", "merge-evaluation", "judge-export",
    "judge-import", "aggregate", "linear-ppo-ifeval-generate",
    "linear-ppo-ifeval-merge", "audit", "lora-ppo-ifeval-sft",
}


def _config_arg(argv):
    for index, arg in enumerate(argv):
        if arg == "--config" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return DEFAULT_CONFIG


def _with_default_config(argv):
    if any(arg == "--config" or arg.startswith("--config=") for arg in argv):
        return argv
    return [*argv, "--config", DEFAULT_CONFIG]


def _option(argv, name):
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _guard_execution(argv, c):
    stage, budget = argv[0], c["v6"]["preference_budget"]
    method = _option(argv, "--method")
    if method == "full-pool-lora":
        raise SystemExit("V6 rejects LoRA Reward/PPO controls as primary methods")
    if stage == "freeze-study":
        selected_budget, design = _option(argv, "--budget"), _option(argv, "--design-lambda")
        if selected_budget != str(budget) or design is None or float(design) != 0.001:
            raise SystemExit(
                f"V6 freeze-study requires --budget {budget} --design-lambda 0.001")
    if stage == "linear-ppo-policy" and _option(argv, "--stop-after-episodes") != "16384":
        raise SystemExit("V6 linear PPO must stop at exactly 16384 episodes")
    episode = _option(argv, "--evaluation-episode")
    if episode is not None and method in ("empirical", "prism") and episode != "16384":
        raise SystemExit("V6 primary linear-policy evaluation must use episode 16384")


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            description="V6: V5 protocol with budget-capped PRISM Design")
        parser.add_argument("stage", choices=sorted(LOCAL_STAGES | EXECUTION_STAGES))
        parser.print_help()
        return
    stage = argv[0]
    if stage not in LOCAL_STAGES | EXECUTION_STAGES:
        raise SystemExit(f"V6 rejects unknown or out-of-scope stage: {stage}")
    c = load_config(_config_arg(argv))
    if stage == "validate":
        value = {"structural_validation": "passed", "plan": plan(c)}
    elif stage == "plan":
        value = plan(c)
    else:
        _guard_execution(argv, c)
        return v3_main(_with_default_config(argv))
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
