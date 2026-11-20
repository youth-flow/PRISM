"""Narrow V5 CLI around the audited V3 linear-Reward execution engine."""
import argparse
import json
from pathlib import Path

from prism.v3.cli import main as v3_main

from .config import load_config, plan
from .registry import load_registry, summary


DEFAULT_CONFIG = str(Path(__file__).resolve().parents[3] / "configs/experiment.v5.yaml")
LOCAL_STAGES = {"validate", "plan", "results"}
EXECUTION_STAGES = {
    "import-upstream",
    "resource", "status", "download-data", "prepare-data", "extract-features",
    "merge-features", "geometry", "geometry-audit", "design", "design-report",
    "freeze-study", "sample", "reward", "audit-training", "prepare-shared",
    "merge-shared", "shared-smoke", "normalize-reward", "model-smoke",
    "full-pool-reward", "full-pool-reward-report", "linear-ppo-admit",
    "linear-ppo-smoke", "linear-ppo-policy", "linear-ppo-audit",
    "evaluate-generate", "merge-evaluation", "judge-export", "judge-import",
    "aggregate", "linear-ppo-ifeval-generate", "linear-ppo-ifeval-merge", "audit",
    "lora-ppo-ifeval-sft",
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
    stage = argv[0]
    method = _option(argv, "--method")
    if method == "full-pool-lora":
        raise SystemExit("V5 rejects LoRA Reward/PPO controls as executable primary methods")
    if stage == "freeze-study":
        budget, design = _option(argv, "--budget"), _option(argv, "--design-lambda")
        expected = str(c["v5"]["preference_budget"])
        if budget != expected or design is None or float(design) != 0.001:
            raise SystemExit(
                f"V5 freeze-study requires --budget {expected} --design-lambda 0.001")
    if stage == "linear-ppo-policy":
        stop = _option(argv, "--stop-after-episodes")
        if stop != "16384":
            raise SystemExit("V5 linear PPO must stop at exactly 16384 episodes")
    episode = _option(argv, "--evaluation-episode")
    if episode is not None and method in ("empirical", "prism") and episode != "16384":
        raise SystemExit("V5 primary linear-policy evaluation must use episode 16384")


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            description="V5: OLMo-1B + HH + frozen-linear Reward budget studies + 16k PPO")
        parser.add_argument("stage", choices=sorted(LOCAL_STAGES | EXECUTION_STAGES))
        parser.print_help()
        return
    stage = argv[0]
    if stage not in LOCAL_STAGES | EXECUTION_STAGES:
        raise SystemExit(f"V5 rejects non-linear or unknown stage: {stage}")
    c = load_config(_config_arg(argv))
    if stage == "validate":
        value = {
            "structural_validation": "passed",
            "plan": plan(c),
            "registry": summary(load_registry(c["_v5_registry_path"])),
        }
    elif stage == "plan":
        value = plan(c)
    elif stage == "results":
        value = summary(load_registry(c["_v5_registry_path"]))
    else:
        # The V3 engine is reused only after the stricter V5 contract passes.
        _guard_execution(argv, c)
        return v3_main(_with_default_config(argv))
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
