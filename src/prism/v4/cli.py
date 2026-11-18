"""V4 explicit entry point with runtime, cohort and production acceptance gates."""
from pathlib import Path

from prism.v3.cli import main as shared_main
from .config import load_config, require_runtime


def guard(c, args):
    if not args.execute or args.stage in ("validate", "plan", "resource", "status"):
        return
    if args.stage.startswith("lora-reward-"):
        raise ValueError("V4 currently tests the frozen linear Reward; LoRA Reward is a separate study")
    if args.stage == "policy":
        from .acceptance import require_policy_admission
        require_policy_admission(c)
    if args.stage in ("download-model", "prepare-data", "extract-features", "model-smoke", "policy-smoke",
                      "prepare-shared", "shared-smoke", "evaluate-generate", "runtime-check"):
        require_runtime(c)


def main(argv=None):
    from .acceptance import (model_smoke, policy_smoke, runtime_check, download_model, cohort_audit,
                              prepare_reference_data)
    handlers = {"runtime-check": ("cpu", lambda c, a: runtime_check(c)),
                "download-model": ("cpu", lambda c, a: download_model(c)),
                "model-smoke": ("gpu", lambda c, a: model_smoke(c)),
                "policy-smoke": ("gpu", lambda c, a: policy_smoke(c)),
                "prepare-data": ("cpu", lambda c, a: prepare_reference_data(c, a.reference_run)),
                "cohort-audit": ("cpu", lambda c, a: cohort_audit(c, a.reference_run))}
    return shared_main(argv, config_loader=load_config, stage_guard=guard, handlers=handlers,
                       default_config=str(Path(__file__).resolve().parents[3] / "configs/experiment.v4.yaml"))
