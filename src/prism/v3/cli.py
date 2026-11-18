"""Explicit V3 stage entry point; planning never submits or executes work."""
import argparse
import json
from pathlib import Path

from .config import load_config, plan


GPU_STAGES = {"extract-features", "prepare-shared", "shared-smoke", "policy", "evaluate-generate", "model-smoke",
              "lora-reward-smoke", "lora-reward-train", "lora-reward-score",
              "lora-ppo-normalize", "lora-ppo-smoke", "lora-ppo-policy"}
STAGES = ("validate", "plan", "resource", "status", "download-data", "prepare-data", "extract-features",
          "merge-features", "geometry", "geometry-audit", "design", "design-report", "freeze-study", "sample", "reward",
          "prepare-shared", "merge-shared", "normalize-reward", "policy", "evaluate-generate", "merge-evaluation",
          "judge-export", "judge-import", "aggregate", "audit", "model-smoke", "shared-smoke", "audit-training",
          "full-pool-reward", "full-pool-reward-report", "lora-reward-data", "lora-reward-smoke",
          "lora-reward-train", "lora-reward-score", "lora-reward-report",
          "lora-ppo-normalize", "lora-ppo-smoke", "lora-ppo-policy", "lora-ppo-admit")


def stage_resource(stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown V3 stage: {stage}")
    return "gpu" if stage in GPU_STAGES else "cpu"


def main(argv=None):
    parser = argparse.ArgumentParser(description="HH human-preference V3 stages; no implicit paid or compute jobs")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[3] / "configs/experiment.v3.yaml"))
    parser.add_argument("--execute", action="store_true", help="Explicitly execute a stage; never sets a paid-call allowance")
    parser.add_argument("--for-stage", choices=STAGES)
    parser.add_argument("--method", choices=["sft", "empirical", "prism"])
    parser.add_argument("--source-manifest")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--design-lambda", type=float)
    parser.add_argument("--reward-l2", type=float)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--user-budget-confirmed", action="store_true", help="Record the user's N decision after reviewing Design")
    parser.add_argument("--user-lora-ppo-confirmed", action="store_true",
                        help="Bind explicit continuation authority after reviewing the LoRA PPO episode-8192 gate")
    parser.add_argument("--stop-after-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results")
    parser.add_argument("--complete", action="store_true", help="Require every formal stage in the final audit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    c = load_config(args.config)
    if args.stage == "validate":
        value = {"structural_validation": "passed", "counts": plan(c),
                 "formal_compute_started": False, "note": "Numerical/model/data acceptance is performed by explicit stages."}
    elif args.stage == "plan":
        value = plan(c)
        value["phases"] = [
            {"phase": 1, "stages": ["download-data", "prepare-data", "extract-features", "merge-features", "prepare-shared"]},
            {"phase": 2, "stages": ["geometry", "design", "design-report"], "next": "user chooses N; assistant selects ridge"},
            {"phase": 3, "stages": ["freeze-study", "sample", "reward", "normalize-reward"], "blocked_until": "explicit user budget selection"},
            {"phase": 4, "stages": ["policy"], "units": 2, "acceptance": "critic prefit; engineering and development probes; true optimizer steps"},
            {"phase": 5, "stages": ["evaluate-generate", "merge-evaluation", "judge-export"]},
            {"phase": 6, "stages": ["judge-import", "aggregate", "audit"], "local_paid_judge": "requires a separate explicit V3 ceiling"}]
    elif args.stage == "resource":
        if not args.for_stage:
            parser.error("resource requires --for-stage")
        print(stage_resource(args.for_stage))
        return
    elif args.stage == "status":
        from .audit import status
        value = status(c)
    else:
        if not args.execute:
            parser.error(f"{args.stage} requires --execute; plan/validate do not execute work")
        if args.stage == "download-data":
            from .data import download_raw
            value = download_raw(c)
        elif args.stage == "prepare-data":
            from .data import prepare_data
            value = prepare_data(c, source_manifest=args.source_manifest)
        elif args.stage == "extract-features":
            from .features import extract_features
            value = extract_features(c, shard_index=args.shard_index, shards=c["features"]["shards"] if args.shards is None else args.shards)
        elif args.stage == "merge-features":
            from .features import merge_features
            value = merge_features(c, shards=c["features"]["shards"] if args.shards is None else args.shards)
        elif args.stage == "geometry":
            from .geometry import build_geometry
            from .audit import audit_geometry
            value = {"geometry": build_geometry(c), "independent_audit": audit_geometry(c)}
        elif args.stage == "geometry-audit":
            from .audit import audit_geometry
            value = audit_geometry(c)
        elif args.stage == "design":
            from .geometry import solve_design
            values = c["design"]["lambdas"] if args.design_lambda is None else [args.design_lambda]
            value = [solve_design(c, design_lambda=v) for v in values]
        elif args.stage == "design-report":
            from .geometry import design_report
            value = design_report(c)
        elif args.stage == "freeze-study":
            if args.budget is None:
                parser.error("freeze-study requires the user's --budget")
            from .study import freeze_study
            value = freeze_study(c, args.budget, args.design_lambda, user_confirmed=args.user_budget_confirmed)
        elif args.stage == "sample":
            from .study import sample
            value = sample(c)
        elif args.stage == "reward":
            from .reward import fit_rewards
            value = fit_rewards(c)
        elif args.stage == "audit-training":
            from .training_audit import audit_training
            value = audit_training(c)
        elif args.stage == "full-pool-reward":
            if args.reward_l2 is None:
                parser.error("full-pool-reward requires --reward-l2 from the original grid")
            from .full_pool_reward import fit_full_pool_candidate
            value = fit_full_pool_candidate(c, args.reward_l2)
        elif args.stage == "full-pool-reward-report":
            from .full_pool_reward import summarize_full_pool
            value = summarize_full_pool(c)
        elif args.stage.startswith("lora-reward-"):
            from . import lora_reward
            if args.stage == "lora-reward-data":
                value = lora_reward.prepare(c)
            elif args.stage == "lora-reward-smoke":
                value = lora_reward.smoke(c)
            elif args.stage == "lora-reward-train":
                value = lora_reward.train(c, resume=args.resume)
            else:
                if args.shards is None:
                    parser.error(f"{args.stage} requires --shards")
                from . import lora_reward_evaluation
                value = (lora_reward_evaluation.score(c, args.shard_index, args.shards) if args.stage == "lora-reward-score"
                         else lora_reward_evaluation.summarize(c, args.shards))
        elif args.stage.startswith("lora-ppo-"):
            if args.stage == "lora-ppo-policy" and args.stop_after_episodes is None:
                parser.error("lora-ppo-policy requires an explicit --stop-after-episodes checkpoint")
            from . import lora_ppo
            if args.stage == "lora-ppo-normalize":
                value = lora_ppo.normalize_reward(c)
            elif args.stage == "lora-ppo-smoke":
                value = lora_ppo.policy_smoke(c)
            elif args.stage == "lora-ppo-policy":
                value = lora_ppo.train_policy(c, stop_after_episodes=args.stop_after_episodes, resume=args.resume)
            else:
                value = lora_ppo.admit_continuation(c, user_confirmed=args.user_lora_ppo_confirmed)
        elif args.stage == "prepare-shared":
            if args.shards is None:
                from .ppo import prepare_shared
                value = prepare_shared(c)
            else:
                from .shared import prepare_shared_part
                value = prepare_shared_part(c, args.shard_index, args.shards)
        elif args.stage == "merge-shared":
            if args.shards is None:
                parser.error("merge-shared requires --shards")
            from .shared import merge_shared
            value = merge_shared(c, args.shards)
        elif args.stage == "shared-smoke":
            from .shared import shared_smoke
            value = shared_smoke(c)
        elif args.stage == "normalize-reward":
            from .reward import normalize_rewards
            value = normalize_rewards(c)
        elif args.stage == "policy":
            if args.method not in ("empirical", "prism"):
                parser.error("policy requires --method empirical or prism")
            from .ppo import train_policy
            value = train_policy(c, args.method, stop_after_episodes=args.stop_after_episodes, resume=args.resume)
        elif args.stage == "evaluate-generate":
            if args.method is None:
                parser.error("evaluate-generate requires --method")
            from .evaluation import evaluate_generate
            value = evaluate_generate(c, args.method, shard_index=args.shard_index, shards=1 if args.shards is None else args.shards)
        elif args.stage == "merge-evaluation":
            if args.method is None:
                parser.error("merge-evaluation requires --method")
            from .evaluation import merge_evaluation
            value = merge_evaluation(c, args.method, 1 if args.shards is None else args.shards)
        elif args.stage == "judge-export":
            from .evaluation import export_judge
            value = export_judge(c)
        elif args.stage == "judge-import":
            if not args.results:
                parser.error("judge-import requires --results")
            from .evaluation import import_judge
            value = import_judge(c, args.results)
        elif args.stage == "aggregate":
            from .evaluation import aggregate
            value = aggregate(c)
        elif args.stage == "audit":
            from .audit import audit
            value = audit(c, require_complete=args.complete)
        elif args.stage == "model-smoke":
            from .smoke import model_smoke
            value = model_smoke(c)
        else:
            raise AssertionError("Unimplemented V3 CLI stage")
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
