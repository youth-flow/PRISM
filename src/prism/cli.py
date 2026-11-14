import argparse
import json
from pathlib import Path

from . import pipeline
from .io import METHODS, load_config


def main():
    parser = argparse.ArgumentParser(description="PRISM experiment stages (one cell per job)")
    parser.add_argument("stage", choices=["validate", "plan", "resource", "api-env-names", "stage-assets", "prepare-data", "model-smoke", "rollout", "rollout-shard", "merge-rollout", "design", "sample", "label", "reward", "policy", "evaluate-reference", "evaluate-generate", "evaluate-judge", "aggregate"])
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2]/"configs/experiment.yaml"))
    parser.add_argument("--for-stage", help="Stage to classify with the resource command")
    parser.add_argument("--judge-role", choices=["training", "evaluation"], help="Role for API environment-name metadata")
    parser.add_argument("--json", action="store_true", help="Machine-readable plan and count summary")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ready", action="store_true", help="Require all four choices and check inputs")
    parser.add_argument("--source-manifest", help="Pinned raw-source manifest for offline prepare-data")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shards", type=int, default=1)
    a = parser.parse_args()
    c, missing = load_config(a.config, ready=a.ready or a.stage not in ("validate", "plan", "resource", "api-env-names", "stage-assets", "prepare-data"))
    if a.stage == "validate":
        print(json.dumps({"structural_validation": "passed", "prepared_inputs_present": not missing, "full_input_validation": bool(a.ready), "remaining_choices": missing, "gpu_validation": "not_performed"}, ensure_ascii=False, indent=2))
        return
    if a.stage == "plan":
        from .workflow import stage_plan
        jobs = stage_plan(c)
        result = {"jobs": jobs, "paid_api_calls_enabled": c.get("execution", {}).get("allow_paid_api_calls", False), "note": "Plan only; no jobs submitted or APIs called."}
        if c.get("schema_version") == 2:
            from .config import plan_summary
            result["counts"] = plan_summary(c)
        if a.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if "counts" in result:
                print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
            for job in jobs:
                cell = " ".join(f"--{key} {job[key]}" for key in ("method", "budget", "seed") if key in job)
                print(f"{job['resource']:3}  {job['stage']} {cell}".rstrip())
            print(result["note"])
        return
    if a.stage == "resource":
        from .workflow import stage_resource
        if not a.for_stage:
            parser.error("resource requires --for-stage")
        print(stage_resource(c, a.for_stage))
        return
    if a.stage == "api-env-names":
        roles = [a.judge_role] if a.judge_role else ["training", "evaluation"]
        names = [c["judges"][role]["api_key_env"] for role in roles
                 if "judges" in c and c["judges"][role]["backend"] != "hf"]
        print(" ".join(dict.fromkeys(names)))
        return
    if a.stage == "aggregate":
        from .evaluation import aggregate
        print(aggregate(c))
    elif a.stage == "evaluate-reference":
        from .evaluation_v2 import evaluate_reference
        print(evaluate_reference(c))
    elif a.stage == "rollout-shard":
        if a.shard_index is None:
            parser.error("rollout-shard requires --shard-index")
        print(pipeline.rollout(c, shard_index=a.shard_index, shards=a.shards))
    elif a.stage == "merge-rollout":
        print(pipeline.merge_rollout(c, a.shards))
    elif a.stage == "prepare-data":
        print(pipeline.prepare_data(c, source_manifest=a.source_manifest))
    elif a.stage in ("stage-assets", "model-smoke", "rollout", "design"):
        print(getattr(pipeline, a.stage.replace("-", "_"))(c))
    else:
        if a.method is None or a.budget is None or a.seed is None:
            parser.error("Cell stages require --method, --budget, --seed")
        if a.stage.startswith("evaluate"):
            from . import evaluation
            function = getattr(evaluation, a.stage.replace("-", "_"))
        else:
            function = getattr(pipeline, a.stage)
        print(function(c, a.method, a.budget, a.seed))


if __name__ == "__main__":
    main()
