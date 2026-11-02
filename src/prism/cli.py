import argparse
import json

from . import pipeline
from .io import METHODS, load_config


def main():
    parser = argparse.ArgumentParser(description="PRISM experiment stages (one cell per job)")
    parser.add_argument("stage", choices=["validate", "plan", "stage-assets", "model-smoke", "rollout", "design", "sample", "label", "reward", "policy", "evaluate-generate", "evaluate-judge", "aggregate"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ready", action="store_true", help="Require all four choices and check inputs")
    a = parser.parse_args()
    c, missing = load_config(a.config, ready=a.ready or a.stage not in ("validate", "plan"))
    if a.stage == "validate":
        print(json.dumps({"structural_validation": "passed", "ready": not missing, "remaining_choices": missing}, ensure_ascii=False, indent=2))
        return
    if a.stage == "plan":
        print("Shared: stage-assets -> model-smoke -> rollout -> design")
        for method in METHODS:
            for budget in c["budgets"]:
                for seed in c["seeds"]:
                    print(f"{method} n={budget} seed={seed}: sample -> label -> reward -> policy -> evaluate-generate -> evaluate-judge")
        print("Finally: aggregate. These are plans only; no jobs submitted.")
        return
    if a.stage == "aggregate":
        from .evaluation import aggregate
        print(aggregate(c))
    elif a.stage in ("stage-assets", "model-smoke", "rollout", "design"):
        getattr(pipeline, a.stage.replace("-", "_"))(c)
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
