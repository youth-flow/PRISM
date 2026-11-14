"""Config-derived stage plans. No scheduler, model, or API is invoked here."""


def stage_resource(c, stage):
    if stage in ("model-smoke", "rollout", "rollout-shard", "policy", "evaluate-reference", "evaluate-generate"):
        return "gpu"
    if stage in ("label", "evaluate-judge"):
        from .judges import requires_gpu
        return "gpu" if requires_gpu(c, "training" if stage == "label" else "evaluation") else "cpu"
    if stage in ("stage-assets", "prepare-data", "merge-rollout", "design", "sample", "cluster-label", "reward", "aggregate", "validate", "plan"):
        return "cpu"
    raise ValueError(f"Unknown experiment stage: {stage}")


def _has_automatic_data_preparation(c):
    """Recognize the prompt-column/IFEval adapter's required source metadata.

    Other configurations may supply already prepared JSONL inputs. In particular,
    exact-match evaluation does not use the IFEval-specific preparation adapter.
    """
    if c.get("schema_version") != 2 or c["choices"]["benchmark"]["kind"] != "ifeval":
        return False
    data = c.get("data", {})
    source_fields = ("id", "revision", "train_split", "test_split", "prompt_field", "id_field")
    benchmark_fields = ("id", "revision", "split")
    return (all(key in data.get("source", {}) for key in source_fields)
            and all(key in data.get("benchmark_source", {}) for key in benchmark_fields)
            and all(key in data for key in ("model_context_tokens", "split_seed")))


def stage_plan(c):
    """Order shared preparation, all generation, then matched comparisons."""
    stages = []
    def add(stage, **cell):
        stages.append({"stage": stage, "resource": stage_resource(c, stage), **cell})
    add("stage-assets")
    if _has_automatic_data_preparation(c):
        add("prepare-data")
    add("model-smoke")
    add("rollout")
    add("design")
    cells = [{"method": method, "budget": budget, "seed": seed}
             for method in c["methods"] for budget in c["budgets"] for seed in c["seeds"]]
    for stage in ("sample", "label", "reward", "policy"):
        for cell in cells:
            add(stage, **cell)
    if c.get("schema_version") == 2:
        add("evaluate-reference")
    # All policy outputs must exist before PRISM-vs-Empirical judging begins.
    for stage in ("evaluate-generate", "evaluate-judge"):
        for cell in cells:
            add(stage, **cell)
    add("aggregate")
    return stages
