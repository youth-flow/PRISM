import numpy as np
import pytest

from prism.io import rows
from prism.v3.artifacts import Run, read_json
from prism.v3.audit import audit
from prism.v3.reward import fit_rewards, normalize_rewards, prediction_metrics, save_npz
from prism.v3.study import freeze_study, independent_design_audit, sample
from v3_helpers import configuration, numerical_run


def test_no_preference_training_before_user_budget(tmp_path):
    c = configuration(tmp_path)
    numerical_run(c)
    with pytest.raises(ValueError, match="user must choose"):
        freeze_study(c, 4)
    with pytest.raises(FileNotFoundError):
        sample(c)
    with pytest.raises(FileNotFoundError):
        fit_rewards(c)


def test_one_budget_two_heads_shared_regularizer_and_fixed_normalization(tmp_path):
    c = configuration(tmp_path)
    run = numerical_run(c)
    selection = freeze_study(c, 4, user_confirmed=True)
    assert selection["independent_design_audit"]["passed"]
    with pytest.raises(ValueError, match="Preserve"):
        freeze_study(c, 3, user_confirmed=True)
    summary = sample(c)
    assert summary["formal_policy_units"] == 2
    assert summary["new_annotation_api_calls"] == 0
    for method in c["study"]["methods"]:
        assert len({r["id"] for r in rows(run.file(f"sampling/{method}.jsonl"))}) == 4
        metadata = read_json(run.file(f"sampling/{method}.json"))
        assert metadata["shared_rng_seed"] is True
        assert metadata["common_gumbel_keys"] is False
        assert metadata["gumbel_key_alignment"] == "positive_support_order"
    report = fit_rewards(c)
    assert report["selected_l2"] in c["reward"]["l2_candidates"]
    assert len(report["candidates"]) == 3
    assert fit_rewards(c) == report
    from prism.v3.training_audit import audit_training
    verified = audit_training(c)
    assert verified["passed"] and verified["selected_l2"] == report["selected_l2"]
    assert all(row["independent_wor_replay_exact"] for row in verified["sampling"].values())
    assert audit_training(c) == verified
    calibration = np.arange(24, dtype=np.float32).reshape(8, 3) / 7
    path = run.file("shared/calibration.features.npz")
    save_npz(path, phi=calibration, ids=np.array([f"cal-{i}" for i in range(8)]))
    run.publish("shared-sft", [path], [run.receipt_path("data")])
    norms = normalize_rewards(c)
    for method in c["study"]["methods"]:
        assert abs(norms[method]["reference_normalized_mean"]) < 1e-12
        assert abs(norms[method]["reference_normalized_std"] - 1) < 1e-12
        assert read_json(run.file(f"normalization/{method}.json"))["frozen_after_fit"]
    assert normalize_rewards(c) == norms
    partial = audit(c)
    assert partial["complete"] is False
    assert partial["status"] == "partial_engineering_readiness"
    with pytest.raises(ValueError, match="not complete"):
        audit(c, require_complete=True)


def test_design_audit_rejects_unconverged_and_invalid_weights():
    v = np.eye(3)
    q = np.array([1.0, 0, 0])
    with pytest.raises(ValueError, match="gap"):
        independent_design_audit(v, np.eye(3), q, 0.01, 1e-4)
    with pytest.raises(ValueError, match="simplex"):
        independent_design_audit(v, np.eye(3), np.array([1.01, -0.01, 0]), 0.01, 1e-4)


def test_probability_validation_metrics_expose_overconfidence():
    v = np.ones((4, 1))
    metrics = prediction_metrics(v, [1, 1, 0, 0], [10])
    assert metrics["accuracy"] == 0.5
    assert metrics["confidence_ece"] > 0.49
    assert metrics["bt_nll"] > 4
    neutral = prediction_metrics(v, [1, 0, 1, 0], [0])
    assert neutral["accuracy"] == 0.5
    assert neutral["bt_nll"] == pytest.approx(np.log(2))


def test_independent_training_audit_rejects_a_false_gradient_claim(tmp_path, monkeypatch):
    from prism.v3.training_audit import audit_training
    c = configuration(tmp_path)
    numerical_run(c)
    freeze_study(c, 4, user_confirmed=True)
    sample(c)

    def false_fit(v, labels, **kwargs):
        return np.zeros(v.shape[1]), {"gradient_inf": 0., "loss": float(np.log(2.))}

    monkeypatch.setattr("prism.v3.reward.fit_bt", false_fit)
    fit_rewards(c)
    with pytest.raises(ValueError, match="gradient/objective"):
        audit_training(c)


def test_audit_checks_actual_completed_policy_not_the_configured_target(tmp_path):
    from prism.v3.artifacts import write_json
    c = configuration(tmp_path)
    run = Run(c)
    p = c["ppo"]
    rounds = p["total_episodes"] // p["rollout_batch_size"]
    steps = rounds * p["num_ppo_epochs"] * p["num_mini_batches"]
    value = {"total_episodes": p["total_episodes"], "completed": False,
             "completed_episodes": p["total_episodes"] // 2, "rollout_rounds": rounds // 2,
             "actual_optimizer_steps": {role: {"step_min": steps // 2, "step_max": steps // 2, "tensor_count": 1}
                                        for role in ("actor", "critic_lora", "critic_head")},
             "gradient_norm_clipping_executed": True, "probability_checks": [
                 {"max_abs_logp_error": 1e-6, "zero_update_approxkl": 1e-11} for _ in range(rounds)]}
    path = run.file("policy/empirical/report.json")
    write_json(path, value)
    run.publish("policy-empirical", [path])
    with pytest.raises(ValueError, match="partial policy"):
        audit(c)
    # Rebuild an independent completed fixture: a completed receipt alone is
    # insufficient; audit must also inspect the engine's actual counters.
    value.update(completed=True, completed_episodes=p["total_episodes"], rollout_rounds=rounds)
    for row in value["actual_optimizer_steps"].values():
        row.update(step_min=steps, step_max=steps)
    second = dict(c)
    second["output"] = str(tmp_path / "completed-run")
    other = Run(second)
    target = other.file("policy/empirical/report.json")
    write_json(target, value)
    other.publish("policy-empirical", [target])
    assert audit(second)["checks"]["policy_empirical"]["actual_steps"] == steps
