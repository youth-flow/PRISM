import numpy as np
import pytest

from prism.io import digest
from prism.v3.full_pool_reward import fit_full_pool_candidate, summarize_full_pool
from prism.v3.reward import fit_rewards
from prism.v3.study import freeze_study, sample
from v3_helpers import configuration, numerical_run


def test_full_pool_uses_all_training_and_preserves_formal_heads(tmp_path):
    c = configuration(tmp_path)
    run = numerical_run(c)
    freeze_study(c, 4, user_confirmed=True)
    sample(c)
    fit_rewards(c)
    protected = [run.file(f"reward/{m}.npz") for m in c["study"]["methods"]]
    protected += [run.receipt_path("reward"), run.file("study/selection.json")]
    before = {str(p): digest(p) for p in protected}
    for penalty in c["reward"]["l2_candidates"]:
        report = fit_full_pool_candidate(c, penalty)
        assert report["training_pairs"] == 6  # Formal N is only 4.
        assert report["development_pairs"] == 4
        assert report["independent_fit_check"]["passed"]
        assert set(report["development_subsets"]) == {"helpful", "helpful-base"}
        assert fit_full_pool_candidate(c, penalty) == report
    result = summarize_full_pool(c)
    assert result["diagnostic_only"] and result["independent_audit_passed"]
    assert not result["backbone_updated"] and not result["PPO_started"]
    assert result["N14000_reference"]["budget"] == 4
    assert len(result["candidates"]) == 3
    assert summarize_full_pool(c) == result
    assert {str(p): digest(p) for p in protected} == before
    with pytest.raises(ValueError, match="predeclared"):
        fit_full_pool_candidate(c, .33)


def test_full_pool_diagnostic_rejects_false_convergence(tmp_path, monkeypatch):
    c = configuration(tmp_path)
    numerical_run(c)

    def false_fit(v, y, **kwargs):
        return np.zeros(v.shape[1]), {"gradient_inf": 0., "loss": float(np.log(2.))}

    monkeypatch.setattr("prism.v3.full_pool_reward.fit_bt", false_fit)
    with pytest.raises(ValueError, match="gradient/objective"):
        fit_full_pool_candidate(c, .1)
