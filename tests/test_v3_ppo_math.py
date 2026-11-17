import numpy as np
import pytest

from prism.v3.ppo_math import gae_reference, learning_rate_multiplier, value_statistics


def test_schedule_warmup_and_nonzero_final_learning_rate():
    values = [learning_rate_multiplier(i, 100, .05, .1) for i in range(1, 101)]
    assert values[:5] == pytest.approx([.2, .4, .6, .8, 1.])
    assert values[-1] == pytest.approx(.1)
    assert np.all(np.diff(values[4:]) <= 0)
    with pytest.raises(ValueError):
        learning_rate_multiplier(0, 100)


def test_mc_return_and_gae_hand_computation():
    reward = np.array([[0., 0., 2., 0.]])
    values = np.array([[.1, .2, .3, 0.]])
    advantage, returns = gae_reference(reward, values, 1., 1.)
    assert returns == pytest.approx(np.array([[2., 2., 2., 0.]]))
    assert advantage == pytest.approx(returns - values)
    advantage, _ = gae_reference(reward, np.zeros_like(values), 1., .95)
    assert advantage == pytest.approx(np.array([[2 * .95 ** 2, 2 * .95, 2., 0.]]))


def test_value_statistics_reports_bias_and_failure_to_beat_constant():
    result = value_statistics([2., 2., 2.], [-1., 0., 1.])
    assert result["bias"] == 2.
    assert result["mse"] > result["constant_baseline_mse"]
    assert result["explained_variance"] == pytest.approx(0.)
    assert value_statistics([2., 2.], [1., 1.])["explained_variance"] is None
