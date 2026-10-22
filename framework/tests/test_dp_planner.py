"""Central-DP planner: what an (epsilon, delta) budget implies for noise before a run starts."""
import math

import pytest

from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.privacy.planner import plan_central_dp


def test_plan_uses_the_platform_accountant_and_the_dp_aggregate_noise_scale():
    plan = plan_central_dp(target_epsilon=4.0, delta=1e-5, rounds=20, clients=80, clip_norm=0.05,
                           trainable_parameters=1026)
    z = float(required_noise_multiplier(4.0, 1.0, 20, 1e-5))
    assert plan["noise_multiplier"] == pytest.approx(z)
    assert plan["per_coordinate_noise_std"] == pytest.approx(z * 0.05 / 80)
    assert plan["expected_noise_norm"] == pytest.approx(z * 0.05 / 80 * math.sqrt(1026))
    assert plan["noise_norm_to_clip_ratio"] == pytest.approx(z * math.sqrt(1026) / 80)
    assert plan["accounted_epsilon"] == pytest.approx(float(get_epsilon(compute_rdp(1.0, z, 20), 1e-5)[0]))
    assert plan["accounted_epsilon"] <= 4.0 + 1e-6


def test_the_ratio_grows_with_width_and_falls_with_cohort_size_and_is_clip_free():
    base = dict(target_epsilon=4.0, delta=1e-5, rounds=20, clients=80, clip_norm=0.05, trainable_parameters=1026)
    wide = plan_central_dp(**{**base, "trainable_parameters": 2050})
    big_cohort = plan_central_dp(**{**base, "clients": 320})
    other_clip = plan_central_dp(**{**base, "clip_norm": 1.0})
    ratio = plan_central_dp(**base)["noise_norm_to_clip_ratio"]
    assert wide["noise_norm_to_clip_ratio"] == pytest.approx(ratio * math.sqrt(2050 / 1026))
    assert big_cohort["noise_norm_to_clip_ratio"] == pytest.approx(ratio / 4)
    assert other_clip["noise_norm_to_clip_ratio"] == pytest.approx(ratio)
    assert other_clip["per_coordinate_noise_std"] == pytest.approx(plan_central_dp(**base)["per_coordinate_noise_std"] * 20)


@pytest.mark.parametrize("bad", [dict(target_epsilon=0), dict(delta=1.0), dict(rounds=0), dict(clients=0),
                                 dict(clip_norm=-1), dict(trainable_parameters=0)])
def test_invalid_inputs_are_refused(bad):
    base = dict(target_epsilon=4.0, delta=1e-5, rounds=20, clients=80, clip_norm=0.05, trainable_parameters=1026)
    with pytest.raises(ValueError):
        plan_central_dp(**{**base, **bad})
