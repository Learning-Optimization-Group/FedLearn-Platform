"""DPFedAvg: central DP for FedAvg with FedLoRA's exact mechanism (clip S, uniform mean, N(0,(zS/N)^2))."""
from collections import OrderedDict
import math

import pytest
import torch

from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.server import DPFedAvg


def _global():
    return OrderedDict(head=torch.zeros(4), bias=torch.zeros(2))


def _update(scale, n=10):
    return OrderedDict(head=torch.full((4,), float(scale)), bias=torch.full((2,), float(scale))), n


def test_zero_noise_is_the_uniform_mean_of_jointly_clipped_deltas():
    strategy = DPFedAvg(_global(), dp_clip_norm=1.0, dp_noise_multiplier=0.0, dp_seed=0)
    small = _update(0.1, n=1)        # norm sqrt(6)*0.1 < 1: unclipped
    large = _update(10.0, n=1000)    # norm sqrt(6)*10 > 1: scaled to norm 1; its 1000 examples carry no extra weight
    out = strategy.aggregate_fit(1, [("a", *small), ("b", *large)])
    clipped_coordinate = 10.0 / (math.sqrt(6) * 10.0)   # 10 per coordinate, scaled by 1/||delta||
    for key in ("head", "bias"):
        assert torch.allclose(out[key], torch.full_like(out[key], (0.1 + clipped_coordinate) / 2), atol=1e-6)


def test_noise_std_is_z_times_clip_over_n():
    big = OrderedDict(w=torch.zeros(200_000))
    strategy = DPFedAvg(big, dp_clip_norm=2.0, dp_noise_multiplier=1.5, dp_seed=7)
    results = [(f"c{i}", OrderedDict(w=torch.zeros(200_000)), 5) for i in range(4)]
    out = strategy.aggregate_fit(1, results)
    assert float(out["w"].std()) == pytest.approx(1.5 * 2.0 / 4, rel=0.02)


def test_target_epsilon_uses_the_platform_accountant_and_exposes_the_eval_card_trace():
    strategy = DPFedAvg(_global(), min_fit_clients=8, dp_clip_norm=0.5, dp_target_epsilon=4.0,
                        dp_delta=1e-5, dp_rounds=20)
    expected_z = float(required_noise_multiplier(4.0, 1.0, 20, 1e-5))
    assert strategy.dp_enabled is True
    assert strategy.dp_noise_multiplier == pytest.approx(expected_z)
    assert strategy.dp_q == 1.0 and strategy.dp_rounds == 20 and strategy.dp_target_epsilon == 4.0
    assert strategy.dp_accounted_epsilon == pytest.approx(
        float(get_epsilon(compute_rdp(1.0, expected_z, 20), 1e-5)[0]))
    assert strategy.dp_accounted_epsilon <= 4.0 + 1e-6


def test_the_delta_reference_advances_with_each_released_global():
    strategy = DPFedAvg(_global(), dp_clip_norm=100.0, dp_noise_multiplier=0.0, dp_seed=0)
    first = strategy.aggregate_fit(1, [("a", *_update(1.0))])
    second = strategy.aggregate_fit(2, [("a", *_update(1.0))])  # same params as the new global: zero delta
    assert torch.equal(first["head"], second["head"])


def test_only_the_submitted_keys_are_privatised_and_returned():
    initial = OrderedDict(head=torch.zeros(3), frozen=torch.ones(3))
    strategy = DPFedAvg(initial, dp_clip_norm=1.0, dp_noise_multiplier=1.0, dp_seed=1)
    out = strategy.aggregate_fit(1, [("a", OrderedDict(head=torch.zeros(3)), 4)])
    assert list(out) == ["head"]


def test_unseeded_noise_does_not_follow_the_disclosed_global_torch_seed():
    draws = []
    for _ in range(2):
        torch.manual_seed(1234)
        strategy = DPFedAvg(OrderedDict(w=torch.zeros(64)), dp_clip_norm=1.0, dp_noise_multiplier=1.0)
        draws.append(strategy.aggregate_fit(1, [("a", OrderedDict(w=torch.zeros(64)), 1)])["w"])
    assert not torch.equal(draws[0], draws[1])


@pytest.mark.parametrize("kwargs, message", [
    (dict(dp_noise_multiplier=1.0), "clip"),
    (dict(dp_clip_norm=1.0), "exactly one"),
    (dict(dp_clip_norm=1.0, dp_noise_multiplier=1.0, dp_target_epsilon=4.0, dp_delta=1e-5, dp_rounds=5), "not both"),
    (dict(dp_clip_norm=1.0, dp_target_epsilon=4.0), "dp_delta and dp_rounds"),
])
def test_an_incomplete_or_contradictory_config_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DPFedAvg(_global(), **kwargs)


def _simulate(strategy_factory, rounds=3):
    from fedlearn.simulation.federation import SimulatedFederation
    from tests.test_simulation_federation import _evaluate, _initial_params, _make_factory
    init = _initial_params()
    sim = SimulatedFederation(strategy=strategy_factory(init), client_factory=_make_factory(8), num_clients=8,
                              clients_per_round=8, seed=0, initial_parameters=init,
                              client_config={"learning_rate": 0.5, "local_epochs": 2}, wire_in_the_loop=True)
    return sim.run(rounds)


def test_through_the_production_coordinator_and_wire_zero_noise_matches_fedavg_on_equal_shards():
    from fedlearn.server.strategy import FedAvg
    from tests.test_simulation_federation import _evaluate
    plain = _simulate(lambda init: FedAvg(init, evaluate_fn=_evaluate, min_fit_clients=1, clients_per_round=8))
    private = _simulate(lambda init: DPFedAvg(init, evaluate_fn=_evaluate, min_fit_clients=1, clients_per_round=8,
                                              dp_clip_norm=1e6, dp_noise_multiplier=0.0, dp_seed=0))
    for key in plain.final_params:
        assert torch.allclose(plain.final_params[key], private.final_params[key], atol=1e-6)


def test_through_the_production_coordinator_a_noisy_dp_run_completes_with_a_finite_model():
    from tests.test_simulation_federation import _evaluate
    result = _simulate(lambda init: DPFedAvg(init, evaluate_fn=_evaluate, min_fit_clients=1, clients_per_round=8,
                                             dp_clip_norm=0.5, dp_target_epsilon=4.0, dp_delta=1e-5, dp_rounds=3))
    assert all(torch.isfinite(v).all() for v in result.final_params.values())
    assert len(result.rounds) == 3
