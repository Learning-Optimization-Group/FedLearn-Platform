# framework/tests/test_gdp_accountant.py
"""Exact Gaussian-DP accounting for full-participation releases (q = 1)."""

import math
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.privacy.gdp_accountant import (
    gdp_delta, gdp_epsilon, gdp_mu, gdp_required_noise_multiplier,
)
from fedlearn.privacy.planner import plan_central_dp
from fedlearn.server.strategy import DPFedAvg


def _hockey_stick_by_quadrature(epsilon, mu):
    """δ(ε) between N(μ,1) and N(0,1) by direct numerical integration, independent of the closed form."""
    x = np.linspace(-40.0, 40.0 + mu, 2_000_001)
    p = np.exp(-0.5 * (x - mu) ** 2) / math.sqrt(2 * math.pi)
    q = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
    return float(np.trapezoid(np.maximum(0.0, p - math.exp(epsilon) * q), x))


@pytest.mark.parametrize("epsilon, mu", [(0.0, 0.5), (0.5, 0.3), (1.0, 1.0), (3.0, 1.5), (6.0, 1.45)])
def test_closed_form_delta_matches_numerical_integration(epsilon, mu):
    assert gdp_delta(epsilon, mu) == pytest.approx(_hockey_stick_by_quadrature(epsilon, mu), rel=1e-6, abs=1e-12)


def test_composition_is_root_steps_over_z():
    assert gdp_mu(2.0, 16) == pytest.approx(2.0)


@pytest.mark.parametrize("claimed, exact", [(8.0, 6.763), (4.0, 3.250), (1.0, 0.741)])
def test_rdp_calibrated_noise_is_looser_than_its_claim(claimed, exact):
    z = required_noise_multiplier(claimed, 1.0, 20, 1e-5)
    assert gdp_epsilon(z, 20, 1e-5) == pytest.approx(exact, abs=1e-3)


@pytest.mark.parametrize("z", [0.6, 1.0, 3.0, 10.0, 30.0])
@pytest.mark.parametrize("steps", [1, 5, 20, 100])
def test_never_above_the_rdp_bound(z, steps):
    rdp = get_epsilon(compute_rdp(1.0, z, steps), 1e-5)[0]
    assert gdp_epsilon(z, steps, 1e-5) <= rdp + 1e-9


@pytest.mark.parametrize("target", [0.5, 1.0, 4.0, 8.0, 16.0])
@pytest.mark.parametrize("steps", [1, 20, 200])
def test_required_noise_round_trips_and_is_minimal(target, steps):
    z = gdp_required_noise_multiplier(target, steps, 1e-5)
    assert gdp_epsilon(z, steps, 1e-5) <= target + 1e-6
    assert gdp_epsilon(z * 0.999, steps, 1e-5) > target


def test_exact_calibration_needs_less_noise_than_rdp():
    for target, saving in [(8.0, 0.13), (4.0, 0.17), (1.0, 0.24)]:
        z_rdp = required_noise_multiplier(target, 1.0, 20, 1e-5)
        z_gdp = gdp_required_noise_multiplier(target, 20, 1e-5)
        assert 1 - z_gdp / z_rdp == pytest.approx(saving, abs=0.01)


def test_tiny_mu_satisfies_any_delta_at_zero_epsilon():
    assert gdp_epsilon(1e6, 1, 1e-5) == 0.0


@pytest.mark.parametrize("call", [
    lambda: gdp_epsilon(0.0, 20, 1e-5),
    lambda: gdp_epsilon(1.0, 0, 1e-5),
    lambda: gdp_epsilon(1.0, 20, 1.0),
    lambda: gdp_required_noise_multiplier(0.0, 20, 1e-5),
    lambda: gdp_delta(-1.0, 1.0),
])
def test_invalid_inputs_are_refused(call):
    with pytest.raises(ValueError):
        call()


# --------------------------------------------------------------------------------------------------
# Opt-in use by the DP strategies and the planner
# --------------------------------------------------------------------------------------------------
def _dp_fedavg(**overrides):
    kwargs = dict(clients_per_round=8, dp_clip_norm=0.05, dp_target_epsilon=4.0, dp_delta=1e-5, dp_rounds=20)
    kwargs.update(overrides)
    return DPFedAvg(OrderedDict(w=torch.zeros(3)), **kwargs)


def test_strategies_default_to_rdp_and_calibrate_exactly_with_gdp():
    rdp, gdp = _dp_fedavg(), _dp_fedavg(dp_accountant="gdp")
    assert rdp.dp_accountant == "rdp" and gdp.dp_accountant == "gdp"
    assert rdp.dp_noise_multiplier == pytest.approx(required_noise_multiplier(4.0, 1.0, 20, 1e-5))
    assert gdp.dp_noise_multiplier == pytest.approx(gdp_required_noise_multiplier(4.0, 20, 1e-5))
    assert gdp.dp_noise_multiplier < rdp.dp_noise_multiplier
    assert gdp.dp_accounted_epsilon == pytest.approx(4.0, abs=1e-6)
    assert rdp.dp_accounted_epsilon <= 4.0 + 1e-6


def test_gdp_accounts_a_raw_noise_multiplier_exactly():
    strategy = _dp_fedavg(dp_target_epsilon=None, dp_noise_multiplier=5.796, dp_accountant="gdp")
    assert strategy.dp_accounted_epsilon == pytest.approx(gdp_epsilon(5.796, 20, 1e-5))


def test_gdp_is_refused_below_full_participation_and_unknown_accountants_are_refused():
    with pytest.raises(ValueError, match="q=1"):
        _dp_fedavg(dp_num_clients=80, dp_accountant="gdp")
    with pytest.raises(ValueError, match="dp_accountant"):
        _dp_fedavg(dp_accountant="pld")


def test_planner_reports_the_accountant_it_used():
    base = dict(target_epsilon=4.0, delta=1e-5, rounds=20, clients=80, clip_norm=0.05, trainable_parameters=1026)
    rdp, gdp = plan_central_dp(**base), plan_central_dp(**base, accountant="gdp")
    assert (rdp["accountant"], gdp["accountant"]) == ("rdp", "gdp")
    assert gdp["noise_multiplier"] == pytest.approx(gdp_required_noise_multiplier(4.0, 20, 1e-5))
    assert gdp["noise_norm_to_clip_ratio"] < rdp["noise_norm_to_clip_ratio"]
    with pytest.raises(ValueError, match="accountant"):
        plan_central_dp(**base, accountant="pld")
