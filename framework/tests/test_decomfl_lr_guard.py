"""DeComFL's learning rate is not dimension-transferable — the server should say so.

Measured in `research/results/decomfl/`:

  * `mu_eta_dimension_scaling.json` — at d=103,002 the reference eta=0.01 diverges to loss ~1e19.
    Scaling eta by sqrt(d0/d) ALONE restores clean convergence (0.9815 AUC, replay check back to
    1.49e-08); scaling mu alone does not. The learning rate is the entire cause.
  * `stability_ladder.json` — at the reference eta the boundary is sharp: stable through
    d=20,602, diverged from d=30,902. The d=30,902 cell reaches 0.9805 AUC and THEN explodes,
    so an accuracy column cannot see it coming.

Both incidents are in the record, and `ondevice_large_d_diverged_unscaled_lr.json` is filed as
"the same error, repeated after documenting it". The strategy knows d and eta at construction and
validated neither. The invariant is S = eta*sqrt(d): it is 0.320 at the production head and at the
rescued d=103,002 cell, 1.435 at the largest measured-stable point, and 1.758 at the smallest
measured-divergent one.
"""

import logging
import math
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.decomfl_strategy import (
    DeComFL,
    LR_DIVERGENT_MIN_S,
    LR_REFERENCE_D,
    LR_REFERENCE_ETA,
    LR_STABLE_MAX_S,
    lr_stability_statistic,
    suggested_eta,
)


def _strategy(d: int, eta: float, **kw) -> DeComFL:
    return DeComFL(
        initial_parameters=OrderedDict([("w", torch.zeros(d))]),
        min_fit_clients=1,
        learning_rate=eta,
        **kw,
    )


def test_stability_statistic_matches_the_measured_reference_point():
    # d=1026 at eta=0.01 is the production frozen head, measured stable.
    assert lr_stability_statistic(LR_REFERENCE_ETA, LR_REFERENCE_D) == pytest.approx(0.3203, abs=1e-4)


def test_suggested_eta_reproduces_the_measured_rescue_value():
    """`mu_eta_dimension_scaling.json` rescued d=103,002 with eta=0.0009980466738393954."""
    assert suggested_eta(103002) == pytest.approx(0.0009980466738393954, rel=1e-9)


def test_suggested_eta_holds_the_stability_statistic_invariant():
    for d in (1026, 10302, 51502, 103002, 1_600_000):
        assert lr_stability_statistic(suggested_eta(d), d) == pytest.approx(0.3203, abs=1e-4)


def test_production_config_is_accepted_silently(caplog):
    with caplog.at_level(logging.WARNING, logger="fedlearn.server.decomfl_strategy"):
        _strategy(LR_REFERENCE_D, LR_REFERENCE_ETA)
    assert not [r for r in caplog.records if "eta" in r.getMessage().lower()], (
        "the measured-stable production config must not warn"
    )


def test_largest_measured_stable_point_is_accepted(caplog):
    """d=20,602 at eta=0.01 converged (stability_ladder.json) — it must not be rejected."""
    with caplog.at_level(logging.WARNING, logger="fedlearn.server.decomfl_strategy"):
        _strategy(20602, 0.01)


def test_warns_in_the_unmeasured_gray_band(caplog):
    d = 25000  # S = 1.581, between the stable max and the divergent min
    s = lr_stability_statistic(0.01, d)
    assert LR_STABLE_MAX_S < s < LR_DIVERGENT_MIN_S, "test fixture must sit inside the gray band"
    with caplog.at_level(logging.WARNING, logger="fedlearn.server.decomfl_strategy"):
        _strategy(d, 0.01)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("eta" in m and "suggest" in m.lower() for m in msgs), (
        f"gray-band config should warn and suggest a learning rate; got {msgs}"
    )


def test_raises_at_the_smallest_measured_divergent_point():
    """d=30,902 at eta=0.01 reached 0.9805 AUC and then exploded to loss 9.2e18."""
    with pytest.raises(ValueError) as e:
        _strategy(30902, 0.01)
    msg = str(e.value)
    assert "0.01" in msg or "eta" in msg.lower()
    assert f"{suggested_eta(30902):.3g}" in msg, "the error must name the learning rate to use"


def test_raises_at_the_recorded_repeat_failure():
    """`ondevice_large_d_diverged_unscaled_lr.json`: NaN at d=1.6M with the d=1026 learning rate."""
    with pytest.raises(ValueError):
        _strategy(1_600_000, 0.01)


def test_escape_hatch_permits_an_unstable_rate_but_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="fedlearn.server.decomfl_strategy"):
        strat = _strategy(30902, 0.01, allow_unstable_lr=True)
    assert strat.eta == 0.01
    assert [r for r in caplog.records if "eta" in r.getMessage().lower()], (
        "an explicit override should still leave a warning in the log"
    )


def test_scaled_rate_is_accepted_at_a_dimension_that_would_otherwise_be_rejected(caplog):
    with caplog.at_level(logging.WARNING, logger="fedlearn.server.decomfl_strategy"):
        strat = _strategy(103002, suggested_eta(103002))
    assert strat.eta == pytest.approx(0.0009980466738393954, rel=1e-9)
    assert not [r for r in caplog.records if "suggest" in r.getMessage().lower()]
