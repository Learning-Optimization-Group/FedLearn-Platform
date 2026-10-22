# framework/tests/test_privacy_audit.py
"""One-run canary audit: the estimator recovers a known Gaussian separation."""
import math

import numpy as np
import pytest

from fedlearn.privacy.audit import canary_audit, gdp_epsilon_for_mu
from fedlearn.privacy.gdp_accountant import gdp_epsilon


def test_recovers_a_known_mu_from_gaussian_scores():
    rng = np.random.default_rng(1)
    out = canary_audit(rng.normal(1.5, 2.0, 4000), rng.normal(0.0, 2.0, 20000), 1e-5)
    assert out["mu"] == pytest.approx(0.75, abs=0.05)
    assert out["mu_interval95"][0] < 0.75 < out["mu_interval95"][1]


def test_epsilon_matches_the_accountant_for_the_same_mu():
    # mu = sqrt(T)/z, so the epsilon for mu must equal gdp_epsilon at that z and T.
    assert gdp_epsilon_for_mu(math.sqrt(20) / 4.835, 1e-5) == pytest.approx(gdp_epsilon(4.835, 20, 1e-5), rel=1e-8)


def test_no_separation_gives_zero_epsilon():
    assert gdp_epsilon_for_mu(-0.3, 1e-5) == 0.0
    rng = np.random.default_rng(2)
    out = canary_audit(rng.normal(0, 1, 3000), rng.normal(0, 1, 3000), 1e-5)
    assert out["epsilon_interval95"][0] == 0.0


@pytest.mark.parametrize("a, b", [([1.0], [0.0, 1.0]), ([1.0, 2.0], [0.0, float("nan")])])
def test_bad_inputs_are_refused(a, b):
    with pytest.raises(ValueError):
        canary_audit(a, b, 1e-5)
