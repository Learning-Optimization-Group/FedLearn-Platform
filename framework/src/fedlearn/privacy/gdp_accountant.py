"""Exact (ε, δ) for full-participation Gaussian releases, via Gaussian differential privacy.

With every client in every round (q = 1), each round's release is a Gaussian mechanism with noise
multiplier z, i.e. (1/z)-GDP. T of them compose exactly to μ-GDP with μ = √T / z (Dong, Roth, Su,
"Gaussian Differential Privacy", arXiv:1905.02383, Corollary 3.3), and μ-GDP is equivalent to the
family of (ε, δ(ε)) guarantees

    δ(ε) = Φ(−ε/μ + μ/2) − e^ε · Φ(−ε/μ − μ/2)

(Corollary 2.13; the analytic Gaussian mechanism of Balle and Wang, 2018). Nothing is bounded or
approximated, so this gives the smallest ε the mechanism actually satisfies. The RDP accountant
(:mod:`fedlearn.privacy.dp_accountant`) with the classic conversion is an upper bound on it: at
T = 20 and δ = 10⁻⁵, the z that RDP calibrates for ε 8/4/1 is exactly ε 6.76/3.25/0.74.

This applies ONLY to q = 1. Subsampled rounds do not compose to an exact Gaussian, so callers with
q < 1 must stay on the RDP accountant. Like the RDP accountant, it takes z as noise std divided by
L2 sensitivity under whichever neighbour relation the mechanism is calibrated for (see
``dp_mechanism.ADJACENCY_SENSITIVITY_FACTOR``).

Public API:
    gdp_mu(noise_multiplier, steps) -> float
    gdp_delta(epsilon, mu) -> float
    gdp_epsilon(noise_multiplier, steps, delta) -> float
    gdp_required_noise_multiplier(target_epsilon, steps, delta) -> float
"""

from __future__ import annotations

import math

from fedlearn.privacy.dp_accountant import _log_erfc

__all__ = ["gdp_delta", "gdp_epsilon", "gdp_mu", "gdp_required_noise_multiplier"]

_LOG_HALF = math.log(0.5)


def _log_phi(x: float) -> float:
    """log Φ(x) for the standard normal CDF, stable far into the lower tail."""
    return _LOG_HALF + _log_erfc(-x / math.sqrt(2.0))


def _check_steps_delta(steps: int, delta: float) -> None:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps!r}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")


def gdp_mu(noise_multiplier: float, steps: int) -> float:
    """μ of ``steps`` composed Gaussian releases with noise multiplier z: √steps / z."""
    if not noise_multiplier > 0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps!r}")
    return math.sqrt(steps) / noise_multiplier


def gdp_delta(epsilon: float, mu: float) -> float:
    """The δ at which μ-GDP gives ε; decreasing in ε, increasing in μ."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon!r}")
    if not mu > 0:
        raise ValueError(f"mu must be > 0, got {mu!r}")
    log_first = _log_phi(-epsilon / mu + mu / 2.0)
    log_second = epsilon + _log_phi(-epsilon / mu - mu / 2.0)
    if log_second >= log_first:  # equal up to rounding: the two terms cancel
        return 0.0
    return math.exp(log_first) * -math.expm1(log_second - log_first)


def gdp_epsilon(noise_multiplier: float, steps: int, delta: float, *, tol: float = 1e-10) -> float:
    """Smallest ε with δ(ε) <= ``delta`` for ``steps`` full-participation releases at multiplier z."""
    _check_steps_delta(steps, delta)
    mu = gdp_mu(noise_multiplier, steps)
    if gdp_delta(0.0, mu) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while gdp_delta(hi, mu) > delta:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            raise ValueError(f"epsilon exceeds 1e6 for z={noise_multiplier}, steps={steps}, delta={delta}")
    while hi - lo > tol * max(1.0, hi):
        mid = (lo + hi) / 2.0
        if gdp_delta(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return hi  # the conservative side: δ(hi) <= delta


def gdp_required_noise_multiplier(target_epsilon: float, steps: int, delta: float, *,
                                  tol: float = 1e-12) -> float:
    """Smallest z whose ``steps`` full-participation releases satisfy (target_epsilon, delta)."""
    if not target_epsilon > 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon!r}")
    _check_steps_delta(steps, delta)
    lo, hi = 1e-8, 1.0  # bracket μ: δ(target; lo) <= delta < δ(target; hi)
    while gdp_delta(target_epsilon, hi) <= delta:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            raise ValueError(f"target_epsilon={target_epsilon} needs no noise at steps={steps}, delta={delta}")
    while hi - lo > tol * hi:
        mid = (lo + hi) / 2.0
        if gdp_delta(target_epsilon, mid) <= delta:
            lo = mid
        else:
            hi = mid
    return math.sqrt(steps) / lo  # the largest feasible μ, so the smallest z that meets the target
