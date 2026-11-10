"""Formal Differential Privacy Audit Witness and Empirical Distinguishing Test.

Demonstrates mathematically and empirically that the unprivatized adaptive median quantile
defect enables an adversary to distinguish neighboring client datasets with statistical certainty,
proving that the claimed (epsilon=4.0, delta=1e-5) privacy profile is violated by orders of magnitude.
"""
import math
from typing import Tuple

import numpy as np
import pytest
import torch

from fedlearn.server.robust_aggregation import adaptive_centered_clip, centered_clip


def std_norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gaussian_privacy_profile(shift: float, sigma: float, epsilon: float) -> float:
    """Computes exact delta(epsilon) for distinguishing N(0, sigma^2) and N(shift, sigma^2).

    Formula (Balle & Wang 2018):
        delta(eps) = Phi(shift/(2*sigma) - eps*sigma/shift) - exp(eps) * Phi(-shift/(2*sigma) - eps*sigma/shift)
    """
    if shift <= 0 or sigma <= 0:
        return 0.0
    a = float(shift)
    s = float(sigma)
    e = float(epsilon)

    term1 = a / (2.0 * s) - (e * s) / a
    term2 = -a / (2.0 * s) - (e * s) / a

    delta = float(std_norm_cdf(term1) - math.exp(e) * std_norm_cdf(term2))
    return max(0.0, delta)


def test_analytic_privacy_witness_shows_massive_delta_breach():
    """Analytic witness: shift a=0.60 vs declared sensitivity 0.05 creates massive delta breach."""
    shift = 0.60
    target_eps = 4.0
    declared_delta = 1e-5

    # Case A: Nominal z = 0.5 with false sensitivity 0.05 -> sigma = 0.025
    sigma_nominal = 0.5 * 0.05  # 0.025
    snr_nominal = shift / sigma_nominal  # 24.0 standard deviations!
    distinguish_acc_nominal = float(std_norm_cdf(snr_nominal / 2.0))  # Phi(12.0)

    # Distinguishing accuracy is practically 1.0 (separated by 24 standard deviations)
    assert snr_nominal == pytest.approx(24.0, abs=1e-5)
    assert distinguish_acc_nominal > 0.9999999999

    delta_nominal = gaussian_privacy_profile(shift, sigma_nominal, target_eps)
    # delta is practically 1.0 (total privacy collapse)
    assert delta_nominal > 0.999

    # Case B: Even if noise multiplier is raised to certified z = 6.48041 with false sensitivity 0.05
    sigma_z648 = 6.480408963711757 * 0.05  # 0.32402045
    delta_z648 = gaussian_privacy_profile(shift, sigma_z648, target_eps)

    # True delta is ~0.05317, which is >5300x larger than declared delta 1e-5!
    assert delta_z648 == pytest.approx(0.05317, abs=1e-3)
    assert delta_z648 > 5000 * declared_delta

    # Case C: Conservative clamp calibration z = 0.5 with assumed sensitivity 0.20 -> sigma = 0.100
    sigma_clamp = 0.5 * 0.20  # 0.100
    delta_clamp = gaussian_privacy_profile(shift, sigma_clamp, target_eps)
    # True delta is ~0.98348, which is >98,000x larger than declared delta 1e-5!
    assert delta_clamp == pytest.approx(0.98348, abs=1e-3)
    assert delta_clamp > 90000 * declared_delta


def test_empirical_distinguisher_achieves_perfect_accuracy_on_unnoised_adaptive_query():
    """Empirical distinguisher: 100 trials distinguishing D vs D' from observable noisy aggregates."""
    # Construct Codex counterexample datasets (N=20, d=1)
    D = torch.cat([torch.zeros(9, 1), torch.full((11, 1), 2.0)], dim=0)
    D_prime = torch.cat([torch.zeros(10, 1), torch.full((10, 1), 2.0)], dim=0)
    centre = torch.zeros(1)

    # 1. Faulty adaptive query
    agg_D, _ = adaptive_centered_clip(D, centre=centre, quantile=0.5, min_tau=0.2, max_tau=2.0)
    agg_D_prime, _ = adaptive_centered_clip(D_prime, centre=centre, quantile=0.5, min_tau=0.2, max_tau=2.0)
    shift_faulty = float(torch.abs(agg_D - agg_D_prime).item())
    assert shift_faulty == pytest.approx(0.60, abs=1e-5)

    # Calibrated noise under FALSE sensitivity 0.05
    sigma_faulty = 0.5 * 0.05  # 0.025
    rng = np.random.default_rng(20261109)

    # Draw 500 samples from D + noise and 500 from D' + noise
    samples_D = agg_D.item() + rng.normal(0.0, sigma_faulty, size=500)
    samples_D_prime = agg_D_prime.item() + rng.normal(0.0, sigma_faulty, size=500)

    # Simple threshold distinguisher at the midpoint
    threshold = (agg_D.item() + agg_D_prime.item()) / 2.0
    correct_D = np.mean(samples_D > threshold) if agg_D.item() > agg_D_prime.item() else np.mean(samples_D < threshold)
    correct_D_prime = np.mean(samples_D_prime < threshold) if agg_D.item() > agg_D_prime.item() else np.mean(samples_D_prime > threshold)

    empirical_acc = 0.5 * (correct_D + correct_D_prime)
    # The distinguisher must achieve 100% empirical accuracy
    assert empirical_acc == 1.0, f"Empirical distinguishing accuracy {empirical_acc} must be 1.0 (24-sigma separation)"

    # Even under conservative clamp sensitivity Delta_max = 2*tau_max/N = 2*2.0/20 = 0.20 (sigma = 0.10)
    sigma_clamp = 0.5 * 0.20  # 0.10
    samples_D_clamp = agg_D.item() + rng.normal(0.0, sigma_clamp, size=500)
    samples_D_prime_clamp = agg_D_prime.item() + rng.normal(0.0, sigma_clamp, size=500)
    correct_D_clamp = np.mean(samples_D_clamp > threshold) if agg_D.item() > agg_D_prime.item() else np.mean(samples_D_clamp < threshold)
    correct_D_prime_clamp = np.mean(samples_D_prime_clamp < threshold) if agg_D.item() > agg_D_prime.item() else np.mean(samples_D_prime_clamp > threshold)
    acc_clamp = 0.5 * (correct_D_clamp + correct_D_prime_clamp)
    # Theoretical Phi(3.0) is ~99.87%; empirical accuracy must exceed 99%
    assert acc_clamp >= 0.99, f"Clamp distinguishing accuracy {acc_clamp} must be >= 0.99 (6-sigma separation)"

    # 2. In contrast, under the CERTIFIED origin-centered query with true sensitivity 0.05
    agg_cert_D = centered_clip(D, centre=centre, tau=0.5, iterations=1)
    agg_cert_D_prime = centered_clip(D_prime, centre=centre, tau=0.5, iterations=1)
    shift_cert = float(torch.abs(agg_cert_D - agg_cert_D_prime).item())
    assert shift_cert == pytest.approx(0.025, abs=1e-5)
    assert shift_cert <= 0.05  # Bounded strictly within sensitivity
