"""Smoke test for the FR-13/DA-14 `dp_on_head` benchmark — does a SMALL trainable head escape the
high-dimension central-DP collapse that FedLoRA (d=26112) suffers?

Fast, seeded, and deterministic: a tiny frozen-backbone head-only FedAvg run with the REAL DP
mechanism (`fedlearn.privacy.dp_mechanism.dp_aggregate`) and the REAL RDP accountant
(`fedlearn.privacy.dp_accountant`). The three assertions the DA-14 task pins:

1. the accountant-solved noise multiplier z is finite and positive;
2. the no-DP control accuracy exceeds the tightest-ε accuracy OR the SNR ordering is correct
   (tighter ε -> more noise -> lower utility SNR);
3. the head dimension d (trainable scalars) is SMALL — well under 10x the FedLoRA d=26112 baseline.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from benchmarks.dp_on_head import (  # noqa: E402
    FEDLORA_REFERENCE_D,
    run_sweep,
    solve_noise_multiplier,
)


def test_accountant_solves_finite_positive_z():
    # (1) The from-scratch RDP accountant must return a finite, positive z for a real target budget.
    z = solve_noise_multiplier(target_epsilon=1.0, q=1.0, rounds=3, delta=1e-5)
    assert math.isfinite(z)
    assert z > 0.0


def test_head_escapes_or_snr_ordered_and_d_is_small():
    # Two budgets + the implicit no-DP control, at a tiny/fast/seeded scale.
    out = run_sweep(
        epsilons=[8.0, 1.0], rounds=3, clients=8,
        d_in=128, d_hidden=16, n_classes=3, clip=0.4, delta=1e-5,
        seed=1234, dp_seed=777,
    )
    results = out["results"]
    ctrl = next(r for r in results if r["target_epsilon"] is None)
    loose = next(r for r in results if r["target_epsilon"] == 8.0)
    tight = next(r for r in results if r["target_epsilon"] == 1.0)

    # (3) Head d is small: identical across configs and far below 10x the FedLoRA collapse baseline.
    d = tight["aggregatable_coords_d"]
    assert d == ctrl["aggregatable_coords_d"] == loose["aggregatable_coords_d"]
    assert 0 < d < 10 * FEDLORA_REFERENCE_D

    # (1) again at the sweep level: solved z + its SNR are finite and positive.
    assert math.isfinite(tight["noise_multiplier_z"]) and tight["noise_multiplier_z"] > 0.0
    assert math.isfinite(tight["utility_snr"]) and tight["utility_snr"] > 0.0
    # The accountant round-trips: the accounted ε certifies back to (about) the requested budget.
    assert tight["accounted_epsilon"] == pytest.approx(1.0, abs=1e-3)

    # (2) no-DP control accuracy > tightest-ε accuracy  OR  the SNR ordering is correct.
    escapes_by_accuracy = ctrl["final_accuracy"] > tight["final_accuracy"]
    snr_ordered = tight["utility_snr"] < loose["utility_snr"]  # tighter ε -> lower SNR
    assert escapes_by_accuracy or snr_ordered

    # Frozen-backbone invariants must survive the DP path: the backbone never rode the wire and the
    # DP mechanism only touched the head keys.
    assert tight["backbone_unchanged"] is True
    assert tight["wire_is_head_only"] is True


def test_sweep_is_deterministic():
    kw = dict(epsilons=[1.0], rounds=3, clients=8, d_in=128, d_hidden=16, n_classes=3,
              clip=0.4, delta=1e-5, seed=1234, dp_seed=777)
    a = run_sweep(**kw)["results"]
    b = run_sweep(**kw)["results"]
    assert [r["per_round_accuracy"] for r in a] == [r["per_round_accuracy"] for r in b]
    assert [r["noise_multiplier_z"] for r in a] == [r["noise_multiplier_z"] for r in b]
