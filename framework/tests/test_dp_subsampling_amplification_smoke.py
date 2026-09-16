"""Smoke test for the C1/DA-14 `dp_subsampling_amplification` benchmark — Poisson client
subsampling (q<1) AMPLIFIES privacy on the small federated head.

Fast, seeded, deterministic. Everything privacy-related routes through the REAL from-scratch RDP
accountant (`fedlearn.privacy.dp_accountant`) and the REAL central-DP mechanism
(`fedlearn.privacy.dp_mechanism.dp_aggregate`); only the utility target is synthetic. The
assertions the task pins:

1. at a FIXED noise multiplier z, the accountant's certified ε is STRICTLY DECREASING as q shrinks
   (subsampling amplification — a smaller sampling rate certifies tighter privacy);
2. at a FIXED target ε, the accountant-solved noise multiplier z is STRICTLY DECREASING as q shrinks
   (the dual — subsampling lets you inject LESS noise for the same budget);
3. the q=1 (full-participation) row matches the `dp_on_head` full-participation baseline exactly.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from benchmarks.dp_on_head import accounted_epsilon, solve_noise_multiplier  # noqa: E402
from benchmarks.dp_subsampling_amplification import (  # noqa: E402
    certified_epsilon,
    run_sweep,
    solved_noise_multiplier,
)

_QS = [1.0, 0.5, 0.25, 0.1]


def test_certified_epsilon_strictly_decreases_as_q_shrinks_at_fixed_z():
    # (1) Fixed z, sweep q downward: the accountant's subsampled RDP certifies a tighter ε as q falls.
    z, rounds, delta = 2.0, 8, 1e-5
    eps = [certified_epsilon(q, z, rounds, delta) for q in _QS]
    assert all(math.isfinite(e) and e > 0.0 for e in eps)
    assert all(eps[i] > eps[i + 1] for i in range(len(eps) - 1)), eps


def test_solved_z_strictly_decreases_as_q_shrinks_at_fixed_epsilon():
    # (2) Fixed target ε, sweep q downward: the required noise multiplier shrinks (less noise needed).
    target, rounds, delta = 4.0, 8, 1e-5
    zs = [solved_noise_multiplier(target, q, rounds, delta) for q in _QS]
    assert all(math.isfinite(zz) and zz > 0.0 for zz in zs)
    assert all(zs[i] > zs[i + 1] for i in range(len(zs) - 1)), zs
    # The dual round-trips: certified ε at the solved z lands (about) back on the target budget.
    for q, zz in zip(_QS, zs):
        assert certified_epsilon(q, zz, rounds, delta) == pytest.approx(target, abs=1e-3)


def test_q1_matches_full_participation_baseline():
    # (3) At q=1 both levers reduce to the non-subsampled Gaussian — i.e. dp_on_head's q=1 baseline.
    z, target, rounds, delta = 2.0, 4.0, 8, 1e-5
    assert certified_epsilon(1.0, z, rounds, delta) == accounted_epsilon(1.0, z, rounds, delta)
    assert solved_noise_multiplier(target, 1.0, rounds, delta) == solve_noise_multiplier(
        target, 1.0, rounds, delta
    )


def test_run_sweep_reports_both_levers_and_amplification_factor():
    out = run_sweep(
        q_values=_QS, fixed_z=2.0, target_epsilon=4.0, rounds=8, clients=16,
        d_in=128, d_hidden=16, n_classes=3, clip=0.4, delta=1e-5, seed=1234,
        dp_seed=777, sample_seed=99, with_utility=False,
    )
    res = out["results"]
    assert [r["sampling_rate_q"] for r in res] == _QS

    ce = [r["certified_epsilon_fixed_z"] for r in res]
    zz = [r["solved_z_fixed_epsilon"] for r in res]
    assert all(ce[i] > ce[i + 1] for i in range(len(ce) - 1)), ce   # amplification
    assert all(zz[i] > zz[i + 1] for i in range(len(zz) - 1)), zz   # dual

    # The q=1 row is exactly the dp_on_head full-participation baseline.
    assert res[0]["certified_epsilon_fixed_z"] == accounted_epsilon(1.0, 2.0, 8, 1e-5)
    assert res[0]["solved_z_fixed_epsilon"] == solve_noise_multiplier(4.0, 1.0, 8, 1e-5)

    # Headline factors are reported and real (> 1).
    assert out["meta"]["amplification_factor"] > 1.0
    assert out["meta"]["noise_reduction_factor"] > 1.0
    # Each dual solved-z round-trips to the target budget.
    for r in res:
        assert r["dual_accounted_epsilon"] == pytest.approx(4.0, abs=1e-3)


def test_utility_run_q1_is_full_participation_and_deterministic():
    kw = dict(
        q_values=[1.0, 0.25], fixed_z=2.0, target_epsilon=4.0, rounds=4, clients=16,
        d_in=128, d_hidden=16, n_classes=3, clip=0.4, delta=1e-5, seed=1234,
        dp_seed=777, sample_seed=99, with_utility=True,
    )
    a = run_sweep(**kw)["results"]
    b = run_sweep(**kw)["results"]
    # Deterministic under fixed seeds (data, backbone, Poisson mask, and DP noise all seeded).
    assert [r["final_accuracy"] for r in a] == [r["final_accuracy"] for r in b]
    assert [r["mean_participants"] for r in a] == [r["mean_participants"] for r in b]

    q1 = next(r for r in a if r["sampling_rate_q"] == 1.0)
    # q=1 => every enrolled client participates every round (the full-participation utility baseline).
    assert q1["mean_participants"] == 16.0
    # Frozen-backbone invariants survive the subsampled DP path.
    assert q1["backbone_unchanged"] is True
    assert q1["wire_is_head_only"] is True

    q_sub = next(r for r in a if r["sampling_rate_q"] == 0.25)
    # Subsampling really does drop participants below the full cohort.
    assert 0.0 < q_sub["mean_participants"] < 16.0
