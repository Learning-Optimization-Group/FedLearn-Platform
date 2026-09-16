"""Smoke test for the FR-13/DA-14 `dp_head_cohort_sweep` benchmark — does the DP utility-SNR
crossing (~1) land at a MODERATE, reachable cohort N on the small frozen-backbone HEAD?

This is the complement of `dp_on_head` (which FIXED the cohort N and swept ε): here ε and d are
fixed and the cohort size N is swept. Both reuse the REAL DP mechanism
(`fedlearn.privacy.dp_mechanism.dp_aggregate`) and the from-scratch RDP accountant
(`fedlearn.privacy.dp_accountant`). The task's pinned assertions, at a fast/seeded/tiny scale:

1. the utility SNR = N/(z·√d) is STRICTLY INCREASING in N at the fixed (z-for-ε, d) — and, at q=1,
   the accountant-solved z is CONSTANT across the sweep (it depends only on ε, rounds, δ, not N);
2. final utility is NON-DECREASING (within noise) as N grows — the head recovers accuracy as the
   cohort averages the DP noise down;
3. the SNR=1 crossing N (smallest N with SNR≥1) is FINITE and MODEST — far below the N the
   FedLoRA d=26112 baseline would need at the same z (its crossing is z·√26112).
"""
import math

import pytest

torch = pytest.importorskip("torch")

from benchmarks.dp_head_cohort_sweep import (  # noqa: E402
    FEDLORA_REFERENCE_D,
    crossing_cohort,
    run_cohort_sweep,
)

# A fast, tiny, seeded config whose crossing brackets the swept range (see the benchmark's own
# sizing: at ε=8, rounds=3, d=51 the SNR=1 crossing is N≈8.5, so N∈{4,8,16,24} straddles it).
_SMOKE = dict(
    target_epsilon=8.0, n_values=(4, 8, 16, 24), rounds=3,
    d_in=128, d_hidden=16, n_classes=3, clip=0.4, delta=1e-5,
    lr=0.5, local_epochs=5, sep=2.0, seed=1234, dp_seed=777,
)


def test_snr_strictly_increasing_and_z_constant():
    out = run_cohort_sweep(**_SMOKE)
    rows = out["results"]
    ns = [r["n"] for r in rows]
    assert ns == sorted(ns)  # swept in increasing N

    # (1) SNR = N/(z·√d) strictly increases in N.
    snrs = [r["utility_snr"] for r in rows]
    assert all(a < b for a, b in zip(snrs, snrs[1:])), snrs

    # At q=1 the accountant's solved z is independent of N — it must be identical across the sweep,
    # and d is identical (same head), so the SNR denominator z·√d is a fixed constant.
    zs = [r["noise_multiplier_z"] for r in rows]
    assert all(math.isfinite(z) and z > 0.0 for z in zs)
    assert len(set(round(z, 12) for z in zs)) == 1, zs
    assert len({r["aggregatable_coords_d"] for r in rows}) == 1

    # The accountant round-trips: the certified ε returns (about) the requested target.
    assert out["meta"]["accounted_epsilon"] == pytest.approx(_SMOKE["target_epsilon"], abs=1e-3)


def test_utility_non_decreasing_as_cohort_grows():
    out = run_cohort_sweep(**_SMOKE)
    rows = out["results"]

    # (2) Utility recovers as N grows: the largest cohort's (noise-smoothed) accuracy strictly
    # exceeds the smallest cohort's, and the sequence never drops materially below the running max.
    accs = [r["dp_last3_avg_accuracy"] for r in rows]
    assert accs[-1] > accs[0], accs
    running = accs[0]
    for a in accs:
        running = max(running, a)
        assert a >= running - 0.06, accs  # non-decreasing within single-seed round noise

    # Frozen-backbone invariants survive the DP path on every cohort.
    assert all(r["backbone_unchanged"] for r in rows)
    assert all(r["wire_is_head_only"] for r in rows)


def test_crossing_is_finite_and_modest():
    out = run_cohort_sweep(**_SMOKE)
    meta = out["meta"]

    # (3) The SNR=1 crossing N is finite, within the swept range, and modest.
    crossing = meta["crossing_n"]
    assert crossing is not None
    assert crossing in _SMOKE["n_values"]
    assert crossing <= max(_SMOKE["n_values"])

    # It is FAR below the cohort FedLoRA's d=26112 would need at the same z (q=1) — the whole point.
    fedlora_crossing = meta["fedlora_predicted_crossing_n"]
    assert crossing < fedlora_crossing
    assert crossing < FEDLORA_REFERENCE_D

    # The helper agrees with the reported crossing.
    assert crossing_cohort(out["results"]) == crossing


def test_sweep_is_deterministic():
    a = run_cohort_sweep(**_SMOKE)["results"]
    b = run_cohort_sweep(**_SMOKE)["results"]
    assert [r["dp_per_round_accuracy"] for r in a] == [r["dp_per_round_accuracy"] for r in b]
    assert [r["noise_multiplier_z"] for r in a] == [r["noise_multiplier_z"] for r in b]
    assert [r["utility_snr"] for r in a] == [r["utility_snr"] for r in b]
