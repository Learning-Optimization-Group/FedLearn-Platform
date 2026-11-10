"""Unit tests for the Production Federated Learning Benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "framework" / "src"))
sys.path.insert(0, str(REPO_ROOT / "research" / "benchmarks"))

from run_production_fl_benchmark import (
    build_orthonormal_projection,
    LinearClassifier,
    evaluate_model,
    run_production_experiment,
)


def test_build_orthonormal_projection_properties():
    """Verifies that the generated projection matrix is strictly orthonormal."""
    in_dim, out_dim = 512, 128
    p = build_orthonormal_projection(in_dim=in_dim, out_dim=out_dim, seed=20261107)
    assert p.shape == (in_dim, out_dim)

    # P^T @ P should be identity of size out_dim
    ptp = p.T @ p
    eye = torch.eye(out_dim)
    diff = torch.norm(ptp - eye, p="fro").item()
    assert diff < 1e-5


def test_linear_classifier_flat_param_roundtrip():
    """Verifies parameter serialization and restoration in LinearClassifier."""
    model = LinearClassifier(in_dim=64, num_classes=5)
    flat = model.get_flat_params()
    assert flat.numel() == 64 * 5 + 5

    new_flat = torch.randn_like(flat)
    model.set_flat_params(new_flat)
    recovered = model.get_flat_params()
    assert torch.equal(new_flat, recovered)


def test_production_experiment_smoke():
    """Runs a 2-round deterministic smoke test verifying FedAdam and policy execution."""
    torch.manual_seed(42)
    n_samples = 200
    n_test = 50
    dim = 64
    num_classes = 5

    # Synthetic features with class separation
    train_x = torch.randn(n_samples, dim)
    train_y = torch.randint(0, num_classes, (n_samples,))
    for c in range(num_classes):
        mask = train_y == c
        train_x[mask, c * 10:(c + 1) * 10] += 3.0

    test_x = torch.randn(n_test, dim)
    test_y = torch.randint(0, num_classes, (n_test,))
    for c in range(num_classes):
        mask = test_y == c
        test_x[mask, c * 10:(c + 1) * 10] += 3.0

    partitions = [np.array_split(np.arange(n_samples), 4)[i] for i in range(4)]

    res = run_production_experiment(
        policy="Mean512",
        private=False,
        attacked=False,
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        partitions=partitions,
        num_rounds=2,
        clip_c=2.0,
        seed=42,
    )

    assert res["final_accuracy"] > 0.40
    assert len(res["round_history"]) == 1  # evaluated at round 2
    assert "final_loss" in res


def test_adaptive_centered_clip_bounds_outliers():
    """Verifies that adaptive_centered_clip scales radius to honest dispersion and suppresses outliers."""
    from fedlearn.server.robust_aggregation import adaptive_centered_clip

    torch.manual_seed(42)
    # 8 honest clients with small updates (norm ~1.0)
    honest = torch.randn(8, 20) * 0.2
    # 2 Byzantine attackers with huge malicious updates (norm ~50.0)
    byzantine = torch.ones(2, 20) * 10.0

    stacked = torch.cat([honest, byzantine], dim=0)
    centre = torch.zeros(20)

    agg, tau = adaptive_centered_clip(stacked, centre=centre, quantile=0.5, min_tau=0.1)

    # Applied tau should adapt to honest cluster (median distance ~0.8-1.5, not 50.0)
    assert 0.1 <= tau <= 2.5
    # The aggregate norm should be bounded tightly by tau
    assert torch.norm(agg).item() < 3.0


def test_production_experiment_private_all_policies_json_serializable():
    """Verify that all policies in private mode return strictly JSON serializable results."""
    torch.manual_seed(42)
    n_samples, n_test, dim, num_classes = 100, 30, 32, 3
    train_x = torch.randn(n_samples, dim)
    train_y = torch.randint(0, num_classes, (n_samples,))
    test_x = torch.randn(n_test, dim)
    test_y = torch.randint(0, num_classes, (n_test,))
    partitions = [np.array_split(np.arange(n_samples), 4)[i] for i in range(4)]
    proj = build_orthonormal_projection(dim, 16)

    policies = ["Mean512", "Projected128", "Trim16", "CenterQuarter"]
    for pol in policies:
        res = run_production_experiment(
            policy=pol,
            private=True,
            attacked=False,
            train_features=train_x,
            train_labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            partitions=partitions,
            proj_matrix=proj,
            num_rounds=2,
            clip_c=2.0,
            target_epsilon=4.0,
            target_delta=1e-5,
            seed=42,
        )

        # Must have committed numeric accountant traces
        assert isinstance(res["accounted_epsilon"], float) and res["accounted_epsilon"] > 0
        assert res["delta"] == 1e-5
        assert isinstance(res["noise_multiplier_z"], float) and res["noise_multiplier_z"] > 0
        assert isinstance(res["sensitivity"], float) and res["sensitivity"] > 0
        assert isinstance(res["final_accuracy"], float)

        # Crucial test: must serialize to JSON without throwing TypeError
        serialized = json.dumps(res)
        assert len(serialized) > 100
        reloaded = json.loads(serialized)
        assert reloaded["policy"] == pol
        assert reloaded["delta"] == 1e-5


def test_production_benchmark_centerquarter_satisfies_certified_sensitivity():
    """Verify that CenterQuarter uses certified centered clipping and satisfies Delta bound."""
    from fedlearn.server.robust_aggregation import centered_clip

    # 4 clients with updates
    D = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
    # Neighbor differing in 1 client
    D_prime = torch.tensor([[0.0], [0.0], [0.0], [2.0]])

    tau = 0.5  # clip_c / 4.0
    certified_sens = (2.0 * tau) / 4.0  # 0.25

    agg_D = centered_clip(D, centre=torch.zeros(1), tau=tau, iterations=1)
    agg_D_prime = centered_clip(D_prime, centre=torch.zeros(1), tau=tau, iterations=1)

    shift = float(torch.abs(agg_D - agg_D_prime).item())
    assert shift <= certified_sens + 1e-7, f"Shift {shift} must be <= certified sensitivity {certified_sens}"
    assert shift == pytest.approx(0.125, abs=1e-5)


def test_production_experiment_center_mode_and_accounting_integration():
    """Integration test verifying center norm tracking, trimming on N=20, and dual regimes."""
    torch.manual_seed(42)
    n_samples, n_test, dim, num_classes = 200, 40, 16, 2
    train_x = torch.randn(n_samples, dim)
    train_y = torch.randint(0, num_classes, (n_samples,))
    test_x = torch.randn(n_test, dim)
    test_y = torch.randint(0, num_classes, (n_test,))
    # 20 clients
    partitions = [np.array_split(np.arange(n_samples), 20)[i] for i in range(20)]

    # 1. Test CenterQuarter_Origin (origin mode): all center norms must be exactly 0.0
    res_origin = run_production_experiment(
        policy="CenterQuarter_Origin",
        private=True,
        attacked=False,
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        partitions=partitions,
        num_rounds=3,
        clip_c=2.0,
        target_epsilon=4.0,
        target_delta=1e-5,
        seed=42,
    )
    assert res_origin["center_mode"] == "origin"
    assert len(res_origin["center_norms_per_round"]) == 3
    assert all(cn == 0.0 for cn in res_origin["center_norms_per_round"])
    assert res_origin["sensitivity"] == pytest.approx((2.0 * 0.5) / 20.0, rel=1e-6)
    assert res_origin["accounted_epsilon"] <= 4.0001
    assert res_origin["noise_sigma"] == pytest.approx(
        res_origin["noise_multiplier_z"] * res_origin["sensitivity"], rel=1e-6
    )
    # Strict JSON without NaN
    json.dumps(res_origin, allow_nan=False)

    # 2. Test CenterQuarter_History: rounds >= 2 must have non-zero center norms if aggregate is non-zero
    res_history = run_production_experiment(
        policy="CenterQuarter_History",
        private=False,
        attacked=False,
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        partitions=partitions,
        num_rounds=3,
        clip_c=2.0,
        seed=42,
    )
    assert res_history["center_mode"] == "history"
    assert res_history["center_norms_per_round"][0] == 0.0
    assert res_history["center_norms_per_round"][1] > 0.0

    # 3. Test EqualRadiusMean: sensitivity and center mode match origin
    res_eq = run_production_experiment(
        policy="EqualRadiusMean",
        private=True,
        attacked=False,
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        partitions=partitions,
        num_rounds=2,
        clip_c=2.0,
        target_epsilon=4.0,
        target_delta=1e-5,
        seed=42,
    )
    assert res_eq["center_mode"] == "origin"
    assert res_eq["sensitivity"] == res_origin["sensitivity"]

    # 4. Test Trim16 with N=20: trims 2*4=8 clients, surviving=12, sensitivity = 2*2.0/12 = 1/3
    res_trim = run_production_experiment(
        policy="Trim16",
        private=True,
        attacked=False,
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        partitions=partitions,
        num_rounds=2,
        clip_c=2.0,
        z_noise=0.5,
        seed=42,
    )
    assert res_trim["sensitivity"] == pytest.approx(2.0 * 2.0 / 12.0, rel=1e-6)
    assert res_trim["noise_sigma"] == pytest.approx(0.5 * (4.0 / 12.0), rel=1e-6)


