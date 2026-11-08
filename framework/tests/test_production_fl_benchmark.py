"""Unit tests for the Production Federated Learning Benchmark."""
from __future__ import annotations

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
