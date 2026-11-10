"""Unit, mathematical, and algorithmic tests for Intrinsic Subspace Differential Privacy."""
import math
from collections import OrderedDict

import pytest
import torch

from fedlearn.privacy.subspace import SubspaceProjector, build_orthonormal_basis
from fedlearn.privacy.subspace_dp import subspace_dp_aggregate
from fedlearn.server.composed_strategy import ComposedStrategy
from fedlearn.server.strategy import DPFedAvg, FedLoRA


def test_build_orthonormal_basis_properties():
    d, k = 1024, 64
    P = build_orthonormal_basis(d, k, seed=42)
    assert P.shape == (d, k)

    # Check orthonormality: P^T P == I_k
    PtP = P.T @ P
    eye = torch.eye(k)
    diff = torch.norm(PtP - eye).item()
    assert diff < 1e-4

    # Check reproducibility
    P_same = build_orthonormal_basis(d, k, seed=42)
    assert torch.equal(P, P_same)

    P_diff = build_orthonormal_basis(d, k, seed=43)
    assert not torch.equal(P, P_diff)

    # Check invalid dimensions
    with pytest.raises(ValueError, match="cannot exceed ambient"):
        build_orthonormal_basis(64, 128, seed=42)


def test_subspace_projector_projection_and_reconstruction():
    d, k = 512, 32
    projector = SubspaceProjector(in_dim=d, subspace_dim=k, base_seed=123)

    # Contraction property: ||P^T x||_2 <= ||x||_2
    x = torch.randn(d)
    z = projector.project_vector(x)
    assert z.shape == (k,)
    assert torch.norm(z) <= torch.norm(x) + 1e-5

    # Dict interface test
    template = OrderedDict(
        w1=torch.randn(16, 16),  # 256
        w2=torch.randn(16, 16),  # 256 -> total 512
    )
    keys = ["w1", "w2"]
    z_dict = projector.project_dict(template, keys)
    assert z_dict.shape == (k,)

    recon_dict = projector.reconstruct_dict(z_dict, template, keys)
    assert set(recon_dict.keys()) == set(keys)
    assert recon_dict["w1"].shape == template["w1"].shape
    assert recon_dict["w2"].shape == template["w2"].shape


def test_subspace_snr_amplification_and_noise_reduction():
    """Verify that subspace noise energy is reduced by k / d compared to ambient noise."""
    d, k = 1000, 100  # d / k = 10x noise energy reduction, sqrt(10) ~= 3.16x SNR boost
    projector = SubspaceProjector(in_dim=d, subspace_dim=k, base_seed=99)
    P = projector.get_basis(0)

    # Sample N noise vectors in R^k and project to R^d: P * xi
    torch.manual_seed(42)
    trials = 2000
    sigma = 0.5
    xi_k = torch.randn(trials, k) * sigma
    recon_noise = xi_k @ P.T  # [trials, d]

    # Sample direct noise in R^d
    xi_d = torch.randn(trials, d) * sigma

    # Expected Euclidean norm squared
    var_subspace = torch.mean(torch.sum(recon_noise ** 2, dim=1)).item()
    var_ambient = torch.mean(torch.sum(xi_d ** 2, dim=1)).item()

    # Theoretical values: var_subspace = k * sigma^2 = 100 * 0.25 = 25.0
    # var_ambient = d * sigma^2 = 1000 * 0.25 = 250.0
    ratio = var_subspace / var_ambient
    assert ratio == pytest.approx(0.10, rel=0.10)
    assert projector.snr_amplification == pytest.approx(math.sqrt(10.0))


def test_subspace_dp_aggregate_executes_cleanly():
    d1, d2 = 64, 64  # total 128
    k = 16
    glob = OrderedDict(w=torch.zeros(d1, d2))
    keys = ["w"]

    # 4 clients
    results = [
        ("c0", OrderedDict(w=torch.ones(d1, d2) * 0.1), 10),
        ("c1", OrderedDict(w=torch.ones(d1, d2) * -0.1), 10),
        ("c2", OrderedDict(w=torch.ones(d1, d2) * 0.05), 10),
        ("c3", OrderedDict(w=torch.ones(d1, d2) * -0.05), 10),
    ]

    out = subspace_dp_aggregate(
        results=results,
        global_params=glob,
        aggregatable_keys=keys,
        clip_norm=0.5,
        noise_multiplier=1.0,
        subspace_dim=k,
        server_round=1,
    )

    assert "w" in out
    assert out["w"].shape == (d1, d2)
    assert torch.isfinite(out["w"]).all()


def test_fedlora_with_subspace_dp():
    # Setup miniature FFA_LORA adapter
    initial = OrderedDict(
        lora_A=torch.randn(4, 16),       # Frozen A
        lora_B=torch.zeros(16, 4),       # Trainable B (64 params)
        head=torch.zeros(16),            # Trainable head (16 params) -> total aggregatable = 80
    )
    strategy = FedLoRA(
        initial_parameters=initial,
        min_fit_clients=2,
        subspace_dim=16,  # 80 -> 16 subspace
        dp_enabled=True,
        dp_clip_norm=0.5,
        dp_noise_multiplier=1.0,
        dp_seed=123,
    )

    results = [
        ("c0", OrderedDict(lora_B=torch.ones(16, 4) * 0.1, head=torch.ones(16) * 0.1), 1),
        ("c1", OrderedDict(lora_B=torch.ones(16, 4) * 0.2, head=torch.ones(16) * 0.2), 1),
    ]

    aggregated = strategy.aggregate_fit(server_round=1, results=results)
    assert aggregated is not None
    # Frozen A is preserved bit-identically
    assert torch.equal(aggregated["lora_A"], initial["lora_A"])
    # Trainable params moved
    assert not torch.equal(aggregated["lora_B"], initial["lora_B"])
    assert torch.isfinite(aggregated["lora_B"]).all()


def test_composed_strategy_with_subspace_dp():
    glob = OrderedDict(
        fc1=torch.zeros(32, 16),  # 512
        fc2=torch.zeros(16, 4),   # 64 -> total 576
    )
    s = ComposedStrategy(
        initial_parameters=glob,
        rule="mean",
        clip_norm=0.5,
        central_noise_multiplier=1.0,
        subspace_dim=32,  # 576 -> 32
        noise_seed=42,
    )

    results = [
        ("c0", OrderedDict(fc1=torch.ones(32, 16) * 0.1, fc2=torch.ones(16, 4) * 0.1), 1),
        ("c1", OrderedDict(fc1=torch.ones(32, 16) * 0.2, fc2=torch.ones(16, 4) * 0.2), 1),
    ]

    out = s.aggregate_fit(1, results)
    assert out is not None
    stats = s.last_round_stats
    assert stats["subspace_dim"] == 32
    assert stats["snr_amplification"] == pytest.approx(math.sqrt(576 / 32))
    assert stats["snr_amplification"] > 4.0
    # Per-client orig_norms are suppressed during central DP
    assert stats["orig_norms"] is None


def test_key_order_permutation_invariant():
    """Verify that SubspaceProjector and DPFedAvg produce identical projections regardless of client key order."""
    init = OrderedDict(a=torch.zeros(4), b=torch.zeros(4))
    projector = SubspaceProjector(in_dim=8, subspace_dim=2, base_seed=42)

    update_ab = OrderedDict(a=torch.tensor([1.0, 2.0, 3.0, 4.0]), b=torch.tensor([5.0, 6.0, 7.0, 8.0]))
    update_ba = OrderedDict(b=torch.tensor([5.0, 6.0, 7.0, 8.0]), a=torch.tensor([1.0, 2.0, 3.0, 4.0]))

    z_ab = projector.project_dict(update_ab, ["a", "b"])
    z_ba = projector.project_dict(update_ba, ["b", "a"])
    assert torch.equal(z_ab, z_ba)

    recon_ab = projector.reconstruct_dict(z_ab, init, ["a", "b"])
    recon_ba = projector.reconstruct_dict(z_ba, init, ["b", "a"])
    assert torch.equal(recon_ab["a"], recon_ba["a"])
    assert torch.equal(recon_ab["b"], recon_ba["b"])


def test_extreme_inputs_overflow_defense():
    """Verify that extreme float32 inputs near 3e38 are handled safely without producing NaNs."""
    d, k = 64, 4
    glob = OrderedDict(w=torch.zeros(d))
    # Extreme input
    extreme_val = torch.ones(d) * 3e38
    results = [("c_poison", OrderedDict(w=extreme_val), 1)]

    # Must either clip safely or raise ValueError, never silently output NaNs
    try:
        out = subspace_dp_aggregate(
            results=results,
            global_params=glob,
            aggregatable_keys=["w"],
            clip_norm=1.0,
            noise_multiplier=0.0,
            subspace_dim=k,
            server_round=1,
        )
        assert torch.isfinite(out["w"]).all()
    except ValueError as e:
        assert "Non-finite" in str(e)


def test_composed_strategy_preview_matches_subspace_projection():
    """Preview must compute the exact noise-free subspace projection."""
    d, k = 64, 4
    glob = OrderedDict(w=torch.zeros(d))
    s = ComposedStrategy(
        initial_parameters=glob,
        rule="mean",
        clip_norm=1.0,
        central_noise_multiplier=0.0,
        subspace_dim=k,
        noise_seed=1,
    )
    x = torch.randn(d)
    results = [("c0", OrderedDict(w=x), 1)]
    preview_out = s.preview(results, server_round=1)["w"]
    actual_out = s.aggregate_fit(1, results)["w"]
    assert torch.allclose(preview_out, actual_out, atol=1e-5)


def test_centered_clip_rejects_nan_and_inf_tau():
    """centered_clip must reject NaN and Inf tau with a clean ValueError."""
    from fedlearn.server.robust_aggregation import centered_clip
    stacked = torch.ones(4, 10)
    centre = torch.zeros(10)

    with pytest.raises(ValueError, match="positive finite float"):
        centered_clip(stacked, centre, tau=float("nan"))

    with pytest.raises(ValueError, match="positive finite float"):
        centered_clip(stacked, centre, tau=float("inf"))

    with pytest.raises(ValueError, match="positive finite float"):
        centered_clip(stacked, centre, tau=-1.0)
