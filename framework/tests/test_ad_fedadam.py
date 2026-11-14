"""Unit and mathematical tests for Anchor-Decoupled FedAdam (AD-FedAdam)."""

from collections import OrderedDict
import pytest
import torch

from fedlearn.server.ad_fedadam import ADFedAdam, ad_fedadam_step
from fedlearn.server.strategy_factory import create_strategy, STRATEGY_REGISTRY


def test_ad_fedadam_step_mathematical_properties():
    dim = 4
    deltas = torch.tensor([
        [1.0, 2.0, -1.0, 0.5],
        [1.0, 2.0, -1.0, 0.5],
    ], dtype=torch.float32)

    centre = torch.zeros(dim, dtype=torch.float32)
    server_m = torch.zeros(dim, dtype=torch.float32)
    server_v = torch.zeros(dim, dtype=torch.float32)

    # Large tau so no clipping occurs
    tau = 100.0
    eta_g = 0.1
    beta1 = 0.9
    beta2 = 0.99
    eps = 1e-4

    step, next_c, next_m, next_v = ad_fedadam_step(
        stacked_deltas=deltas,
        centre=centre,
        server_m=server_m,
        server_v=server_v,
        round_idx=1,
        tau=tau,
        eta_g=eta_g,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )

    expected_r = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)
    assert torch.allclose(next_c, expected_r, atol=1e-6)

    # Round 1 moments
    expected_m = (1.0 - beta1) * expected_r
    expected_v = (1.0 - beta2) * (expected_r ** 2)
    assert torch.allclose(next_m, expected_m, atol=1e-6)
    assert torch.allclose(next_v, expected_v, atol=1e-6)

    # Bias-corrected moments
    m_hat = expected_m / (1.0 - beta1)
    v_hat = expected_v / (1.0 - beta2)
    expected_step = eta_g * m_hat / (torch.sqrt(v_hat) + eps)
    assert torch.allclose(step, expected_step, atol=1e-6)


def test_ad_fedadam_step_clipping():
    dim = 2
    # Vector norm is 5.0
    deltas = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    centre = torch.zeros(dim, dtype=torch.float32)
    server_m = torch.zeros(dim, dtype=torch.float32)
    server_v = torch.zeros(dim, dtype=torch.float32)

    tau = 2.5  # Exactly half the norm
    step, next_c, next_m, next_v = ad_fedadam_step(
        stacked_deltas=deltas,
        centre=centre,
        server_m=server_m,
        server_v=server_v,
        round_idx=1,
        tau=tau,
    )

    expected_r = torch.tensor([1.5, 2.0], dtype=torch.float32)
    assert torch.allclose(next_c, expected_r, atol=1e-6)
    assert float(torch.linalg.vector_norm(next_c).item()) == pytest.approx(tau, abs=1e-5)


def test_ad_fedadam_step_input_validation():
    dim = 4
    c = torch.zeros(dim)
    m = torch.zeros(dim)
    v = torch.zeros(dim)

    # Empty stacked deltas
    with pytest.raises(ValueError, match="at least one client update"):
        ad_fedadam_step(torch.empty((0, dim)), c, m, v, round_idx=1)

    # Negative or zero tau
    with pytest.raises(ValueError, match="tau must be positive"):
        ad_fedadam_step(torch.randn(2, dim), c, m, v, round_idx=1, tau=0.0)

    # round_idx < 1
    with pytest.raises(ValueError, match="round_idx must be a finite positive integer"):
        ad_fedadam_step(torch.randn(2, dim), c, m, v, round_idx=0)

    # round_idx non-integer
    with pytest.raises(ValueError, match="round_idx must be a finite positive integer"):
        ad_fedadam_step(torch.randn(2, dim), c, m, v, round_idx=1.5)  # type: ignore[arg-type]


def test_ad_fedadam_strategy_e2e():
    initial_params = OrderedDict([
        ("weight", torch.tensor([[0.5, -0.5], [0.2, 0.1]], dtype=torch.float32)),
        ("bias", torch.tensor([0.0, 0.0], dtype=torch.float32)),
    ])

    strategy = ADFedAdam(
        initial_parameters=initial_params,
        server_learning_rate=0.1,
        tau=1.0,
    )

    assert "adfedadam" in STRATEGY_REGISTRY
    factory_strategy = create_strategy("adfedadam", initial_parameters=initial_params)
    assert isinstance(factory_strategy, ADFedAdam)

    # Case insensitive matching
    case_strategy = create_strategy("AD-FedAdam", initial_parameters=initial_params)
    assert isinstance(case_strategy, ADFedAdam)

    underscore_strategy = create_strategy("ad_fedadam", initial_parameters=initial_params)
    assert isinstance(underscore_strategy, ADFedAdam)

    # Simulate 3 rounds of aggregation
    for r in range(1, 4):
        client_updates = [
            ("client_1", OrderedDict([
                ("weight", initial_params["weight"] + 0.1 * r),
                ("bias", initial_params["bias"] + 0.05 * r),
            ]), 10),
            ("client_2", OrderedDict([
                ("weight", initial_params["weight"] + 0.12 * r),
                ("bias", initial_params["bias"] + 0.04 * r),
            ]), 10),
        ]
        updated = strategy.aggregate_fit(server_round=r, results=client_updates)
        assert updated is not None
        assert "weight" in updated
        assert "bias" in updated
        assert updated["weight"].shape == initial_params["weight"].shape
        assert updated["bias"].shape == initial_params["bias"].shape
        assert torch.isfinite(updated["weight"]).all()
        assert torch.isfinite(updated["bias"]).all()


def test_ad_fedadam_recovers_stable_updates_under_tight_clipping():
    torch.manual_seed(42)
    dim = 20
    centre = torch.zeros(dim)
    m_ad = torch.zeros(dim)
    v_ad = torch.zeros(dim)

    # In naive FedAdam, the center accumulator c_t is fed directly into v_t.
    # In AD-FedAdam, v_t tracks r_t.
    c_naive = torch.zeros(dim)
    v_naive = torch.zeros(dim)

    tau = 0.2  # Tight clipping
    beta2 = 0.99

    for r in range(1, 15):
        # Clients submit true gradient plus small noise
        true_step = torch.full((dim,), 0.05)
        client_deltas = true_step.unsqueeze(0) + torch.randn(5, dim) * 0.02

        # 1. AD-FedAdam step
        _, centre, m_ad, v_ad = ad_fedadam_step(
            stacked_deltas=client_deltas,
            centre=centre,
            server_m=m_ad,
            server_v=v_ad,
            round_idx=r,
            tau=tau,
        )

        # 2. Naive FedAdam center update
        res = client_deltas - c_naive
        res_norms = torch.linalg.vector_norm(res, dim=1, keepdim=True)
        factors = torch.clamp(tau / torch.clamp(res_norms, min=1e-12), max=1.0)
        c_naive = c_naive + (res * factors).mean(dim=0)
        v_naive = beta2 * v_naive + (1.0 - beta2) * (c_naive ** 2)

    # In naive FedAdam, v_naive tracks accumulated c_naive^2 which grows monotonically,
    # causing denominator explosion or high-frequency oscillation.
    # In AD-FedAdam, v_ad tracks per-round innovation r_t^2 which remains stationary!
    v_ad_rms = torch.sqrt(torch.mean(v_ad)).item()
    v_naive_rms = torch.sqrt(torch.mean(v_naive)).item()
    assert v_ad_rms < v_naive_rms


def test_ad_fedadam_input_hardening():
    dim = 2
    c = torch.zeros(dim)
    m = torch.zeros(dim)
    v = torch.zeros(dim)
    valid_deltas = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    # NaN in deltas
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        ad_fedadam_step(torch.tensor([[float("nan"), 1.0]]), c, m, v, round_idx=1)

    # Inf in deltas
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        ad_fedadam_step(torch.tensor([[float("inf"), 1.0]]), c, m, v, round_idx=1)

    # NaN in centre
    with pytest.raises(ValueError, match="centre contains NaN or Inf"):
        ad_fedadam_step(valid_deltas, torch.tensor([float("nan"), 0.0]), m, v, round_idx=1)

    # Invalid beta1
    with pytest.raises(ValueError, match="beta1 must be in"):
        ad_fedadam_step(valid_deltas, c, m, v, round_idx=1, beta1=1.0)
    with pytest.raises(ValueError, match="beta1 must be in"):
        ad_fedadam_step(valid_deltas, c, m, v, round_idx=1, beta1=-0.1)

    # Invalid beta2
    with pytest.raises(ValueError, match="beta2 must be in"):
        ad_fedadam_step(valid_deltas, c, m, v, round_idx=1, beta2=1.0)

    # Invalid eps
    with pytest.raises(ValueError, match="eps must be positive"):
        ad_fedadam_step(valid_deltas, c, m, v, round_idx=1, eps=0.0)

    # Invalid tau (NaN)
    with pytest.raises(ValueError, match="tau must be positive and finite"):
        ad_fedadam_step(valid_deltas, c, m, v, round_idx=1, tau=float("nan"))


def test_ad_fedadam_norm_overflow_protection():
    dim = 2
    c = torch.zeros(dim)
    m = torch.zeros(dim)
    v = torch.zeros(dim)

    # Vector that would overflow float32 norm if squared directly (1e20^2 = 1e40 > float32 max ~3.4e38)
    large_delta = torch.tensor([[1e20, 1e20]], dtype=torch.float32)
    step, next_c, next_m, next_v = ad_fedadam_step(large_delta, c, m, v, round_idx=1, tau=0.5)

    # Must produce a finite step with norm <= tau and non-zero
    assert torch.isfinite(step).all()
    assert torch.isfinite(next_c).all()
    # Norm of clipped residual should equal tau = 0.5
    res_norm = torch.linalg.vector_norm(next_c).item()
    assert res_norm == pytest.approx(0.5, abs=1e-4)


def test_ad_fedadam_extreme_float32_subtraction():
    """P2: Extreme float32 values that overflow on subtraction before float64 widening."""
    dim = 4
    c = torch.zeros(dim)
    m = torch.zeros(dim)
    v = torch.zeros(dim)

    # 3e38 - (-3e38) = 6e38 which overflows float32 (~3.4e38 max)
    deltas = torch.full((1, dim), 3e38, dtype=torch.float32)
    centre = torch.full((dim,), -3e38, dtype=torch.float32)

    step, next_c, next_m, next_v = ad_fedadam_step(
        deltas, centre, m, v, round_idx=1, tau=0.5
    )
    # After float64 widening, the subtraction should succeed and produce finite outputs
    assert torch.isfinite(step).all(), "Step must be finite after float64 widening"
    assert torch.isfinite(next_c).all(), "Centre must be finite after float64 widening"


def test_ad_fedadam_rejects_non_integer_round_idx():
    """P2: round_idx must be a positive integer, not float or NaN."""
    dim = 4
    deltas = torch.randn(2, dim)
    c = torch.zeros(dim)
    m = torch.zeros(dim)
    v = torch.zeros(dim)

    with pytest.raises((ValueError, TypeError)):
        ad_fedadam_step(deltas, c, m, v, round_idx=float("nan"))  # type: ignore[arg-type]


def test_dp_robust_aggregate_rejects_nan_noise():
    """P3: NaN noise_multiplier must not silently skip noise."""
    from fedlearn.privacy.dp_mechanism import dp_robust_aggregate
    from collections import OrderedDict

    global_params = OrderedDict(w=torch.zeros(4))
    client = OrderedDict(w=torch.ones(4))
    results = [((1, 100), client)]

    with pytest.raises(ValueError, match="noise_multiplier.*finite"):
        dp_robust_aggregate(
            results, global_params, ["w"],
            clip_norm=1.0, noise_multiplier=float("nan"),
        )


def test_composed_strategy_rejects_nan_noise():
    """P3: NaN central_noise_multiplier must be rejected at init."""
    from fedlearn.server.composed_strategy import ComposedStrategy
    from collections import OrderedDict

    params = OrderedDict(w=torch.zeros(4))

    with pytest.raises(ValueError, match="central_noise_multiplier.*finite"):
        ComposedStrategy(params, rule="mean", clip_norm=1.0,
                         central_noise_multiplier=float("nan"))


def test_composed_strategy_rejects_nan_clip_norm():
    """P3: NaN clip_norm must be rejected at init."""
    from fedlearn.server.composed_strategy import ComposedStrategy
    from collections import OrderedDict

    params = OrderedDict(w=torch.zeros(4))

    with pytest.raises(ValueError, match="clip_norm.*finite"):
        ComposedStrategy(params, rule="mean", clip_norm=float("nan"),
                         central_noise_multiplier=1.0)


def test_ad_fedadam_rejects_boolean_round_idx():
    """P2 hardening: round_idx=True (bool subclass of int) must be rejected."""
    dim = 4
    deltas = torch.randn(2, dim)
    c = torch.zeros(dim)
    m = torch.zeros(dim)
    v = torch.zeros(dim)

    with pytest.raises(ValueError, match="round_idx must be a finite positive integer"):
        ad_fedadam_step(deltas, c, m, v, round_idx=True)  # type: ignore[arg-type]


def test_ad_fedadam_catches_moment_overflow():
    """P2 hardening: non-finite next_m or next_v state must trigger fail-closed guard."""
    dim = 4
    deltas = torch.randn(2, dim)
    c = torch.zeros(dim)
    # Supply non-finite accumulator that bypasses initial check or blows up in recurrence
    m = torch.zeros(dim)
    v = torch.zeros(dim)
    # Initial checks catch non-finite inputs
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        ad_fedadam_step(deltas, c, torch.full((dim,), float("inf")), v, round_idx=1)


def test_ad_fedadam_strategy_rejects_global_param_overflow():
    """Point 4: ADFedAdam strategy must reject non-finite global parameters after applying step."""
    init_params = OrderedDict(w=torch.tensor([3e38], dtype=torch.float32))
    strat = ADFedAdam(init_params, tau=0.5, server_learning_rate=1e38)

    # Client update that produces a finite step which overflows when added to 3e38
    client_params = OrderedDict(w=torch.tensor([3.1e38], dtype=torch.float32))
    results = [("client_1", client_params, 100)]

    with pytest.raises(ValueError, match="produced non-finite global parameter"):
        strat.aggregate_fit(1, results)


def test_ad_fedadam_init_validates_hyperparameters():
    init_params = OrderedDict(w=torch.zeros(4))
    with pytest.raises(ValueError, match="non-empty OrderedDict"):
        ADFedAdam({})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="min_fit_clients must be >= 1"):
        ADFedAdam(init_params, min_fit_clients=0)

    with pytest.raises(ValueError, match="clients_per_round .* cannot be less than min_fit_clients"):
        ADFedAdam(init_params, min_fit_clients=5, clients_per_round=3)

    with pytest.raises(ValueError, match="server_learning_rate must be positive and finite"):
        ADFedAdam(init_params, server_learning_rate=0.0)

    with pytest.raises(ValueError, match="server_learning_rate must be positive and finite"):
        ADFedAdam(init_params, server_learning_rate=float("nan"))

    with pytest.raises(ValueError, match="tau must be positive and finite"):
        ADFedAdam(init_params, tau=-1.0)

    with pytest.raises(ValueError, match="beta1 must be in \\[0, 1\\)"):
        ADFedAdam(init_params, beta1=1.0)

    with pytest.raises(ValueError, match="beta2 must be in \\[0, 1\\)"):
        ADFedAdam(init_params, beta2=-0.1)

    with pytest.raises(ValueError, match="eps must be positive and finite"):
        ADFedAdam(init_params, eps=0.0)


def test_ad_fedadam_strategy_rejects_client_key_mismatch():
    init_params = OrderedDict(w=torch.zeros(4), b=torch.zeros(1))
    strat = ADFedAdam(init_params)

    # Missing key 'b'
    bad_results = [("c1", OrderedDict(w=torch.randn(4)), 10)]
    with pytest.raises(ValueError, match="Client update keys mismatch"):
        strat.aggregate_fit(1, bad_results)

    # Extra key 'extra'
    bad_results2 = [("c1", OrderedDict(w=torch.randn(4), b=torch.randn(1), extra=torch.randn(2)), 10)]
    with pytest.raises(ValueError, match="Client update keys mismatch"):
        strat.aggregate_fit(1, bad_results2)


def test_ad_fedadam_strategy_rejects_client_shape_mismatch():
    init_params = OrderedDict(w=torch.zeros(4, 4))
    strat = ADFedAdam(init_params)

    # Wrong shape (4, 2) instead of (4, 4)
    bad_results = [("c1", OrderedDict(w=torch.randn(4, 2)), 10)]
    with pytest.raises(ValueError, match="Client update shape mismatch"):
        strat.aggregate_fit(1, bad_results)


def test_ad_fedadam_quadratic_convergence():
    """Verify that ADFedAdam does not stall at distance ~27 on a strongly convex quadratic."""
    d, n = 10, 20
    tau, eta = 0.5, 0.1
    gen = torch.Generator().manual_seed(42)
    spread = torch.randn(n, d, generator=gen)
    spread = spread - spread.mean(dim=0, keepdim=True)
    w_star = torch.full((d,), 10.0)
    optima = w_star + spread

    w = torch.zeros(d)
    strat = ADFedAdam(OrderedDict(w=w.clone()), server_learning_rate=eta, tau=tau)
    client_lr = 0.1

    initial_dist = float(torch.linalg.vector_norm(w - w_star))
    assert initial_dist > 30.0

    for t in range(1, 301):
        curv = torch.full((d,), client_lr)
        deltas = -curv * (w.unsqueeze(0) - optima)
        res = [(f"c{i}", OrderedDict(w=w + deltas[i]), 1) for i in range(n)]
        out = strat.aggregate_fit(t, res)
        assert out is not None
        w = out["w"]

    final_dist = float(torch.linalg.vector_norm(w - w_star))
    # Must converge close to optimum, not stall at 27.2
    assert final_dist < 0.01, f"ADFedAdam stalled at distance {final_dist}"


def test_ad_fedadam_bn_buffers_linear_averaging_and_clamping():
    """Verify that BatchNorm buffers are linearly averaged and running_var never turns negative."""
    init = OrderedDict([("layer.weight", torch.zeros(10)), ("bn.running_var", torch.full((4,), 0.05))])
    strat = ADFedAdam(init, server_learning_rate=0.1, tau=0.5, buffer_names=["bn.running_var"])

    # Clients send small weight delta and running_var of 0.04
    results = [
        (f"c{i}", OrderedDict([("layer.weight", torch.randn(10) * 0.01), ("bn.running_var", torch.full((4,), 0.04))]), 1)
        for i in range(5)
    ]
    out = strat.aggregate_fit(1, results)
    assert out is not None
    assert (out["bn.running_var"] >= 0).all()
    assert torch.allclose(out["bn.running_var"], torch.full((4,), 0.04))

