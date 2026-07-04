# framework/tests/test_dp_mechanism.py
"""FR-13 — central differential privacy on the FFA-LoRA adapter delta.

Covers the mechanism (`fedlearn.privacy.dp_mechanism.dp_aggregate`) and its `FedLoRA`
integration. The security-critical invariant under test: DP noise + clipping touch ONLY the
released adapter B + head; the frozen `lora_A` is carried through bit-identical, so
`avg(B) @ A == avg(B @ A)` (the FFA invariant) stays exact under DP.
"""

from collections import OrderedDict

import pytest
import torch

from fedlearn.server.strategy import FedLoRA
from fedlearn.privacy.dp_mechanism import dp_aggregate


A_KEY = "m.lora_A.weight"
B_KEY = "m.lora_B.weight"
H_KEY = "score.weight"


# --------------------------------------------------------------------------------------------------
# builders (mirror tests/test_fedlora_strategy.py shapes: global carries A+B+head, client B+head)
# --------------------------------------------------------------------------------------------------
def ffa_initial(a=((1.0, 2.0),), b=((0.0,), (0.0,)), h=(0.0,)):
    return OrderedDict(
        [
            (A_KEY, torch.tensor(a)),   # frozen shared A  (1 x 2)
            (B_KEY, torch.tensor(b)),   # global B         (2 x 1), starts at 0
            (H_KEY, torch.tensor(h)),   # head             (1,)
        ]
    )


def ffa_upload(bval, hval):
    # A client uploads only B + head under FFA.
    return OrderedDict(
        [
            (B_KEY, torch.tensor([[bval], [bval]])),
            (H_KEY, torch.tensor([hval])),
        ]
    )


def two_clients(b1=2.0, h1=2.0, b2=4.0, h2=4.0, n1=100, n2=100):
    return [(ffa_upload(b1, h1), n1), (ffa_upload(b2, h2), n2)]


# --------------------------------------------------------------------------------------------------
# 1. Default-off is byte-for-byte the current weighted-average + frozen-A re-attach.
# --------------------------------------------------------------------------------------------------
def test_dp_disabled_matches_weighted_average_and_reattach():
    init = ffa_initial()
    s = FedLoRA(initial_parameters=init, aggregation="FFA_LORA", dp_enabled=False)
    # num_examples-weighted: (2*300 + 4*100) / 400 = 2.5  (the historical FedAvg-weighted result).
    out = s.aggregate_fit(1, [(ffa_upload(2.0, 2.0), 300), (ffa_upload(4.0, 4.0), 100)])
    assert torch.allclose(out[B_KEY], torch.full((2, 1), 2.5))
    assert torch.allclose(out[H_KEY], torch.tensor([2.5]))
    assert torch.equal(out[A_KEY], init[A_KEY])  # frozen A re-attached unchanged


def test_dp_default_is_off():
    # The dp_* kwargs default such that FedLoRA is DP-off unless explicitly enabled.
    s = FedLoRA(initial_parameters=ffa_initial(), aggregation="FFA_LORA")
    assert s.dp_enabled is False


# --------------------------------------------------------------------------------------------------
# 2. FFA-safety under DP: frozen A bit-identical; avg(B)@A == avg(B@A) preserved.
# --------------------------------------------------------------------------------------------------
def test_dp_freezes_a_bit_identical():
    init = ffa_initial()
    s = FedLoRA(
        initial_parameters=init,
        aggregation="FFA_LORA",
        dp_enabled=True,
        dp_clip_norm=10.0,
        dp_noise_multiplier=1.0,
        dp_seed=7,
    )
    out = s.aggregate_fit(1, two_clients())
    assert torch.equal(out[A_KEY], init[A_KEY])  # zero noise on A => bit-identical


def test_ffa_invariant_avgB_at_A_equals_avg_BatA_under_dp():
    # z=0 => deterministic clip + uniform-average, no noise. Global B starts at 0 so each client's
    # delta == its B, and (huge clip) nothing is clipped.
    A = torch.tensor([[1.0, 2.0, 3.0]])  # shared frozen A (1 x 3)
    init = OrderedDict([(A_KEY, A), (B_KEY, torch.zeros(2, 1)), (H_KEY, torch.zeros(1))])
    s = FedLoRA(
        initial_parameters=init,
        aggregation="FFA_LORA",
        dp_enabled=True,
        dp_clip_norm=1e9,
        dp_noise_multiplier=0.0,
    )
    B1 = torch.tensor([[0.1], [0.2]])
    B2 = torch.tensor([[0.3], [0.4]])
    up1 = OrderedDict([(B_KEY, B1), (H_KEY, torch.tensor([0.5]))])
    up2 = OrderedDict([(B_KEY, B2), (H_KEY, torch.tensor([0.7]))])
    out = s.aggregate_fit(1, [(up1, 100), (up2, 100)])

    B_out = out[B_KEY]
    assert torch.allclose(B_out, (B1 + B2) / 2)                 # uniform average of B deltas
    # FFA invariant: averaging B then applying the shared frozen A == averaging (B @ A).
    assert torch.allclose(B_out @ A, (B1 @ A + B2 @ A) / 2)
    assert torch.equal(out[A_KEY], A)


# --------------------------------------------------------------------------------------------------
# 3. Noise lands on aggregatable keys only: per-coord var(B/head) ~= (z*S/N)^2; var(A) == 0.
# --------------------------------------------------------------------------------------------------
def test_noise_variance_on_aggregatable_keys_matches_calibration():
    global_params = ffa_initial()          # B=0, head=0, A(1x2)
    agg_keys = [B_KEY, H_KEY]
    results = two_clients(b1=1.0, h1=1.0, b2=3.0, h2=3.0)  # small deltas => no clipping
    S, z, N = 10.0, 1.0, 2
    expected_var = (z * S / N) ** 2        # = 25.0

    draws_b, draws_h = [], []
    M = 4000
    for seed in range(M):
        g = torch.Generator().manual_seed(seed)
        out = dp_aggregate(
            results, global_params, agg_keys, clip_norm=S, noise_multiplier=z, generator=g
        )
        draws_b.append(out[B_KEY])
        draws_h.append(out[H_KEY])

    var_b = torch.stack(draws_b).var(dim=0)
    var_h = torch.stack(draws_h).var(dim=0)
    assert torch.allclose(var_b, torch.full_like(var_b, expected_var), rtol=0.12)
    assert torch.allclose(var_h, torch.full_like(var_h, expected_var), rtol=0.12)


def test_lora_a_has_exactly_zero_variance_across_seeds():
    init = ffa_initial()
    a_draws = []
    for seed in range(40):
        s = FedLoRA(
            initial_parameters=init,
            aggregation="FFA_LORA",
            dp_enabled=True,
            dp_clip_norm=10.0,
            dp_noise_multiplier=5.0,
            dp_seed=seed,
        )
        out = s.aggregate_fit(1, two_clients())
        a_draws.append(out[A_KEY])
        assert torch.equal(out[A_KEY], init[A_KEY])  # every draw is the frozen A, exactly
    var_a = torch.stack(a_draws).var(dim=0, unbiased=False)
    assert torch.equal(var_a, torch.zeros_like(init[A_KEY]))  # variance is EXACTLY zero


# --------------------------------------------------------------------------------------------------
# 4. Clip bounds: a client with ||delta|| >> S is scaled so its clipped delta norm <= S.
# --------------------------------------------------------------------------------------------------
def test_clip_bounds_scale_oversized_delta():
    global_params = ffa_initial()  # B=0, head=0
    agg_keys = [B_KEY, H_KEY]
    S = 1.0
    huge = OrderedDict(
        [(B_KEY, torch.tensor([[1000.0], [1000.0]])), (H_KEY, torch.tensor([1000.0]))]
    )
    # Single client, z=0 => out - global == the clipped delta (uniform avg over N=1).
    out = dp_aggregate([("c0", huge, 100)], global_params, agg_keys, clip_norm=S, noise_multiplier=0.0)
    clipped = OrderedDict((k, out[k] - global_params[k].float()) for k in agg_keys)
    joint_norm = torch.sqrt(sum((t * t).sum() for t in clipped.values()))
    assert float(joint_norm) <= S + 1e-5


def test_within_budget_delta_passes_unchanged():
    global_params = ffa_initial()
    agg_keys = [B_KEY, H_KEY]
    small = OrderedDict([(B_KEY, torch.tensor([[0.1], [0.1]])), (H_KEY, torch.tensor([0.1]))])
    out = dp_aggregate([("c0", small, 100)], global_params, agg_keys, clip_norm=100.0, noise_multiplier=0.0)
    assert torch.allclose(out[B_KEY], small[B_KEY])   # delta unclipped, global is 0
    assert torch.allclose(out[H_KEY], small[H_KEY])


# --------------------------------------------------------------------------------------------------
# 5. Determinism: same dp_seed => identical; different dp_seed => different.
# --------------------------------------------------------------------------------------------------
def test_same_seed_identical_diff_seed_differs():
    init = ffa_initial()

    def run(seed):
        s = FedLoRA(
            initial_parameters=init,
            aggregation="FFA_LORA",
            dp_enabled=True,
            dp_clip_norm=10.0,
            dp_noise_multiplier=1.0,
            dp_seed=seed,
        )
        return s.aggregate_fit(1, two_clients())

    a, b, c = run(123), run(123), run(999)
    assert torch.equal(a[B_KEY], b[B_KEY]) and torch.equal(a[H_KEY], b[H_KEY])
    assert not torch.equal(a[B_KEY], c[B_KEY])


def test_generator_determinism_at_mechanism_level():
    global_params = ffa_initial()
    agg_keys = [B_KEY, H_KEY]
    results = two_clients()
    o1 = dp_aggregate(results, global_params, agg_keys, 10.0, 1.0, torch.Generator().manual_seed(5))
    o2 = dp_aggregate(results, global_params, agg_keys, 10.0, 1.0, torch.Generator().manual_seed(5))
    o3 = dp_aggregate(results, global_params, agg_keys, 10.0, 1.0, torch.Generator().manual_seed(6))
    assert torch.equal(o1[B_KEY], o2[B_KEY])
    assert not torch.equal(o1[B_KEY], o3[B_KEY])


# --------------------------------------------------------------------------------------------------
# 6. z=0 edge: DP-on with noise_multiplier=0 == clip + UNIFORM average, no noise (not weighted).
# --------------------------------------------------------------------------------------------------
def test_z_zero_is_uniform_average_no_noise():
    global_params = ffa_initial()  # B=0, head=0
    agg_keys = [B_KEY, H_KEY]
    # Different num_examples on purpose: uniform average ignores the counts.
    results = [(ffa_upload(2.0, 2.0), 300), (ffa_upload(4.0, 4.0), 100)]
    out = dp_aggregate(results, global_params, agg_keys, clip_norm=1e9, noise_multiplier=0.0)
    # Uniform (1/N): (2 + 4)/2 = 3.0  — NOT the num_examples-weighted 2.5.
    assert torch.allclose(out[B_KEY], torch.full((2, 1), 3.0))
    assert torch.allclose(out[H_KEY], torch.tensor([3.0]))
    # No generator supplied and z=0 => fully deterministic.
    out2 = dp_aggregate(results, global_params, agg_keys, clip_norm=1e9, noise_multiplier=0.0)
    assert torch.equal(out[B_KEY], out2[B_KEY])


def test_fedlora_dp_z0_uniform_not_num_examples_weighted():
    init = ffa_initial()
    s = FedLoRA(
        initial_parameters=init,
        aggregation="FFA_LORA",
        dp_enabled=True,
        dp_clip_norm=1e9,
        dp_noise_multiplier=0.0,
    )
    out = s.aggregate_fit(1, [(ffa_upload(2.0, 2.0), 300), (ffa_upload(4.0, 4.0), 100)])
    assert torch.allclose(out[H_KEY], torch.tensor([3.0]))  # uniform (weighted would be 2.5)
    assert torch.equal(out[A_KEY], init[A_KEY])


# --------------------------------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------------------------------
def test_dp_enabled_validates_clip_and_noise():
    init = ffa_initial()
    with pytest.raises(ValueError):
        FedLoRA(initial_parameters=init, dp_enabled=True, dp_clip_norm=None, dp_noise_multiplier=1.0)
    with pytest.raises(ValueError):
        FedLoRA(initial_parameters=init, dp_enabled=True, dp_clip_norm=-1.0, dp_noise_multiplier=1.0)
    with pytest.raises(ValueError):
        FedLoRA(initial_parameters=init, dp_enabled=True, dp_clip_norm=1.0, dp_noise_multiplier=-0.5)


def test_mechanism_rejects_bad_args():
    gp = ffa_initial()
    with pytest.raises(ValueError):
        dp_aggregate([], gp, [B_KEY], 1.0, 1.0)                       # no clients
    with pytest.raises(ValueError):
        dp_aggregate(two_clients(), gp, [B_KEY], 0.0, 1.0)           # clip_norm must be > 0
    with pytest.raises(ValueError):
        dp_aggregate(two_clients(), gp, [B_KEY], 1.0, -1.0)          # noise_multiplier must be >= 0
    with pytest.raises(ValueError):
        dp_aggregate(two_clients(), gp, [], 1.0, 1.0)                # no aggregatable keys
