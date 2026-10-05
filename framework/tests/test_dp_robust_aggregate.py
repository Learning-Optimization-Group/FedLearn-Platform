# framework/tests/test_dp_robust_aggregate.py
"""Central DP over a robust aggregate (capstone matched study).

Covers ``replace_one_sensitivity`` and ``dp_robust_aggregate`` in
``fedlearn.privacy.dp_mechanism``. Under replace-one adjacency with each client's delta jointly
clipped to L2 norm ``S``:

* the clipped mean moves by at most ``2S/n``;
* a coordinate-wise trimmed mean that drops ``k = floor(beta*n)`` values from each end moves by at
  most ``2S/(n - 2k)``: each coordinate of the trimmed SUM moves by at most the change in the one
  replaced client's value, so the joint L2 change is at most ``||x - x'||_2 / (n - 2k)``.

Expected values below are derived by hand, never with the code under test.
"""

import pytest

from fedlearn.privacy import dp_mechanism as dpm


# --------------------------------------------------------------------------------------------------
# replace_one_sensitivity
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rule, n, clip, trim, want",
    [
        ("mean", 10, 1.0, 0.0, 0.2),            # 2S/n
        ("mean", 80, 0.4, 0.0, 0.01),           # 2*0.4/80
        ("trimmed_mean", 10, 1.0, 0.2, 1 / 3),  # k=2, n-2k=6 -> 2/6
        ("trimmed_mean", 40, 1.0, 0.2, 1 / 12),  # k=8, n-2k=24 -> 2/24
        ("trimmed_mean", 8, 1.0, 0.2, 1 / 3),   # k=floor(1.6)=1, n-2k=6 -> 2/6 (round would give 2/4)
        ("trimmed_mean", 10, 1.0, 0.0, 0.2),    # beta=0 is the plain mean
    ],
)
def test_replace_one_sensitivity_matches_hand_derived_bound(rule, n, clip, trim, want):
    assert dpm.replace_one_sensitivity(rule, n, clip, trim) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize(
    "rule, n, clip, trim",
    [
        ("trimmed_mean", 4, 1.0, 0.5),   # k=2 removes all 4 clients
        ("median", 10, 1.0, 0.0),        # no 1/n sensitivity bound: deliberately unsupported
        ("mean", 10, 0.0, 0.0),          # clip must be positive
        ("mean", 0, 1.0, 0.0),           # needs at least one client
        ("trimmed_mean", 10, 1.0, -0.1),  # negative trim ratio
    ],
)
def test_replace_one_sensitivity_rejects_invalid_configurations(rule, n, clip, trim):
    with pytest.raises(ValueError):
        dpm.replace_one_sensitivity(rule, n, clip, trim)


# --------------------------------------------------------------------------------------------------
# dp_robust_aggregate
# --------------------------------------------------------------------------------------------------
from collections import OrderedDict  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402

W, B = "head.weight", "head.bias"
GLOBAL = OrderedDict([(W, torch.tensor([1.0, 1.0])), (B, torch.tensor([0.5])), ("frozen", torch.tensor([7.0]))])


def _client(dw, db, num_examples=100, cid="c"):
    """A client upload whose delta from GLOBAL is (dw, db) on the aggregatable keys."""
    return (
        cid,
        OrderedDict([(W, GLOBAL[W] + torch.tensor(dw)), (B, GLOBAL[B] + torch.tensor([db]))]),
        num_examples,
    )


def _agg(results, rule, trim=0.0, clip=100.0, z=0.0, keys=(W, B), generator=None):
    return dpm.dp_robust_aggregate(
        results, GLOBAL, list(keys), clip_norm=clip, noise_multiplier=z,
        rule=rule, trim_ratio=trim, generator=generator,
    )


FIVE = [
    _client([0.0, 1.0], 0.0, cid="c0"),
    _client([0.1, 2.0], -1.0, cid="c1"),
    _client([0.2, 3.0], 5.0, cid="c2"),
    _client([0.3, 4.0], 2.0, cid="c3"),
    _client([9.0, -9.0], 1.0, cid="c4"),
]


@pytest.mark.parametrize(
    "rule, trim, want_w, want_b",
    [
        # beta=0.2, n=5 -> drop 1 per end: w0 {.1,.2,.3}->.2, w1 {1,2,3}->2, b {0,1,2}->1
        ("trimmed_mean", 0.2, [1.2, 3.0], [1.5]),
        # plain mean: w0 9.6/5=1.92, w1 1/5=0.2, b 7/5=1.4
        ("mean", 0.0, [2.92, 1.2], [1.9]),
    ],
)
def test_noiseless_aggregate_matches_hand_computed_rule(rule, trim, want_w, want_b):
    out = _agg(FIVE, rule, trim)
    assert list(out.keys()) == [W, B]  # only the aggregatable keys, in order
    assert torch.allclose(out[W], torch.tensor(want_w), atol=1e-6)
    assert torch.allclose(out[B], torch.tensor(want_b), atol=1e-6)


def test_each_delta_is_clipped_jointly_before_aggregation():
    # c0's delta (0.6, 0.8 | 1.0) has per-key norms 1 and 1 but joint norm sqrt(2): a joint clip to
    # S=1 scales it by 1/sqrt(2); a per-key clip, or a clip after averaging, would leave it alone.
    out = _agg([_client([0.6, 0.8], 1.0, cid="a"), _client([0.0, 0.0], 0.0, cid="b")], "mean", clip=1.0)
    s = 1 / math.sqrt(2) / 2
    assert torch.allclose(out[W] - GLOBAL[W], torch.tensor([0.6 * s, 0.8 * s]), atol=1e-6)
    assert torch.allclose(out[B] - GLOBAL[B], torch.tensor([1.0 * s]), atol=1e-6)


@pytest.mark.parametrize("rule, trim, want", [("trimmed_mean", 0.2, 2 / 3), ("mean", 0.0, 2 / 5)])
def test_replacing_one_client_can_move_the_aggregate_by_exactly_the_bound(rule, trim, want):
    # One coordinate, S=1, n=5: values {-1,-1,-1,+1,+1}; replace one -1 by +1.
    before = [_client([0.0, 0.0], v, cid=f"c{i}") for i, v in enumerate([-1.0, -1.0, -1.0, 1.0, 1.0])]
    after = [_client([0.0, 0.0], 1.0, cid="c0")] + before[1:]
    moved = (_agg(after, rule, trim, clip=1.0, keys=(B,))[B] - _agg(before, rule, trim, clip=1.0, keys=(B,))[B]).abs()
    assert moved.item() == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("rule, trim", [("trimmed_mean", 0.2), ("trimmed_mean", 0.1), ("mean", 0.0)])
def test_replacing_any_client_never_exceeds_the_bound(rule, trim):
    gen = torch.Generator().manual_seed(1234)
    clip = 0.4
    for trial in range(300):
        n = [5, 10, 13, 40][trial % 4]
        bound = 2 * clip / (n - 2 * int(trim * n))  # hand formula, not the code under test
        scales = torch.rand(n, generator=gen) * 3  # some clients far outside the clip
        results = [
            _client((torch.randn(2, generator=gen) * s).tolist(), float(torch.randn(1, generator=gen) * s), cid=f"c{i}")
            for i, s in enumerate(scales)
        ]
        swap = int(torch.randint(n, (1,), generator=gen))
        replaced = list(results)
        replaced[swap] = _client((torch.randn(2, generator=gen) * 5).tolist(), float(torch.randn(1, generator=gen) * 5), cid="x")
        a, b = _agg(results, rule, trim, clip=clip), _agg(replaced, rule, trim, clip=clip)
        moved = math.sqrt(sum(float(((a[k] - b[k]) ** 2).sum()) for k in (W, B)))
        assert moved <= bound * (1 + 1e-5), f"trial {trial}: moved {moved} > bound {bound}"


@pytest.mark.parametrize("rule, trim, want_std", [("trimmed_mean", 0.2, 0.5), ("mean", 0.0, 0.3)])
def test_noise_std_is_multiplier_times_rule_sensitivity(rule, trim, want_std):
    # n=10, S=1, z=1.5: trimmed 1.5*2/6=0.5, mean 1.5*2/10=0.3. Identical clients -> noiseless
    # aggregate equals GLOBAL, so the output minus GLOBAL is pure noise over 40,000 coordinates.
    big = OrderedDict([("w", torch.zeros(40_000))])
    results = [(f"c{i}", OrderedDict([("w", torch.zeros(40_000))]), 10) for i in range(10)]
    out = dpm.dp_robust_aggregate(results, big, ["w"], clip_norm=1.0, noise_multiplier=1.5,
                                  rule=rule, trim_ratio=trim, generator=torch.Generator().manual_seed(7))
    assert out["w"].std().item() == pytest.approx(want_std, rel=0.03)
    assert abs(out["w"].mean().item()) < 0.01


def test_noise_is_reproducible_from_the_generator_seed():
    run = lambda seed: _agg(FIVE, "trimmed_mean", 0.2, z=2.0, generator=torch.Generator().manual_seed(seed))
    a, b, c = run(11), run(11), run(12)
    assert all(torch.equal(a[k], b[k]) for k in (W, B))
    assert not torch.equal(a[W], c[W])


def test_reported_example_counts_do_not_change_the_aggregate():
    # An attacker controls its reported num_examples; the rules must stay unweighted.
    inflated = [FIVE[0][:2] + (10**6,)] + FIVE[1:]
    for rule, trim in (("trimmed_mean", 0.2), ("mean", 0.0)):
        a, b = _agg(FIVE, rule, trim), _agg(inflated, rule, trim)
        assert all(torch.equal(a[k], b[k]) for k in (W, B))


def test_rejects_a_non_finite_update():
    with pytest.raises(ValueError):
        _agg(FIVE[:4] + [_client([float("nan"), 0.0], 0.0, cid="bad")], "trimmed_mean", 0.2)


@pytest.mark.parametrize("kwargs", [dict(z=-1.0), dict(rule="median"), dict(results=[]), dict(keys=())])
def test_rejects_invalid_arguments(kwargs):
    rule = kwargs.pop("rule", "mean")
    results = kwargs.pop("results", FIVE)
    with pytest.raises(ValueError):
        _agg(results, rule, **kwargs)


def test_missing_key_raises_key_error():
    with pytest.raises(KeyError):
        _agg(FIVE, "mean", keys=(W, "absent"))  # not in GLOBAL
    short = FIVE[:4] + [("c4", OrderedDict([(W, GLOBAL[W])]), 100)]  # client lacks B
    with pytest.raises(KeyError):
        _agg(short, "mean")
