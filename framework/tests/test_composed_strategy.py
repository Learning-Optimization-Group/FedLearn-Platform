"""ComposedStrategy: bit-exact with dp_robust_aggregate; float32-equivalent to FedAvg, FedOpt and
RobustAggregator (it aggregates deltas, they aggregate absolute parameters — translation
equivariance makes the two identical in exact arithmetic but not bit for bit)."""
from collections import OrderedDict

import pytest
import torch

from fedlearn.privacy.dp_mechanism import dp_robust_aggregate
from fedlearn.server.composed_strategy import CENTRAL_DP_RULES, RULES, ComposedStrategy
from fedlearn.server.robust_aggregation import RobustAggregator
from fedlearn.server.strategy import FedAvg, FedOpt
from fedlearn.server.strategy_factory import create_strategy

KEYS = ("w", "b")


def _global():
    return OrderedDict(w=torch.tensor([[0.5, -0.5, 1.0]]), b=torch.tensor([0.1]))


def _results(seed, n=10, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        p = OrderedDict((k, v + scale * torch.randn(v.shape, generator=g)) for k, v in _global().items())
        out.append((f"c{i}", p, 10 + i))
    return out


def _copy(results):
    return [(c, OrderedDict((k, v.clone()) for k, v in p.items()), n) for c, p, n in results]


def test_rule_tables_match_the_config_module():
    from fedlearn.simulation import config
    assert RULES == config.RULES and CENTRAL_DP_RULES == config.CENTRAL_DP_RULES


@pytest.mark.parametrize("rule, trim", [("mean", 0.0), ("trimmed_mean", 0.2)])
@pytest.mark.parametrize("z", [0.0, 1.3])
def test_bit_exact_with_dp_robust_aggregate_over_three_rounds(rule, trim, z):
    s = ComposedStrategy(_global(), rule=rule, clip_norm=0.4, rule_params={"trim_ratio": trim},
                         central_noise_multiplier=z, noise_seed=7)
    gen = torch.Generator().manual_seed(7)
    ref = _global()
    for r in range(3):
        res = _results(r)
        want = dp_robust_aggregate(_copy(res), ref, list(KEYS), clip_norm=0.4, noise_multiplier=z,
                                   rule=rule, trim_ratio=trim, generator=gen)
        got = s.aggregate_fit(r, _copy(res))
        assert all(torch.equal(got[k], want[k]) for k in KEYS)
        ref = want


def test_weighted_mean_without_clip_matches_fedavg():
    s, f = ComposedStrategy(_global(), rule="weighted_mean"), FedAvg(_global())
    res = _results(1)
    got, want = s.aggregate_fit(0, _copy(res)), f.aggregate_fit(0, _copy(res))
    assert all(torch.allclose(got[k], want[k], atol=1e-6) for k in KEYS)


@pytest.mark.parametrize("variant", ["adam", "yogi"])
def test_weighted_mean_with_optimizer_matches_fedopt(variant):
    s = ComposedStrategy(_global(), rule="weighted_mean", optimizer=variant, optimizer_lr=0.1)
    f = FedOpt(_global(), server_learning_rate=0.1, variant=variant)
    for r in range(3):
        res = _results(10 + r)
        got, want = s.aggregate_fit(r, _copy(res)), f.aggregate_fit(r, _copy(res))
        assert all(torch.allclose(got[k], want[k], atol=1e-5) for k in KEYS)


@pytest.mark.parametrize("rule, kw", [
    ("median", {}), ("trimmed_mean", {"trim_ratio": 0.2}),
    ("krum", {"byzantine_fraction": 0.2}), ("multi_krum", {"byzantine_fraction": 0.2}),
    ("bulyan", {"byzantine_fraction": 0.1}), ("centered_clip", {"tau": 0.3}),
])
def test_robust_rules_match_robust_aggregator(rule, kw):
    s = ComposedStrategy(_global(), rule=rule, clip_norm=0.5, rule_params=kw)
    ra = RobustAggregator(_global(), method=rule, clip_norm=0.5,
                          trim_ratio=kw.get("trim_ratio", 0.1),
                          byzantine_fraction=kw.get("byzantine_fraction", 0.0),
                          centered_clip_tau=kw.get("tau", 1.0))
    res = _results(3, n=12)
    got, want = s.aggregate_fit(0, _copy(res)), ra.aggregate_fit(0, _copy(res))
    assert all(torch.allclose(got[k], want[k], atol=1e-6) for k in KEYS)


@pytest.mark.parametrize("rule", ["median", "krum", "weighted_mean"])
def test_central_noise_rejects_rules_without_a_bound(rule):
    with pytest.raises(ValueError):
        ComposedStrategy(_global(), rule=rule, clip_norm=0.4, central_noise_multiplier=1.0)


def test_central_noise_requires_a_clip_norm():
    with pytest.raises(ValueError):
        ComposedStrategy(_global(), rule="mean", central_noise_multiplier=1.0)


def test_stats_report_clipping_noise_and_kept_share():
    s = ComposedStrategy(_global(), rule="trimmed_mean", clip_norm=0.1, rule_params={"trim_ratio": 0.2},
                         central_noise_multiplier=2.0, noise_seed=1)
    s.aggregate_fit(0, _results(5))
    st = s.last_round_stats
    assert st["aggregated"] == 10 and st["clip_rate"] == 1.0          # every update exceeds 0.1
    assert 0.0 < st["mean_shrink"] < 1.0
    assert st["sensitivity"] == pytest.approx(2 * 0.1 / 6)
    assert st["noise_std"] == pytest.approx(2.0 * 2 * 0.1 / 6)
    assert set(st["kept_share"]) == {f"c{i}" for i in range(10)}
    assert sum(st["kept_share"].values()) == pytest.approx(6.0)        # 6 of 10 kept per coordinate
    assert set(st["timings"]) == {"clip", "rule", "noise", "optimizer"}


def test_a_non_finite_client_is_dropped_and_recorded():
    s = ComposedStrategy(_global(), rule="mean", clip_norm=0.4)
    res = _results(2)
    res[0][1]["w"][0, 0] = float("nan")
    s.aggregate_fit(0, res)
    assert s.last_round_stats["dropped"] == ["c0"] and s.last_round_stats["aggregated"] == 9


def test_a_rule_precondition_is_a_recorded_refusal_not_a_crash():
    # Krum with f = int(0.5 * 4) = 2 needs n >= 7; four clients cannot satisfy it.
    s = ComposedStrategy(_global(), rule="krum", clip_norm=0.4, rule_params={"byzantine_fraction": 0.5})
    assert s.aggregate_fit(0, _results(1, n=4)) is None
    st = s.last_round_stats
    assert st["failed"] is True and "krum" in st["message"] and "2f + 3" in st["message"]
    assert s.preview(_results(1, n=4)) is None
    # the global is untouched, so the next valid round aggregates against the initial model
    t = ComposedStrategy(_global(), rule="krum", clip_norm=0.4, rule_params={"byzantine_fraction": 0.1})
    s.rule_params["byzantine_fraction"] = 0.1
    a, b = s.aggregate_fit(1, _results(2)), t.aggregate_fit(1, _results(2))
    assert all(torch.equal(a[k], b[k]) for k in KEYS)


def test_preview_does_not_change_state_or_draw_noise():
    s = ComposedStrategy(_global(), rule="mean", clip_norm=0.4, central_noise_multiplier=1.0, noise_seed=3)
    res = _results(4)
    s.preview(_copy(res))
    t = ComposedStrategy(_global(), rule="mean", clip_norm=0.4, central_noise_multiplier=1.0, noise_seed=3)
    a, b = s.aggregate_fit(0, _copy(res)), t.aggregate_fit(0, _copy(res))
    assert all(torch.equal(a[k], b[k]) for k in KEYS)


def test_registered_in_the_strategy_factory():
    assert isinstance(create_strategy("composed", initial_parameters=_global()), ComposedStrategy)


def test_norm_band_drops_out_of_band_uploads_before_aggregating():
    glob = _global()
    near = OrderedDict((k, v + 0.5) for k, v in glob.items())        # delta norm 0.5 * sqrt(4) = 1.0
    far = OrderedDict((k, v + 5.0) for k, v in glob.items())         # delta norm 10.0
    s = ComposedStrategy(glob, rule="mean", norm_band=(0.5, 2.0))
    out = s.aggregate_fit(1, [("a", near, 1), ("b", far, 1)])
    assert s.last_round_stats["filtered"] == ["b"] and s.last_round_stats["aggregated"] == 1
    assert all(torch.allclose(out[k], near[k]) for k in KEYS)


def test_norm_band_must_be_ordered_and_non_negative():
    for band in [(2.0, 1.0), (-1.0, 1.0)]:
        with pytest.raises(ValueError, match="norm_band"):
            ComposedStrategy(_global(), norm_band=band)
