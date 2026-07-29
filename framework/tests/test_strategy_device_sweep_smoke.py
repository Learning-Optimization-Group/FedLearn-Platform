"""Tests for the strategy x device sweep harness.

The harness exists to close a gap in a load-bearing claim: 'the CPU beats the GPU at frozen-head
scale' rested on CPU numbers for only two of six strategies. If the harness silently ran the same
aggregation for every named strategy, it would manufacture agreement rather than measure it, so the
tests below pin that each name produces a genuinely different aggregator.
"""
import os
import sys
from collections import OrderedDict

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import strategy_device_sweep as S  # noqa: E402


def _sd(v=1.0, n=4):
    return OrderedDict([("net.weight", torch.full((n,), v)), ("net.bias", torch.zeros(1))])


def test_every_named_strategy_constructs():
    for name in S.STRATEGIES:
        obj = S.build_strategy(name, _sd(), 10, 0.001, 1e-3)
        assert obj is not None, name


def test_unknown_strategy_fails_loudly():
    with pytest.raises(SystemExit):
        S.build_strategy("NotAStrategy", _sd(), 10, 0.001, 1e-3)


def test_strategies_are_not_all_the_same_class():
    """If these collapsed to one class the sweep would report five identical columns."""
    classes = {type(S.build_strategy(n, _sd(), 10, 0.001, 1e-3)).__name__ for n in S.STRATEGIES}
    assert len(classes) >= 3, f"expected distinct aggregators, got {classes}"


def _ups(n_honest, attacker_value=None):
    """Build a FRESH update list. Never reuse one across two aggregate_fit calls -- see
    test_aggregate_is_destructive_to_its_input."""
    u = [(_sd(1.0), 10) for _ in range(n_honest)]
    if attacker_value is not None:
        u.append((_sd(attacker_value), 10))
    return u


def test_aggregate_is_destructive_to_its_input():
    """FedAvgAggregator.aggregate() calls params.clear() on every client dict to free memory, so
    the update list is EMPTIED as a side effect. Reusing a list across two strategies silently
    yields an empty second aggregate. Benign in production (the coordinator passes results once)
    but a real footgun, and it broke the first version of the tests below."""
    ups = _ups(2)
    S.build_strategy("FedAvg", _sd(), 2, 0.001, 1e-3).aggregate_fit(1, ups)
    assert all(len(p) == 0 for p, _ in ups), "expected inputs to be cleared"


def test_robust_median_actually_differs_from_fedavg_under_an_outlier():
    """The load-bearing behavioural check. A coordinate-wise median must reject an outlier that a
    weighted mean absorbs -- otherwise 'Robust' is FedAvg wearing a different name and any CPU/GPU
    comparison between them is meaningless."""
    fa = S.build_strategy("FedAvg", _sd(), 5, 0.001, 1e-3).aggregate_fit(1, _ups(4, 1000.0))
    rb = S.build_strategy("Robust", _sd(), 5, 0.001, 1e-3).aggregate_fit(1, _ups(4, 1000.0))
    fa_w = float(fa["net.weight"][0])
    rb_w = float(rb["net.weight"][0])
    assert fa_w > 100.0, f"FedAvg should be dragged by the outlier, got {fa_w}"
    assert rb_w < 2.0, f"median should reject the outlier, got {rb_w}"


def test_trimmed_mean_also_rejects_the_outlier():
    rb = S.build_strategy("RobustTrimmed", _sd(), 9, 0.001, 1e-3).aggregate_fit(1, _ups(8, 1000.0))
    assert float(rb["net.weight"][0]) < 2.0


def test_fedprox_server_aggregation_matches_fedavg():
    """FedProx's mu is a CLIENT-side proximal term; its server aggregation is FedAvg by design.
    Pinning this means any CPU/GPU difference measured for FedProx isolates the client loop."""
    def mixed():
        return [(_sd(1.0), 10), (_sd(3.0), 30)]
    fa = S.build_strategy("FedAvg", _sd(), 2, 0.001, 1e-3).aggregate_fit(1, mixed())
    fp = S.build_strategy("FedProx", _sd(), 2, 0.001, 1e-3).aggregate_fit(1, mixed())
    assert torch.allclose(fa["net.weight"], fp["net.weight"], atol=1e-6)


def test_run_strategy_returns_the_expected_record_shape():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 8, generator=g)
    y = (x[:, 0] > 0).long()
    r = S.run_strategy("FedAvg", train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8,
                       n_classes=2, hidden=0, clients=2, clients_per_round=2, alpha=10.0,
                       rounds=4, local_epochs=1, lr=0.1, batch_size=16, seed=0,
                       device="cpu", eval_every=2)
    for k in ("strategy", "device", "d", "wall_seconds", "final_auc", "per_round"):
        assert k in r, k
    assert r["d"] == 8 * 2 + 2


def test_all_strategies_run_end_to_end_on_cpu():
    """Every strategy must complete a real run, not merely construct."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 8, generator=g)
    y = (x[:, 0] > 0).long()
    for name in S.STRATEGIES:
        r = S.run_strategy(name, train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8,
                           n_classes=2, hidden=0, clients=4, clients_per_round=4, alpha=10.0,
                           rounds=3, local_epochs=1, lr=0.1, batch_size=16, seed=0,
                           device="cpu", eval_every=3)
        assert r["per_round"], f"{name} produced no evaluations"


# ------------------------------------------------- FedProx must actually BE FedProx

def _task(n=128, feat=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, feat, generator=g)
    return x, (x[:, 0] + 0.5 * x[:, 1] > 0).long()


def test_fedprox_with_mu_differs_from_fedavg():
    """THE TEST WHOSE ABSENCE LET A WRONG RESULT SHIP.

    An earlier version of this harness used a generic SGD client loop and never applied the
    proximal term, so FedProx and FedAvg produced BIT-IDENTICAL results across three seeds and
    were reported as two independent strategies agreeing. They were the same strategy run twice.

    FedProx's mu is a CLIENT-side term: mu*(w - w_global) added to each gradient before the step
    (framework/src/fedlearn/client/local_trainer.py). With mu > 0 the trajectory must differ."""
    x, y = _task()
    kw = dict(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2, hidden=0,
              clients=4, clients_per_round=4, alpha=10.0, rounds=6, local_epochs=2,
              lr=0.1, batch_size=32, seed=0, device="cpu", eval_every=6)
    fa = S.run_strategy("FedAvg", **kw)
    fp = S.run_strategy("FedProx", proximal_mu=0.5, **kw)
    assert fa["final_auc"] != fp["final_auc"], (
        "FedProx with mu=0.5 produced an identical result to FedAvg -- the proximal term is "
        "not being applied by the client loop")


def test_fedprox_with_mu_zero_matches_fedavg_exactly():
    """The other half of the contract: mu=0 must reduce to plain local SGD, so FedProx at mu=0
    is FedAvg exactly. Without this, 'differs from FedAvg' could be satisfied by any bug."""
    x, y = _task()
    kw = dict(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2, hidden=0,
              clients=4, clients_per_round=4, alpha=10.0, rounds=6, local_epochs=2,
              lr=0.1, batch_size=32, seed=0, device="cpu", eval_every=6)
    fa = S.run_strategy("FedAvg", **kw)
    fp = S.run_strategy("FedProx", proximal_mu=0.0, **kw)
    assert fa["final_auc"] == fp["final_auc"]


def test_larger_mu_pulls_harder_toward_the_global_model():
    """A behavioural check on direction, not just difference. The proximal term penalises
    departure from the round's starting weights, so a larger mu must keep local models closer
    to the global model -- measurable as a smaller mean update norm per round."""
    x, y = _task()
    kw = dict(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2, hidden=0,
              clients=4, clients_per_round=4, alpha=10.0, rounds=8, local_epochs=3,
              lr=0.1, batch_size=32, seed=0, device="cpu", eval_every=8)
    lo = S.run_strategy("FedProx", proximal_mu=0.0, **kw)["mean_update_norm"]
    hi = S.run_strategy("FedProx", proximal_mu=5.0, **kw)["mean_update_norm"]
    assert hi < lo, f"expected mu=5.0 to restrain local drift; got {hi} vs {lo} at mu=0"
