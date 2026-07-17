"""Coordinator contract for a FAILED round: aggregate_fit returning None is non-fatal.

A strategy may declare a round failed by returning ``None`` from ``aggregate_fit`` (distinct from
a malformed input, which is rejected at ingress). When that happens the coordinator must NOT wedge:

  * the global model is left byte-for-byte UNTOUCHED (a failed aggregation cannot corrupt or wipe
    the last good global);
  * ``evaluate`` is NOT invoked on a non-existent aggregate (unpacking a None eval would crash the
    round inside the lock — the FR-22 bug class, but here on the aggregate-None branch);
  * ``latest_metrics`` is reset to None (no stale metrics leak as if the round succeeded);
  * the round counter advances exactly once and the completion event fires, so the server steps on
    to the next round instead of hanging or retrying the failed one forever;
  * the pending-update buffer is cleared, so nothing leaks into the next round.

Both dispatch triggers must honour this — the FedAvg trigger (``_trigger_aggregation_and_evaluation``)
and the DeComFL trigger (``_trigger_decomfl_aggregation_and_evaluation``). The DeComFL trigger has an
extra invariant: a failed round must NOT write ``gradient_history[round]`` (clients replay that history
to rebuild locally; recording an entry for a round whose model never advanced would desync them).

The existing coordinator suite covers evaluate()->None (FR-22) and malformed-input rejection
(FR-17/FR-5), but not aggregate_fit()->None itself; this pins that untested branch.
"""
from collections import OrderedDict
from unittest.mock import MagicMock

import torch

from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.strategy import Strategy


def make_params(val: float) -> OrderedDict:
    return OrderedDict([("w", torch.tensor([val]))])


def test_fedavg_failed_round_leaves_global_untouched_and_advances():
    """FedAvg trigger: aggregate_fit -> None must not touch the global, must skip evaluate, must
    advance the round, and must clear the pending buffer."""
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = None                 # strategy declares the round failed
    strategy.evaluate.return_value = (0.5, {"accuracy": 0.9})  # must NOT be consulted on a None aggregate

    coord = FLCoordinator(strategy=strategy, min_clients_for_aggregation=2, clients_per_round=2)
    good_global = make_params(3.14)
    coord.set_initial_parameters(good_global)

    # Two distinct clients complete the round -> aggregation fires and returns None.
    coord.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
    coord.submit_client_update("c2", make_params(2.0), 100, trained_on_round=1)

    # Global model is the SAME object, byte-for-byte unchanged: a failed round cannot corrupt it.
    assert coord.get_global_model_params() is good_global
    assert torch.equal(coord.get_global_model_params()["w"], torch.tensor([3.14]))

    strategy.aggregate_fit.assert_called_once()
    strategy.evaluate.assert_not_called()          # never evaluate a non-existent aggregate
    assert coord.get_latest_metrics() is None       # no stale/fabricated metrics on a failed round
    assert coord.current_round == 2                  # round advanced exactly once (server steps on)
    assert coord._round_complete_event.is_set()     # waiting main loop released, not hung
    assert coord._client_updates_received == []      # pending buffer cleared, no leak into next round


def test_decomfl_failed_round_untouched_global_and_no_gradient_history():
    """DeComFL trigger: a failed round must leave the global untouched, advance the round, and — the
    DeComFL-specific invariant — NOT record gradient_history for that round (clients replay it)."""
    strat = DeComFL(
        OrderedDict({"w": torch.zeros(3)}), evaluate_fn=None, min_fit_clients=1,
        clients_per_round=1, num_local_steps=1, num_perturbations=2,
        learning_rate=0.01, smoothing_param=0.001, seed=1,
    )
    # Force the failed-round branch while keeping a real DeComFL (so gradient_history / K / P exist).
    strat.aggregate_fit = MagicMock(return_value=None)

    coord = FLCoordinator(strategy=strat, min_clients_for_aggregation=1, clients_per_round=1)
    good_global = OrderedDict({"w": torch.zeros(3)})
    coord.set_initial_parameters(good_global)

    # A single 1x2 (K x P) submission completes the DeComFL round -> aggregation fires, returns None.
    coord.submit_decomfl_update("c1", [[0.1, 0.2]], 100, coord.current_round)

    assert coord.get_global_model_params() is good_global       # global untouched on a failed round
    assert torch.equal(coord.get_global_model_params()["w"], torch.zeros(3))
    strat.aggregate_fit.assert_called_once()
    assert 1 not in strat.gradient_history                      # NO history written for the failed round
    assert coord.get_latest_metrics() is None
    assert coord.current_round == 2                             # round advanced
    assert coord._round_complete_event.is_set()
    assert coord._client_updates_received == []                 # pending buffer cleared
