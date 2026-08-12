"""P0-1c: a wall-clock-free seam for resolving an incomplete round.

Written before the implementation (TDD).

``FLCoordinator`` resolves a round in one of two ways. The happy path is inline: the submit
that brings the count to ``clients_per_round`` calls the aggregation trigger directly, so a
*full* round never touches a clock. The unhappy path — a client dropped out — is reachable
only from ``wait_for_round_to_complete()``, which polls on a 1-second tick until
``round_timeout_s`` (default 120s) elapses.

That second path is correct for a deployed server, where a missing client really is a
wall-clock event. It is unusable for simulation, where dropout is *modelled*: 1000 dropout
rounds at a 120-second deadline is 33 hours of sleeping, and the result would depend on
scheduler timing rather than on the seed.

So the timeout policy (*when* to give up) is separated from the resolution mechanism (*what to
do* about it). ``resolve_round_incomplete(reason)`` is the mechanism, callable directly;
``_handle_round_timeout()`` becomes the wall-clock-triggered caller. Deployed behaviour is
unchanged — these tests exist partly to prove that.
"""

from collections import OrderedDict

import torch

from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.strategy import FedAvg


def _params(v: float) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict([("w", torch.tensor([v, v], dtype=torch.float32))])


def _coordinator(clients_per_round=3, min_clients=1) -> FLCoordinator:
    strategy = FedAvg(
        initial_parameters=_params(0.0),
        min_fit_clients=min_clients,
        clients_per_round=clients_per_round,
    )
    coord = FLCoordinator(
        strategy=strategy,
        min_clients_for_aggregation=min_clients,
        clients_per_round=clients_per_round,
        round_timeout_s=120.0,
    )
    coord.set_initial_parameters(_params(0.0))
    coord.start_round()
    return coord


class TestResolveRoundIncomplete:
    def test_force_aggregates_the_clients_that_did_report(self):
        coord = _coordinator(clients_per_round=3, min_clients=1)
        coord.submit_client_update("c0", _params(1.0), num_examples=10, trained_on_round=1)
        coord.submit_client_update("c1", _params(3.0), num_examples=10, trained_on_round=1)

        assert coord.current_round == 1, "round must not advance on a partial cohort"

        coord.resolve_round_incomplete("simulated dropout")

        assert coord.current_round == 2, "round did not advance after force-resolution"
        # Equal weights, so the aggregate is the mean of the two that reported.
        got = coord.get_global_model_for_client()[0]["w"]
        assert torch.allclose(got, torch.tensor([2.0, 2.0]))
        assert coord.last_round_failed is True
        assert "simulated dropout" in (coord.last_round_message or "")

    def test_stops_the_run_when_too_few_reported(self):
        coord = _coordinator(clients_per_round=3, min_clients=2)
        coord.submit_client_update("c0", _params(1.0), num_examples=10, trained_on_round=1)

        coord.resolve_round_incomplete("simulated dropout")

        assert coord.stop_requested is True
        assert coord.last_round_failed is True

    def test_is_a_noop_when_the_round_already_completed(self):
        """A round completed inline must not be aggregated twice."""
        coord = _coordinator(clients_per_round=2, min_clients=1)
        coord.submit_client_update("c0", _params(1.0), num_examples=10, trained_on_round=1)
        coord.submit_client_update("c1", _params(3.0), num_examples=10, trained_on_round=1)
        assert coord.current_round == 2

        coord.resolve_round_incomplete("should be ignored")

        assert coord.current_round == 2, "double aggregation — the round advanced twice"
        assert coord.last_round_failed is False

    def test_does_not_consult_the_clock(self):
        """Resolution must be immediate regardless of round_timeout_s."""
        import time

        coord = _coordinator(clients_per_round=3, min_clients=1)
        coord.round_timeout_s = 10_000.0
        coord.submit_client_update("c0", _params(1.0), num_examples=10, trained_on_round=1)

        t0 = time.monotonic()
        coord.resolve_round_incomplete("simulated dropout")
        assert time.monotonic() - t0 < 1.0
        assert coord.current_round == 2

    def test_zero_reports_stops_rather_than_aggregating_nothing(self):
        """An empty cohort must never reach the strategy — FedAvg on [] wipes the global model."""
        coord = _coordinator(clients_per_round=3, min_clients=1)
        before = coord.get_global_model_for_client()[0]

        coord.resolve_round_incomplete("total dropout")

        assert coord.stop_requested is True
        after = coord.get_global_model_for_client()[0]
        # stop_requested makes the getter return None; the stored params must be intact.
        assert after is None
        assert coord._global_model_params is not None
        assert torch.equal(coord._global_model_params["w"], before["w"])


class TestTimeoutPathIsUnchanged:
    def test_timeout_handler_delegates_to_the_same_mechanism(self):
        """Deployed behaviour must be identical — the refactor moved code, not policy."""
        coord = _coordinator(clients_per_round=3, min_clients=1)
        coord.submit_client_update("c0", _params(2.0), num_examples=10, trained_on_round=1)

        coord._handle_round_timeout()

        assert coord.current_round == 2
        assert coord.last_round_failed is True
        msg = coord.last_round_message or ""
        assert "timed out" in msg, "the timeout path must still say it timed out"
        assert f"{coord.round_timeout_s:.1f}s" in msg
