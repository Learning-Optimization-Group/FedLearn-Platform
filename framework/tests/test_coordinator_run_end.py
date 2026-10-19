"""A run ends at its last round.

Observed live (2026-09-24, Robust, 4 clients): the final aggregation advanced the coordinator to round 4 a moment
before the server loop called mark_training_complete(). A laptop that fetched the global model in that gap was
handed "round 4", trained it and submitted it after completion; the coordinator appended it to a round that would
never aggregate, and the servicer logged "Coordinator accepted update". So: no round beyond the run is handed out,
an update arriving after the run is not counted, and submit_client_update says whether it counted an update so the
servicer's log is true.
"""
import logging
from collections import OrderedDict
from unittest.mock import MagicMock

import torch

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.communication.serializer import parameters_to_proto
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.server.strategy import Strategy


def _params(v):
    return OrderedDict([("w", torch.tensor([v]))])


def _coordinator(num_rounds, clients=1):
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = _params(1.0)
    strategy.evaluate.return_value = (0.5, {"accuracy": 0.9})
    c = FLCoordinator(strategy=strategy, min_clients_for_aggregation=clients, clients_per_round=clients,
                      num_rounds=num_rounds)
    c.set_initial_parameters(_params(0.0))
    return c


def test_no_round_is_handed_out_after_the_last_one():
    c = _coordinator(num_rounds=1)
    c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1)   # completes round 1 inline

    # Before the server loop gets to mark_training_complete(), the run is already over.
    assert c.get_global_model_for_client() == (None, -1, {})


def test_a_round_within_the_run_is_still_handed_out():
    c = _coordinator(num_rounds=2)
    c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1)

    params, current_round, _ = c.get_global_model_for_client()
    assert current_round == 2 and params is not None


def test_without_a_run_length_the_coordinator_keeps_handing_out_rounds():
    c = _coordinator(num_rounds=None)
    c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1)

    assert c.get_global_model_for_client()[1] == 2


def test_an_update_arriving_after_the_run_ended_is_not_counted():
    c = _coordinator(num_rounds=3, clients=2)
    c.mark_training_complete()

    assert c.submit_client_update("c1", _params(0.5), 8, trained_on_round=c.current_round) is False
    assert c._client_updates_received == []


def test_submit_reports_whether_it_counted_the_update():
    c = _coordinator(num_rounds=3, clients=2)
    assert c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1) is True
    assert c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1) is False   # duplicate
    assert c.submit_client_update("c2", _params(0.5), 8, trained_on_round=0) is False   # stale
    assert c.submit_client_update("c2", _params(0.5), 8, trained_on_round=5) is False   # ahead
    assert c.submit_client_update("c2", _params(0.5), 0, trained_on_round=1) is False   # no examples
    assert len(c._client_updates_received) == 1


class _Context:
    def abort(self, code, details):
        raise AssertionError(f"unexpected abort {code}: {details}")


def _submit(coordinator_says, caplog):
    coordinator = MagicMock()
    coordinator.submit_client_update.return_value = coordinator_says
    servicer = FederatedLearningServiceServicer(coordinator)
    request = pb.SubmitModelUpdateRequest(client_id="c", trained_on_round=4,
                                          parameters=parameters_to_proto(_params(0.5), 8))
    with caplog.at_level(logging.INFO):
        servicer.SubmitModelUpdate(request, _Context())
    return caplog.text


def test_the_servicer_logs_an_update_as_accepted_only_when_it_was_counted(caplog):
    assert "Coordinator accepted update" in _submit(True, caplog)


def test_the_servicer_says_so_when_an_update_was_not_counted(caplog):
    text = _submit(False, caplog)
    assert "Coordinator accepted update" not in text
    assert "not counted" in text


def test_the_server_builds_its_coordinator_with_the_runs_length():
    from fedlearn.server.server import ServerConfig, build_coordinator
    strategy = MagicMock(spec=Strategy)
    strategy.min_fit_clients, strategy.clients_per_round = 2, 3

    coordinator = build_coordinator(strategy, ServerConfig(num_rounds=5))

    assert coordinator.num_rounds == 5
    assert (coordinator.min_clients, coordinator.clients_per_round) == (2, 3)


def test_status_reports_completion_as_soon_as_the_last_round_aggregated():
    """Clients decide whether to fetch another round from the status. In the moment before the server loop marks
    completion, it must already say the run is over, or a client asks for a round the download then refuses."""
    c = _coordinator(num_rounds=1)
    c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1)

    assert c.get_server_status()["training_complete"] is True


def test_an_update_for_the_round_after_the_last_is_not_counted_even_before_completion_is_marked():
    c = _coordinator(num_rounds=1)
    c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1)

    assert c.submit_client_update("c1", _params(0.5), 8, trained_on_round=c.current_round) is False
    assert c._client_updates_received == []


# --- the DeComFL path has the same window --------------------------------------------------------------------

def test_the_decomfl_config_is_not_handed_out_for_a_round_past_the_run(monkeypatch):
    from fedlearn.server.decomfl_strategy import DeComFL
    coordinator = MagicMock()
    coordinator.strategy = MagicMock(spec=DeComFL)
    coordinator.stop_requested = False
    coordinator.run_is_over.return_value = True
    servicer = FederatedLearningServiceServicer(coordinator)

    response = servicer.GetDeComFLConfig(pb.GetDeComFLConfigRequest(client_id="c"), _Context())

    assert response.current_round == -1


def test_run_is_over_once_the_last_round_aggregated_or_the_run_stopped():
    c = _coordinator(num_rounds=1)
    assert c.run_is_over() is False
    c.submit_client_update("c1", _params(0.5), 8, trained_on_round=1)
    assert c.run_is_over() is True

    stopped = _coordinator(num_rounds=3)
    stopped.signal_stop()
    assert stopped.run_is_over() is True


def test_a_decomfl_update_after_the_run_is_not_counted():
    c = _coordinator(num_rounds=3, clients=2)
    c.mark_training_complete()

    c.submit_decomfl_update("c1", [[0.1]], 8, trained_on_round=c.current_round)

    assert c._client_updates_received == []
