"""Slow evaluation must not hold client upload acknowledgments or heartbeats."""
from collections import OrderedDict
import threading
from unittest.mock import MagicMock
import torch
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.strategy import Strategy


def params(value):
    return OrderedDict(w=torch.tensor([value], dtype=torch.float32))


def setup_blocked_evaluation():
    entered, release = threading.Event(), threading.Event()
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = params(3.)
    def evaluate(*args):
        entered.set()
        assert release.wait(5), 'test did not release evaluator'
        return .4, {'accuracy': .7}
    strategy.evaluate.side_effect = evaluate
    c = FLCoordinator(strategy, 2, 2, num_rounds=2, async_round_completion=True)
    c.set_initial_parameters(params(0.))
    return c, strategy, entered, release


def test_upload_ack_and_identity_heartbeat_remain_responsive_during_evaluation():
    c, strategy, entered, release = setup_blocked_evaluation()
    try:
        assert c.submit_client_update('a', params(1.), 2, 1)
        assert c.submit_client_update('b', params(5.), 2, 1)
        assert entered.wait(2)
        assert c.current_round == 1
        assert c.bind_or_check_identity(0, 'a')
        assert c.update_client_heartbeat('a', 'TRAINING', 0, 0, 1)[:2] == (True, False)
        assert not c.submit_client_update('b', params(99.), 2, 1)
        assert not c.submit_client_update('c', params(99.), 2, 1)
        c.resolve_round_incomplete('dropout timeout while evaluating')
        assert not c.stop_requested
        assert strategy.aggregate_fit.call_count == 1
    finally:
        release.set()
    assert c._round_complete_event.wait(2)
    assert c.current_round == 2
    assert c.get_latest_metrics() == {'loss': .4, 'accuracy': .7}
    assert torch.equal(c.get_global_model_params()['w'], params(3.)['w'])


def test_stop_during_evaluation_does_not_publish_or_advance_a_late_result():
    c, strategy, entered, release = setup_blocked_evaluation()
    try:
        c.submit_client_update('a', params(1.), 2, 1)
        c.submit_client_update('b', params(5.), 2, 1)
        assert entered.wait(2)
        c.signal_stop()
        assert c._round_complete_event.is_set()
    finally:
        release.set()
    c._aggregation_worker.join(2)
    assert not c._aggregation_worker.is_alive()
    assert c.current_round == 1
    assert torch.equal(c.get_global_model_params()['w'], params(0.)['w'])


def test_evaluation_failure_is_a_terminal_round_failure():
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = params(1.)
    strategy.evaluate.side_effect = RuntimeError('evaluation broken')
    c = FLCoordinator(strategy, 1, 1, async_round_completion=True)
    c.set_initial_parameters(params(0.))
    assert c.submit_client_update('a', params(1.), 2, 1)
    assert c._round_complete_event.wait(2)
    assert c.stop_requested and c.last_round_failed
    assert not c.training_complete
    assert c.current_round == 1
    assert 'evaluation' in c.last_round_message


def test_network_server_does_not_publish_a_failed_worker_as_training_complete(monkeypatch):
    from fedlearn.server import server as network
    strategy = MagicMock(spec=Strategy)
    strategy.initial_parameters = params(0.)
    c = FLCoordinator(strategy, 1, 1)
    c.stop_requested = True
    c.last_round_failed = True
    monkeypatch.setattr(c, 'wait_for_round_to_complete', lambda: None)
    monkeypatch.setattr(network, 'build_coordinator', lambda *_: c)
    monkeypatch.setattr(network.grpc, 'server', lambda *a, **kw: MagicMock())
    monkeypatch.setenv('FEDLEARN_GRPC_USE_TLS', '0')
    monkeypatch.setenv('FEDLEARN_REQUIRE_CLIENT_AUTH', '0')
    monkeypatch.setenv('FEDLEARN_COMPLETION_DRAIN_SECONDS', '0')
    history, final = network.start_server('localhost:0', network.ServerConfig(num_rounds=1), strategy)
    assert not c.training_complete
    assert history == [] and final == {}


def test_next_round_early_upload_survives_main_loop_start_round():
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = params(3.)
    strategy.evaluate.return_value = None
    c = FLCoordinator(strategy, 2, 2, async_round_completion=True)
    c.set_initial_parameters(params(0.))
    c.start_round()
    c.submit_client_update('a', params(1.), 2, 1)
    c.submit_client_update('b', params(5.), 2, 1)
    assert c._round_complete_event.wait(2)
    assert c.current_round == 2
    assert c.submit_client_update('a', params(2.), 2, 2)
    c.start_round()  # network clients can arrive before the main loop resumes
    assert len(c._client_updates_received) == 1
    assert not c._round_complete_event.is_set()


def test_completed_round_metrics_survive_clients_advancing_before_main_loop():
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = params(3.)
    strategy.evaluate.side_effect = lambda round_, _: (.1 * round_, {'round': round_})
    c = FLCoordinator(strategy, 2, 2, async_round_completion=True)
    c.set_initial_parameters(params(0.))
    c.start_round(1)
    for round_ in (1, 2):
        c.submit_client_update('a', params(1.), 2, round_)
        c.submit_client_update('b', params(5.), 2, round_)
        c._aggregation_worker.join(2)
        assert not c._aggregation_worker.is_alive()
    c.start_round(2)
    assert c._round_complete_event.is_set()
    assert c.get_latest_metrics(1)['round'] == 1
    assert c.get_latest_metrics(2)['round'] == 2
