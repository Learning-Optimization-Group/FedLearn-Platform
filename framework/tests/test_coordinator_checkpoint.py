"""A model becomes available for the next round only after durable commit."""
from collections import OrderedDict
import hashlib
import threading

import pytest
import torch

from fedlearn.server.checkpoint import CheckpointIdentity, RoundCheckpointStore
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.server import ServerConfig, build_coordinator
from fedlearn.server.strategy import FedAvg


def parameters(value=0.0):
    return OrderedDict(w=torch.tensor([value], dtype=torch.float32))


def store_at(path):
    return RoundCheckpointStore(path, CheckpointIdentity("run-1", hashlib.sha256(b"config").hexdigest()))


def coordinator(store, *, asynchronous=True, rounds=3):
    strategy = FedAvg(parameters(), evaluate_fn=lambda round_, state: (.1 * round_, {"round": round_}),
                      min_fit_clients=2, clients_per_round=2)
    result = FLCoordinator(strategy, 2, 2, num_rounds=rounds,
                           async_round_completion=asynchronous, checkpoint_store=store)
    result.set_initial_parameters(parameters())
    return result


def complete_round(result, round_=1):
    assert result.submit_client_update("a", parameters(round_), 2, round_)
    assert result.submit_client_update("b", parameters(round_ + 2), 2, round_)
    if result._aggregation_worker is not None:
        result._aggregation_worker.join(3)
        assert not result._aggregation_worker.is_alive()


@pytest.mark.parametrize("asynchronous", [True, False])
def test_round_commit_contains_model_metrics_and_identity_bindings(tmp_path, asynchronous):
    store = store_at(tmp_path / "private")
    result = coordinator(store, asynchronous=asynchronous)
    assert result.bind_or_check_identity(0, "a")
    assert result.bind_or_check_identity(1, "b")
    complete_round(result)
    saved = store.load_latest()
    assert saved.completed_round == 1
    assert saved.history == [(1, {"loss": .1, "round": 1})]
    assert saved.partition_to_client == {0: "a", 1: "b"}
    assert torch.equal(saved.parameters["w"], parameters(2)["w"])
    assert result.current_round == 2


@pytest.mark.parametrize("asynchronous", [True, False])
def test_disk_failure_does_not_publish_next_round(tmp_path, monkeypatch, asynchronous):
    store = store_at(tmp_path / "private")
    result = coordinator(store, asynchronous=asynchronous)

    def failed_write(snapshot):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", failed_write)
    complete_round(result)
    assert result.stop_requested
    assert result.last_round_failed
    assert result.current_round == 1
    assert torch.equal(result.get_global_model_params()["w"], parameters()["w"])
    assert store.load_latest() is None
    assert not result.submit_client_update("a", parameters(99), 2, 2)


@pytest.mark.parametrize("asynchronous", [True, False])
def test_stop_during_storage_does_not_publish_late_model(tmp_path, monkeypatch, asynchronous):
    store = store_at(tmp_path / "private")
    result = coordinator(store, asynchronous=asynchronous)
    entered, release = threading.Event(), threading.Event()
    real_save = store.save

    def blocked_write(snapshot):
        entered.set()
        assert release.wait(4), "test did not release checkpoint writer"
        return real_save(snapshot)

    monkeypatch.setattr(store, "save", blocked_write)
    submission = threading.Thread(target=lambda: result.submit_client_update("b", parameters(3), 2, 1))
    try:
        result.submit_client_update("a", parameters(1), 2, 1)
        submission.start()
        assert entered.wait(2)
        assert result.current_round == 1
        if asynchronous:
            assert result.update_client_heartbeat("a", "WAITING", 1, 1, 1)[:2] == (True, False)
            assert not result.submit_client_update("c", parameters(7), 2, 1)
            assert result.get_global_model_for_client()[1] == 1
        result.signal_stop()
        assert result._round_complete_event.is_set()
    finally:
        release.set()
        submission.join(3)
        assert not submission.is_alive()
        if result._aggregation_worker is not None:
            result._aggregation_worker.join(3)
    assert result.current_round == 1
    assert torch.equal(result.get_global_model_params()["w"], parameters()["w"])


@pytest.mark.parametrize("asynchronous", [True, False])
def test_empty_aggregate_cannot_advance_a_checkpointed_run(tmp_path, monkeypatch, asynchronous):
    store = store_at(tmp_path / "private")
    result = coordinator(store, asynchronous=asynchronous)
    monkeypatch.setattr(result.strategy, "aggregate_fit", lambda *args: None)
    complete_round(result)
    assert result.stop_requested and result.last_round_failed
    assert result.current_round == 1
    assert store.load_latest() is None


def test_restored_coordinator_rejects_stale_retries_and_keeps_history(tmp_path):
    store = store_at(tmp_path / "private")
    original = coordinator(store)
    original.bind_or_check_identity(0, "a")
    complete_round(original)
    restored = coordinator(store)
    restored.restore_checkpoint(store.load_latest())
    assert restored.current_round == 2
    assert restored.get_latest_metrics(1) == {"loss": .1, "round": 1}
    assert not restored.bind_or_check_identity(0, "impostor")
    assert not restored.submit_client_update("a", parameters(9), 2, 1)
    complete_round(restored, 2)
    assert store.load_latest().completed_round == 2
    assert [number for number, _ in store.load_latest().history] == [1, 2]


def test_restore_refuses_a_changed_round_budget_without_mutation(tmp_path):
    store = store_at(tmp_path / "private")
    original = coordinator(store)
    complete_round(original)
    restored = coordinator(store, rounds=4)
    with pytest.raises(ValueError, match="round"):
        restored.restore_checkpoint(store.load_latest())
    assert restored.current_round == 1
    assert torch.equal(restored.get_global_model_params()["w"], parameters()["w"])


def test_server_configuration_requires_explicit_resume_of_existing_state(tmp_path):
    store = store_at(tmp_path / "private")
    original = coordinator(store)
    complete_round(original)
    strategy = FedAvg(parameters(), min_fit_clients=2, clients_per_round=2)
    config = ServerConfig(num_rounds=3, checkpoint_directory=str(store.directory),
                          checkpoint_identity=store.identity)
    with pytest.raises(ValueError, match="resume"):
        build_coordinator(strategy, config)
    config.resume_from_checkpoint = True
    restored = build_coordinator(strategy, config)
    assert restored.current_round == 2
    assert restored.get_latest_metrics(1)["round"] == 1


def test_server_refuses_missing_resume_checkpoint_before_listening(tmp_path):
    store = store_at(tmp_path / "private")
    config = ServerConfig(num_rounds=3, checkpoint_directory=str(store.directory),
                          checkpoint_identity=store.identity, resume_from_checkpoint=True)
    with pytest.raises(ValueError, match="no.*checkpoint"):
        build_coordinator(FedAvg(parameters()), config)


def test_model_and_round_cannot_change_after_stop_returns(tmp_path, monkeypatch):
    """Force a context switch at the model assignment, after the worker's Stop check."""
    store = store_at(tmp_path / "private")
    result = coordinator(store)
    entered, release, stop_started, stop_returned = (threading.Event() for _ in range(4))
    original_setattr = FLCoordinator.__setattr__
    state_at_stop = []

    def pause_model_assignment(instance, name, value):
        if instance is result and name == "_global_model_params":
            entered.set()
            assert release.wait(4), "test did not release model publication"
        original_setattr(instance, name, value)

    def stop():
        stop_started.set()
        result.signal_stop()
        state_at_stop.append((result.current_round, result._global_model_params["w"].clone()))
        stop_returned.set()

    monkeypatch.setattr(FLCoordinator, "__setattr__", pause_model_assignment)
    stopper = threading.Thread(target=stop)
    try:
        result.submit_client_update("a", parameters(1), 2, 1)
        result.submit_client_update("b", parameters(3), 2, 1)
        assert entered.wait(2)
        stopper.start()
        assert stop_started.wait(2)
        # Cancellation either finishes first (and prevents publication), or waits for
        # an already committed publication. It must not finish before a later mutation.
        stop_returned.wait(.2)
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(3)
        if result._aggregation_worker is not None:
            result._aggregation_worker.join(3)
    assert stop_returned.is_set()
    assert result.current_round == state_at_stop[0][0]
    assert torch.equal(result._global_model_params["w"], state_at_stop[0][1])
