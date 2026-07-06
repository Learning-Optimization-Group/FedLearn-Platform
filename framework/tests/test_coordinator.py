import time
import pytest
import torch
from collections import OrderedDict
from unittest.mock import MagicMock, patch
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.strategy import Strategy


def make_params(val: float) -> OrderedDict:
    return OrderedDict([("w", torch.tensor([val]))])


def make_mock_strategy():
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = make_params(1.0)
    strategy.evaluate.return_value = (0.5, {"accuracy": 0.9})
    return strategy


class TestFLCoordinator:

    def setup_method(self):
        self.strategy = make_mock_strategy()
        # 2 clients needed per round
        self.coordinator = FLCoordinator(
            strategy=self.strategy,
            min_clients_for_aggregation=2,
            clients_per_round=2
        )
        self.coordinator.set_initial_parameters(make_params(0.0))

    def test_register_client_returns_true(self):
        result = self.coordinator.register_client("client-1")
        assert result is True

    def test_submit_client_update_stale_update_is_ignored(self):
        # coordinator is at round 1. Submitting an update for round 0 (stale).
        self.coordinator.submit_client_update("c1", make_params(1.0), 100, trained_on_round=0)
        assert len(self.coordinator._client_updates_received) == 0

    def test_submit_client_update_future_round_is_ignored(self):
        self.coordinator.submit_client_update("c1", make_params(1.0), 100, trained_on_round=5)
        assert len(self.coordinator._client_updates_received) == 0

    def test_submit_client_update_zero_examples_is_ignored(self):
        self.coordinator.submit_client_update("c1", make_params(1.0), 0, trained_on_round=1)
        assert len(self.coordinator._client_updates_received) == 0

    def test_submit_client_update_dedups_a_retried_submission(self):
        """FR-5: a retried FedAvg submit (ABORTED/UNAVAILABLE are client-retryable, so the
        server can see the same client's update twice in a round) must be counted ONCE. A
        double append both inflates that client's weight in the weighted average and can trip
        the clients_per_round aggregation trigger with fewer than N distinct clients. Mirrors
        the DeComFL submit-path dedup; first accepted update wins, duplicate is an idempotent
        no-op."""
        self.coordinator.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
        self.coordinator.submit_client_update("c1", make_params(2.0), 100, trained_on_round=1)
        # Counted once (first-write-wins), not twice.
        assert len(self.coordinator._client_updates_received) == 1
        ids = [cid for cid, _p, _n in self.coordinator._client_updates_received]
        assert ids == ["c1"]
        # And the duplicate did NOT prematurely trigger aggregation (needs 2 DISTINCT clients).
        self.strategy.aggregate_fit.assert_not_called()

    def test_distinct_clients_still_trigger_aggregation(self):
        """Complementary happy path: two DISTINCT clients complete the round and aggregation
        fires exactly once — dedup must not block legitimate distinct submissions."""
        self.coordinator.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
        self.coordinator.submit_client_update("c2", make_params(2.0), 100, trained_on_round=1)
        self.strategy.aggregate_fit.assert_called_once()

    def test_submit_client_update_caps_num_examples_at_max(self):
        # Submit one valid update - it should be stored (not trigger aggregation yet since 2 clients needed)
        self.coordinator.submit_client_update("c1", make_params(1.0), 200_000, trained_on_round=1)
        assert len(self.coordinator._client_updates_received) == 1
        # The stored num_examples should be capped. SE-3: entries are (client_id, params, num_examples).
        _, _, stored_count = self.coordinator._client_updates_received[0]
        assert stored_count <= FLCoordinator.MAX_NUM_EXAMPLES

    def test_submit_client_update_triggers_aggregation_when_all_clients_report(self):
        # First update: does NOT trigger aggregation (only 1 of 2)
        self.coordinator.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
        assert self.strategy.aggregate_fit.call_count == 0

        # Second update: SHOULD trigger aggregation
        self.coordinator.submit_client_update("c2", make_params(2.0), 100, trained_on_round=1)
        assert self.strategy.aggregate_fit.call_count == 1

    def test_aggregation_advances_round_counter(self):
        assert self.coordinator.current_round == 1
        self.coordinator.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
        self.coordinator.submit_client_update("c2", make_params(2.0), 100, trained_on_round=1)
        assert self.coordinator.current_round == 2

    def test_get_latest_metrics_after_aggregation(self):
        self.coordinator.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
        self.coordinator.submit_client_update("c2", make_params(2.0), 100, trained_on_round=1)
        metrics = self.coordinator.get_latest_metrics()
        assert metrics is not None
        assert "loss" in metrics
        assert metrics["loss"] == pytest.approx(0.5)

    def test_signal_stop_releases_waiting_thread(self):
        # start_round clears the event; signal_stop should set it
        self.coordinator.start_round()
        self.coordinator.signal_stop()
        assert self.coordinator.stop_requested is True
        # Event should be set so wait_for_round_to_complete returns quickly
        assert self.coordinator._round_complete_event.is_set()

    def test_update_client_heartbeat_stores_entry(self):
        ok, should_stop, msg = self.coordinator.update_client_heartbeat(
            "c1", "TRAINING", 10, 100, 1)
        assert ok is True
        assert should_stop is False
        assert "c1" in self.coordinator.client_heartbeats

    def test_is_client_alive_returns_true_for_recent_heartbeat(self):
        self.coordinator.update_client_heartbeat("c1", "TRAINING", 5, 100, 1)
        assert self.coordinator.is_client_alive("c1") is True

    def test_is_client_alive_returns_false_for_unknown_client(self):
        assert self.coordinator.is_client_alive("ghost") is False

    def test_get_active_clients_excludes_expired(self):
        # Inject a stale heartbeat manually
        self.coordinator.client_heartbeats["stale"] = {
            "status": "DONE", "current_step": 0, "total_steps": 0,
            "current_round": 1, "last_seen": time.time() - 9999
        }
        self.coordinator.update_client_heartbeat("fresh", "TRAINING", 1, 10, 1)
        active = self.coordinator.get_active_clients()
        assert "fresh" in active
        assert "stale" not in active
