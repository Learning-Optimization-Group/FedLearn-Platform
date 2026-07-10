"""
Lifecycle tests for the DeComFL client end-of-run handling.

Covers the durable fix for the "retry-loop-on-completion" wart: when a DeComFL
run finishes and the server tears its RPCs down, the client must recognise the
terminal condition and exit cleanly (success) instead of spinning forever on the
post-completion CANCELLED / UNAVAILABLE. A genuine mid-run disconnect must still
be treated as a disconnect (bounded, no infinite loop), NOT as a clean completion.

The client-loop tests drive ``start_decomfl_client`` with a scripted test-double
for the gRPC transport (patched in for the real ``GrpcClient``), so the loop's
decision logic is exercised without a live server. The server-side tests exercise
the real coordinator / servicer terminal-state reporting the client relies on.
"""

from collections import OrderedDict
from unittest.mock import MagicMock

import grpc
import pytest
import torch

import fedlearn.client.decomfl_start as ds
from fedlearn.client.decomfl_start import start_decomfl_client
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.strategy import Strategy
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.communication.generated import fedlearn_pb2


# Terminal-outcome contract of start_decomfl_client (kept as literals here so the
# RED run fails on behaviour, not on an import of a not-yet-defined symbol).
COMPLETED = "completed"
DISCONNECTED = "disconnected"
ERROR = "error"

# Safety valve: if the client ever retry-loops, the double aborts rather than
# hanging the test. Must exceed the client's bounded-retry budget.
_LOOP_GUARD = 8


class _FakeRpcError(grpc.RpcError):
    """A gRPC RpcError with a controllable status code / details, for tests."""

    def __init__(self, code, details="fake failure"):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _FakeClient:
    """Minimal DeComFLClient stand-in; only the lifecycle hooks are exercised."""

    def __init__(self):
        self.grpc_client = None

    def set_grpc_client(self, c):
        self.grpc_client = c

    def fit(self, parameters, config):  # not reached in terminal/error scenarios
        return [[0.0]], 1

    def load_global_model(self, params):  # FR-1 initial sync; no-op for the lifecycle doubles
        pass


class _FakeComm:
    """
    Test-double for GrpcClient. ``config_action`` is a zero-arg callable invoked
    on every get_decomfl_config() call (it returns a config tuple or raises).
    ``complete`` is what the server-status probe reports.
    """

    def __init__(self, config_action, complete=False, register_ok=True):
        self._config_action = config_action
        self._complete = complete
        self._register_ok = register_ok
        self.config_calls = 0
        self.completion_probed = False
        self.current_round = 0
        self.heartbeat_started = False
        self.heartbeat_stopped = False
        self.closed = False
        self.statuses = []

    # --- lifecycle ---
    def register(self):
        return self._register_ok

    def get_global_model(self):
        # FR-1 initial global-model sync — return a non-empty global so the loop proceeds to the
        # round logic the lifecycle tests actually exercise.
        return (OrderedDict([("w", torch.tensor([0.0]))]), None, None)

    def should_stop_training(self):
        # FR-10 heartbeat-driven stop; the lifecycle tests drive completion via the config RPC, not this.
        return False

    def start_heartbeat(self):
        self.heartbeat_started = True

    def stop_heartbeat(self):
        self.heartbeat_stopped = True

    def close(self):
        self.closed = True

    def update_status(self, status, step, total):
        self.statuses.append(status)

    # --- round loop ---
    def get_decomfl_config(self):
        self.config_calls += 1
        if self.config_calls > _LOOP_GUARD:
            raise RuntimeError("loop-guard: client retried far too many times")
        return self._config_action()

    def submit_gradient_scalars(self, grads, num_examples, round_num):
        return True

    # --- terminal-state probe (the durable signal) ---
    def server_reports_complete(self):
        self.completion_probed = True
        return self._complete


@pytest.fixture(autouse=True)
def _fast_and_isolated(monkeypatch):
    """Never really sleep, and never construct a real gRPC channel."""
    monkeypatch.setattr(ds.time, "sleep", lambda *a, **k: None)


def _run(comm):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ds, "GrpcClient", lambda client_id, server_address: comm)
    try:
        return start_decomfl_client("addr:1", _FakeClient(), "c0")
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# Client round-loop: run completion -> clean exit
# ---------------------------------------------------------------------------
class TestClientCompletionExit:

    def test_minus_one_sentinel_exits_cleanly(self):
        # Server still alive, reports "no more rounds" via current_round == -1.
        comm = _FakeComm(lambda: (-1, [], [], {}))
        outcome = _run(comm)
        assert outcome == COMPLETED
        assert comm.config_calls == 1
        assert comm.heartbeat_stopped and comm.closed

    def test_cancelled_then_status_probe_complete_exits_cleanly(self):
        # End-of-run teardown surfaces as CANCELLED on the config RPC, but the
        # status probe reports the run is complete -> clean success, no retry.
        def action():
            raise _FakeRpcError(grpc.StatusCode.CANCELLED)

        comm = _FakeComm(action, complete=True)
        outcome = _run(comm)
        assert outcome == COMPLETED
        assert comm.completion_probed is True
        assert comm.config_calls == 1  # broke immediately; did NOT retry-loop


# ---------------------------------------------------------------------------
# Client round-loop: genuine mid-run trouble is NOT mistaken for completion
# ---------------------------------------------------------------------------
class TestClientDisconnectHandling:

    def test_midrun_unavailable_is_disconnect_not_completion(self):
        def action():
            raise _FakeRpcError(grpc.StatusCode.UNAVAILABLE)

        comm = _FakeComm(action, complete=False)
        outcome = _run(comm)
        assert outcome == DISCONNECTED
        assert comm.config_calls == 1  # no infinite loop

    def test_midrun_cancelled_without_completion_is_disconnect(self):
        # CANCELLED but the server does NOT confirm completion (e.g. crash):
        # must stop and report a disconnect, never a false success.
        def action():
            raise _FakeRpcError(grpc.StatusCode.CANCELLED)

        comm = _FakeComm(action, complete=False)
        outcome = _run(comm)
        assert outcome == DISCONNECTED
        assert comm.completion_probed is True
        assert comm.config_calls <= _LOOP_GUARD  # bounded, not infinite

    def test_transient_error_retries_bounded_then_gives_up(self):
        # A non-terminal transient error (server hiccup) with no completion:
        # the client rejoins a bounded number of times, then shuts down.
        def action():
            raise _FakeRpcError(grpc.StatusCode.INTERNAL)

        comm = _FakeComm(action, complete=False)
        outcome = _run(comm)
        assert outcome == DISCONNECTED
        assert 2 <= comm.config_calls <= _LOOP_GUARD  # retried, but bounded

    def test_register_failure_returns_error(self):
        comm = _FakeComm(lambda: (-1, [], [], {}), register_ok=False)
        outcome = _run(comm)
        assert outcome == ERROR
        assert comm.config_calls == 0


# ---------------------------------------------------------------------------
# Server-side terminal-state reporting the client depends on
# ---------------------------------------------------------------------------
def _mock_strategy():
    s = MagicMock(spec=Strategy)
    s.aggregate_fit.return_value = OrderedDict([("w", torch.tensor([1.0]))])
    s.evaluate.return_value = (0.5, {"accuracy": 0.9})
    return s


def _coordinator():
    return FLCoordinator(
        strategy=_mock_strategy(),
        min_clients_for_aggregation=1,
        clients_per_round=1,
    )


class TestCoordinatorTerminalState:

    def test_status_reports_not_complete_initially(self):
        c = _coordinator()
        assert c.get_server_status().get("training_complete") is False

    def test_mark_training_complete_sets_flag_and_stop(self):
        c = _coordinator()
        c.mark_training_complete()
        st = c.get_server_status()
        assert st.get("training_complete") is True
        # Drives the existing GetDeComFLConfig -> -1 sentinel path.
        assert c.stop_requested is True


class TestServerStatusRpc:

    def test_reports_training_complete_after_finish(self):
        coord = _coordinator()
        servicer = FederatedLearningServiceServicer(coord)
        ctx = MagicMock()
        req = fedlearn_pb2.GetServerStatusRequest()
        States = fedlearn_pb2.GetServerStatusResponse.ServerState

        before = servicer.GetServerStatus(req, ctx)
        assert before.server_state != States.TRAINING_COMPLETE

        coord.mark_training_complete()
        after = servicer.GetServerStatus(req, ctx)
        assert after.server_state == States.TRAINING_COMPLETE


class TestServerMarksCompletion:

    def test_start_server_marks_completion_on_normal_finish(self, monkeypatch):
        import fedlearn.server.server as server_mod

        fake_server = MagicMock()
        monkeypatch.setattr(server_mod.grpc, "server", lambda *a, **k: fake_server)
        monkeypatch.setattr(server_mod.time, "sleep", lambda *a, **k: None)
        # Don't actually block waiting for clients.
        monkeypatch.setattr(
            server_mod.FLCoordinator, "wait_for_round_to_complete", lambda self: None
        )
        marked = []
        monkeypatch.setattr(
            server_mod.FLCoordinator,
            "mark_training_complete",
            lambda self: marked.append(True),
            raising=False,
        )

        strategy = MagicMock()
        strategy.min_fit_clients = 1
        strategy.clients_per_round = 1
        strategy.initial_parameters = OrderedDict([("w", torch.tensor([0.0]))])
        strategy.aggregate_fit.return_value = None
        strategy.evaluate.return_value = (0.0, {})

        server_mod.start_server(
            "[::]:0", server_mod.ServerConfig(num_rounds=1), strategy
        )

        assert marked == [True]
