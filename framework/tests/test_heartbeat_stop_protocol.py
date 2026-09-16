"""FR-10 — the server-driven heartbeat STOP protocol must actually stop a training client.

The heartbeat response has always carried ``should_stop``, but the wire was decorative end to end:

  * server side — ``FLCoordinator.update_client_heartbeat`` hardcoded ``should_stop = False``,
    so a stopped coordinator (``signal_stop()`` / quorum-lost timeout) never told any client;
  * client side — ``send_heartbeat`` read ``res.should_stop`` but ``_heartbeat_loop`` discarded
    the return, and no flag/Event existed for the fit loop to check, so even a True halted nothing.

These tests pin the real protocol: coordinator ``stop_requested`` -> HeartbeatResponse.should_stop
-> the client's ``_stop_training`` Event -> the K-local-steps fit loop breaks between steps. The
e2e test uses the REAL coordinator + REAL servicer over a live gRPC socket (no server-side mocks)
and the real ``GrpcClient`` heartbeat thread, preserving the two-stub contract: the training stub
is blocked inside fit() the whole time — only the parallel heartbeat stub carries the stop.

Also here: the dual-stub-shared status triple (``current_status``/``current_step``/``total_steps``)
was written by ``update_status`` with three bare attribute stores while the heartbeat thread read
them mid-write — a torn (status of one phase, step of another) heartbeat. The snapshot both sides
now use must be atomic.
"""
import concurrent.futures
import socket
import threading
import time
from collections import OrderedDict
from unittest.mock import Mock

import grpc
import pytest
import torch
import torch.nn as nn

from fedlearn.client.decomfl_client import DeComFLClient
from fedlearn.client.grpc_client import GrpcClient
from fedlearn.communication.generated import fedlearn_pb2_grpc as pbg
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer


class LogReg(nn.Module):
    """Linear(4, 3) toy model (same shape as test_decomfl_convergence; d=15, fast ZO steps)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):  # noqa: D401
        return self.fc(x)


class _SlowLoader:
    """Yields the whole toy batch each iteration, sleeping first so every local step has a
    measurable duration — a full K-step run takes many seconds, while a stop-aborted run must
    return almost immediately. Matches DeComFLClient.fit's ``next(iter(loader))`` contract."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, delay_s: float) -> None:
        self.X, self.y = X, y
        self.delay_s = delay_s
        self.dataset = X  # len() -> num_examples

    def __iter__(self):
        while True:
            time.sleep(self.delay_s)
            yield self.X, self.y


def _toy_batch(n: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 4, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return X, y


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Server side: the coordinator must report its real stop state, not a constant.
# ---------------------------------------------------------------------------

def test_update_client_heartbeat_reflects_coordinator_stop():
    coord = FLCoordinator(Mock(), min_clients_for_aggregation=1, clients_per_round=1,
                          round_timeout_s=30)

    ok, should_stop, msg = coord.update_client_heartbeat("c1", "training", 1, 10, 1)
    assert ok is True
    assert should_stop is False          # not stopped -> clients keep training
    assert isinstance(msg, str)

    coord.signal_stop()                  # the global stop every stop path funnels through

    ok, should_stop, msg = coord.update_client_heartbeat("c1", "training", 2, 10, 1)
    assert ok is True                    # tuple shape + ack preserved for existing callers
    assert should_stop is True, (
        "a stopped coordinator must ask every heart-beating client to halt "
        "(update_client_heartbeat hardcoded should_stop=False)"
    )
    assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# Client side: the fit loop must check the stop Event between local steps.
# ---------------------------------------------------------------------------

def test_fit_loop_aborts_when_stop_event_already_set():
    # No server needed: the Event is set before fit() starts, so the K-local-steps loop must
    # break out on its very first between-steps check instead of grinding through all K steps.
    X, y = _toy_batch()
    client = DeComFLClient(model=LogReg(), train_loader=_SlowLoader(X, y, delay_s=0.0), device="cpu")
    comm = GrpcClient(client_id="stop-pre", server_address="127.0.0.1:1")
    try:
        client.set_grpc_client(comm)
        comm._stop_training.set()

        K = 50
        seeds = [[100 + k] for k in range(K)]
        scalars, _ = client.fit(None, {"seeds": seeds, "learning_rate": 0.01})

        assert len(scalars) == 0, (
            f"fit ran {len(scalars)} local steps despite the stop Event being set before it started"
        )
    finally:
        comm.close()


# ---------------------------------------------------------------------------
# End to end: coordinator stop -> real servicer -> real heartbeat thread -> fit-loop abort.
# ---------------------------------------------------------------------------

@pytest.fixture
def live_server():
    init = OrderedDict((k, v.clone()) for k, v in LogReg().state_dict().items())
    strat = DeComFL(init, evaluate_fn=None, min_fit_clients=1, clients_per_round=1,
                    num_local_steps=1, num_perturbations=1, learning_rate=0.01,
                    smoothing_param=0.001, seed=42)
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=1,
                          round_timeout_s=300)
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_FederatedLearningServiceServicer_to_server(FederatedLearningServiceServicer(coord), server)
    port = _free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    try:
        yield f"127.0.0.1:{port}", coord
    finally:
        server.stop(grace=None)


def test_coordinator_stop_reaches_running_fit_loop_within_a_heartbeat(live_server):
    addr, coord = live_server

    X, y = _toy_batch()
    # 400 steps x >=30ms each: a run that ignores the stop takes >= 12s. The abort path must
    # come in orders of magnitude under that (one heartbeat interval + one local step + slack).
    K = 400
    step_delay_s = 0.03
    seeds = [[1000 + k] for k in range(K)]

    fl_client = DeComFLClient(model=LogReg(), train_loader=_SlowLoader(X, y, step_delay_s),
                              device="cpu")
    comm = GrpcClient(client_id="stop-e2e", server_address=addr)
    comm.heartbeat_interval = 0.05
    try:
        fl_client.set_grpc_client(comm)
        comm.start_heartbeat()   # parallel heartbeat stub — the ONLY channel carrying the stop

        result = {}

        def run_fit():
            # Training stub stays blocked in fit() for the whole duration (two-stub contract).
            scalars, _ = fl_client.fit(None, {"seeds": seeds, "learning_rate": 0.01})
            result["scalars"] = scalars

        fit_thread = threading.Thread(target=run_fit, daemon=True)
        fit_thread.start()

        # Let training genuinely get going (>= 2 local steps reported) before pulling the plug.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and comm.current_step < 2:
            time.sleep(0.01)
        assert comm.current_step >= 2, "fit loop never started reporting local steps"

        coord.signal_stop()
        stop_at = time.monotonic()

        fit_thread.join(timeout=4.0)
        assert not fit_thread.is_alive(), (
            "fit loop kept training after the coordinator stop — the server-driven stop "
            "protocol never reached the client (decorative should_stop)"
        )
        aborted_after = time.monotonic() - stop_at
        assert aborted_after < 3.0, f"fit aborted only {aborted_after:.2f}s after the stop"

        # The stop travelled via the heartbeat: the client's training Event must be set...
        assert comm._stop_training.is_set(), (
            "client never latched the server's should_stop into its training Event"
        )
        # ...and the loop broke out early rather than finishing all K local steps.
        assert len(result["scalars"]) < K, "fit returned all K steps — it never aborted"
    finally:
        comm.close()


# ---------------------------------------------------------------------------
# The dual-stub-shared status triple must never be read torn.
# ---------------------------------------------------------------------------

def test_update_status_never_yields_a_torn_status_tuple():
    # update_status writes (status, step, total) as three separate attribute stores; the
    # heartbeat thread reads them concurrently. Writers only ever publish COHERENT triples,
    # so any observed mix (phase of one write, step of another) is a torn read. The reader
    # uses _status_snapshot() — the exact read path send_heartbeat builds its request from.
    #
    # NOTE on teeth: on today's GIL builds consecutive attribute stores/loads are rarely
    # preempted (the eval breaker fires at calls/backward jumps), so an unsynchronized
    # implementation tears only sporadically here — but tears become routine on free-threaded
    # (PEP 703) builds or as soon as any call lands between the stores. The switch-interval
    # shrink below maximises preemption pressure; the deterministic lock-contract test that
    # follows guarantees regression coverage on every build.
    import sys
    comm = GrpcClient(client_id="race", server_address="127.0.0.1:1")
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        coherent = {("idle", 0, 0), ("phase-a", 1, 100), ("phase-b", 2, 200)}
        stop = threading.Event()

        def writer(status: str, step: int, total: int):
            while not stop.is_set():
                comm.update_status(status, step, total)

        writers = [
            threading.Thread(target=writer, args=("phase-a", 1, 100), daemon=True),
            threading.Thread(target=writer, args=("phase-b", 2, 200), daemon=True),
        ]
        for w in writers:
            w.start()

        try:
            for _ in range(30_000):
                status, step, total, _round = comm._status_snapshot()
                assert (status, step, total) in coherent, (
                    f"torn status read: ({status!r}, {step}, {total}) mixes two writes"
                )
        finally:
            stop.set()
            for w in writers:
                w.join(timeout=5)
    finally:
        sys.setswitchinterval(old_interval)
        comm.close()


def test_status_writer_and_heartbeat_snapshot_mutually_exclude():
    # Deterministic guard for the same invariant: the writer (update_status, training thread)
    # and the reader (_status_snapshot, heartbeat thread) must synchronise on the SAME lock,
    # so neither can ever run inside the other's critical section. White-box on purpose —
    # if either path silently drops the shared _status_lock, this fails on any build,
    # independent of scheduler luck.
    comm = GrpcClient(client_id="lock-contract", server_address="127.0.0.1:1")
    try:
        entered_write = threading.Event()
        entered_read = threading.Event()
        snap = {}

        with comm._status_lock:
            w = threading.Thread(
                target=lambda: (comm.update_status("guarded", 7, 70), entered_write.set()),
                daemon=True)
            r = threading.Thread(
                target=lambda: (snap.setdefault("v", comm._status_snapshot()), entered_read.set()),
                daemon=True)
            w.start()
            r.start()
            # While the test holds the lock, neither side may complete its critical section.
            assert not entered_write.wait(timeout=0.2), "update_status ignored _status_lock"
            assert not entered_read.wait(timeout=0.05), "_status_snapshot ignored _status_lock"

        assert entered_write.wait(timeout=2.0) and entered_read.wait(timeout=2.0)
        w.join(timeout=2)
        r.join(timeout=2)
        status, step, total, _round = comm._status_snapshot()
        assert (status, step, total) == ("guarded", 7, 70)
        assert snap["v"][:3] in {("idle", 0, 0), ("guarded", 7, 70)}  # coherent either way
    finally:
        comm.close()
