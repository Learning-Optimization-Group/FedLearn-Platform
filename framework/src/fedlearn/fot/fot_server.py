"""Standalone FoT gRPC server + N-round loop.

The control plane spawns this exactly like fl_server.py (its stdout is streamed to the dashboard
via STOMP and its Process handle is tracked for /stop). It serves FoTService and runs an N-round
text-federation loop, emitting one machine-readable JSON event per stdout line so the dashboard
can tail progress. Torch-free; the LLM distiller is behind the AgentBackend seam.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent import futures

import grpc

from fedlearn.communication.generated import fot_pb2_grpc
from fedlearn.fot.distiller import TraceDistiller
from fedlearn.fot.fot_servicer import FotServicer


def emit(event: str, **fields) -> None:
    """Write one JSON event line to stdout (consumed line-by-line by the control plane)."""
    sys.stdout.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")
    sys.stdout.flush()


def build_server(servicer: FotServicer, address: str, *, max_workers: int = 8) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    fot_pb2_grpc.add_FoTServiceServicer_to_server(servicer, server)
    server.add_insecure_port(address)  # plaintext, consistent with the gradient FL server
    return server


def run_rounds(
    servicer: FotServicer,
    *,
    num_rounds: int,
    round_seconds: float = 0.0,
    sleep=time.sleep,
) -> None:
    """Run the N-round distill loop, emitting JSON events. Clients submit traces between rounds."""
    for r in range(num_rounds):
        emit("round_started", round=r)
        if round_seconds > 0:
            sleep(round_seconds)  # window for clients to SubmitReasoningTrace
        emit("traces_collected", round=r, count=servicer.pending_count())
        lib = servicer.distill_round()
        emit("insights_extracted", round=r, version=lib.version, num_insights=len(lib))
    emit(
        "run_complete",
        rounds=num_rounds,
        version=servicer.library.version,
        num_insights=len(servicer.library),
    )


def start_fot_server(
    address: str,
    *,
    num_rounds: int,
    round_seconds: float = 5.0,
    backend_name: str = "stub",
    quorum: int = 2,
) -> None:
    """Build + serve the FoT server and run the round loop. Used by the spawn entrypoint."""
    from fedlearn.fot.backend import get_backend

    servicer = FotServicer(TraceDistiller(get_backend(backend_name), quorum=quorum))
    server = build_server(servicer, address)
    server.start()
    emit("server_started", address=address, num_rounds=num_rounds)
    try:
        run_rounds(servicer, num_rounds=num_rounds, round_seconds=round_seconds)
    finally:
        server.stop(grace=2.0)
