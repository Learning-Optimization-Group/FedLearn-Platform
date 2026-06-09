"""Tests for the FoT server round loop + stdout JSON event protocol."""
import io
import json
from contextlib import redirect_stdout

from fedlearn.communication.generated import fot_pb2
from fedlearn.fot import fot_server
from fedlearn.fot.backend import DeterministicStubBackend
from fedlearn.fot.distiller import TraceDistiller
from fedlearn.fot.fot_servicer import FotServicer
from fedlearn.fot.model import ReasoningTrace


class _Ctx:
    pass


def _servicer(quorum=1):
    return FotServicer(TraceDistiller(DeterministicStubBackend(default="{}"), quorum=quorum))


def _events(servicer, *, num_rounds):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fot_server.run_rounds(servicer, num_rounds=num_rounds, round_seconds=0.0)
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def test_run_rounds_emits_expected_event_sequence():
    events = _events(_servicer(), num_rounds=2)
    assert [e["event"] for e in events] == [
        "round_started", "traces_collected", "insights_extracted",
        "round_started", "traces_collected", "insights_extracted",
        "run_complete",
    ]
    assert events[-1]["rounds"] == 2


def test_run_rounds_distills_submitted_traces():
    svc = _servicer(quorum=2)
    for cid in ("c1", "c2"):
        tr = ReasoningTrace(f"{cid}-0", cid, "run", 0, "task", {"insight_a": "Validate inputs."})
        svc.SubmitReasoningTrace(
            fot_pb2.SubmitReasoningTraceRequest(client_id=cid, round=0, trace_json=tr.to_json()), _Ctx()
        )
    events = _events(svc, num_rounds=1)
    assert len(svc.library) == 1
    extracted = next(e for e in events if e["event"] == "insights_extracted")
    assert extracted["num_insights"] == 1


def test_build_server_constructs_and_stops():
    server = fot_server.build_server(_servicer(), "localhost:0")  # OS-assigned port
    assert server is not None
    server.stop(grace=0)
