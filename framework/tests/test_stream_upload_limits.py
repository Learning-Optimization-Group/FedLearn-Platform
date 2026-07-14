"""SE-18: bound the streamed model-upload buffer so a client can't exhaust server memory.

``SubmitModelUpdateStream`` reassembles a client's ``ModelUpdateChunk`` stream into a single in-memory
buffer. With no cap, a malicious or buggy client that never sends ``is_final_chunk`` (or sends huge /
endless chunks) grows that buffer without bound -> OOM DoS. The server must abort
``RESOURCE_EXHAUSTED`` as soon as the cumulative bytes OR the chunk count exceeds a configurable cap,
BEFORE the buffer grows past it — while leaving legitimately-sized uploads untouched.

The abort must sit outside the servicer's broad ``except Exception`` (which would otherwise remap it to
INTERNAL) — the same ordering constraint as the SE-15 identity gate.
"""
import grpc
import pytest
from unittest.mock import MagicMock

from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer


class _Abort(Exception):
    pass


class _Ctx:
    """Fake ServicerContext whose abort() records the code and raises like the real one."""

    def __init__(self):
        self.aborted = None

    def invocation_metadata(self):
        return []

    def abort(self, code, details):
        self.aborted = (code, details)
        raise _Abort(details)


def _servicer(max_bytes=None, max_chunks=None):
    # partition_extractor=None -> identity binding disabled, so the stream reaches the cap logic.
    s = FederatedLearningServiceServicer(MagicMock(), partition_extractor=None)
    if max_bytes is not None:
        s._max_upload_bytes = max_bytes
    if max_chunks is not None:
        s._max_upload_chunks = max_chunks
    return s


def _chunk(data=b"", is_final=False, client_id="c0", total_chunks=1, total_bytes=0, num_examples=1):
    return fedlearn_pb2.ModelUpdateChunk(
        client_id=client_id, trained_on_round=1, total_chunks=total_chunks,
        chunk_data=data, is_final_chunk=is_final, num_examples=num_examples, total_bytes=total_bytes,
    )


def test_aborts_resource_exhausted_when_cumulative_bytes_exceed_cap():
    s = _servicer(max_bytes=10, max_chunks=10_000)
    chunks = [_chunk(b"AAAAA"), _chunk(b"BBBBB"), _chunk(b"CCCCC", is_final=True)]  # 15 > 10
    ctx = _Ctx()
    with pytest.raises(_Abort):
        s.SubmitModelUpdateStream(iter(chunks), ctx)
    assert ctx.aborted[0] == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_aborts_resource_exhausted_when_chunk_count_exceeds_cap():
    s = _servicer(max_bytes=10**9, max_chunks=2)
    chunks = [_chunk(b"a"), _chunk(b"b"), _chunk(b"c"), _chunk(b"d", is_final=True)]  # 4 > 2
    ctx = _Ctx()
    with pytest.raises(_Abort):
        s.SubmitModelUpdateStream(iter(chunks), ctx)
    assert ctx.aborted[0] == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_aborts_early_on_declared_oversize_before_buffering():
    # An honest client that DECLARES a huge total_bytes is rejected on the first chunk.
    s = _servicer(max_bytes=100, max_chunks=10_000)
    ctx = _Ctx()
    with pytest.raises(_Abort):
        s.SubmitModelUpdateStream(iter([_chunk(b"x", total_bytes=10_000)]), ctx)
    assert ctx.aborted[0] == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_within_limits_upload_is_not_rejected(monkeypatch):
    # A legitimately-sized upload must pass the guard and reach decode+submit unharmed.
    s = _servicer(max_bytes=10**9, max_chunks=10_000)
    monkeypatch.setattr("fedlearn.server.grpc_servicer.chunks_to_parameters",
                        lambda data, compressed=False: ({"w": object()}, 42))
    chunks = [_chunk(b"AAAAA"), _chunk(b"BBBBB", is_final=True, num_examples=42)]
    ctx = _Ctx()
    resp = s.SubmitModelUpdateStream(iter(chunks), ctx)
    assert resp.received is True
    assert ctx.aborted is None
    s.coordinator.submit_client_update.assert_called_once()
