"""sha256 integrity on the GetGlobalModelStream download path (FR-8, download half).

The v2 ModelChunk contract declares ``sha256 = 10`` as "hash of the full reassembled
blob; receiver verifies (integrity)", and the mobile C++ client already enforces it
(FedLearnClient.cpp getGlobalModelStream, skip-if-empty). These tests pin the Python
side of the same contract over a REAL gRPC socket (same substrate as
test_v2_grpc_e2e.py — an actual grpc.Server + client channel, no stub mocks):

- the server populates ``sha256`` with the hex digest of the full payload on every
  chunk (mobile reads the first chunk, the Python client the final one);
- the Python client verifies the reassembled bytes and REFUSES to deserialize a
  tampered stream (raises before decode);
- an empty ``sha256`` (an older server) still decodes — verification is skip-if-absent,
  so the field stays purely additive and backward-compatible.

The wire format itself is SAFETENSORS (FR-8 download half): the server emits a
deterministic safetensors blob with ``codec='safetensors'`` + ``total_bytes`` framing
(symmetric with the upload path and the libtorch-free mobile C++ core), and the client
decodes it — transparently falling back to a legacy torch.save pickle blob when an old
server sends one (version gate), see test_download_stream_is_safetensors_not_pickle and
test_new_client_decodes_legacy_pickle_stream below.
"""
import concurrent.futures
import hashlib
import io
import socket
from collections import OrderedDict

import grpc
import numpy as np
import pytest
import torch

from fedlearn.client.grpc_client import GrpcClient
from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.communication.generated import fedlearn_pb2_grpc as pbg
from fedlearn.communication.safetensors_codec import load_safetensors
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.server.strategy import FedAvg

INIT_PARAMS = OrderedDict(
    w=torch.arange(16, dtype=torch.float32).reshape(4, 4),
    b=torch.full((3,), 0.5),
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_coordinator() -> FLCoordinator:
    init = OrderedDict((k, v.clone()) for k, v in INIT_PARAMS.items())
    strat = FedAvg(initial_parameters=init, min_fit_clients=1, clients_per_round=1)
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=1,
                          round_timeout_s=30)
    coord.set_initial_parameters(strat.initialize_parameters())
    return coord


class _TamperedPayloadServicer(FederatedLearningServiceServicer):
    """Wire-tamper simulation: streams the real servicer's chunks but flips one byte
    of the payload while declaring the sha256 of the ORIGINAL (untampered) bytes —
    exactly what an on-path corruption/tamper looks like to the client."""

    def GetGlobalModelStream(self, request, context):
        chunks = list(super().GetGlobalModelStream(request, context))
        original_sha = hashlib.sha256(
            b"".join(c.chunk_data for c in chunks)).hexdigest()
        flipped = False
        for chunk in chunks:
            if not flipped and len(chunk.chunk_data) > 0:
                data = bytearray(chunk.chunk_data)
                data[len(data) // 2] ^= 0xFF
                chunk.chunk_data = bytes(data)
                flipped = True
            chunk.sha256 = original_sha
            yield chunk


class _WrongHashServicer(FederatedLearningServiceServicer):
    """Streams CLEAN payload bytes but declares a bogus sha256: the client must trust
    the hash check, not the decodability of the payload."""

    def GetGlobalModelStream(self, request, context):
        for chunk in super().GetGlobalModelStream(request, context):
            chunk.sha256 = "0" * 64
            yield chunk


class _LegacyNoHashServicer(FederatedLearningServiceServicer):
    """Simulates a pre-integrity server: chunks arrive with sha256 unset."""

    def GetGlobalModelStream(self, request, context):
        for chunk in super().GetGlobalModelStream(request, context):
            chunk.sha256 = ""
            yield chunk


@pytest.fixture
def serve():
    """Factory fixture: stand up a real gRPC server for a given servicer class and
    yield its address. All servers are torn down at test exit."""
    servers = []

    def _serve(servicer_cls) -> str:
        server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
        pbg.add_FederatedLearningServiceServicer_to_server(
            servicer_cls(_make_coordinator()), server)
        port = _free_port()
        server.add_insecure_port(f"127.0.0.1:{port}")
        server.start()
        servers.append(server)
        return f"127.0.0.1:{port}"

    yield _serve
    for server in servers:
        server.stop(grace=None)


@pytest.fixture
def client_for(serve):
    """Factory fixture: a real GrpcClient wired to a freshly served servicer class."""
    clients = []

    def _client(servicer_cls) -> GrpcClient:
        client = GrpcClient("integrity-test-client", serve(servicer_cls))
        clients.append(client)
        return client

    yield _client
    for client in clients:
        client.close()


def test_download_stream_populates_and_verifies_sha256(serve, client_for):
    """Round-trip: the real server declares the payload hash on the wire (first AND
    final chunk — mobile reads the first, the Python client the final) and the real
    client verifies it and returns the exact model."""
    addr = serve(FederatedLearningServiceServicer)

    # Raw wire check: the declared hash matches the reassembled payload.
    with grpc.insecure_channel(addr) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        chunks = list(stub.GetGlobalModelStream(
            pb.GetGlobalModelRequest(client_id="raw-probe")))
    assert chunks, "server streamed no chunks"
    payload = b"".join(c.chunk_data for c in chunks)
    expected = hashlib.sha256(payload).hexdigest()
    assert chunks[0].sha256 == expected, "first chunk must declare the payload sha256"
    assert chunks[-1].sha256 == expected, "final chunk must declare the payload sha256"
    assert chunks[-1].is_final_chunk

    # Full client path: verification passes and the model round-trips exactly.
    client = client_for(FederatedLearningServiceServicer)
    params, current_round, _config = client.get_global_model()
    assert params is not None
    assert set(params.keys()) == set(INIT_PARAMS.keys())
    for name, tensor in INIT_PARAMS.items():
        assert torch.equal(params[name], tensor)
    assert current_round == 1


def test_tampered_payload_is_rejected_before_deserialization(client_for):
    """Security half of FR-8: a byte flipped in transit (declared hash = original
    bytes) must raise an integrity error — NOT a torch.load/unpickling error, which
    would mean the tampered blob reached the deserializer."""
    client = client_for(_TamperedPayloadServicer)
    with pytest.raises((ValueError, RuntimeError), match=r"(?i)sha256|integrity"):
        client.get_global_model()


def test_wrong_declared_hash_is_rejected_even_if_payload_decodes(client_for):
    """A clean (decodable) payload with a mismatching declared hash must still be
    rejected: the client trusts the integrity check, not decodability."""
    client = client_for(_WrongHashServicer)
    with pytest.raises((ValueError, RuntimeError), match=r"(?i)sha256|integrity"):
        client.get_global_model()


def test_empty_hash_from_legacy_server_still_decodes(client_for):
    """Backward compat: an older server that never sets sha256 must not trip a false
    rejection — the check is skip-if-absent."""
    client = client_for(_LegacyNoHashServicer)
    params, current_round, _config = client.get_global_model()
    assert params is not None
    for name, tensor in INIT_PARAMS.items():
        assert torch.equal(params[name], tensor)
    assert current_round == 1


# ---------------------------------------------------------------------------
# FR-8 (download half, cross-segment): the wire format itself — safetensors, not pickle.
# ---------------------------------------------------------------------------

class _LegacyPickleServicer(FederatedLearningServiceServicer):
    """A pre-FR-8 server: streams a torch.save PICKLE blob with codec unset (the old wire).
    Used to prove a NEW client still decodes an OLD server during a staged rollout — the
    Python-side version gate must not hard-break a mixed-version deployment."""

    def GetGlobalModelStream(self, request, context):
        params, current_round, config = self.coordinator.get_global_model_for_client()
        buffer = io.BytesIO()
        torch.save({'parameters': params, 'num_examples': 0}, buffer)
        data = buffer.getvalue()
        yield pb.ModelChunk(
            chunk_index=0, total_chunks=1, chunk_data=data, is_final_chunk=True,
            current_round=current_round, config=config,
            sha256=hashlib.sha256(data).hexdigest(),
        )


def test_download_stream_is_safetensors_not_pickle(serve):
    """The global-model download stream must carry a deterministic SAFETENSORS blob with
    the codec/total_bytes framing set — never a torch.save/pickle blob. This makes the
    download symmetric with the upload path and decodable by the libtorch-free mobile C++
    core, whose FedLearnClient::validateCodec rejects any first chunk whose codec is not
    'safetensors'/'lz4+safetensors' (so today's empty-codec pickle stream throws there)."""
    addr = serve(FederatedLearningServiceServicer)
    with grpc.insecure_channel(addr) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        chunks = list(stub.GetGlobalModelStream(
            pb.GetGlobalModelRequest(client_id="fmt-probe")))
    assert chunks, "server streamed no chunks"
    payload = b"".join(c.chunk_data for c in chunks)

    # (a) NOT a torch.save pickle/zip blob (raw pickle -> 0x80; torch.save zip -> 'PK').
    assert payload[:2] != b"PK" and payload[0] != 0x80, \
        "download payload is still a torch.save pickle/zip blob"

    # (b) decodes as safetensors to the exact global model.
    named, _meta = load_safetensors(payload)
    got = dict(named)
    assert set(got.keys()) == set(INIT_PARAMS.keys())
    for name, tensor in INIT_PARAMS.items():
        assert np.array_equal(got[name], tensor.numpy())

    # (c) the mobile C++ framing contract: codec + total_bytes declared on the first chunk.
    assert chunks[0].codec == "safetensors", "first chunk must declare codec='safetensors'"
    assert chunks[0].total_bytes == len(payload), "first chunk must declare total_bytes"


def test_new_client_decodes_legacy_pickle_stream(client_for):
    """Version gate: a new client talking to an OLD (pre-FR-8) server that still emits a
    torch.save pickle blob must transparently fall back and decode it — the migration must
    not hard-break a mixed-version deployment on the Python side."""
    client = client_for(_LegacyPickleServicer)
    params, current_round, _config = client.get_global_model()
    assert params is not None
    for name, tensor in INIT_PARAMS.items():
        assert torch.equal(params[name], tensor)
    assert current_round == 1
