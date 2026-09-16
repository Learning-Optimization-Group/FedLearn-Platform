"""SE-1 slice 2 — the connection-token interceptor gates a REAL gRPC server.

Stands up an actual grpc.Server with the interceptor + the real FL servicer, and confirms over the
wire that RegisterClient is refused UNAUTHENTICATED without a token and accepted with the golden
Java-minted token in x-connection-token metadata.
"""
import concurrent.futures
import json
import pathlib
import socket
from collections import OrderedDict

import grpc
import pytest
import torch

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.communication.generated import fedlearn_pb2_grpc as pbg
from fedlearn.security.interceptor import METADATA_KEY, ConnectionTokenInterceptor
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import SERVER_PROTOCOL_VERSION, FederatedLearningServiceServicer

_GOLDEN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "golden_connection_token.json").read_text())


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(expected_run_id=None):
    """Start a real FL server whose interceptor is bound to expected_run_id. Returns (addr, server)."""
    strat = DeComFL(OrderedDict(w=torch.zeros(4)), evaluate_fn=lambda r, p: (1.0, {}),
                    min_fit_clients=1, clients_per_round=1, num_local_steps=1, num_perturbations=2,
                    learning_rate=0.01, smoothing_param=0.01, seed=42)
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=1, round_timeout_s=30)
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[ConnectionTokenInterceptor(_GOLDEN["secret_base64"], expected_run_id=expected_run_id)],
    )
    pbg.add_FederatedLearningServiceServicer_to_server(FederatedLearningServiceServicer(coord), server)
    port = _free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    return f"127.0.0.1:{port}", server


@pytest.fixture
def authed_server():
    addr, server = _start_server()
    try:
        yield addr
    finally:
        server.stop(grace=None)


def _register_request():
    return pb.RegisterClientRequest(
        client_id="mobile-1", run_id="run-1", protocol_version=SERVER_PROTOCOL_VERSION,
        enrollment_token="ignored-by-interceptor")


def test_register_without_token_is_unauthenticated(authed_server):
    with grpc.insecure_channel(authed_server) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.RegisterClient(_register_request())
        assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_register_with_valid_token_succeeds(authed_server):
    with grpc.insecure_channel(authed_server) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        reg = stub.RegisterClient(_register_request(),
                                  metadata=[(METADATA_KEY, _GOLDEN["token"])])
        assert reg.status == pb.RegisterClientResponse.Status.ACCEPTED


def test_register_with_forged_token_is_unauthenticated(authed_server):
    # SE-13: a tampered token must be refused over the wire, not just in a unit test. Flip a char in
    # the PAYLOAD (not the last sig char, whose low bits are base64 padding) so the signature over the
    # tampered payload no longer matches — the real "attacker edited the claims" forgery.
    header, payload, sig = _GOLDEN["token"].split(".")
    tampered_payload = payload[:5] + ("A" if payload[5] != "A" else "B") + payload[6:]
    forged = ".".join([header, tampered_payload, sig])
    with grpc.insecure_channel(authed_server) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.RegisterClient(_register_request(), metadata=[(METADATA_KEY, forged)])
        assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_token_for_a_different_run_is_permission_denied():
    # FR-7: a valid token minted for run 111...  presented to a server serving a DIFFERENT run must be
    # refused PERMISSION_DENIED (authenticated, but not for this federation).
    addr, server = _start_server(expected_run_id="99999999-9999-9999-9999-999999999999")
    try:
        with grpc.insecure_channel(addr) as channel:
            stub = pbg.FederatedLearningServiceStub(channel)
            with pytest.raises(grpc.RpcError) as excinfo:
                stub.RegisterClient(_register_request(), metadata=[(METADATA_KEY, _GOLDEN["token"])])
            assert excinfo.value.code() == grpc.StatusCode.PERMISSION_DENIED
    finally:
        server.stop(grace=None)


def test_token_for_the_served_run_is_accepted():
    # The same token IS accepted when the server serves that run (111...).
    addr, server = _start_server(expected_run_id=_GOLDEN["claims"]["runId"])
    try:
        with grpc.insecure_channel(addr) as channel:
            stub = pbg.FederatedLearningServiceStub(channel)
            reg = stub.RegisterClient(_register_request(), metadata=[(METADATA_KEY, _GOLDEN["token"])])
            assert reg.status == pb.RegisterClientResponse.Status.ACCEPTED
    finally:
        server.stop(grace=None)


def test_client_interceptor_and_server_interceptor_interoperate(authed_server):
    # Slice 3: the CLIENT interceptor attaches the token so no explicit per-call metadata is needed;
    # the server accepts it. Proves the two halves interoperate end to end.
    from fedlearn.security.client_interceptor import maybe_wrap_channel
    base = grpc.insecure_channel(authed_server)
    channel = maybe_wrap_channel(base, token=_GOLDEN["token"])
    try:
        stub = pbg.FederatedLearningServiceStub(channel)
        reg = stub.RegisterClient(_register_request())          # no metadata= here — the interceptor adds it
        assert reg.status == pb.RegisterClientResponse.Status.ACCEPTED
    finally:
        base.close()
