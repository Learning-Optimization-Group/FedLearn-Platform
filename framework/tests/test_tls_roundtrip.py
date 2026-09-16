"""SE-2 — a real TLS handshake: the gRPC server serves TLS and a client that trusts the CA connects.

Generates a throwaway self-signed cert at runtime (cryptography), stands up the FL servicer over TLS
via grpc.ssl_server_credentials, and drives a real RegisterClient over grpc.secure_channel — proving
the TLS mechanism (server credentials + client CA verification) actually interoperates end to end.
"""
import concurrent.futures
import datetime
import socket
from collections import OrderedDict

import grpc
import pytest
import torch
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.communication.generated import fedlearn_pb2_grpc as pbg
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import SERVER_PROTOCOL_VERSION, FederatedLearningServiceServicer


def _self_signed_cert():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime(2026, 1, 1)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(datetime.datetime(2099, 1, 1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.TraditionalOpenSSL,
                                serialization.NoEncryption())
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return key_pem, cert_pem


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_tls_handshake_and_register_over_secure_channel():
    key_pem, cert_pem = _self_signed_cert()
    strat = DeComFL(OrderedDict(w=torch.zeros(4)), evaluate_fn=lambda r, p: (1.0, {}),
                    min_fit_clients=1, clients_per_round=1, num_local_steps=1, num_perturbations=2,
                    learning_rate=0.01, smoothing_param=0.01, seed=42)
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=1, round_timeout_s=30)

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_FederatedLearningServiceServicer_to_server(FederatedLearningServiceServicer(coord), server)
    creds = grpc.ssl_server_credentials([(key_pem, cert_pem)])
    port = _free_port()
    server.add_secure_port(f"localhost:{port}", creds)
    server.start()
    try:
        channel_creds = grpc.ssl_channel_credentials(root_certificates=cert_pem)  # client trusts the CA
        with grpc.secure_channel(f"localhost:{port}", channel_creds) as channel:
            stub = pbg.FederatedLearningServiceStub(channel)
            reg = stub.RegisterClient(pb.RegisterClientRequest(
                client_id="c", run_id="r", protocol_version=SERVER_PROTOCOL_VERSION,
                enrollment_token="t"), timeout=10)
            assert reg.status == pb.RegisterClientResponse.Status.ACCEPTED
    finally:
        server.stop(grace=None)


def test_plaintext_client_cannot_talk_to_a_tls_server():
    # Negative: an insecure client against a TLS server must NOT succeed (proves TLS is actually on).
    key_pem, cert_pem = _self_signed_cert()
    strat = DeComFL(OrderedDict(w=torch.zeros(4)), evaluate_fn=lambda r, p: (1.0, {}),
                    min_fit_clients=1, clients_per_round=1, num_local_steps=1, num_perturbations=2,
                    learning_rate=0.01, smoothing_param=0.01, seed=42)
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=1, round_timeout_s=30)
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_FederatedLearningServiceServicer_to_server(FederatedLearningServiceServicer(coord), server)
    port = _free_port()
    server.add_secure_port(f"localhost:{port}", grpc.ssl_server_credentials([(key_pem, cert_pem)]))
    server.start()
    try:
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = pbg.FederatedLearningServiceStub(channel)
            with pytest.raises(grpc.RpcError):
                stub.RegisterClient(pb.RegisterClientRequest(
                    client_id="c", run_id="r", protocol_version=SERVER_PROTOCOL_VERSION,
                    enrollment_token="t"), timeout=5)
    finally:
        server.stop(grace=None)
