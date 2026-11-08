"""Multi-client Federated Learning integration test over real gRPC wire with FedAvg aggregation."""
from __future__ import annotations

import concurrent.futures
import socket
from collections import OrderedDict

import grpc
import pytest
import torch

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.communication.generated import fedlearn_pb2_grpc as pbg
from fedlearn.communication.serializer import parameters_to_proto
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.server.strategy import FedAvg


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def multi_client_server():
    """Starts a live gRPC server expecting 2 clients per round."""
    init_weights = OrderedDict(
        weight=torch.zeros(4, 2),
        bias=torch.zeros(4),
    )
    strategy = FedAvg(
        initial_parameters=init_weights,
        min_fit_clients=2,
        clients_per_round=2,
    )
    coord = FLCoordinator(
        strategy=strategy,
        min_clients_for_aggregation=2,
        clients_per_round=2,
        round_timeout_s=10,
    )
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=8))
    servicer = FederatedLearningServiceServicer(coord)
    pbg.add_FederatedLearningServiceServicer_to_server(servicer, server)

    port = _free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()

    try:
        yield f"127.0.0.1:{port}", coord, strategy
    finally:
        server.stop(grace=None)


def test_multi_client_fedavg_round_over_grpc(multi_client_server):
    """Verifies that multiple independent clients can complete a real gRPC round over TCP."""
    endpoint, coord, strategy = multi_client_server

    channel_a = grpc.insecure_channel(endpoint)
    channel_b = grpc.insecure_channel(endpoint)
    stub_a = pbg.FederatedLearningServiceStub(channel_a)
    stub_b = pbg.FederatedLearningServiceStub(channel_b)

    # 1. Register Client A and Client B
    reg_a = stub_a.RegisterClient(pb.RegisterClientRequest(client_id="client-A", protocol_version=2))
    assert reg_a.status == pb.RegisterClientResponse.ACCEPTED
    reg_b = stub_b.RegisterClient(pb.RegisterClientRequest(client_id="client-B", protocol_version=2))
    assert reg_b.status == pb.RegisterClientResponse.ACCEPTED

    # 2. Both clients check status
    status_a = stub_a.GetServerStatus(pb.GetServerStatusRequest())
    assert status_a.current_round == 1

    # 3. Client A submits update (params = +1.0)
    params_a = OrderedDict(weight=torch.ones(4, 2) * 1.0, bias=torch.ones(4) * 1.0)
    proto_a = parameters_to_proto(params_a, num_examples=100)
    resp_a = stub_a.SubmitModelUpdate(
        pb.SubmitModelUpdateRequest(
            client_id="client-A",
            trained_on_round=1,
            parameters=proto_a,
        )
    )
    assert resp_a.received is True

    # 4. Client B submits update (params = +3.0)
    params_b = OrderedDict(weight=torch.ones(4, 2) * 3.0, bias=torch.ones(4) * 3.0)
    proto_b = parameters_to_proto(params_b, num_examples=100)
    resp_b = stub_b.SubmitModelUpdate(
        pb.SubmitModelUpdateRequest(
            client_id="client-B",
            trained_on_round=1,
            parameters=proto_b,
        )
    )
    assert resp_b.received is True

    # 5. Server should now have aggregated the updates and advanced to Round 2
    status_after = stub_a.GetServerStatus(pb.GetServerStatusRequest())
    assert status_after.current_round == 2

    # 6. Verify aggregated weights: (100 * 1.0 + 100 * 3.0) / 200 = 2.0
    latest_params = coord.get_global_model_params()
    assert latest_params is not None
    expected_weight = torch.ones(4, 2) * 2.0
    assert torch.allclose(latest_params["weight"], expected_weight, atol=1e-5)

    channel_a.close()
    channel_b.close()


def test_multi_client_weighted_fedavg_round_over_grpc():
    """Verifies that 3 clients with heterogeneous sample counts aggregate correctly with sample weighting."""
    init_weights = OrderedDict(
        weight=torch.zeros(2, 2),
        bias=torch.zeros(2),
    )
    strategy = FedAvg(
        initial_parameters=init_weights,
        min_fit_clients=3,
        clients_per_round=3,
    )
    coord = FLCoordinator(
        strategy=strategy,
        min_clients_for_aggregation=3,
        clients_per_round=3,
        round_timeout_s=10,
    )
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=8))
    servicer = FederatedLearningServiceServicer(coord)
    pbg.add_FederatedLearningServiceServicer_to_server(servicer, server)

    port = _free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()

    channels = [grpc.insecure_channel(f"127.0.0.1:{port}") for _ in range(3)]
    stubs = [pbg.FederatedLearningServiceStub(ch) for ch in channels]

    try:
        # Register 3 clients
        for i, stub in enumerate(stubs):
            reg = stub.RegisterClient(pb.RegisterClientRequest(client_id=f"client-{i}", protocol_version=2))
            assert reg.status == pb.RegisterClientResponse.ACCEPTED

        # Client 0: 50 samples, value 1.0
        # Client 1: 100 samples, value 2.0
        # Client 2: 150 samples, value 4.0
        # Weighted avg: (50*1 + 100*2 + 150*4) / 300 = (50 + 200 + 600) / 300 = 850 / 300 = 2.83333...
        samples = [50, 100, 150]
        vals = [1.0, 2.0, 4.0]
        for i, stub in enumerate(stubs):
            p = OrderedDict(weight=torch.ones(2, 2) * vals[i], bias=torch.ones(2) * vals[i])
            proto = parameters_to_proto(p, num_examples=samples[i])
            resp = stub.SubmitModelUpdate(
                pb.SubmitModelUpdateRequest(
                    client_id=f"client-{i}",
                    trained_on_round=1,
                    parameters=proto,
                )
            )
            assert resp.received is True

        status = stubs[0].GetServerStatus(pb.GetServerStatusRequest())
        assert status.current_round == 2

        latest = coord.get_global_model_params()
        assert latest is not None
        expected = torch.ones(2, 2) * (850.0 / 300.0)
        assert torch.allclose(latest["weight"], expected, atol=1e-5)
    finally:
        for ch in channels:
            ch.close()
        server.stop(grace=None)

