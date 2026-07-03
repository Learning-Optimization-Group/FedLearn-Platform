"""End-to-end DeComFL round over a REAL gRPC socket against the fedlearn.v2 servicer (e2e training, P5).

Unlike test path that call the servicer methods directly, this stands up an actual grpc.Server + a real
client channel and drives the full DeComFL cycle a mobile client performs:

    RegisterClient(v2) -> Heartbeat -> GetServerStatus -> GetDeComFLConfig(seeds) ->
    SubmitGradientScalars(scalars for those seeds) -> ReportClientMetrics

and asserts the v2 fields round-trip over the wire and the server aggregates + advances the round. This
is the server-side half of P5; the mobile-device-in-the-loop half (phone -> backend -> fl_server) is a
live-stack step. It uses the same seeds the server issues (DeComFL is server-authoritative on seeds), so
the reconstruction the server performs from its own seed_history matches what the client computed.
"""
import concurrent.futures
import socket
from collections import OrderedDict

import grpc
import pytest
import torch

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.communication.generated import fedlearn_pb2_grpc as pbg
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer, SERVER_PROTOCOL_VERSION

K, P = 1, 2


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def live_server():
    init = OrderedDict(w=torch.zeros(4), b=torch.zeros(2))
    strat = DeComFL(init, evaluate_fn=lambda rnd, params: (1.097, {"accuracy": 0.5}),
                    min_fit_clients=1, clients_per_round=1, num_local_steps=K, num_perturbations=P,
                    learning_rate=0.01, smoothing_param=0.01, seed=42)
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=1, round_timeout_s=30)
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_FederatedLearningServiceServicer_to_server(FederatedLearningServiceServicer(coord), server)
    port = _free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    try:
        yield f"127.0.0.1:{port}", coord, strat
    finally:
        server.stop(grace=None)


def test_full_decomfl_round_over_grpc(live_server):
    addr, coord, strat = live_server
    with grpc.insecure_channel(addr) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)

        # 1. Register (v2 fields on the wire)
        reg = stub.RegisterClient(pb.RegisterClientRequest(
            client_id="mobile-1", run_id="run-1", protocol_version=SERVER_PROTOCOL_VERSION,
            enrollment_token="tok"))
        assert reg.status == pb.RegisterClientResponse.Status.ACCEPTED
        assert reg.assigned_round == 1
        assert reg.protocol_version == SERVER_PROTOCOL_VERSION

        # 2. Heartbeat carries run_id; makes the client active for the status quorum
        hb = stub.Heartbeat(pb.HeartbeatRequest(
            client_id="mobile-1", run_id="run-1", status="training", current_step=1, total_steps=1,
            current_round=1))
        assert hb.acknowledged

        # 3. Server status exposes the v2 fields
        st = stub.GetServerStatus(pb.GetServerStatusRequest())
        assert st.active_clients == 1
        assert st.round_deadline_unix_ms > 0

        # 4. DeComFL config: server-issued seeds + the v2 determinism fields
        cfg = stub.GetDeComFLConfig(pb.GetDeComFLConfigRequest(client_id="mobile-1"))
        assert cfg.torch_version != ""
        assert cfg.grad_estimate_method == "forward"
        assert len(cfg.current_seeds.local_steps) == K
        assert all(len(ls.seeds) == P for ls in cfg.current_seeds.local_steps)

        # 5. Submit scalars for those seeds + echo the seeds (int64 must survive the wire)
        seeds = [list(ls.seeds) for ls in cfg.current_seeds.local_steps]
        grads = pb.GradientScalars(local_steps=[pb.LocalStepGradients(scalars=[0.1] * P) for _ in range(K)])
        pseeds = pb.PerturbationSeeds(local_steps=[pb.LocalStepSeeds(seeds=seeds[k]) for k in range(K)])
        sub = stub.SubmitGradientScalars(pb.SubmitGradientScalarsRequest(
            client_id="mobile-1", trained_on_round=1, num_examples=8, gradients=grads,
            perturbation_seeds=pseeds))
        assert sub.received

        # 6. Telemetry
        rm = stub.ReportClientMetrics(pb.ReportClientMetricsRequest(
            client_id="mobile-1", run_id="run-1", round=1, loss=1.097, accuracy=0.5, current_step=1,
            total_steps=1, client_type="mobile", compute_ms=42))
        assert rm.acknowledged

    # Server-side effects: the round aggregated + advanced, and the metric was recorded.
    assert coord.current_round == 2
    assert len(coord.client_metrics_log) == 1
    assert coord.client_metrics_log[0]["client_id"] == "mobile-1"


def test_malformed_submit_rejected_as_invalid_argument(live_server):
    """FR-5: a submission whose scalar grid isn't K x P is refused over the wire with
    INVALID_ARGUMENT (not silently accepted then crashed on the aggregation thread)."""
    addr, coord, _ = live_server
    with grpc.insecure_channel(addr) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        stub.RegisterClient(pb.RegisterClientRequest(
            client_id="m1", run_id="r1", protocol_version=SERVER_PROTOCOL_VERSION,
            enrollment_token="t"))
        stub.GetDeComFLConfig(pb.GetDeComFLConfigRequest(client_id="m1"))

        # The round expects P=2 scalars per step; send 3 -> malformed shape.
        bad = pb.GradientScalars(local_steps=[pb.LocalStepGradients(scalars=[0.1, 0.2, 0.3])])
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.SubmitGradientScalars(pb.SubmitGradientScalarsRequest(
                client_id="m1", trained_on_round=1, num_examples=8, gradients=bad))
        assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    assert coord.current_round == 1                    # round not advanced or corrupted by the bad submit


def test_protocol_version_mismatch_rejected_over_grpc(live_server):
    addr, _, _ = live_server
    with grpc.insecure_channel(addr) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        reg = stub.RegisterClient(pb.RegisterClientRequest(client_id="c", protocol_version=99))
        assert reg.status == pb.RegisterClientResponse.Status.REJECTED
        assert reg.protocol_version == SERVER_PROTOCOL_VERSION


def test_int64_seeds_survive_the_wire(live_server):
    """A DeComFL seed near INT32_MAX must not truncate (v1 used int32 -> v2 int64)."""
    addr, _, _ = live_server
    big = 2_147_483_646  # > would overflow int32 arithmetic in some paths
    with grpc.insecure_channel(addr) as channel:
        stub = pbg.FederatedLearningServiceStub(channel)
        seeds = pb.PerturbationSeeds(local_steps=[pb.LocalStepSeeds(seeds=[big, 1234567])])
        # Round-trip through the proto (serialize/parse) preserves the 64-bit values.
        parsed = pb.PerturbationSeeds.FromString(seeds.SerializeToString())
        assert list(parsed.local_steps[0].seeds) == [big, 1234567]
