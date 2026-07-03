"""FR-6 — gRPC servicer status-code hygiene + typed strategy dispatch.

A server that is NOT configured for DeComFL must reject the DeComFL-only RPCs with
FAILED_PRECONDITION and still return the RPC's OWN response type — previously SubmitGradientScalars
returned a GetDeComFLConfigResponse in that branch, a latent wrong-type bug masked only because the
non-OK status makes the client discard the body.
"""
from unittest.mock import MagicMock

import grpc

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.server.strategy import Strategy


class _FakeContext:
    """Minimal unary-RPC context that records the status the servicer sets."""
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _non_decomfl_servicer():
    # A strategy that is a Strategy but NOT a DeComFL, so isinstance(strategy, DeComFL) is False.
    strat = MagicMock(spec=Strategy)
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=1)
    return FederatedLearningServiceServicer(coord)


def test_submit_gradient_scalars_on_non_decomfl_returns_its_own_response_type():
    servicer = _non_decomfl_servicer()
    ctx = _FakeContext()
    req = pb.SubmitGradientScalarsRequest(
        client_id="c", trained_on_round=1, num_examples=1,
        gradients=pb.GradientScalars(local_steps=[pb.LocalStepGradients(scalars=[0.1])]),
    )

    resp = servicer.SubmitGradientScalars(req, ctx)

    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
    assert isinstance(resp, pb.SubmitGradientScalarsResponse)   # not GetDeComFLConfigResponse (FR-6)


def test_get_decomfl_config_on_non_decomfl_returns_its_own_response_type():
    # Guard the sibling handler stays correct after the isinstance dispatch change.
    servicer = _non_decomfl_servicer()
    ctx = _FakeContext()

    resp = servicer.GetDeComFLConfig(pb.GetDeComFLConfigRequest(client_id="c"), ctx)

    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
    assert isinstance(resp, pb.GetDeComFLConfigResponse)
