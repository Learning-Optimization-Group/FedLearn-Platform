"""FR-6 — gRPC servicer status-code hygiene + typed strategy dispatch.

A server that is NOT configured for DeComFL must reject the DeComFL-only RPCs with
FAILED_PRECONDITION and still return the RPC's OWN response type — previously SubmitGradientScalars
returned a GetDeComFLConfigResponse in that branch, a latent wrong-type bug masked only because the
non-OK status makes the client discard the body.
"""
from collections import OrderedDict
from unittest.mock import MagicMock

import grpc
import pytest
import torch

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.communication.serializer import parameters_to_proto
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.server.strategy import Strategy


class _Aborted(Exception):
    pass


class _AbortContext:
    """Fake unary context whose abort() records the code and raises, like real gRPC abort()."""
    def __init__(self):
        self.code = None
        self.details = None

    def abort(self, code, details):
        self.code = code
        self.details = details
        raise _Aborted(details)


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


def test_submit_model_update_with_non_finite_params_is_invalid_argument():
    # FR-6 follow-on: a client-sent malformed/non-finite FedAvg payload is the client's fault; the
    # serializer raises ValueError, which must surface as INVALID_ARGUMENT, not a generic INTERNAL.
    servicer = FederatedLearningServiceServicer(MagicMock())   # proto_to_parameters raises before the coordinator
    ctx = _AbortContext()
    poisoned = parameters_to_proto(OrderedDict({"w": torch.tensor([1.0, float("nan"), 3.0])}), 5)
    req = pb.SubmitModelUpdateRequest(client_id="c", trained_on_round=1, parameters=poisoned)
    with pytest.raises(_Aborted):
        servicer.SubmitModelUpdate(req, ctx)
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


# --- a deliberate abort inside a handler's try must reach the client as itself ---------------------------------
# gRPC's context.abort() works by raising. Raised inside a handler's broad try, it was caught by `except Exception`,
# logged as "RPC failed" with a stack trace, and replaced by INTERNAL: a client at the end of a run was told
# "An internal server error occurred" and retried instead of seeing "Training complete".

def _model_servicer(answer):
    coordinator = MagicMock()
    coordinator.get_global_model_for_client.return_value = answer
    return FederatedLearningServiceServicer(coordinator)


@pytest.mark.parametrize("answer, code, details", [
    ((None, -1, {}), grpc.StatusCode.UNAVAILABLE, "Training complete"),
    ((None, 1, {}), grpc.StatusCode.UNAVAILABLE, "Server not initialized"),
], ids=["run_over", "no_model_yet"])
def test_the_model_stream_refuses_with_its_own_status(answer, code, details, caplog):
    ctx = _AbortContext()
    with pytest.raises(_Aborted):
        list(_model_servicer(answer).GetGlobalModelStream(pb.GetGlobalModelRequest(client_id="c"), ctx))
    assert (ctx.code, ctx.details) == (code, details)
    assert "RPC failed" not in caplog.text


def test_the_unary_model_download_refuses_a_missing_model_as_unavailable(caplog):
    ctx = _AbortContext()
    with pytest.raises(_Aborted):
        _model_servicer((None, 1, {})).GetGlobalModel(pb.GetGlobalModelRequest(client_id="c"), ctx)
    assert ctx.code == grpc.StatusCode.UNAVAILABLE
    assert "RPC failed" not in caplog.text


def test_the_unary_model_download_tells_a_client_to_stream_a_model_too_large_for_it(monkeypatch, caplog):
    import fedlearn.server.grpc_servicer as servicer_module

    def too_large(*_args, **_kwargs):
        raise MemoryError

    monkeypatch.setattr(servicer_module, "parameters_to_proto", too_large)
    ctx = _AbortContext()
    with pytest.raises(_Aborted):
        _model_servicer((OrderedDict({"w": torch.zeros(2)}), 1, {})).GetGlobalModel(
            pb.GetGlobalModelRequest(client_id="c"), ctx)
    assert ctx.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert "RPC failed" not in caplog.text
