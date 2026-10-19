"""A DeComFL client can hold each round to a statement made before the run (wikis/mobile/08).

The server's per-round DeComFL config decides everything a client trains: rate, smoothing, the K x P seed matrix
and the gradient estimator. A caller with an execution contract must be able to check that config before any
training happens, and must be able to see the estimator the server announced at all.
"""
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from fedlearn.client.decomfl_client import DeComFLClient
from fedlearn.client.grpc_client import GrpcClient
from fedlearn.communication.generated import fedlearn_pb2


class _Stub:
    def __init__(self, response):
        self.response = response

    def GetDeComFLConfig(self, request):   # noqa: N802 - gRPC method name
        return self.response


def test_the_client_config_carries_the_servers_gradient_estimator():
    response = fedlearn_pb2.GetDeComFLConfigResponse(
        current_round=1,
        current_seeds=fedlearn_pb2.PerturbationSeeds(local_steps=[fedlearn_pb2.LocalStepSeeds(seeds=[7, 8])]),
        config={"learning_rate": "0.001", "smoothing_param": "0.001"},
        grad_estimate_method="forward",
    )
    client = GrpcClient("localhost:1", "c")
    client.stub = _Stub(response)

    _round, seeds, _history, config = client.get_decomfl_config()

    assert config["grad_estimate_method"] == "forward"
    assert seeds == [[7, 8]]


def _client():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,))), batch_size=8)
    return DeComFLClient(model=model, train_loader=loader, smoothing_param=0.001)


def test_a_round_check_runs_before_any_training():
    client = _client()
    before = OrderedDict((k, v.clone()) for k, v in client.model.state_dict().items())
    seen = []

    def refuse(config):
        seen.append(config)
        raise ValueError("the server asked for training the execution contract does not state")

    client.round_check = refuse
    config = {"seeds": [[1, 2]], "learning_rate": "0.001", "smoothing_param": "0.001"}
    with pytest.raises(ValueError, match="execution contract"):
        client.fit(None, config)

    assert seen == [config]
    assert all(torch.equal(before[k], v) for k, v in client.model.state_dict().items())


def test_without_a_round_check_the_client_trains_as_before():
    client = _client()
    assert client.round_check is None
    scalars, _examples = client.fit(None, {"seeds": [[1, 2]], "learning_rate": "0.001", "smoothing_param": "0.001"})
    assert len(scalars) == 1 and len(scalars[0]) == 2
