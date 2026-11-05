"""Protobuf model maps have no stable iteration order; named state remains equivalent."""
from collections import OrderedDict

import torch

from fedlearn.server.checkpoint import capture_round
from fedlearn.server.strategy import FedAvg


def test_capture_uses_original_layout_for_equivalent_unordered_wire_state():
    initial = OrderedDict(weight=torch.zeros(2), bias=torch.zeros(1))
    received = OrderedDict(bias=torch.ones(1), weight=torch.ones(2))
    snapshot = capture_round(FedAvg(initial), completed_round=1, total_rounds=2,
                             parameters=received, history=[(1, None)])
    assert list(snapshot.parameters) == ["weight", "bias"]
    assert torch.equal(snapshot.parameters["weight"], received["weight"])
    assert list(received) == ["bias", "weight"]
