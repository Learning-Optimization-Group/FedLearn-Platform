"""SE-3 — model-poisoning defense: non-finite (NaN/Inf) values are rejected before they can reach
aggregation, on both the FedAvg (serializer) and DeComFL (scalar submit) paths. One malformed or
malicious client must not be able to destroy the global model for every honest client in a round.
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.communication.serializer import (
    chunks_to_parameters,
    parameters_to_chunks,
    parameters_to_proto,
    proto_to_parameters,
)
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL


# --- FedAvg path: the serializer must reject non-finite tensor values -------------------------
def test_proto_to_parameters_rejects_nan():
    poisoned = OrderedDict({"w": torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float32)})
    proto = parameters_to_proto(poisoned, 10)
    with pytest.raises(ValueError, match="non-finite"):
        proto_to_parameters(proto)


def test_proto_to_parameters_rejects_inf():
    poisoned = OrderedDict({"w": torch.tensor([1.0, float("inf")], dtype=torch.float32)})
    proto = parameters_to_proto(poisoned, 10)
    with pytest.raises(ValueError, match="non-finite"):
        proto_to_parameters(proto)


def test_chunks_to_parameters_rejects_nan():
    poisoned = OrderedDict({"w": torch.tensor([[1.0, float("nan")], [3.0, 4.0]], dtype=torch.float32)})
    blob = b"".join(c["chunk_data"] for c in parameters_to_chunks(poisoned, num_examples=5, chunk_size=1024 * 1024))
    with pytest.raises(ValueError, match="non-finite"):
        chunks_to_parameters(blob, compressed=False)


def test_finite_parameters_still_round_trip():
    clean = OrderedDict({"w": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)})
    params, n = proto_to_parameters(parameters_to_proto(clean, 7))
    assert n == 7
    assert torch.allclose(params["w"], clean["w"])


# --- DeComFL path: the coordinator must reject a submission with non-finite gradient scalars ----
def _decomfl_coordinator(clients_per_round: int) -> FLCoordinator:
    strat = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(3)}),
        evaluate_fn=None, min_fit_clients=1, clients_per_round=clients_per_round,
        num_local_steps=1, num_perturbations=2, learning_rate=0.01, smoothing_param=0.001, seed=1,
    )
    return FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=clients_per_round)


def test_decomfl_submit_rejects_non_finite_scalars_without_corrupting_the_round():
    coord = _decomfl_coordinator(clients_per_round=2)

    coord.submit_decomfl_update("good", [[0.1, 0.2]], 100, coord.current_round)
    assert len(coord._client_updates_received) == 1  # honest update accepted

    coord.submit_decomfl_update("poisoned", [[float("nan"), 0.2]], 100, coord.current_round)
    assert len(coord._client_updates_received) == 1  # NaN submission rejected, round not corrupted
