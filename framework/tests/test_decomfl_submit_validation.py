"""FR-5 — DeComFL submission validation at the coordinator ingress.

Two integrity gaps, independent of the poisoning-value defenses (SE-3):

  1. Shape: a submission whose gradient-scalar grid is not exactly K x P would otherwise reach
     aggregate_fit (which indexes grad_scalars[k][p]) and crash the round on the aggregation thread
     — AFTER the client was already told received=True. It is rejected here so the servicer can map
     it to a gRPC INVALID_ARGUMENT the client actually sees.

  2. Dedup: a client that submits twice in one round would otherwise be appended twice and
     double-counted in the averaged update, inflating its own weight. The second submission is
     ignored (the first accepted one is retained).
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.coordinator import FLCoordinator, MalformedDeComFLSubmission
from fedlearn.server.decomfl_strategy import DeComFL


def _coord(clients_per_round: int, K: int = 1, P: int = 2) -> FLCoordinator:
    strat = DeComFL(
        OrderedDict({"w": torch.zeros(3)}), evaluate_fn=None, min_fit_clients=1,
        clients_per_round=clients_per_round, num_local_steps=K, num_perturbations=P,
        learning_rate=0.01, smoothing_param=0.001, seed=1,
    )
    return FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=clients_per_round)


def test_submit_with_wrong_perturbation_count_is_malformed():
    coord = _coord(clients_per_round=2, K=1, P=2)
    with pytest.raises(MalformedDeComFLSubmission):
        coord.submit_decomfl_update("c", [[0.1]], 100, coord.current_round)          # P=1, expected 2


def test_submit_with_wrong_local_step_count_is_malformed():
    coord = _coord(clients_per_round=2, K=1, P=2)
    with pytest.raises(MalformedDeComFLSubmission):
        coord.submit_decomfl_update("c", [[0.1, 0.2], [0.3, 0.4]], 100, coord.current_round)  # K=2, exp 1


def test_duplicate_submission_from_same_client_is_ignored():
    coord = _coord(clients_per_round=3, K=1, P=2)   # 3 so a single client's two submits never aggregate
    coord.submit_decomfl_update("c", [[0.1, 0.2]], 100, coord.current_round)
    coord.submit_decomfl_update("c", [[0.9, 0.9]], 100, coord.current_round)          # duplicate
    assert len(coord._client_updates_received) == 1                                   # not double-counted
    _, stored, _ = coord._client_updates_received[0]
    assert stored == [[0.1, 0.2]]                                                     # first update retained


def test_valid_shape_is_accepted():
    coord = _coord(clients_per_round=2, K=1, P=2)
    coord.submit_decomfl_update("c", [[0.1, 0.2]], 100, coord.current_round)
    assert len(coord._client_updates_received) == 1
