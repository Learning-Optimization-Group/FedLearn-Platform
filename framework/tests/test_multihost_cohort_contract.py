"""Opt-in fixed-cohort admission and deterministic transport aggregation."""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.composed_strategy import ComposedStrategy
from fedlearn.server.coordinator import FLCoordinator


def params(value=0.0):
    return OrderedDict(head=torch.tensor([value], dtype=torch.float32))


def strategy(ids=("b", "a")):
    result = ComposedStrategy(params(), rule="mean", min_fit_clients=2, clients_per_round=2)
    result.allowed_client_ids = ids
    return result


def coordinator(aggregation_strategy=None, *, async_completion=False, minimum=2, count=2):
    aggregation_strategy = strategy() if aggregation_strategy is None else aggregation_strategy
    result = FLCoordinator(aggregation_strategy, minimum, count, num_rounds=1,
                           async_round_completion=async_completion)
    result.set_initial_parameters(params())
    return result


@pytest.mark.parametrize("ids", [None, "ab", {"a", "b"}, [], ["a", "a"], ["", "b"], [1, "b"]])
def test_malformed_membership_fails_before_admission(ids):
    with pytest.raises(ValueError, match="allowed_client_ids"):
        coordinator(strategy(ids))


@pytest.mark.parametrize("minimum,count", [(1, 2), (2, 3), (3, 2)])
def test_fixed_membership_requires_an_exact_full_cohort(minimum, count):
    with pytest.raises(ValueError, match="cohort"):
        coordinator(minimum=minimum, count=count)


def test_unknown_registration_and_unregistered_submissions_do_not_count():
    server = coordinator()
    assert server.register_client("outsider") is False
    assert server.submit_client_update("outsider", params(10), 20, 1) is False
    assert server.submit_client_update("a", params(10), 20, 1) is False
    assert server.register_client("a") is True
    assert server.submit_client_update("a", params(2), 20, 1) is True
    assert server.get_global_model_params()["head"].item() == 0
    assert server.current_round == 1


@pytest.mark.parametrize("bad", [
    OrderedDict(other=torch.tensor([1.0])),
    OrderedDict(head=torch.tensor([1.0]), extra=torch.tensor([1.0])),
    OrderedDict(head=torch.tensor([1.0, 2.0])),
    OrderedDict(head=torch.tensor([1.0], dtype=torch.float64)),
    OrderedDict(head=torch.tensor([1], dtype=torch.int64)),
    OrderedDict(head=torch.tensor([float("nan")])),
    OrderedDict(head=torch.tensor([float("inf")])),
])
def test_incompatible_updates_are_rejected_without_consuming_the_member(bad):
    server = coordinator()
    server.register_client("a")
    server.register_client("b")
    with pytest.raises(ValueError):
        server.submit_client_update("a", bad, 4, 1)
    assert server.submit_client_update("a", params(2), 4, 1)
    assert server.submit_client_update("b", params(6), 400, 1)
    assert server.get_global_model_params()["head"].item() == 4


class OrderedReduction(ComposedStrategy):
    """A reducer whose output exposes the order it actually receives."""

    def aggregate_fit(self, server_round, results):
        digits = {"a": 1, "b": 2}
        value = 0
        for client_id, _parameters, _examples in results:
            value = 10 * value + digits[client_id]
        return params(value)


@pytest.mark.parametrize("async_completion", [False, True])
def test_declared_identity_order_controls_real_reducer_input(async_completion):
    reducer = OrderedReduction(params(), min_fit_clients=2, clients_per_round=2)
    reducer.allowed_client_ids = ["b", "a"]
    server = coordinator(reducer, async_completion=async_completion)
    for cid in ("a", "b"):
        server.register_client(cid)
        assert server.submit_client_update(cid, params(1), 1, 1)
    if async_completion:
        server._aggregation_worker.join(timeout=5)
        assert not server._aggregation_worker.is_alive()
    assert server.get_global_model_params()["head"].item() == 21
    assert server.current_round == 2


@pytest.mark.parametrize("async_completion", [False, True])
def test_unequal_shards_still_receive_uniform_weight_with_complete_receipts(async_completion):
    server = coordinator(async_completion=async_completion)
    for cid in ("a", "b"):
        server.register_client(cid)
    assert not server.submit_client_update("a", params(30), 9, 0)
    assert not server.submit_client_update("a", params(30), 9, 2)
    assert server.submit_client_update("a", params(2), 1, 1)
    assert not server.submit_client_update("a", params(100), 1000, 1)
    assert server.submit_client_update("b", params(6), 99, 1)
    if async_completion:
        server._aggregation_worker.join(timeout=5)
        assert not server._aggregation_worker.is_alive()
    assert server.get_global_model_params()["head"].item() == 4
    assert not server.submit_client_update("b", params(100), 99, 1)


def test_missing_member_deadline_stops_without_publishing_partial_model():
    server = coordinator()
    server.register_client("a")
    assert server.submit_client_update("a", params(8), 99, 1)
    server.resolve_round_incomplete("test deadline")
    assert server.stop_requested
    assert server.last_round_failed
    assert not server.training_complete
    assert server.current_round == 1
    assert server.get_global_model_params()["head"].item() == 0


def test_no_manifest_retains_permissive_generic_admission():
    generic = ComposedStrategy(params(), min_fit_clients=1, clients_per_round=1)
    server = coordinator(generic, minimum=1, count=1)
    assert server.register_client("any identity")
    assert server.submit_client_update("never registered", params(7), 1, 1)
    assert server.get_global_model_params()["head"].item() == 7


def test_tensor_head_contract_refuses_scalar_updates_without_consuming_a_member():
    server = coordinator()
    server.register_client("a")
    server.register_client("b")
    with pytest.raises(ValueError, match="tensor head"):
        server.submit_decomfl_update("a", [[0.1]], 5, 1)
    assert server.submit_client_update("a", params(2), 1, 1)
    assert server.submit_client_update("b", params(6), 99, 1)
    assert server.get_global_model_params()["head"].item() == 4
