"""SE-3 — model-poisoning defense: non-finite (NaN/Inf) values are rejected before they can reach
aggregation, on both the FedAvg (serializer) and DeComFL (scalar submit) paths. One malformed or
malicious client must not be able to destroy the global model for every honest client in a round.
"""
from collections import OrderedDict
from unittest.mock import Mock

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


# --- DeComFL path, layer 2: a finite-but-LARGE adversarial scalar must be CLAMPED (not rejected) --
# to a bounded magnitude at ingress, before it is stored. Both downstream consumers — aggregate_fit
# (steps the real global model) and _calculate_average_gradients (feeds gradient_history, which
# clients replay to rebuild locally) — read the stored scalars, so clamping once here keeps the
# server model and the client-rebuilt model in lockstep. Clamp preserves liveness; NaN is dropped.
def test_decomfl_submit_bounds_a_large_poisoning_scalar():
    # Threshold-independent: an astronomically large scalar must be reduced, whatever the default is.
    coord = _decomfl_coordinator(clients_per_round=2)
    coord.submit_decomfl_update("attacker", [[1e12, -1e12]], 100, coord.current_round)
    _, stored, _ = coord._client_updates_received[0]        # update accepted (clamped, not dropped)
    assert abs(stored[0][0]) < 1e12, "large poisoning scalar was not bounded"
    assert abs(stored[0][1]) < 1e12


def test_decomfl_submit_clamps_symmetrically_to_the_threshold():
    coord = _decomfl_coordinator(clients_per_round=2)
    tau = coord.grad_clip_threshold
    coord.submit_decomfl_update("attacker", [[tau * 1e6, -tau * 1e6]], 100, coord.current_round)
    _, stored, _ = coord._client_updates_received[0]
    assert stored == [[tau, -tau]]                          # clamped into [-tau, tau], sign preserved


def test_decomfl_submit_leaves_in_range_honest_scalars_untouched():
    # Guard against convergence bias: a scalar within [-tau, tau] must pass through as the identity.
    coord = _decomfl_coordinator(clients_per_round=2)
    coord.submit_decomfl_update("good", [[0.1, -0.2]], 100, coord.current_round)
    _, stored, _ = coord._client_updates_received[0]
    assert stored == [[0.1, -0.2]]


# --- FedAvg path (coordinator): identity-tagging, self-defending isfinite, and delta norm-clip -----
# The serializer already rejects non-finite params on the gRPC path; these cover the coordinator being
# self-defending (so a direct caller can't bypass it), attributability, and the configurable L2 clip.
def _fedavg_coordinator(clients_per_round: int = 2, l2_clip=None) -> FLCoordinator:
    # A bare Mock strategy is fine: submitting fewer than clients_per_round never triggers aggregation,
    # so the strategy is never invoked — we assert only on the coordinator's ingress behavior.
    return FLCoordinator(
        Mock(), min_clients_for_aggregation=1, clients_per_round=clients_per_round,
        client_update_l2_clip=l2_clip,
    )


def test_fedavg_submit_tags_the_authenticated_client_identity():
    coord = _fedavg_coordinator()
    coord.submit_client_update("alice", OrderedDict({"w": torch.tensor([1.0, 2.0])}), 100, coord.current_round)
    entry = coord._client_updates_received[0]
    assert len(entry) == 3, "an accepted update must carry (client_id, params, num_examples)"
    assert entry[0] == "alice", "the update is attributable to the submitting client"


def test_fedavg_submit_rejects_non_finite_params():
    coord = _fedavg_coordinator()
    poisoned = OrderedDict({"w": torch.tensor([1.0, float("nan"), 3.0])})
    with pytest.raises(ValueError, match="non-finite"):
        coord.submit_client_update("attacker", poisoned, 100, coord.current_round)
    assert len(coord._client_updates_received) == 0, "the poisoned update never reaches aggregation"


def test_fedavg_submit_clips_an_over_norm_delta_to_the_bound():
    coord = _fedavg_coordinator(l2_clip=1.0)
    coord._global_model_params = OrderedDict({"w": torch.zeros(4)})
    # Delta from the zero global has L2 norm 10.0 — 10x over the 1.0 budget.
    coord.submit_client_update("c", OrderedDict({"w": torch.tensor([5.0, 5.0, 5.0, 5.0])}), 100, coord.current_round)
    _, stored, _ = coord._client_updates_received[0]
    delta_norm = torch.sqrt(sum((t * t).sum() for t in stored.values())).item()  # global is 0, so ||delta||=||stored||
    assert delta_norm <= 1.0 + 1e-4, f"over-norm delta was not clipped to the bound (got {delta_norm})"


def test_fedavg_submit_leaves_an_in_bound_delta_unchanged():
    coord = _fedavg_coordinator(l2_clip=100.0)
    coord._global_model_params = OrderedDict({"w": torch.zeros(2)})
    coord.submit_client_update("c", OrderedDict({"w": torch.tensor([1.0, 2.0])}), 100, coord.current_round)  # ||delta||=sqrt(5)
    _, stored, _ = coord._client_updates_received[0]
    assert torch.allclose(stored["w"], torch.tensor([1.0, 2.0])), "an in-budget update must pass through unchanged"


def test_fedavg_submit_rejects_a_float32_overflowing_value_before_the_clip():
    # Review finding: a value finite in float64 but overflowing float32 (the aggregation precision)
    # must be rejected at ingress. With the clip enabled it would otherwise downcast to inf, be scaled
    # by 0 to NaN, and poison the round — a bypass of the very defense the clip adds.
    coord = _fedavg_coordinator(l2_clip=1.0)
    coord._global_model_params = OrderedDict({"w": torch.zeros(4)})
    poisoned = OrderedDict({"w": torch.full((4,), 1e300, dtype=torch.float64)})
    with pytest.raises(ValueError, match="non-finite"):
        coord.submit_client_update("attacker", poisoned, 100, coord.current_round)
    assert len(coord._client_updates_received) == 0, "the float32-overflow update never reaches aggregation"


def test_fedavg_submit_still_clips_a_huge_but_representable_delta_rather_than_rejecting():
    # Guard the fix from over-rejecting: a large-but-float32-finite delta must still be CLIPPED
    # (bounded), not rejected, and the clip must never manufacture a non-finite value.
    coord = _fedavg_coordinator(l2_clip=1.0)
    coord._global_model_params = OrderedDict({"w": torch.zeros(4)})
    huge = OrderedDict({"w": torch.full((4,), 1e30)})   # float32-finite; ||delta|| = 2e30 >> 1.0
    coord.submit_client_update("c", huge, 100, coord.current_round)
    _, stored, _ = coord._client_updates_received[0]
    norm = torch.sqrt(sum((t * t).sum() for t in stored.values())).item()
    assert norm <= 1.0 + 1e-4, f"a representable-but-huge delta must be clipped to the bound (got {norm})"
    assert torch.isfinite(stored["w"]).all(), "clipping must not manufacture a non-finite value"
