"""DA-11: federate only a model's requires_grad (trainable) subset under FedAvg.

The wire payload is estimators.params.trainable_state (FR-14 layout). This module adds the pieces
FedAvg lacks for a partially-trainable (frozen-backbone) model: a fail-loud, shape-aware guard
(``validate_subset_update``) so a client whose trainable keys OR shapes differ from the server's is
rejected (never silently averaged), a per-client pre-aggregation sweep (``guard_client_updates``),
and a non-strict reconstruction that writes the aggregated subset back onto a net whose frozen
params stay intact. The averaging itself is unchanged — FedAvgAggregator already averages over the
keys it is handed.

The guard MUST run per-client, BEFORE aggregation: FedAvgAggregator.aggregate() derives its output
key-set from the FIRST client's update and silently skips (`if key in params`) any key a LATER
client is missing (see ``strategy.FedAvgAggregator.aggregate``) — so validating the AGGREGATED
output can never catch a non-first client's bad payload; it always reflects the first client's key
set. ``guard_client_updates`` is the fix: call it on the raw per-client payload list before handing
them to the aggregator.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from fedlearn.estimators.params import trainable_state


class SubsetDimMismatch(ValueError):
    """A client's trainable subset (keys/order) does not match the server's expected trainable layout."""


def expected_trainable_keys(model: nn.Module) -> list[str]:
    """The ordered trainable (requires_grad) parameter names — the server's expected wire keys."""
    return list(trainable_state(model).keys())


def validate_subset_update(update: "OrderedDict[str, torch.Tensor]", model: nn.Module) -> None:
    """Raise SubsetDimMismatch unless `update` matches model's expected trainable layout on BOTH
    axes: the key SET AND each tensor's shape. A same-key/wrong-shape update (e.g. a misconfigured
    client's head) is exactly as much a contract violation as a wrong key set, so it raises this
    same typed error instead of falling through to load_state_dict's raw, untyped RuntimeError.

    Key comparison is order-INSENSITIVE (set-based), and deliberately so. The FedAvg subset path is
    entirely by-NAME — FedAvgAggregator averages per key and apply_trainable_subset writes back via
    load_state_dict(strict=False), neither of which indexes positionally — so key ORDER carries no
    safety signal here. It also can't be relied on: a small (non-transformer) head takes the UNARY
    upload path (grpc_client._submit_update_unary → serializer.parameters_to_proto), and a protobuf
    ``map<string, Tensor>`` field iterates in an UNSPECIFIED order, so trainable_state()'s
    named_parameters order does NOT survive that transport. An order-sensitive check therefore
    false-rejected every legitimate head update that traveled the unary path — the exact DA-14
    frozen-backbone use case. This mirrors distribution.reconstruct_frozen_backbone, which already
    compares frozen-backbone keys as a set. (The order-CRITICAL DeComFL flat-vector layout is a
    different path — estimators.params.param_layout/flat_params — and is untouched by this.)
    """
    expected = trainable_state(model)
    update_keys = set(update.keys())
    expected_keys = set(expected.keys())
    if update_keys != expected_keys:
        raise SubsetDimMismatch(
            f"trainable-subset mismatch: client sent {sorted(update_keys)} but the server expects "
            f"{sorted(expected_keys)} (requires_grad params). Send "
            f"estimators.params.trainable_state(model), NOT a full state_dict()."
        )
    for name, expected_tensor in expected.items():
        actual_shape = tuple(update[name].shape)
        expected_shape = tuple(expected_tensor.shape)
        if actual_shape != expected_shape:
            raise SubsetDimMismatch(
                f"trainable-subset shape mismatch for {name!r}: client sent shape {actual_shape} "
                f"but the server expects {expected_shape} (model parameter shape)."
            )


def guard_client_updates(
    client_payloads: "list[OrderedDict[str, torch.Tensor]]", model: nn.Module
) -> None:
    """Validate EVERY client's raw payload against model's expected trainable layout BEFORE
    aggregation runs (FINDING 1). This is where the fail-loud guarantee actually lives:
    FedAvgAggregator.aggregate() derives its output key-set from the FIRST client's update and
    silently skips (`if key in params`) any key a LATER client is missing, so a non-first client
    with a bad payload would otherwise be averaged with no error. Raises on the first bad client
    (SubsetDimMismatch), so a malformed client is rejected and the round never proceeds to
    aggregation."""
    for payload in client_payloads:
        validate_subset_update(payload, model)


def apply_trainable_subset(model: nn.Module, subset: "OrderedDict[str, torch.Tensor]") -> None:
    """Write an aggregated trainable subset back onto model (non-strict, so the frozen backbone is
    preserved). Validates keys+shapes against the model's expected trainable layout first
    (fail-loud); with that guard in place, `unexpected` from load_state_dict can never be
    non-empty at this call site (see subset_federation module docstring / DA-11 review)."""
    validate_subset_update(subset, model)
    model.load_state_dict(subset, strict=False)
