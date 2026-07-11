"""DA-11: federate only a model's requires_grad (trainable) subset under FedAvg.

The wire payload is estimators.params.trainable_state (FR-14 layout). This module adds the two
pieces FedAvg lacks for a partially-trainable (frozen-backbone) model: a fail-loud key-guard so a
client whose trainable keys differ from the server's is rejected (never silently averaged), and a
non-strict reconstruction that writes the aggregated subset back onto a net whose frozen params stay
intact. The averaging itself is unchanged — FedAvgAggregator already averages over the keys it is handed.
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


def validate_subset_update(update_keys: list[str], expected_keys: list[str]) -> None:
    """Raise SubsetDimMismatch unless update_keys == expected_keys (order-sensitive)."""
    if list(update_keys) != list(expected_keys):
        raise SubsetDimMismatch(
            f"trainable-subset mismatch: client sent {list(update_keys)} but the server expects "
            f"{list(expected_keys)} (requires_grad params in named_parameters order). Send "
            f"estimators.params.trainable_state(model), NOT a full state_dict()."
        )
