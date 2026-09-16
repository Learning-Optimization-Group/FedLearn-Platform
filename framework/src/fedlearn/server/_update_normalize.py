"""Canonical client-update normalizer shared by every aggregation path.

Four aggregation sites accept the same client-update wire shapes and coerce them the same way:
FedAvg (``FedAvgAggregator.aggregate``), the FedLoRA key probe (``FedLoRA._client_keys``), the
Byzantine-robust path (``robust_aggregation._normalize_updates``), and central-DP
(``privacy.dp_mechanism``). This module is the single implementation they all call, so the
front-matter (2-/3-tuple unpacking, JSON decode, tensor construction, error message) lives in
exactly one place instead of four near-identical copies.

An update entry is either ``(client_id, params, num_examples)`` or ``(params, num_examples)``
(``client_id`` defaults to ``None``), and ``params`` is either an ``OrderedDict[str, Tensor]``
already or a JSON string that decodes to a plain ``{name: list}`` dict. Every entry is coerced to
``(client_id, OrderedDict[str, Tensor], num_examples)``; callers that don't need ``num_examples``
(the DP uniform average, the FedLoRA key probe) simply ignore the third element.
"""

import json
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch


def normalize_update(
    entry: tuple,
) -> Tuple[Optional[str], "OrderedDict[str, torch.Tensor]", int]:
    """Coerce ONE update entry into ``(client_id, state_dict, num_examples)``.

    ``entry`` is ``(client_id, params, num_examples)`` or ``(params, num_examples)`` (``client_id``
    then defaults to ``None``); ``params`` is an ``OrderedDict[str, Tensor]`` or a JSON string
    decoding to ``{name: list}``. A JSON payload that fails to decode/convert raises ``ValueError``
    naming the offending client.
    """
    if len(entry) == 3:
        client_id, params, num_examples = entry
    else:
        params, num_examples = entry
        client_id = None

    if isinstance(params, str):
        try:
            decoded = json.loads(params)
            params = OrderedDict((k, torch.tensor(v)) for k, v in decoded.items())
        except Exception as e:  # noqa: BLE001 — surface the offending client id
            raise ValueError(f"Failed to deserialize parameters from {client_id}: {e}")

    return client_id, params, num_examples


def normalize_updates(
    results: "List[tuple]",
) -> "List[Tuple[Optional[str], OrderedDict[str, torch.Tensor], int]]":
    """Apply :func:`normalize_update` across a list of update entries."""
    return [normalize_update(entry) for entry in results]
