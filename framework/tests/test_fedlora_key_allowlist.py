"""FR-23: FedLoRA must aggregate ONLY the server's expected adapter surface.

`_assert_homogeneous` compares clients against EACH OTHER, never against the server's known adapter
key set (which it holds in initial_parameters). So a min-clients=1 client — or a colluding full
cohort — can append keys outside the adapter (e.g. poisoned base-model weights) that get averaged
into the global, broadcast to every peer, and packaged into the LORA_ADAPTER registry bundle. A
server-side allowlist (client keys ⊆ initial_parameters) closes that surface.
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.strategy import FedLoRA


def _adapter() -> OrderedDict:
    # Minimal FFA adapter: frozen A + trainable B + head (matches the repo's key convention).
    return OrderedDict([
        ("lora_A.l", torch.randn(2, 4)),
        ("lora_B.l", torch.zeros(3, 2)),
        ("head.w", torch.zeros(5)),
    ])


def _make_fedlora(aggregation="FFA_LORA") -> FedLoRA:
    return FedLoRA(initial_parameters=_adapter(), aggregation=aggregation, clients_per_round=1)


def test_fedlora_rejects_smuggled_non_adapter_keys():
    """A single client (or colluding cohort) appending a key outside the adapter surface is refused."""
    strat = _make_fedlora()
    smuggled = OrderedDict([
        ("lora_A.l", torch.zeros(2, 4)),
        ("lora_B.l", torch.ones(3, 2)),
        ("head.w", torch.ones(5)),
        ("base_model.embed_tokens.weight", torch.full((10, 4), 99.0)),  # poisoned base-model tensor
    ])
    with pytest.raises(ValueError, match="allowlist|surface|outside|adapter"):
        strat.aggregate_fit(1, [("c1", smuggled, 10)])


def test_fedlora_accepts_honest_adapter_only_update():
    """Regression: an honest client that sends exactly the adapter surface still aggregates."""
    strat = _make_fedlora()
    honest = OrderedDict([
        ("lora_A.l", torch.zeros(2, 4)),
        ("lora_B.l", torch.ones(3, 2)),
        ("head.w", torch.ones(5)),
    ])
    out = strat.aggregate_fit(1, [("c1", honest, 10)])
    assert set(out.keys()) == {"lora_A.l", "lora_B.l", "head.w"}
