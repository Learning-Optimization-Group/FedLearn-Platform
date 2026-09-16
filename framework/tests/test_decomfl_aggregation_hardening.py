"""Adversarial hardening for the DeComFL zeroth-order path (paper C7 — the cross-language contract).

Three previously-untested edges of the FR-14 trainable-layout guard and the aggregate/rebuild
lockstep:

  1. trainable_state must exclude *buffers* (BatchNorm running stats), not only frozen params. The
     existing layout-contract tests exercise frozen params (fc2 requires_grad=False); buffers are a
     distinct mechanism (named_buffers, absent from named_parameters) that the FR-14 docstring calls
     out explicitly. Pinned here so d_server == d_client holds for a buffer-bearing model.

  2. The buffer half of the silent-divergence bug: building a DeComFL server from a full state_dict
     that carries a *float* buffer inflates the server's flat vector past the client's trainable dim.
     validate_participant_dim must fail loud (the frozen-param case is covered in
     test_decomfl_layout_contract; the buffer case was not).

  3. Multi-client (N>1) server<->client-rebuild lockstep through the coordinator's REAL
     _calculate_average_gradients: the averaged gradient_history a client replays must reproduce the
     server's aggregate_fit step exactly. Existing correctness tests either use a single client for
     the rebuild comparison or hand-craft a single gradient_history, so the num_clients normalization
     in the averaged-history path was never checked against aggregate_fit for N>1.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from fedlearn.estimators.params import num_trainable, trainable_state
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.client.decomfl_client import DeComFLClient


class _BNNet(nn.Module):
    """BatchNorm (trainable weight+bias, + float running_mean/var and int64 num_batches_tracked
    buffers) followed by a Linear head. 8 bn params + 15 head = 23 trainable; the buffers are NOT."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not exercised
        return self.head(self.bn(x))


class _BufNet(nn.Module):
    """Linear(3,2) = 8 trainable params + a registered FLOAT buffer of 5. The float buffer keeps a
    full state_dict() torch.cat-able (unlike BatchNorm's int64 counter) while still inflating it past
    the 8 trainable dims — the exact 'buffers inflate the server's flat vector' shape FR-14 warns of."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)
        self.register_buffer("prior", torch.zeros(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not exercised
        return self.fc(x) + self.prior[:2]


def test_trainable_state_excludes_buffers_and_keeps_model_dim_equal_to_num_trainable():
    m = _BNNet()
    ts = trainable_state(m)

    buffer_names = {name for name, _ in m.named_buffers()}
    assert buffer_names == {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
    # No buffer (float running stats OR the int64 counter) leaks into the trainable layout.
    assert buffer_names.isdisjoint(ts.keys())
    assert set(ts.keys()) == {"bn.weight", "bn.bias", "head.weight", "head.bias"}

    # The full state_dict carries the three extra buffer entries; trainable_state does not.
    assert set(m.state_dict().keys()) - set(ts.keys()) == buffer_names

    # d_server (from trainable_state) == d_client (num_trainable) — the FR-14 invariant.
    assert sum(t.numel() for t in ts.values()) == num_trainable(m) == 23
    server = DeComFL(initial_parameters=ts, min_fit_clients=1,
                     num_local_steps=1, num_perturbations=1, seed=1)
    assert server.model_dim == num_trainable(m)
    server.validate_participant_dim(num_trainable(m))          # no raise
    server.validate_participant_dim(num_trainable(m), "phone")  # no raise


def test_float_buffer_state_dict_inflates_model_dim_and_is_detected_not_silent():
    m = _BufNet()
    assert num_trainable(m) == 8

    # Correct: server built from the trainable layout matches the client dim.
    good = DeComFL(initial_parameters=trainable_state(m), min_fit_clients=1,
                   num_local_steps=1, num_perturbations=1, seed=1)
    assert good.model_dim == 8
    good.validate_participant_dim(8)  # no raise

    # BUG shape: a full state_dict (incl. the float 'prior' buffer) inflates the flat vector to 13.
    bad = DeComFL(initial_parameters=OrderedDict(m.state_dict()), min_fit_clients=1,
                  num_local_steps=1, num_perturbations=1, seed=1)
    assert bad.model_dim == 13
    assert bad.model_dim > num_trainable(m)  # would silently misalign the shared-seed z
    import pytest
    with pytest.raises(ValueError, match="dimension mismatch"):
        bad.validate_participant_dim(num_trainable(m), client_id="phone")


class _TinyNet(nn.Module):
    """Linear(3,1) -> fc.weight [1,3] + fc.bias [1]; d=4 (matches the other DeComFL test nets)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not exercised
        return self.fc(x)


def test_multi_client_aggregate_matches_client_rebuild_from_averaged_history():
    """N=3 clients: the server's aggregate_fit step must equal a client replaying the averaged
    gradient_history the coordinator actually hands out (_calculate_average_gradients). This exercises
    the num_clients normalization that the single-client rebuild tests never hit: aggregate_fit divides
    by (num_clients * P) in one shot, while the client divides by num_clients (in the averaged history)
    and by P (via lr/P) separately — the two must land on the same global.
    """
    torch.manual_seed(0)
    model = _TinyNet()
    init = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
    K, P, eta = 2, 3, 0.05
    strat = DeComFL(initial_parameters=init, evaluate_fn=None, min_fit_clients=1, clients_per_round=3,
                    num_local_steps=K, num_perturbations=P, learning_rate=eta, smoothing_param=0.001,
                    seed=123)
    seeds = strat.generate_seeds(1)
    strat.seed_history[1] = seeds

    grads_a = [[0.20 * (k + 1) + 0.03 * p for p in range(P)] for k in range(K)]
    grads_b = [[-0.10 * (k + 1) + 0.05 * p for p in range(P)] for k in range(K)]
    grads_c = [[0.07 * (k + 1) - 0.02 * p for p in range(P)] for k in range(K)]
    results = [("a", grads_a, 100), ("b", grads_b, 100), ("c", grads_c, 100)]

    x0 = strat.global_params_flat.detach().cpu().clone()
    strat.aggregate_fit(1, results)
    server_flat = strat.global_params_flat.detach().cpu().clone()
    assert not torch.allclose(server_flat, x0, atol=1e-6), "aggregate_fit did not move the global model"

    # The averaged gradient_history a client is actually handed to rebuild locally.
    coord = FLCoordinator(strat, min_clients_for_aggregation=1, clients_per_round=3)
    avg_gradients = coord._calculate_average_gradients(results)

    ref_model = _TinyNet()
    ref_model.load_state_dict(init)
    client = DeComFLClient(model=ref_model, train_loader=None, device="cpu")
    client.rebuild_model([{"round_number": 1, "seeds": seeds, "gradients": avg_gradients}],
                         learning_rate=eta)
    client_flat = client.x_current.detach().cpu().clone()

    assert torch.allclose(server_flat, client_flat, atol=1e-6), (
        f"server aggregate {server_flat.tolist()} != client rebuild {client_flat.tolist()} "
        "for a 3-client round"
    )
