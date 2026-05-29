"""DeComFL correctness regression tests (the v2 P0 fix).

Pins the three contracts from docs/v2/specs/2026-05-29-decomfl-correctness-design.md as they
apply to the origin/main code:
  * Bug 1 (1/P): the server's aggregated trajectory must equal the client's rebuild trajectory.
  * Bug 2 (RNG): server and client regenerate identical, CPU-canonical perturbations from a seed.
  * B-2: constructing the strategy must not mutate the process-global RNG.
  * Bug 3 (serializer): chunked save/load is symmetric (already fixed on origin/main; hardened here).
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.client.decomfl_client import DeComFLClient
from fedlearn.estimators.perturbation import canonical_perturbation
from fedlearn.communication.serializer import parameters_to_chunks, chunks_to_parameters


class TinyNet(nn.Module):
    """Linear(3, 1) -> state_dict keys fc.weight [1,3] + fc.bias [1] (matches make_params; d=4)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):  # noqa: D401
        return self.fc(x)


def _make_strategy(model: nn.Module, K: int, P: int, eta: float) -> DeComFL:
    init = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
    return DeComFL(
        initial_parameters=init,
        evaluate_fn=None,
        min_fit_clients=1,
        clients_per_round=2,
        num_local_steps=K,
        num_perturbations=P,
        learning_rate=eta,
        smoothing_param=0.001,
        seed=123,
    )


# ---------------------------------------------------------------------------
# Bug 2 — perturbations are CPU-canonical and identical server-side and client-side
# ---------------------------------------------------------------------------
def test_server_and_client_perturbations_match_canonical():
    torch.manual_seed(0)
    model = TinyNet()
    strat = _make_strategy(model, K=1, P=1, eta=0.01)
    estimator = DeComFLClient(model=model, train_loader=None, device="cpu").zo_estimator
    d = len(strat.global_params_flat)

    for seed in (0, 7, 1234567):
        golden = canonical_perturbation(seed, d)
        server_z = strat._generate_perturbation(seed).cpu()
        client_z = estimator.generate_perturbation(seed, d).cpu()
        assert torch.equal(server_z, golden), f"server z != canonical at seed {seed}"
        assert torch.equal(client_z, golden), f"client z != canonical at seed {seed}"


# ---------------------------------------------------------------------------
# Bug 1 — the canary: server trajectory == client rebuild trajectory (fails while *P is present)
# ---------------------------------------------------------------------------
def test_server_trajectory_matches_client_rebuild():
    torch.manual_seed(0)
    model = TinyNet()
    K, P, eta = 2, 4, 0.05
    strat = _make_strategy(model, K, P, eta)

    seeds = strat.generate_seeds(0)
    strat.seed_history.append(seeds)
    grads = [[0.1 * (k + 1) + 0.01 * p for p in range(P)] for k in range(K)]
    strat.aggregate_fit(0, [("c1", grads, 100)])
    server_flat = strat.global_params_flat.detach().cpu()

    client = DeComFLClient(model=model, train_loader=None, device="cpu")
    client.rebuild_model([{"round_number": 0, "seeds": seeds, "gradients": grads}], learning_rate=eta)
    client_flat = client.x_current.detach().cpu()

    assert torch.allclose(server_flat, client_flat, atol=1e-6), (
        f"server {server_flat} != client rebuild {client_flat}"
    )


# ---------------------------------------------------------------------------
# C-1 — the (hoisted) aggregate must equal a corrected naive O(K*P*N) reference (multi-client)
# ---------------------------------------------------------------------------
def test_aggregate_equals_corrected_naive_multi_client():
    torch.manual_seed(0)
    model = TinyNet()
    K, P, eta = 2, 3, 0.05
    strat = _make_strategy(model, K, P, eta)
    d = len(strat.global_params_flat)

    seeds = strat.generate_seeds(0)
    strat.seed_history.append(seeds)
    grads_a = [[0.2 * (k + 1) + 0.03 * p for p in range(P)] for k in range(K)]
    grads_b = [[-0.1 * (k + 1) + 0.05 * p for p in range(P)] for k in range(K)]

    x0 = strat.global_params_flat.detach().cpu().clone()
    strat.aggregate_fit(0, [("a", grads_a, 100), ("b", grads_b, 100)])
    got = strat.global_params_flat.detach().cpu()

    # Corrected naive reference (no *P; average over N clients and P perturbations).
    x = x0.clone()
    n = 2
    for k in range(K):
        delta = torch.zeros_like(x)
        for grads in (grads_a, grads_b):
            for p in range(P):
                z = canonical_perturbation(seeds[k][p], d)
                delta += grads[k][p] * z
        delta = delta / (n * P)
        x = x - eta * delta
    assert torch.allclose(got, x, atol=1e-6), f"aggregate {got} != naive {x}"


# ---------------------------------------------------------------------------
# B-2 — constructing the strategy must not mutate the process-global RNG
# ---------------------------------------------------------------------------
def test_init_does_not_mutate_global_rng():
    # Build the model OUTSIDE the measured region — nn.Linear init legitimately draws from the
    # global torch RNG; we are isolating the *strategy* constructor.
    init = OrderedDict((k, v.clone()) for k, v in TinyNet().state_dict().items())

    def build_strategy() -> DeComFL:
        return DeComFL(
            initial_parameters=init,
            evaluate_fn=None,
            min_fit_clients=1,
            clients_per_round=2,
            num_local_steps=2,
            num_perturbations=3,
            learning_rate=0.01,
            smoothing_param=0.001,
            seed=123,
        )

    torch.manual_seed(999)
    np.random.seed(999)
    torch_before = torch.randn(3)
    numpy_before = np.random.rand(3)

    torch.manual_seed(999)
    np.random.seed(999)
    _ = build_strategy()
    torch_after = torch.randn(3)
    numpy_after = np.random.rand(3)

    assert torch.equal(torch_before, torch_after), "strategy construction reseeded the global torch RNG"
    assert np.array_equal(numpy_before, numpy_after), "strategy construction reseeded the global numpy RNG"


# ---------------------------------------------------------------------------
# Bug 3 — chunked serializer is symmetric for a transformer-shaped, multi-chunk state_dict
# ---------------------------------------------------------------------------
def test_chunked_roundtrip_transformer_shaped_multichunk():
    torch.manual_seed(0)
    state = OrderedDict([
        ("embedding.weight", torch.randn(256, 64, dtype=torch.float32)),
        ("layer.0.attn.weight", torch.randn(64, 64, dtype=torch.float32)),
        ("layer.0.mlp.bias", torch.randn(64, dtype=torch.float32)),
        ("head.weight", torch.randn(10, 64, dtype=torch.float32)),
    ])
    chunks = list(parameters_to_chunks(state, num_examples=512, chunk_size=64 * 1024))
    assert len(chunks) > 1, "expected a multi-chunk payload"
    blob = b"".join(c["chunk_data"] for c in chunks)
    recovered, num_examples = chunks_to_parameters(blob, compressed=False)

    assert num_examples == 512
    assert set(recovered.keys()) == set(state.keys())
    for key in state:
        assert torch.allclose(state[key], recovered[key], atol=1e-6), key
