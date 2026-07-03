"""DeComFL convergence + client/server state-sync regression tests (FR-1, FR-2).

These pin the paper invariant the audit found broken on the default full-participation path:

  * FR-1 — every party shares the same initial model x_0. A client must start from the
    server's global model, not its own constructor-time random init.
  * FR-2 — after a client reverts its un-averaged local steps (which is correct), it must
    replay each round's AVERAGED update so that DURING round r both the client and the server
    hold x_{r-1}; the client computes its ZO gradient scalars at exactly the model the server
    currently holds as global.

On the pre-fix code a continuously-participating client is pinned at x_0: it never applies any
round's averaged update, so its gradient scalars are directional derivatives at x_0 while the
server's global model walks away — the aggregated model is meaningless and the loss does not
train. `test_participating_client_stays_synced_and_converges` fails at round 2 on the old code
(client desynced) and passes after the fix.

The toy problem is multinomial logistic regression (Linear -> CrossEntropyLoss), which is convex
in the weights — the estimator hardcodes CrossEntropyLoss, and a convex objective gives a clean,
deterministic "loss must decrease" signal.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.client.decomfl_client import DeComFLClient


class LogReg(nn.Module):
    """Linear(4, 3): multinomial logistic regression. state_dict = fc.weight [3,4] + fc.bias [3], d=15.

    Buffer-free and fully trainable, so the client's requires_grad flatten equals the server's
    full-state_dict flatten (no dimension mismatch to confound the sync assertion).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):  # noqa: D401
        return self.fc(x)


def _flatten(params: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
    """Flatten a state_dict exactly as the server (DeComFL._flatten_params) does."""
    return torch.cat([t.reshape(-1) for t in params.values()]).cpu()


def _toy_dataset(n: int = 96, seed: int = 0):
    """A deterministic, linearly-separable 3-class problem in R^4 (learnable + convex)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 4, generator=g)
    teacher = torch.tensor([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ])
    y = (X @ teacher.T).argmax(dim=1)
    return X, y


class _WholeSetLoader:
    """Yields the whole toy set as one batch each iteration (matches DeComFLClient.fit's next(iter))."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X, self.y = X, y
        self.dataset = X  # len() -> num_examples

    def __iter__(self):
        while True:
            yield self.X, self.y

    def __len__(self) -> int:
        return int(self.X.shape[0])


def test_fit_applies_the_servers_smoothing_param_to_the_estimator():
    # FR-10: μ (smoothing_param) is server-authoritative — the client must ZO-estimate with the μ the
    # server sends in the DeComFL config, not its construction-time default. A mismatched μ makes the
    # client's gradient scalars directional derivatives of a differently-smoothed function than the
    # server reconstructs, degrading the aggregate (same invariant class as shared seeds and lr).
    X, y = _toy_dataset()
    client = DeComFLClient(model=LogReg(), train_loader=_WholeSetLoader(X, y), device="cpu")
    assert client.zo_estimator.mu == 0.001                       # construction-time default

    client.fit(None, {"seeds": [[11, 22]], "learning_rate": 0.01, "smoothing_param": 0.05})

    assert client.zo_estimator.mu == 0.05                        # applied from the server config


def _make_strategy(init_model: nn.Module, K: int, P: int, eta: float, seed: int = 7) -> DeComFL:
    init = OrderedDict((k, v.clone()) for k, v in init_model.state_dict().items())
    return DeComFL(
        initial_parameters=init,
        evaluate_fn=None,
        min_fit_clients=1,
        clients_per_round=1,
        num_local_steps=K,
        num_perturbations=P,
        learning_rate=eta,
        smoothing_param=0.001,
        seed=seed,
    )


def _global_loss(strategy: DeComFL, X: torch.Tensor, y: torch.Tensor) -> float:
    """Cross-entropy loss of the server's current global model on the toy set."""
    model = LogReg()
    params = strategy._unflatten_params(strategy.global_params_flat, strategy.initial_parameters)
    model.load_state_dict(params)
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(X), y).item()


def _run_rounds(strategy, coordinator, client, client_id, num_rounds, eta, X, y, assert_sync):
    """Drive the DeComFL round protocol in-process, mirroring what grpc_servicer + the coordinator do."""
    for _ in range(num_rounds):
        r = coordinator.current_round
        seeds = strategy.get_or_create_seeds(r)                 # servicer: GetDeComFLConfig
        rebuild = strategy.get_rebuild_history(client_id, r)    # servicer: rebuild history
        if rebuild:
            client.rebuild_model(rebuild, eta)
        if assert_sync:
            # During round r (after rebuild, before fit) both client and server must hold x_{r-1}.
            assert torch.allclose(
                client.x_current.detach().cpu(), strategy.global_params_flat.detach().cpu(), atol=1e-5
            ), (
                f"round {r}: client desynced from server global "
                f"(max abs diff "
                f"{(client.x_current.detach().cpu() - strategy.global_params_flat.detach().cpu()).abs().max():.4g})"
            )
        grads, n = client.fit(None, {"seeds": seeds, "learning_rate": eta})
        coordinator.submit_decomfl_update(client_id, grads, n, r)  # aggregates + advances the round


# ---------------------------------------------------------------------------
# FR-1 — a client syncs its local model to the server's global model (shared x_0)
# ---------------------------------------------------------------------------
def test_load_global_model_syncs_client_to_server_params():
    torch.manual_seed(1)
    server_params = OrderedDict((k, v.clone()) for k, v in LogReg().state_dict().items())

    torch.manual_seed(2024)  # a DIFFERENT random init than the server
    client = DeComFLClient(model=LogReg(), train_loader=None, device="cpu")

    assert not torch.allclose(client.x_current.cpu(), _flatten(server_params), atol=1e-6), (
        "test setup invalid: client and server inits should differ"
    )

    client.load_global_model(server_params)  # FR-1: adopt the server's global model

    assert torch.allclose(client.x_current.cpu(), _flatten(server_params), atol=1e-6), (
        "after load_global_model the client must hold the server's parameters"
    )
    # And the client's actual nn.Module must reflect the same weights (fit() runs forward on it).
    assert torch.allclose(_flatten(client.model.state_dict()), _flatten(server_params), atol=1e-6)


# ---------------------------------------------------------------------------
# FR-2 — a full-participation client stays synced every round and the federation converges
# ---------------------------------------------------------------------------
def test_participating_client_stays_synced_and_converges():
    torch.manual_seed(0)
    K, P, eta, rounds = 1, 24, 0.1, 40
    X, y = _toy_dataset()

    strategy = _make_strategy(LogReg(), K, P, eta)
    coordinator = FLCoordinator(strategy, min_clients_for_aggregation=1, clients_per_round=1)

    # Client starts from a DIFFERENT random init, then syncs to the server's x_0 (FR-1).
    torch.manual_seed(2024)
    client = DeComFLClient(model=LogReg(), train_loader=_WholeSetLoader(X, y), device="cpu")
    client.load_global_model(strategy.initial_parameters)

    loss_before = _global_loss(strategy, X, y)

    # assert_sync=True makes this fail at round 2 on the pre-fix code (client pinned at x_0).
    _run_rounds(strategy, coordinator, client, "c1", rounds, eta, X, y, assert_sync=True)

    loss_after = _global_loss(strategy, X, y)

    assert loss_after < 0.7 * loss_before, (
        f"DeComFL did not train: loss {loss_before:.4f} -> {loss_after:.4f} "
        f"(expected a substantial decrease on a convex toy problem)"
    )


# ---------------------------------------------------------------------------
# FR-1 (late join) — a client that first participates after the server has advanced must adopt
# the CURRENT global model and NOT replay pre-join rounds on top of it (that would double-apply).
# ---------------------------------------------------------------------------
def test_late_join_client_syncs_to_current_global_without_double_apply():
    torch.manual_seed(0)
    K, P, eta = 1, 24, 0.1
    X, y = _toy_dataset()

    strategy = _make_strategy(LogReg(), K, P, eta)
    coordinator = FLCoordinator(strategy, min_clients_for_aggregation=1, clients_per_round=1)

    # A resident client advances the server past round 1 (populating seed/gradient history).
    torch.manual_seed(2024)
    resident = DeComFLClient(model=LogReg(), train_loader=_WholeSetLoader(X, y), device="cpu")
    resident.load_global_model(strategy.initial_parameters)
    _run_rounds(strategy, coordinator, resident, "resident", 4, eta, X, y, assert_sync=False)

    R = coordinator.current_round
    assert R > 1, "resident should have advanced the server past round 1"
    server_global = strategy.global_params_flat.detach().cpu().clone()  # x_{R-1}

    # A brand-new client joins at round R: it downloads the CURRENT global model (what
    # get_global_model would return) and must NOT be handed the pre-join rounds to rebuild.
    torch.manual_seed(7)
    newcomer = DeComFLClient(model=LogReg(), train_loader=_WholeSetLoader(X, y), device="cpu")
    newcomer.load_global_model(
        strategy._unflatten_params(strategy.global_params_flat, strategy.initial_parameters)
    )

    rebuild = strategy.get_rebuild_history("newcomer", R)
    assert rebuild == [], (
        f"a late-joining client already holds the current global; it must not replay pre-join "
        f"rounds (got {len(rebuild)}), which would double-apply on top of the downloaded model"
    )
    # With no spurious rebuild, the newcomer computes its gradients at exactly the server's global.
    assert torch.allclose(newcomer.x_current.detach().cpu(), server_global, atol=1e-6)
