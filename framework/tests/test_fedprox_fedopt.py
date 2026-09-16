"""FR-11: FedProx + FedOpt (FedAdam / FedYogi) strategy tests.

Pins the three definition-of-done invariants:

  * FedProx(mu=0) aggregation is BITWISE-IDENTICAL to FedAvg — the proximal term is a
    purely client-side objective change, so mu never touches server aggregation.
  * FedOpt keeps persistent Adam-style optimiser state (m, v) over the aggregated
    pseudo-gradient (global_old - aggregated) and applies it across rounds — a second
    round's update differs from the first because (m, v) accumulated.
  * Both strategies drive a 2-client convex (multinomial logistic regression + CE) toy
    federation to a lower loss, analogous to the DeComFL convergence test.

The math the reviewer checks:
  FedProx local objective   F_i(w) + (mu/2)||w - w_global||^2
                            => gradient contribution mu*(w - w_global); mu=0 => plain SGD.
  FedOpt pseudo-gradient     g_t = w_global(old) - aggregate(client models)   (== -Delta_t)
        m_t = b1*m_{t-1} + (1-b1)*g_t
        FedAdam  v_t = b2*v_{t-1} + (1-b2)*g_t^2
        FedYogi  v_t = v_{t-1} - (1-b2)*sign(v_{t-1} - g_t^2)*g_t^2
        w_new  = w_global(old) - eta * m_t / (sqrt(v_t) + tau)
"""
from collections import OrderedDict

import math
import torch
import torch.nn as nn

from fedlearn.server.strategy import FedAvg, FedProx, FedOpt
from fedlearn.server.strategy_factory import create_strategy
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.client.local_trainer import LocalTrainer


# --------------------------------------------------------------------------- helpers
def make_params(vals):
    """A 2-tensor state_dict (fc.weight [1,3], fc.bias [1]) filled from `vals` (len 4)."""
    w, b = vals[:3], vals[3:]
    return OrderedDict([
        ("fc.weight", torch.tensor([w], dtype=torch.float32)),
        ("fc.bias", torch.tensor(b, dtype=torch.float32)),
    ])


def clone_params(p):
    return OrderedDict((k, v.clone()) for k, v in p.items())


class LogReg(nn.Module):
    """Linear(4,3): multinomial logistic regression — convex in the weights (clean 'loss falls')."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _toy_dataset(n=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 4, generator=g)
    teacher = torch.tensor([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ])
    y = (X @ teacher.T).argmax(dim=1)
    return X, y


class _ListLoader:
    """Finite loader: yields the toy partition as a single full batch. `.dataset` gives len."""

    def __init__(self, X, y):
        self.X, self.y = X, y
        self.dataset = X

    def __iter__(self):
        yield self.X, self.y

    def __len__(self):
        return int(self.X.shape[0])


def _ce_loss(params, X, y):
    model = LogReg()
    model.load_state_dict(params)
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(X), y).item()


# --------------------------------------------------------------------------- FedProx: mu=0 == FedAvg
def test_fedprox_mu0_aggregation_is_bitwise_identical_to_fedavg():
    init = make_params([0, 0, 0, 0])

    def results():
        # Fresh copies every call — FedAvgAggregator mutates (clears) the client dicts it consumes.
        return [
            (make_params([1.0, 2.0, 3.0, 4.0]), 100),
            (make_params([5.0, 6.0, 7.0, 8.0]), 300),
        ]

    fedavg = FedAvg(initial_parameters=clone_params(init), min_fit_clients=2, clients_per_round=2)
    fedprox = FedProx(initial_parameters=clone_params(init), min_fit_clients=2, clients_per_round=2,
                      proximal_mu=0.0)

    a = fedavg.aggregate_fit(1, results())
    b = fedprox.aggregate_fit(1, results())

    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"FedProx(mu=0) diverged from FedAvg on {k}"


def test_fedprox_aggregation_independent_of_mu():
    # The proximal term is client-side only: server aggregation must be identical for any mu.
    init = make_params([0, 0, 0, 0])

    def results():
        return [
            (make_params([1.0, 2.0, 3.0, 4.0]), 100),
            (make_params([5.0, 6.0, 7.0, 8.0]), 300),
        ]

    a = FedProx(initial_parameters=clone_params(init), proximal_mu=0.0).aggregate_fit(1, results())
    b = FedProx(initial_parameters=clone_params(init), proximal_mu=0.9).aggregate_fit(1, results())
    for k in a:
        assert torch.equal(a[k], b[k])


def test_fedprox_client_config_carries_mu_as_string():
    s = FedProx(initial_parameters=make_params([0, 0, 0, 0]), proximal_mu=0.25, learning_rate=0.1)
    cfg = s.get_client_config()
    # proto config is map<string,string> — values must be strings.
    assert cfg["proximal_mu"] == "0.25"
    assert cfg["learning_rate"] == "0.1"
    assert all(isinstance(v, str) for v in cfg.values())


def test_fedprox_empty_results_returns_none():
    s = FedProx(initial_parameters=make_params([0, 0, 0, 0]), proximal_mu=0.1)
    assert s.aggregate_fit(1, []) is None


# --------------------------------------------------------------------------- FedProx: client proximal term
def test_local_trainer_mu0_matches_plain_sgd():
    # mu=0 must reduce to ordinary local SGD (FedProx -> FedAvg at the client).
    torch.manual_seed(3)
    X, y = _toy_dataset(n=32)
    init = LogReg()
    init_sd = clone_params(init.state_dict())

    trainer = LocalTrainer(model=LogReg(), train_loader=_ListLoader(X, y), device="cpu")
    out, n = trainer.fit(clone_params(init_sd), {"learning_rate": 0.1, "proximal_mu": 0.0, "local_epochs": 3})

    # Reference: identical plain SGD on a fresh model with the same init/data/lr/steps.
    ref = LogReg()
    ref.load_state_dict(clone_params(init_sd))
    opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(ref(X), y)
        loss.backward()
        opt.step()

    for k in out:
        assert torch.allclose(out[k], ref.state_dict()[k], atol=1e-6), f"mu=0 diverged from plain SGD on {k}"
    assert n == 32


def test_local_trainer_proximal_term_pulls_toward_global_anchor():
    # With mu>0 the FedProx penalty keeps the local solution closer to w_global than mu=0.
    torch.manual_seed(5)
    X, y = _toy_dataset(n=64)
    init = LogReg()
    anchor = clone_params(init.state_dict())

    def run(mu):
        t = LocalTrainer(model=LogReg(), train_loader=_ListLoader(X, y), device="cpu")
        out, _ = t.fit(clone_params(anchor), {"learning_rate": 0.3, "proximal_mu": mu, "local_epochs": 8})
        return out

    def dist(params):
        return sum((params[k] - anchor[k]).pow(2).sum().item() for k in params)

    d_far = dist(run(0.0))
    d_near = dist(run(5.0))
    assert d_near < d_far, f"proximal term did not regularise toward the anchor ({d_near:.4f} !< {d_far:.4f})"


# --------------------------------------------------------------------------- FedOpt: persistent (m, v) state
def test_fedopt_maintains_and_applies_state_across_rounds():
    init = make_params([2.0, 2.0, 2.0, 2.0])
    s = FedOpt(initial_parameters=clone_params(init), variant="adam",
               server_learning_rate=1.0, beta1=0.9, beta2=0.99, tau=1e-3)

    # Before any round there is no accumulated state.
    assert s._m is None and s._v is None

    def zero_updates():
        # Both clients report an all-zeros model -> aggregate x_bar = 0 -> g = global_old - 0 = global_old.
        return [(make_params([0, 0, 0, 0]), 100), (make_params([0, 0, 0, 0]), 100)]

    g0 = clone_params(init)                       # global before round 1
    s.aggregate_fit(1, zero_updates())
    m1 = clone_params(s._m)
    v1 = clone_params(s._v)
    g1 = clone_params(s._global)                  # global after round 1
    update1 = {k: (g1[k] - g0[k]) for k in g0}

    s.aggregate_fit(2, zero_updates())
    m2 = clone_params(s._m)
    v2 = clone_params(s._v)
    g2 = clone_params(s._global)
    update2 = {k: (g2[k] - g1[k]) for k in g1}

    for k in m1:
        # State is non-trivial and EVOLVES between rounds (accumulation, not a reset).
        assert not torch.allclose(m1[k], torch.zeros_like(m1[k]))
        assert not torch.allclose(v1[k], torch.zeros_like(v1[k]))
        assert not torch.allclose(m1[k], m2[k]), f"m did not evolve on {k}"
        assert not torch.allclose(v1[k], v2[k]), f"v did not evolve on {k}"
        # The APPLIED update differs round-to-round precisely because state accumulated.
        assert not torch.allclose(update1[k], update2[k]), f"update did not change on {k}"


def test_fedopt_fedadam_update_matches_closed_form():
    # Ground the FedAdam update rule against an independent scalar computation.
    b1, b2, tau, eta = 0.9, 0.99, 1e-3, 1.0
    init = make_params([2.0, 2.0, 2.0, 2.0])
    s = FedOpt(initial_parameters=clone_params(init), variant="adam",
               server_learning_rate=eta, beta1=b1, beta2=b2, tau=tau)

    s.aggregate_fit(1, [(make_params([0, 0, 0, 0]), 100), (make_params([0, 0, 0, 0]), 100)])

    # g = old(2.0) - aggregate(0.0) = 2.0 (elementwise, every coordinate)
    g = 2.0
    m = (1 - b1) * g                         # 0.2
    v = (1 - b2) * g * g                     # 0.04
    expected = 2.0 - eta * m / (math.sqrt(v) + tau)
    for k in s._global:
        assert torch.allclose(s._global[k], torch.full_like(s._global[k], expected), atol=1e-6)


def test_fedopt_fedyogi_v_update_follows_yogi_rule():
    # Two rounds so the Yogi sign-rule v-update visibly differs from FedAdam.
    b1, b2, tau, eta = 0.9, 0.99, 1e-3, 1.0
    init = make_params([2.0, 2.0, 2.0, 2.0])
    s = FedOpt(initial_parameters=clone_params(init), variant="yogi",
               server_learning_rate=eta, beta1=b1, beta2=b2, tau=tau)

    upd = lambda: [(make_params([0, 0, 0, 0]), 100), (make_params([0, 0, 0, 0]), 100)]

    # Round 1: g1 = 2.0
    s.aggregate_fit(1, upd())
    g1 = 2.0
    m = (1 - b1) * g1
    v = 0.0 - (1 - b2) * math.copysign(1.0, 0.0 - g1 * g1) * (g1 * g1)   # yogi
    w1 = 2.0 - eta * m / (math.sqrt(v) + tau)

    # Round 2: g2 = w1 - 0
    s.aggregate_fit(2, upd())
    g2 = w1
    m = b1 * m + (1 - b1) * g2
    v = v - (1 - b2) * math.copysign(1.0, v - g2 * g2) * (g2 * g2)       # yogi
    w2 = w1 - eta * m / (math.sqrt(v) + tau)

    for k in s._global:
        assert torch.allclose(s._global[k], torch.full_like(s._global[k], w2), atol=1e-6)
        assert torch.allclose(s._v[k], torch.full_like(s._v[k], v), atol=1e-6)


def test_fedopt_empty_results_returns_none():
    s = FedOpt(initial_parameters=make_params([0, 0, 0, 0]), variant="adam")
    assert s.aggregate_fit(1, []) is None


def test_fedopt_rejects_unknown_variant():
    try:
        FedOpt(initial_parameters=make_params([0, 0, 0, 0]), variant="rmsprop")
        assert False, "expected ValueError for unknown variant"
    except ValueError:
        pass


# --------------------------------------------------------------------------- convergence (2-client convex)
def _drive_rounds(strategy, num_rounds, clients, X, y):
    # Give the strategy an evaluator so the FedAvg-path coordinator trigger has a (loss, metrics)
    # to unpack each round (mirrors a real FedAvg-family run, which always supplies one).
    strategy.evaluate_fn = lambda rnd, params: (_ce_loss(params, X, y), {})

    coord = FLCoordinator(strategy, min_clients_for_aggregation=len(clients),
                          clients_per_round=len(clients))
    coord.set_initial_parameters(strategy.initialize_parameters())

    loss_start = _ce_loss(strategy.initialize_parameters(), X, y)
    for _ in range(num_rounds):
        r = coord.current_round
        global_params, _, config = coord.get_global_model_for_client()
        for cid, c in clients.items():
            new_params, n = c.fit(clone_params(global_params), config)
            coord.submit_client_update(cid, new_params, n, r)
    loss_end = _ce_loss(coord.get_global_model_params(), X, y)
    return loss_start, loss_end


def test_fedprox_two_client_convex_convergence():
    torch.manual_seed(0)
    X, y = _toy_dataset(n=96)
    X1, y1, X2, y2 = X[:48], y[:48], X[48:], y[48:]         # heterogeneous partitions

    init = LogReg()
    strategy = FedProx(
        initial_parameters=clone_params(init.state_dict()),
        min_fit_clients=2, clients_per_round=2, proximal_mu=0.1, learning_rate=0.2, local_epochs=2)
    clients = {
        "c1": LocalTrainer(model=LogReg(), train_loader=_ListLoader(X1, y1), device="cpu"),
        "c2": LocalTrainer(model=LogReg(), train_loader=_ListLoader(X2, y2), device="cpu"),
    }
    loss_start, loss_end = _drive_rounds(strategy, 40, clients, X, y)
    assert loss_end < 0.9 * loss_start, f"FedProx did not converge: {loss_start:.4f} -> {loss_end:.4f}"


def test_fedopt_two_client_convex_convergence():
    torch.manual_seed(0)
    X, y = _toy_dataset(n=96)
    X1, y1, X2, y2 = X[:48], y[:48], X[48:], y[48:]

    init = LogReg()
    strategy = FedOpt(
        initial_parameters=clone_params(init.state_dict()),
        min_fit_clients=2, clients_per_round=2, variant="adam",
        server_learning_rate=0.1, beta1=0.9, beta2=0.99, tau=1e-3,
        learning_rate=0.2, local_epochs=2)
    clients = {
        "c1": LocalTrainer(model=LogReg(), train_loader=_ListLoader(X1, y1), device="cpu"),
        "c2": LocalTrainer(model=LogReg(), train_loader=_ListLoader(X2, y2), device="cpu"),
    }
    loss_start, loss_end = _drive_rounds(strategy, 40, clients, X, y)
    assert loss_end < 0.9 * loss_start, f"FedOpt did not converge: {loss_start:.4f} -> {loss_end:.4f}"


# --------------------------------------------------------------------------- factory registration
def test_strategy_factory_creates_fedprox_and_fedopt():
    init = make_params([0, 0, 0, 0])
    fp = create_strategy("fedprox", initial_parameters=clone_params(init), proximal_mu=0.1)
    fo = create_strategy("fedopt", initial_parameters=clone_params(init), variant="yogi")
    assert isinstance(fp, FedProx)
    assert isinstance(fo, FedOpt)
    # Existing strategies still resolve through the same factory.
    assert isinstance(create_strategy("fedavg", initial_parameters=clone_params(init)), FedAvg)


def test_strategy_factory_rejects_unknown_name():
    try:
        create_strategy("does-not-exist", initial_parameters=make_params([0, 0, 0, 0]))
        assert False, "expected ValueError for unknown strategy name"
    except ValueError:
        pass
