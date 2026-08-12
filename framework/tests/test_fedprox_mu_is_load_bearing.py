"""FedProx: prove `mu` actually does something, and that it does the right thing.

Written before reading the implementation's behaviour (TDD), for a specific reason.

Cross-validating FedProx against Flower is nearly vacuous by construction: Flower's `FedProx`
does **not** override `aggregate_fit` — it inherits FedAvg's and only injects `proximal_mu`
into the client config. Ours does the same. So "our FedProx aggregation matches Flower's
FedProx aggregation" is true even if **both** silently drop `mu` and run plain FedAvg. Two
implementations agreeing that mu does nothing is not a validation.

That is not a hypothetical. This repo already carries a recorded concern that FedProx did not
differentiate from FedAvg in benchmarks, and a separate one that the mobile client never
implemented the proximal term at all. So the load-bearing question is not "does the server
aggregate correctly" — it is:

  1. Is `mu` actually delivered to the client? (`get_client_config`)
  2. Does the client actually apply it? (`LocalTrainer.fit`)
  3. Is what it applies the *correct* term, `mu * (w - w_global)`, checked against a
     closed-form reference rather than against itself?
  4. Does `mu = 0` still reduce EXACTLY to FedAvg, so the strategy is a strict generalisation?

None of these needs flwr, so they live here as permanent framework tests rather than in the
research harness.
"""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from fedlearn.client.local_trainer import LocalTrainer
from fedlearn.server.strategy import FedAvg, FedProx


FEATURES, CLASSES, N = 5, 3, 96


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(FEATURES, CLASSES, generator=g)
    x = torch.randn(N, FEATURES, generator=g)
    return x, (x @ w).argmax(1)


X, Y = _data()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FEATURES, CLASSES)

    def forward(self, x):
        return self.fc(x)


class Loader:
    def __init__(self, idx, bs=16):
        self.indices, self.batch_size = list(idx), bs
        self.dataset = self.indices

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            s = self.indices[i:i + self.batch_size]
            yield X[s], Y[s]


def _fresh_params(seed=1234):
    torch.manual_seed(seed)
    return OrderedDict((k, v.detach().clone()) for k, v in Net().state_dict().items())


def _train(params, mu, lr=0.5, epochs=1, seed=0):
    torch.manual_seed(seed)
    trainer = LocalTrainer(model=Net(), train_loader=Loader(range(N)))
    out, _n = trainer.fit(params, {"learning_rate": lr, "proximal_mu": mu,
                                   "local_epochs": epochs})
    return out


# --------------------------------------------------------------------------------------
# 1. mu reaches the client
# --------------------------------------------------------------------------------------

class TestMuIsDelivered:
    def test_strategy_ships_mu_in_the_client_config(self):
        s = FedProx(initial_parameters=_fresh_params(), proximal_mu=0.7,
                    learning_rate=0.1, local_epochs=3)
        cfg = s.get_client_config()
        # The proto config is map<string,string>, so values are stringified on the wire.
        assert float(cfg["proximal_mu"]) == pytest.approx(0.7)
        assert float(cfg["learning_rate"]) == pytest.approx(0.1)
        assert int(cfg["local_epochs"]) == 3

    def test_config_key_matches_flowers_contract(self):
        """Flower's FedProx injects the key `proximal_mu`. Ours must use the same name.

        Not cosmetic: a client written against one framework's convention silently trains
        WITHOUT the penalty under the other, and the run still completes and still converges —
        just as plain FedAvg, with no error anywhere.
        """
        cfg = FedProx(initial_parameters=_fresh_params(), proximal_mu=0.5).get_client_config()
        assert "proximal_mu" in cfg


# --------------------------------------------------------------------------------------
# 2. mu changes the trajectory — the anti-vacuity check
# --------------------------------------------------------------------------------------

class TestMuIsLoadBearing:
    def test_nonzero_mu_changes_the_local_solution(self):
        """If this fails, every FedProx result in the record is a FedAvg result."""
        p0 = _fresh_params()
        a = _train(p0, mu=0.0)
        b = _train(p0, mu=1.0)
        assert not all(torch.equal(a[k], b[k]) for k in a), (
            "mu=1.0 produced the same local model as mu=0.0 — the proximal term is inert"
        )

    def test_larger_mu_keeps_the_local_model_closer_to_the_global_within_the_stable_regime(self):
        """The point of the penalty: it bounds client drift, monotonically in mu.

        Stronger than 'mu changes something' — a sign error would still change the numbers, but
        would push the local model AWAY from the anchor.

        Qualified to ``lr*mu < 2`` deliberately. The penalty is applied as an explicit gradient
        term, so its own iteration has multiplier ``(1 - lr*mu)`` and only contracts inside that
        band. This test originally asserted monotonicity unconditionally and failed at
        ``mu=10, lr=0.5`` (drift 411 vs 0.77) — which is the discretisation behaving as the
        mathematics says, not the implementation misbehaving. See
        ``test_the_stability_boundary_is_where_the_theory_says``.
        """
        p0 = _fresh_params()
        lr = 0.5
        drifts = []
        for mu in (0.0, 0.5, 1.0, 2.0):            # lr*mu = 0, 0.25, 0.5, 1.0 — all < 2
            out = _train(p0, mu=mu, lr=lr)
            drift = sum(
                float(torch.norm(out[k].float() - p0[k].float()) ** 2) for k in p0
            ) ** 0.5
            drifts.append((mu, drift))

        assert all(a[1] > b[1] for a, b in zip(drifts, drifts[1:])), (
            f"drift must fall monotonically as mu rises inside lr*mu<2, got {drifts}"
        )

    def test_the_stability_boundary_is_where_the_theory_says(self):
        """Past ``lr*mu = 2`` the penalty amplifies drift instead of bounding it.

        This is the failure the constructor guard exists for, and it is silent: the run
        completes, the loss curve looks like a bad-hyperparameter run, and nothing reports that
        the term configured to reduce drift increased it by four orders of magnitude.
        """
        p0 = _fresh_params()

        def drift(mu, lr):
            out = _train(p0, mu=mu, lr=lr)
            return sum(float(torch.norm(out[k].float() - p0[k].float()) ** 2) for k in p0) ** 0.5

        baseline = drift(0.0, 0.5)
        stable = drift(3.8, 0.5)       # lr*mu = 1.9
        unstable = drift(20.0, 0.5)    # lr*mu = 10

        assert stable < baseline, "inside the band the penalty must contract"
        assert unstable > baseline * 100, (
            f"past the band the penalty must visibly diverge, got {unstable:.4g} "
            f"vs baseline {baseline:.4g}"
        )

    def test_the_boundary_is_a_property_of_lr_times_mu_not_of_mu(self):
        """Same ``lr*mu`` at different lr must land on the same side of the boundary.

        Confirms the envelope is the discretisation's, not an artifact of one learning rate —
        which is what licenses guarding on the product rather than on mu alone.
        """
        p0 = _fresh_params()

        def drift(mu, lr):
            out = _train(p0, mu=mu, lr=lr)
            return sum(float(torch.norm(out[k].float() - p0[k].float()) ** 2) for k in p0) ** 0.5

        for lr in (0.5, 0.1, 0.01):
            assert drift(1.9 / lr, lr) < drift(0.0, lr), f"lr*mu=1.9 should contract at lr={lr}"
            assert drift(10.0 / lr, lr) > drift(0.0, lr) * 100, (
                f"lr*mu=10 should diverge at lr={lr}"
            )


class TestStabilityGuard:
    """The constructor refuses a (mu, lr) pair that would invert the penalty's purpose.

    Mirrors the DeComFL learning-rate guard (``decomfl_strategy.lr_stability_statistic``): the
    repo's established position is that a silent divergence gets a loud constructor error with a
    deliberate override, not a comment in a docstring.
    """

    def test_unstable_mu_is_rejected_with_an_actionable_message(self):
        with pytest.raises(ValueError, match=r"lr\*mu"):
            FedProx(initial_parameters=_fresh_params(), proximal_mu=20.0, learning_rate=0.5)

    def test_the_error_names_the_largest_safe_mu(self):
        with pytest.raises(ValueError) as exc:
            FedProx(initial_parameters=_fresh_params(), proximal_mu=20.0, learning_rate=0.5)
        assert "Lower mu below 4" in str(exc.value)   # 2.0 / 0.5

    def test_stable_pairs_are_accepted(self):
        FedProx(initial_parameters=_fresh_params(), proximal_mu=1.0, learning_rate=0.5)
        FedProx(initial_parameters=_fresh_params(), proximal_mu=19.0, learning_rate=0.1)

    def test_mu_zero_is_never_rejected(self):
        """mu=0 is exactly FedAvg and must stay constructible at any learning rate."""
        FedProx(initial_parameters=_fresh_params(), proximal_mu=0.0, learning_rate=100.0)

    def test_the_override_exists_and_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            FedProx(initial_parameters=_fresh_params(), proximal_mu=20.0,
                    learning_rate=0.5, allow_unstable_mu=True)
        assert any("AMPLIFY" in r.message for r in caplog.records)

    def test_near_boundary_warns_without_rejecting(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            FedProx(initial_parameters=_fresh_params(), proximal_mu=3.8, learning_rate=0.5)
        assert any("near the boundary" in r.message for r in caplog.records)


# --------------------------------------------------------------------------------------
# 3. the term is the CORRECT one, vs a closed form
# --------------------------------------------------------------------------------------

class TestProximalGradientIsCorrect:
    def test_matches_an_independently_written_closed_form(self):
        """One SGD step with the penalty must equal the analytic update.

        FedProx adds (mu/2)*||w - w_global||^2 to the objective, whose gradient contribution is
        exactly mu*(w - w_global). With w = w_global at the first step, that term is ZERO, so a
        single step from the anchor cannot distinguish a correct implementation from an absent
        one. The reference is therefore computed over TWO steps, where the second step's
        penalty gradient is genuinely non-zero.
        """
        p0 = _fresh_params()
        mu, lr = 2.0, 0.1

        # -- reference: plain autograd, penalty written out by hand -----------------------
        torch.manual_seed(0)
        ref = Net()
        ref.load_state_dict(p0)
        anchor = [t.detach().clone() for t in ref.parameters()]
        opt = torch.optim.SGD(ref.parameters(), lr=lr)
        loader = Loader(range(N))
        for _ in range(2):
            for xb, yb in loader:
                opt.zero_grad()
                loss = nn.functional.cross_entropy(ref(xb), yb)
                # the penalty as an explicit term in the objective, differentiated by autograd
                for p, a in zip(ref.parameters(), anchor):
                    loss = loss + (mu / 2.0) * ((p - a) ** 2).sum()
                loss.backward()
                opt.step()
        expected = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in ref.state_dict().items()
        )

        got = _train(p0, mu=mu, lr=lr, epochs=2)

        for k in expected:
            assert torch.allclose(got[k], expected[k], atol=1e-5), (
                f"key {k}: proximal update does not match the closed form\n"
                f"  got      {got[k].flatten()[:4]}\n"
                f"  expected {expected[k].flatten()[:4]}"
            )

    def test_frozen_parameters_are_skipped(self):
        """The penalty must apply only to trainable params.

        The frozen-backbone and LoRA recipes federate a trainable subset; applying the
        proximal term to frozen weights would regularise parameters that are not being
        learned, which is silently wrong rather than loudly broken.
        """
        torch.manual_seed(0)
        net = Net()
        p0 = OrderedDict((k, v.detach().clone()) for k, v in net.state_dict().items())
        net.fc.bias.requires_grad_(False)
        frozen_before = net.fc.bias.detach().clone()

        trainer = LocalTrainer(model=net, train_loader=Loader(range(N)))
        trainer.fit(p0, {"learning_rate": 0.5, "proximal_mu": 5.0, "local_epochs": 2})

        assert torch.equal(net.fc.bias.detach(), frozen_before), "a frozen param was updated"


# --------------------------------------------------------------------------------------
# 4. mu = 0 is EXACTLY FedAvg
# --------------------------------------------------------------------------------------

class TestMuZeroReducesToFedAvg:
    def test_client_side_mu_zero_is_plain_sgd(self):
        p0 = _fresh_params()
        with_mu0 = _train(p0, mu=0.0)

        torch.manual_seed(0)
        plain = Net()
        plain.load_state_dict(p0)
        opt = torch.optim.SGD(plain.parameters(), lr=0.5)
        for xb, yb in Loader(range(N)):
            opt.zero_grad()
            nn.functional.cross_entropy(plain(xb), yb).backward()
            opt.step()

        for k, v in plain.state_dict().items():
            assert torch.allclose(with_mu0[k], v.detach().cpu(), atol=1e-6)

    def test_server_aggregation_is_bitwise_identical_to_fedavg(self):
        """FedProx's server side must BE FedAvg — matching Flower, which does not override it.

        Recorded explicitly because it makes a cross-framework aggregation comparison nearly
        vacuous: both frameworks would agree even if both ignored mu entirely. The tests above
        are what give the FedProx claim its teeth.
        """
        init = _fresh_params()
        updates = [
            ("c0", OrderedDict((k, v + 0.5) for k, v in init.items()), 30),
            ("c1", OrderedDict((k, v - 1.5) for k, v in init.items()), 70),
        ]
        prox = FedProx(initial_parameters=init, proximal_mu=0.9).aggregate_fit(1, list(updates))
        avg = FedAvg(initial_parameters=init).aggregate_fit(1, list(updates))
        for k in avg:
            assert torch.equal(prox[k], avg[k])
