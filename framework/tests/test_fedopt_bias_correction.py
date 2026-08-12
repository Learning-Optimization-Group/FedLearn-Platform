"""FedOpt: our FedAdam and Flower's FedAdam are not the same optimiser.

Written before the implementation (TDD), off the back of a cross-validation failure.

Extending the P0-3 credibility gate to FedOpt found FedYogi equivalent to Flower's at float32
epsilon (2.4e-07) but FedAdam diverging by **1.5e-01**. Since the two share `m_t` and the update
rule, the difference had to be in `v_t` or in the step scale. It is the step scale:

**Flower's `FedAdam` applies Kingma & Ba bias correction to the learning rate; its `FedYogi`
does not.**

    eta_norm = eta * sqrt(1 - beta_2^(t+1)) / (1 - beta_1^(t+1))

Reddi et al. 2021 ("Adaptive Federated Optimization", arXiv:2003.00295) Algorithm 2 — which our
docstring cites and whose update our code writes out — does **not** include that term. So
neither implementation is wrong; they are two defensible readings, and Flower documents theirs
as a deliberate early-round convergence improvement.

What matters is the magnitude, because it is not a rounding difference. At beta_1=0.9,
beta_2=0.99 the factor is 0.74 at round 1, bottoms near 0.47 around round 12, and is still only
0.93 at round 200. **Flower's FedAdam therefore takes roughly half the effective server step
for the whole of any realistic federated run.** Anyone comparing a FedLearn FedAdam number
against a published Flower FedAdam baseline is comparing different algorithms at
roughly-2x-different server learning rates.

This is directly relevant to a result already in this repo: `research/benchmarks/algo_sweep.py`
records FedOpt at eta=0.01 diverging on ResNet-18/ImageNet-100 (loss 845 by round 11, top-1
stalled at 3.15% against FedAvg's 40.3%). Halving the effective step is exactly the kind of
change that separates divergence from convergence, so that result should be re-run with the
correction before it is attributed to FedAdam as such.

Default stays paper-literal, because the docstring cites Reddi et al. and writes that update.
The correction is opt-in and exact, so a Flower comparison can be made apples-to-apples.
"""

from collections import OrderedDict

import pytest
import torch

from fedlearn.server.strategy import FedOpt, fedopt_bias_correction


def _init(seed=0):
    g = torch.Generator().manual_seed(seed)
    return OrderedDict([
        ("w", torch.randn(4, 3, generator=g)),
        ("b", torch.randn(3, generator=g)),
    ])


def _fresh(upd):
    """A deep copy of an update list.

    Necessary, and a sharp edge worth knowing: ``FedAvgAggregator.aggregate`` calls
    ``params.clear()`` on every client state_dict it consumes (``strategy.py``, "aggressively
    free client memory buffer"). That is deliberate — it is what keeps peak memory flat in a
    1000-client simulation — but it means an update list is single-use. Feeding the same list
    to two strategies silently hands the second one EMPTY dicts, which surfaces later as a
    KeyError deep inside the FedOpt update rather than as anything informative at the call site.
    """
    from copy import deepcopy
    return [(cid, deepcopy(sd), n) for cid, sd, n in upd]


def _updates(init, n_clients=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (f"c{i}",
         OrderedDict((k, v + 0.1 * torch.randn(v.shape, generator=g)) for k, v in init.items()),
         100 * (i + 1))
        for i in range(n_clients)
    ]


class TestBiasCorrectionFactor:
    def test_matches_the_kingma_ba_formula(self):
        b1, b2 = 0.9, 0.99
        for t in (1, 2, 5, 12, 50, 200):
            expected = (1 - b2 ** (t + 1)) ** 0.5 / (1 - b1 ** (t + 1))
            assert fedopt_bias_correction(t, b1, b2) == pytest.approx(expected, rel=1e-12)

    def test_is_substantially_below_one_for_realistic_round_counts(self):
        """The point of recording this: it is a ~2x step difference, not a rounding artifact."""
        b1, b2 = 0.9, 0.99
        assert fedopt_bias_correction(1, b1, b2) == pytest.approx(0.7425, abs=1e-3)
        assert fedopt_bias_correction(12, b1, b2) == pytest.approx(0.4692, abs=1e-3)
        assert fedopt_bias_correction(200, b1, b2) == pytest.approx(0.9313, abs=1e-3)

    def test_converges_to_one(self):
        assert fedopt_bias_correction(20000, 0.9, 0.99) == pytest.approx(1.0, abs=1e-6)


class TestDefaultIsPaperLiteral:
    def test_default_does_not_apply_the_correction(self):
        """Reddi et al. Algorithm 2 has no bias-correction term, and the docstring cites it."""
        init = _init()
        s = FedOpt(initial_parameters=init, variant="adam", server_learning_rate=0.1)
        assert s.bias_correction is False

    def test_correction_changes_the_result(self):
        """Anti-vacuity: the flag must actually alter the update, or the equivalence is empty."""
        init = _init()
        upd = _updates(init)
        plain = FedOpt(initial_parameters=init, variant="adam",
                       server_learning_rate=0.1).aggregate_fit(1, _fresh(upd))
        corrected = FedOpt(initial_parameters=init, variant="adam", server_learning_rate=0.1,
                           bias_correction=True).aggregate_fit(1, _fresh(upd))
        assert not torch.allclose(plain["w"], corrected["w"], atol=1e-7)

    def test_correction_takes_a_smaller_step_early(self):
        """The factor is < 1 early, so the corrected global must move LESS from the start."""
        init = _init()
        upd = _updates(init)
        plain = FedOpt(initial_parameters=init, variant="adam",
                       server_learning_rate=0.1).aggregate_fit(1, _fresh(upd))
        corrected = FedOpt(initial_parameters=init, variant="adam", server_learning_rate=0.1,
                           bias_correction=True).aggregate_fit(1, _fresh(upd))
        d_plain = float(torch.norm(plain["w"] - init["w"]))
        d_corr = float(torch.norm(corrected["w"] - init["w"]))
        assert d_corr < d_plain, f"corrected step {d_corr:.4g} should be smaller than {d_plain:.4g}"


class TestVariantScope:
    def test_yogi_ignores_the_flag_by_default_matching_flower(self):
        """Flower applies the correction in FedAdam ONLY; its FedYogi uses raw eta.

        Mirroring that asymmetry is what made FedYogi cross-validate at 2.4e-07 while FedAdam
        did not, so it is a deliberate asymmetry rather than an oversight to be tidied up.
        """
        init = _init()
        upd = _updates(init)
        a = FedOpt(initial_parameters=init, variant="yogi",
                   server_learning_rate=0.1).aggregate_fit(1, _fresh(upd))
        b = FedOpt(initial_parameters=init, variant="yogi", server_learning_rate=0.1,
                   bias_correction=True).aggregate_fit(1, _fresh(upd))
        assert torch.equal(a["w"], b["w"]), "bias correction must not apply to the yogi variant"


class TestMomentsStillPersist:
    def test_correction_does_not_disturb_moment_accumulation(self):
        """The correction scales eta only; (m, v) must accumulate exactly as before."""
        init = _init()
        s = FedOpt(initial_parameters=init, variant="adam", server_learning_rate=0.1,
                   bias_correction=True)
        for r in (1, 2, 3):
            s.aggregate_fit(r, _updates(init, seed=r))
        # A non-zero second moment after three rounds is the observable sign of accumulation.
        assert s._v is not None
        assert float(s._v["w"].abs().sum()) > 0.0
        assert float(s._m["w"].abs().sum()) > 0.0
