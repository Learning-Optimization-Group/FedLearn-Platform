"""P2-2 slice 3 — wire secure aggregation into DeComFL's aggregation path.

The claim this file exists to pin: running a round under secure aggregation produces the SAME
global model as running it in plaintext, to within the quantisation bound. If that fails, the
privacy is free only in the sense that it does not work.

Why the composition is possible at all: ``aggregate_fit`` consumes exactly ``g_sums[k][p]``
(the sum of every client's scalar for that local step and perturbation) and the client count. It
never reads an individual client's scalars, so handing it a securely-recovered sum instead of a
plaintext one changes nothing downstream.
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.security.lightsecagg import (
    aggregate_shares,
    client_mask,
    mask_values,
    shamir_share,
)
from fedlearn.server.decomfl_strategy import DeComFL


def _strategy(seed: int = 42) -> DeComFL:
    return DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(6, dtype=torch.float32)}),
        num_local_steps=2,
        num_perturbations=3,
        learning_rate=0.01,
        seed=seed,
    )


# 4 clients x K=2 local steps x P=3 perturbations
_SCALARS = {
    "a": [[0.10, -0.20, 0.30], [0.05, 0.15, -0.25]],
    "b": [[-0.05, 0.25, -0.10], [0.20, -0.30, 0.10]],
    "c": [[0.15, 0.10, 0.05], [-0.10, 0.20, 0.15]],
    "d": [[0.00, -0.15, 0.20], [0.25, 0.05, -0.05]],
}


def test_secure_round_matches_the_plaintext_round():
    """The headline: privacy changes the wire, not the model."""
    plain_strategy = _strategy()
    plain_strategy.get_or_create_seeds(1)
    plain = plain_strategy.aggregate_fit(
        1, [(c, _SCALARS[c], 100) for c in sorted(_SCALARS)]
    )

    secure_strategy = _strategy()
    secure_strategy.get_or_create_seeds(1)

    cohort = sorted(_SCALARS)
    K, P, n, t = 2, 3, len(cohort), 3
    masks = {c: client_mask(c, round_seed=1, length=K * P) for c in cohort}
    shares = {
        c: shamir_share(masks[c], num_shares=n, threshold=t, seed=i)
        for i, c in enumerate(cohort)
    }
    masked = {
        c: mask_values([v for step in _SCALARS[c] for v in step], masks[c]) for c in cohort
    }
    summed = {j: aggregate_shares([shares[c][j] for c in cohort]) for j in range(1, t + 1)}

    secure = secure_strategy.aggregate_fit_secure(
        1,
        masked_values=[masked[c] for c in cohort],
        summed_shares=summed,
        threshold=t,
        num_clients=n,
    )

    assert secure is not None
    # Quantisation bound: n/(2*scale) per scalar, propagated through eta * z. Well inside 1e-4.
    assert torch.allclose(secure["w"], plain["w"], atol=1e-4), (
        f"secure round diverged from plaintext:\n  plain  {plain['w']}\n  secure {secure['w']}"
    )


def test_secure_round_is_not_trivially_zero_or_unchanged():
    """Guards against the test passing because NEITHER path moved the model."""
    s = _strategy()
    s.get_or_create_seeds(1)
    before = s.global_params_flat.clone()
    after = s.aggregate_fit(1, [(c, _SCALARS[c], 100) for c in sorted(_SCALARS)])
    assert not torch.allclose(after["w"], before), "the plaintext round did not move the model"


def test_secure_round_refuses_below_threshold():
    s = _strategy()
    s.get_or_create_seeds(1)
    with pytest.raises(ValueError, match="threshold"):
        s.aggregate_fit_secure(
            1,
            masked_values=[torch.zeros(6, dtype=torch.int64)],
            summed_shares={1: torch.zeros(6, dtype=torch.int64)},
            threshold=3,
            num_clients=4,
        )


def test_quantisation_error_does_not_compound_over_a_long_run():
    """Does the per-round quantisation error accumulate into a meaningful drift?

    Each round's secure aggregate sits within n/(2*scale) of the plaintext one, but those errors
    enter the model through ``x -= eta * delta`` and could in principle compound over a run. This
    is the first question a reviewer would ask of a "privacy is free" claim, so it is measured
    rather than argued.

    Measured over 200 rounds: drift grows SUB-LINEARLY (200x the rounds gives ~44x the drift,
    between sqrt(T) and T) and stays ~7 orders of magnitude below the distance to target. The
    assertion below is a regression guard -- it fails if someone lowers the quantisation scale
    far enough to matter, which is exactly the change that would silently degrade the model.
    """
    from fedlearn.security.lightsecagg import (
        aggregate_shares, client_mask, mask_values, shamir_share,
    )
    import random

    D, K, P, N, T, THRESH = 20, 1, 10, 4, 200, 3
    cohort = [f"c{i}" for i in range(N)]

    def fresh():
        torch.manual_seed(0)
        random.seed(0)
        return DeComFL(
            initial_parameters=OrderedDict({"w": torch.zeros(D)}),
            num_local_steps=K, num_perturbations=P, learning_rate=0.05, seed=7,
        )

    plain, secure = fresh(), fresh()
    target = torch.ones(D)

    for r in range(1, T + 1):
        plain.get_or_create_seeds(r)
        secure.get_or_create_seeds(r)
        # Identical scalars for both arms, so any divergence is the masking round-trip alone.
        scal = {
            c: [[float(((plain.global_params_flat - target)
                        @ plain._generate_perturbation(plain.seed_history[r][0][p])).item() / P)
                 for p in range(P)]]
            for c in cohort
        }
        plain.aggregate_fit(r, [(c, scal[c], 100) for c in cohort])

        masks = {c: client_mask(c, round_seed=r, length=K * P) for c in cohort}
        shares = {c: shamir_share(masks[c], num_shares=N, threshold=THRESH, seed=i)
                  for i, c in enumerate(cohort)}
        secure.aggregate_fit_secure(
            r,
            masked_values=[mask_values([v for st in scal[c] for v in st], masks[c])
                           for c in cohort],
            summed_shares={j: aggregate_shares([shares[c][j] for c in cohort])
                           for j in range(1, THRESH + 1)},
            threshold=THRESH, num_clients=N,
        )

    drift = (secure.global_params_flat - plain.global_params_flat).norm().item()
    signal = (plain.global_params_flat - target).norm().item()

    assert drift / signal < 1e-4, (
        f"secure/plaintext drift grew to {drift:.3e} against a {signal:.4f} signal "
        f"({drift / signal:.2e}) -- quantisation error is compounding"
    )
    assert signal < 4.0, "the run did not converge, so the drift comparison is meaningless"
