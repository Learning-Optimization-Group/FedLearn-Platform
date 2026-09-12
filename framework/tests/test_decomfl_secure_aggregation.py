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
