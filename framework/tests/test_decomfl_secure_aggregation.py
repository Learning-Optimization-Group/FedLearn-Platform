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


# ---------------------------------------------------------------------------------------------
# A secure round must leave the same server-side history a plaintext round does
# ---------------------------------------------------------------------------------------------
def _secure_round_inputs(strategy, values, threshold, seed_base=100):
    """Run the client-side masking for one round; return what the server would hold."""
    from fedlearn.security.lightsecagg import (
        aggregate_shares, client_mask, mask_values, shamir_share,
    )
    n = len(values)
    length = strategy.K * strategy.P
    masks = {c: client_mask(str(c), round_seed=seed_base, length=length) for c in values}
    shares = {
        c: shamir_share(masks[c], num_shares=n, threshold=threshold, seed=seed_base + c)
        for c in values
    }
    masked = [mask_values(values[c], masks[c]) for c in values]
    summed = {
        h: aggregate_shares([shares[c][h] for c in values])
        for h in range(1, threshold + 1)
    }
    return masked, summed


def test_a_secure_round_records_gradient_history_so_rejoin_still_works():
    """Without this, a secure round is a hole in the rebuild chain.

    ``get_rebuild_history`` refuses to hand back a torn chain -- correctly, since a silent gap
    would diverge the client's model. But that means a client catching up across a secure round
    gets a hard DeComFLRebuildGap instead of rejoining: the privacy path would break rejoin
    outright rather than degrade it.
    """
    K, P, n, t = 2, 2, 3, 2
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=K, num_perturbations=P,
    )
    server_round = 1
    strategy.get_or_create_seeds(server_round)
    values = {1: [0.1, -0.2, 0.3, 0.05], 2: [0.2, 0.1, -0.1, 0.0], 3: [-0.1, 0.05, 0.2, 0.1]}
    masked, summed = _secure_round_inputs(strategy, values, threshold=t)

    strategy.aggregate_fit_secure(
        server_round=server_round, masked_values=masked, summed_shares=summed,
        threshold=t, num_clients=n,
    )

    assert server_round in strategy.gradient_history, (
        "secure round left no gradient history; a rejoining client hits DeComFLRebuildGap"
    )
    recorded = strategy.gradient_history[server_round]
    assert len(recorded) == K and all(len(row) == P for row in recorded)

    # The recorded value is the AVERAGE, matching what the plaintext path stores -- clients
    # replay it directly, so a sum here would step every rejoining client N times too far.
    expected_avg = [
        [sum(values[c][k * P + p] for c in values) / n for p in range(P)]
        for k in range(K)
    ]
    for k in range(K):
        for p in range(P):
            assert abs(recorded[k][p] - expected_avg[k][p]) < 1e-4, (
                f"gradient_history[{k}][{p}] is not the client-replayable average"
            )


def test_a_client_can_rebuild_across_a_secure_round():
    """The end the previous test protects: catch-up across a secure round must not raise."""
    K, P, n, t = 1, 2, 3, 2
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=K, num_perturbations=P,
    )
    strategy.client_last_round["late"] = 0
    for r in (1, 2):
        strategy.get_or_create_seeds(r)
        values = {c: [0.1 * c, -0.05 * c] for c in (1, 2, 3)}
        masked, summed = _secure_round_inputs(strategy, values, threshold=t, seed_base=100 * r)
        strategy.aggregate_fit_secure(
            server_round=r, masked_values=masked, summed_shares=summed,
            threshold=t, num_clients=n,
        )

    history = strategy.get_rebuild_history("late", current_round=3)
    assert [h["round_number"] for h in history] == [1, 2]
