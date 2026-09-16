"""P2-2 slice 2 — LightSecAgg dropout resilience over the DeComFL scalar channel.

WHAT SLICE 1 COULD NOT DO
    Pairwise masking cancels only over the FULL cohort: one dropout leaves its peers' halves
    uncancelled and the aggregate is garbage. ``test_a_dropped_client_corrupts_the_aggregate...``
    in test_secure_aggregation.py pins exactly that.

WHAT MAKES THIS *LIGHT*SecAgg RATHER THAN CLASSIC SecAgg
    Classic SecAgg repairs a dropout by reconstructing THE DROPPED CLIENT'S mask — one
    reconstruction per dropout, so the server's recovery cost grows with how many left.
    LightSecAgg instead has each client secret-share its OWN mask; the survivors sum the shares
    they hold, and the server performs ONE decode that yields the aggregate of the surviving
    masks directly. Recovery cost is independent of the dropout count.

    That one-shot property rests entirely on Shamir sharing being LINEAR: the sum of shares of
    several secrets is a valid share of the sum of those secrets. Test
    ``test_summed_shares_reconstruct_the_sum_of_secrets`` is the load-bearing one here — if it
    fails, the scheme is classic SecAgg with extra steps.

    Shamir needs a PRIME field, which is why this path uses 2**31 - 1 rather than slice 1's
    power-of-two modulus.
"""
import pytest
import torch

from fedlearn.security.lightsecagg import (
    PRIME,
    aggregate_shares,
    client_mask,
    mask_values,
    recover_aggregate,
    shamir_reconstruct,
    shamir_share,
)


def test_shares_reconstruct_the_original_secret():
    secret = torch.tensor([12345, 67890, 1], dtype=torch.int64)
    shares = shamir_share(secret, num_shares=5, threshold=3, seed=1)

    assert len(shares) == 5
    # Any threshold-sized subset must do, not just the first — pick a scattered one.
    recovered = shamir_reconstruct({i: shares[i] for i in (1, 3, 5)}, threshold=3)
    assert torch.equal(recovered, secret)


def test_summed_shares_reconstruct_the_sum_of_secrets():
    """THE load-bearing property: linearity is what buys one-shot aggregate recovery."""
    secrets = [
        torch.tensor([10, 20], dtype=torch.int64),
        torch.tensor([3, 4], dtype=torch.int64),
        torch.tensor([100, 1], dtype=torch.int64),
    ]
    per_secret = [shamir_share(s, num_shares=4, threshold=2, seed=i) for i, s in enumerate(secrets)]

    # Each holder adds up the shares it received from every dealer -- no dealer involvement.
    summed = {
        holder: sum((sh[holder] for sh in per_secret), torch.zeros(2, dtype=torch.int64)) % PRIME
        for holder in (1, 2, 3, 4)
    }
    recovered = shamir_reconstruct({h: summed[h] for h in (2, 4)}, threshold=2)

    expected = (secrets[0] + secrets[1] + secrets[2]) % PRIME
    assert torch.equal(recovered, expected)


def test_reconstruction_below_threshold_is_refused():
    secret = torch.tensor([42], dtype=torch.int64)
    shares = shamir_share(secret, num_shares=5, threshold=3, seed=7)
    with pytest.raises(ValueError, match="threshold"):
        shamir_reconstruct({i: shares[i] for i in (1, 2)}, threshold=3)


def test_shamir_rejects_a_threshold_larger_than_the_share_count():
    with pytest.raises(ValueError, match="threshold|num_shares"):
        shamir_share(torch.tensor([1], dtype=torch.int64), num_shares=3, threshold=4, seed=0)


# --------------------------------------------------------------------------------------------------
# End to end: the aggregate survives a dropout, which is the entire point of slice 2
# --------------------------------------------------------------------------------------------------
def _run_round(values, dropped, threshold=3, round_seed=99):
    """Simulate one LightSecAgg round over ``values`` (client_id -> list of floats).

    Mirrors the protocol exactly: every client masks and shares, the server then names the
    surviving set, each surviving holder returns ONE summed share vector, and the server does a
    single decode.
    """
    cohort = sorted(values)
    length = len(next(iter(values.values())))
    n = len(cohort)

    masks = {c: client_mask(c, round_seed=round_seed, length=length) for c in cohort}
    shares = {
        # Seeded by cohort POSITION, not hash(c): Python randomises string hashing per
        # process, so hash(c) would exercise a different polynomial on every run.
        c: shamir_share(masks[c], num_shares=n, threshold=threshold, seed=idx)
        for idx, c in enumerate(cohort)
    }
    masked = {c: mask_values(values[c], masks[c]) for c in cohort}

    survivors = [c for c in cohort if c not in dropped]
    # Holder j sums only over the SURVIVING dealers -- one vector per holder, not one per dealer.
    summed = {
        j: aggregate_shares([shares[i][j] for i in survivors])
        for j in range(1, n + 1)
    }
    return recover_aggregate(
        masked_values=[masked[c] for c in survivors],
        summed_shares={j: summed[j] for j in list(summed)[:threshold]},
        threshold=threshold,
    )


def test_aggregate_is_exact_when_nobody_drops():
    values = {"a": [1.0, -2.0], "b": [0.5, 0.25], "c": [-1.5, 3.0], "d": [2.0, 1.0], "e": [0.0, 0.5]}
    out = _run_round(values, dropped=set())
    expected = torch.tensor([sum(v[i] for v in values.values()) for i in range(2)])
    assert torch.allclose(out, expected, atol=1e-5)


def test_aggregate_survives_a_dropout_which_slice_1_could_not():
    """The regression slice 1 pinned as broken: one dropout corrupted everything."""
    values = {"a": [1.0, -2.0], "b": [0.5, 0.25], "c": [-1.5, 3.0], "d": [2.0, 1.0], "e": [9.0, 9.0]}
    out = _run_round(values, dropped={"e"})

    survivors = [v for c, v in values.items() if c != "e"]
    expected = torch.tensor([sum(v[i] for v in survivors) for i in range(2)])
    assert torch.allclose(out, expected, atol=1e-5), "dropped client's mask was not removed"
    # And the dropped client's VALUE must be absent, not merely the mask.
    assert not torch.allclose(out, torch.tensor([sum(v[0] for v in values.values()),
                                                 sum(v[1] for v in values.values())]), atol=1e-3)


def test_aggregate_survives_multiple_dropouts_with_one_decode():
    values = {"a": [1.0], "b": [2.0], "c": [4.0], "d": [8.0], "e": [16.0]}
    out = _run_round(values, dropped={"d", "e"})
    assert torch.allclose(out, torch.tensor([7.0]), atol=1e-5)


def test_recovery_refuses_below_threshold():
    values = {"a": [1.0], "b": [2.0], "c": [4.0], "d": [8.0], "e": [16.0]}
    with pytest.raises(ValueError, match="threshold"):
        recover_aggregate(
            masked_values=[mask_values([1.0], client_mask("a", round_seed=1, length=1))],
            summed_shares={1: torch.tensor([0], dtype=torch.int64)},
            threshold=3,
        )
