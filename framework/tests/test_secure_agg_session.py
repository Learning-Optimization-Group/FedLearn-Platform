"""P2-2 slice 6 — per-round server state for secure aggregation.

The three new RPCs are stateful across a round: keys published in phase 1 are needed to relay in
phase 2, and the survivor set from phase 3a decides which shares phase 3b must sum. That state is
this class, kept out of the gRPC servicer so it can be tested without a channel -- the servicer
handlers become thin adapters over it.

Everything security-relevant is keyed on the SE-15 verified partition, never on the wire
client_id, which the proto itself marks untrusted.
"""
import pytest
import torch

from fedlearn.security.key_agreement import generate_keypair
from fedlearn.security.public_key_registry import PublicKeyConflict
from fedlearn.server.secure_agg_session import SecureAggregationSession


def _keys(n):
    return {p: generate_keypair()[1] for p in range(1, n + 1)}


def test_publishing_a_key_returns_the_cohort_view():
    s = SecureAggregationSession(round_index=1, threshold=3, num_scalars=4)
    ks = _keys(2)
    s.publish_key(partition=1, public_key=ks[1])
    cohort = s.publish_key(partition=2, public_key=ks[2])
    assert cohort == {1: ks[1], 2: ks[2]}


def test_a_partition_cannot_replace_its_published_key():
    s = SecureAggregationSession(round_index=1, threshold=3, num_scalars=4)
    _, first = generate_keypair()
    _, second = generate_keypair()
    s.publish_key(partition=1, public_key=first)
    with pytest.raises(PublicKeyConflict):
        s.publish_key(partition=1, public_key=second)


def test_relayed_shares_are_delivered_only_to_their_recipient():
    s = SecureAggregationSession(round_index=1, threshold=2, num_scalars=2)
    s.relay_shares(sender=1, shares={2: b"for-two", 3: b"for-three"})
    s.relay_shares(sender=2, shares={1: b"two-to-one"})

    assert s.inbound_for(partition=1) == {2: b"two-to-one"}
    assert s.inbound_for(partition=2) == {1: b"for-two"}
    assert s.inbound_for(partition=3) == {1: b"for-three"}


def test_a_sender_cannot_address_a_share_to_itself():
    """Self-addressed shares are meaningless and would skew the holder count."""
    s = SecureAggregationSession(round_index=1, threshold=2, num_scalars=2)
    with pytest.raises(ValueError, match="itself"):
        s.relay_shares(sender=1, shares={1: b"self"})


def test_masked_submissions_define_the_surviving_set():
    s = SecureAggregationSession(round_index=1, threshold=2, num_scalars=2)
    assert s.submit_masked(partition=1, elements=[10, 20]) == [1]
    assert s.submit_masked(partition=3, elements=[30, 40]) == [1, 3]


def test_a_masked_submission_of_the_wrong_length_is_refused():
    s = SecureAggregationSession(round_index=1, threshold=2, num_scalars=4)
    with pytest.raises(ValueError, match="4"):
        s.submit_masked(partition=1, elements=[1, 2])


def test_summed_shares_count_down_to_the_threshold():
    s = SecureAggregationSession(round_index=1, threshold=3, num_scalars=2)
    assert s.submit_summed_share(holder_index=1, share=[1, 1]) == 2
    assert s.submit_summed_share(holder_index=2, share=[2, 2]) == 1
    assert s.submit_summed_share(holder_index=3, share=[3, 3]) == 0


def test_recovery_is_refused_before_the_threshold_is_met():
    s = SecureAggregationSession(round_index=1, threshold=3, num_scalars=2)
    s.submit_masked(partition=1, elements=[1, 1])
    s.close_submissions()   # the freeze is a separate precondition; this test is about the threshold
    s.submit_summed_share(holder_index=1, share=[1, 1])
    with pytest.raises(ValueError, match="threshold"):
        s.recover()


def test_a_full_round_recovers_the_plaintext_sum():
    """End to end through the session, using the real masking primitives."""
    from fedlearn.security.lightsecagg import (
        aggregate_shares, client_mask, mask_values, shamir_share,
    )
    n, t, length = 4, 3, 2
    partitions = [1, 2, 3, 4]
    values = {1: [1.0, -2.0], 2: [0.5, 0.25], 3: [-1.5, 3.0], 4: [2.0, 1.0]}

    s = SecureAggregationSession(round_index=7, threshold=t, num_scalars=length)
    masks = {p: client_mask(str(p), round_seed=7, length=length) for p in partitions}
    shares = {p: shamir_share(masks[p], num_shares=n, threshold=t, seed=p) for p in partitions}

    for p in partitions:
        s.submit_masked(partition=p, elements=mask_values(values[p], masks[p]).tolist())
    s.close_submissions()   # freeze before any holder sums, so every holder sums the same set
    for holder in range(1, t + 1):
        s.submit_summed_share(
            holder_index=holder,
            share=aggregate_shares([shares[p][holder] for p in partitions]).tolist(),
        )

    recovered = s.recover()
    expected = torch.tensor([sum(values[p][i] for p in partitions) for i in range(length)])
    assert torch.allclose(recovered, expected, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# Freezing the surviving set — without this, phase 3b decodes to a well-formed wrong number
# ---------------------------------------------------------------------------------------------
def test_survivors_must_be_frozen_before_holders_can_agree_on_them():
    """The surviving set a client is handed at submit time is partial and still growing.

    Holder A submitting first is told ``[1]``; holder C submitting last is told ``[1,2,3]``. If
    each sums the shares for the set IT was given -- which is exactly what the proto tells it to
    do -- they produce shares of DIFFERENT mask-sums, and the server's single interpolation
    decodes to a well-formed wrong aggregate that nothing downstream can detect.

    Closing the round is what makes the set authoritative.
    """
    session = SecureAggregationSession(round_index=1, threshold=2, num_scalars=2)
    assert not session.is_closed

    session.submit_masked(partition=1, elements=[1, 2])
    session.submit_masked(partition=2, elements=[3, 4])
    session.close_submissions()

    assert session.is_closed
    assert session.survivors == [1, 2]


def test_a_masked_submission_after_the_freeze_is_refused():
    """A late arrival cannot be admitted: holders have already summed over the frozen set, so
    adding a dealer now would leave its mask in the total with no share to cancel it."""
    session = SecureAggregationSession(round_index=1, threshold=2, num_scalars=2)
    session.submit_masked(partition=1, elements=[1, 2])
    session.close_submissions()

    with pytest.raises(ValueError, match="closed"):
        session.submit_masked(partition=2, elements=[3, 4])
    assert session.survivors == [1], "a refused submission still changed the surviving set"


def test_recovery_refuses_to_run_on_an_unfrozen_round():
    """Decoding before the freeze means the holders' shares and the masked total were computed
    over different sets."""
    session = SecureAggregationSession(round_index=1, threshold=1, num_scalars=1)
    session.submit_masked(partition=1, elements=[5])
    session.submit_summed_share(holder_index=1, share=[5])

    with pytest.raises(ValueError, match="not been closed"):
        session.recover()


def test_closing_an_already_closed_round_is_idempotent():
    """The close can be driven by whichever comes first -- the last submission or a deadline --
    so it must tolerate both firing."""
    session = SecureAggregationSession(round_index=1, threshold=1, num_scalars=1)
    session.submit_masked(partition=1, elements=[5])
    session.close_submissions()
    session.close_submissions()
    assert session.survivors == [1]
