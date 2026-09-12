"""P2-2 — the client side of secure aggregation, driven against the REAL servicer.

There is no gRPC channel here, but there are no fakes on the server side either: the stub below
calls the actual servicer handlers, which drive the actual SecureAggregationSession. So this
exercises the whole protocol -- key publication, share sealing and relay, masked submission,
one-shot recovery -- end to end in one process.

What it cannot cover, and what a live server is still needed for: channel behaviour (deadlines,
retries, the interceptor chain) and the real partition extractor. Those are the reason this is
not the same as a deployment test.
"""
from collections import OrderedDict
from unittest.mock import MagicMock

import torch

from fedlearn.client.secure_agg_client import (
    SecureAggregationClient,
    SecureAggregationNotReady,
)
from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer


class _DirectStub:
    """A stub that calls the servicer in-process, with a settable caller partition."""

    def __init__(self, servicer, partition):
        self._s = servicer
        self.partition = partition

    class _Ctx:
        def set_code(self, c): self.code = c
        def set_details(self, d): self.details = d
        def invocation_metadata(self): return ()

    def _call(self, name, request):
        self._s._partition_extractor = lambda ctx, p=self.partition: p
        return getattr(self._s, name)(request, self._Ctx())

    def PublishPublicKey(self, r): return self._call("PublishPublicKey", r)
    def SubmitSecureShares(self, r): return self._call("SubmitSecureShares", r)
    def SubmitAggregatedShare(self, r): return self._call("SubmitAggregatedShare", r)


def _servicer(threshold, K=1, P=2):
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=K, num_perturbations=P,
    )
    coord = FLCoordinator(strategy, min_clients_for_aggregation=1, clients_per_round=1)
    coord.bind_or_check_identity = MagicMock(return_value=True)
    return FederatedLearningServiceServicer(coord, secure_agg_threshold=threshold)


def test_a_full_secure_round_recovers_the_plaintext_sum():
    """The capstone: four clients, real servicer, and the server learns only the sum."""
    n, t, round_num = 4, 3, 1
    servicer = _servicer(threshold=t)
    partitions = [1, 2, 3, 4]
    values = {1: [1.0, -2.0], 2: [0.5, 0.25], 3: [-1.5, 3.0], 4: [2.0, 1.0]}

    clients = {
        p: SecureAggregationClient(_DirectStub(servicer, p), client_id=f"c{p}")
        for p in partitions
    }

    # Phase 1 + 2 for everyone before anyone submits -- keys must all be published before shares
    # are sealed against them.
    for p in partitions:
        clients[p].begin_round(round_num=round_num, threshold=t, num_scalars=2, cohort_size=n)
    for p in partitions:
        clients[p].distribute_shares(round_num=round_num)

    # Relay is a rendezvous: whoever submitted first saw an empty inbox and comes back for it.
    for p in partitions:
        clients[p].collect_shares(round_num=round_num)

    # Phase 3a: masked scalars go in through the session (the submit RPC path carries them).
    session = servicer._secure_session(round_num)
    for p in partitions:
        session.submit_masked(partition=p, elements=clients[p].mask(values[p]))

    # Phase 3b: each holder returns ONE summed vector.
    for p in partitions[:t]:
        clients[p].finish_round(round_num=round_num, survivors=session.survivors)

    recovered = session.recover()
    expected = torch.tensor([sum(values[p][i] for p in partitions) for i in range(2)])
    assert torch.allclose(recovered, expected, atol=1e-5)


def test_the_client_discovers_its_own_partition_from_the_cohort_view():
    """The client is never told its partition; it finds itself by its own published key.

    That avoids a proto field and, more importantly, means a client cannot be told it is
    somebody else.
    """
    servicer = _servicer(threshold=2)
    c = SecureAggregationClient(_DirectStub(servicer, 7), client_id="c7")
    c.begin_round(round_num=1, threshold=1, num_scalars=2, cohort_size=1)
    assert c.partition == 7


def test_masking_hides_the_values_from_anyone_reading_the_wire():
    servicer = _servicer(threshold=2)
    c = SecureAggregationClient(_DirectStub(servicer, 1), client_id="c1")
    c.begin_round(round_num=1, threshold=2, num_scalars=2, cohort_size=2)
    masked = c.mask([1.0, 2.0])
    assert masked != [1_000_000, 2_000_000], "masked payload equals the plaintext quantisation"
    assert all(isinstance(v, int) for v in masked)


def test_the_mask_is_not_derivable_from_public_values():
    """The mask must come from real entropy, not from ``(client_id, round)``.

    ``lightsecagg.client_mask`` derives a mask from ``sha256(round | client_id)`` -- both PUBLIC.
    It is fine as a test helper (its only callers are tests), but a client that used it would
    hand an observer the ability to regenerate its mask and unmask its individual contribution,
    which is the one thing this whole protocol exists to prevent.

    Two clients that agree on every public input must still draw different masks.
    """
    servicer = _servicer(threshold=2)
    a = SecureAggregationClient(_DirectStub(servicer, 1), client_id="same")
    b = SecureAggregationClient(_DirectStub(servicer, 2), client_id="same")
    a.begin_round(round_num=1, threshold=2, num_scalars=8, cohort_size=2)
    b.begin_round(round_num=1, threshold=2, num_scalars=8, cohort_size=2)
    assert not torch.equal(a._mask, b._mask), "mask is a function of public inputs only"


def test_an_inbound_share_is_opened_against_a_LOCALLY_rebuilt_binding():
    """The recipient recomputes the associated data rather than trusting what it was handed.

    ``SealedShare`` carries ``associated_data`` on the wire, but the server relays only the
    ciphertext -- and that is the correct design. If the recipient took the sender's AD at face
    value, the binding to ``(round, sender, recipient)`` would authenticate nothing, since an
    attacker rewriting the routing would rewrite the AD to match.
    """
    servicer = _servicer(threshold=2)
    a = SecureAggregationClient(_DirectStub(servicer, 1), client_id="a")
    b = SecureAggregationClient(_DirectStub(servicer, 2), client_id="b")
    for c in (a, b):
        c.begin_round(round_num=1, threshold=2, num_scalars=4, cohort_size=2)
    for c in (a, b):
        c.distribute_shares(round_num=1)

    # b holds a's share, opened under an AD b rebuilt itself.
    assert 1 in b._held, "b did not open the share a sealed to it"
    assert b._associated_data(round_num=1, sender=1, recipient=2) != \
        b._associated_data(round_num=1, sender=2, recipient=1), "binding is not directional"


def test_a_client_that_drops_after_sharing_is_excluded_without_a_second_decode():
    """The reason to build LightSecAgg rather than classic SecAgg.

    Client 4 completes phases 1 and 2, then vanishes before submitting anything maskable. Its
    mask is in the survivors' held shares but must NOT enter the recovered total, and the server
    must still decode in one interpolation rather than reconstructing the dropped client's mask
    separately.
    """
    n, t, round_num = 4, 3, 1
    servicer = _servicer(threshold=t)
    partitions = [1, 2, 3, 4]
    values = {1: [1.0, -2.0], 2: [0.5, 0.25], 3: [-1.5, 3.0]}
    dropped = 4

    clients = {
        p: SecureAggregationClient(_DirectStub(servicer, p), client_id=f"c{p}")
        for p in partitions
    }
    for p in partitions:
        clients[p].begin_round(round_num=round_num, threshold=t, num_scalars=2, cohort_size=n)
    for p in partitions:
        clients[p].distribute_shares(round_num=round_num)
    for p in partitions:
        clients[p].collect_shares(round_num=round_num)

    session = servicer._secure_session(round_num)
    for p in partitions:
        if p == dropped:
            continue  # gone: no masked submission ever arrives
        session.submit_masked(partition=p, elements=clients[p].mask(values[p]))

    assert session.survivors == [1, 2, 3]
    for p in session.survivors:
        clients[p].finish_round(round_num=round_num, survivors=session.survivors)

    recovered = session.recover()
    expected = torch.tensor([sum(values[p][i] for p in session.survivors) for i in range(2)])
    assert torch.allclose(recovered, expected, atol=1e-5), (
        "the dropped client's mask leaked into the aggregate"
    )


def test_summing_over_a_stale_survivor_set_is_refused_rather_than_decoded_wrong():
    """A holder asked to sum over a dealer it has no share from must refuse.

    Silently summing the subset produces a well-formed vector that decodes to the wrong number --
    the worst possible failure here, because nothing downstream can detect it.
    """
    servicer = _servicer(threshold=2)
    a = SecureAggregationClient(_DirectStub(servicer, 1), client_id="a")
    b = SecureAggregationClient(_DirectStub(servicer, 2), client_id="b")
    for c in (a, b):
        c.begin_round(round_num=1, threshold=2, num_scalars=2, cohort_size=2)
    for c in (a, b):
        c.distribute_shares(round_num=1)
    for c in (a, b):
        c.collect_shares(round_num=1)

    import pytest
    with pytest.raises(SecureAggregationNotReady, match="missing shares from surviving dealers"):
        a.finish_round(round_num=1, survivors=[1, 2, 99])


# ---------------------------------------------------------------------------------------------
# The masked payload's route through the REAL submit RPC
# ---------------------------------------------------------------------------------------------
def test_masked_scalars_reach_the_session_through_SubmitGradientScalars():
    """``masked_gradients`` is declared in the proto; this asserts it is actually consumed.

    Without this route the client can mask perfectly and still have nowhere to send it -- the
    field would be a contract the server ignores, which is worse than not having it, because the
    client would believe it had submitted.
    """
    import pytest
    n, t, round_num = 3, 2, 1
    servicer = _servicer(threshold=t, K=1, P=2)
    partitions = [1, 2, 3]
    values = {1: [1.0, -2.0], 2: [0.5, 0.25], 3: [-1.5, 3.0]}

    clients = {
        p: SecureAggregationClient(_DirectStub(servicer, p), client_id=f"c{p}")
        for p in partitions
    }
    for p in partitions:
        clients[p].begin_round(round_num=round_num, threshold=t, num_scalars=2, cohort_size=n)
    for p in partitions:
        clients[p].distribute_shares(round_num=round_num)
    for p in partitions:
        clients[p].collect_shares(round_num=round_num)

    for p in partitions:
        stub = _DirectStub(servicer, p)
        response = stub._call("SubmitGradientScalars", pb.SubmitGradientScalarsRequest(
            client_id=f"c{p}",
            trained_on_round=round_num,
            num_examples=10,
            masked_gradients=pb.MaskedGradientScalars(
                elements=clients[p].mask(values[p]),
                modulus=2 ** 31 - 1,
                num_local_steps=1,
                num_perturbations=2,
            ),
        ))
        assert response.received, "the server did not accept a masked submission"
        assert sorted(response.surviving_partitions) == sorted(partitions[:p]), \
            "the response does not name the surviving set the client needs for phase 3b"

    session = servicer._secure_session(round_num)
    for p in partitions[:t]:
        clients[p].finish_round(round_num=round_num, survivors=session.survivors)

    recovered = session.recover()
    expected = torch.tensor([sum(values[p][i] for p in partitions) for i in range(2)])
    assert torch.allclose(recovered, expected, atol=1e-5)


def test_a_masked_submission_under_the_wrong_field_is_refused():
    """A modulus mismatch means the client quantised into a different field.

    Accepting it would decode to noise, so it is refused at the door rather than aggregated.
    """
    servicer = _servicer(threshold=2, K=1, P=2)
    c = SecureAggregationClient(_DirectStub(servicer, 1), client_id="c1")
    c.begin_round(round_num=1, threshold=2, num_scalars=2, cohort_size=2)

    stub = _DirectStub(servicer, 1)
    response = stub._call("SubmitGradientScalars", pb.SubmitGradientScalarsRequest(
        client_id="c1", trained_on_round=1, num_examples=10,
        masked_gradients=pb.MaskedGradientScalars(
            elements=c.mask([1.0, 2.0]), modulus=2 ** 32,  # slice 1's power-of-two modulus
            num_local_steps=1, num_perturbations=2,
        ),
    ))
    assert not response.received
