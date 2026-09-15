"""P2-2 — the three secure-aggregation RPCs on the gRPC servicer.

The handlers are thin adapters over SecureAggregationSession; what they own is the part the
session cannot see: the gRPC context, and therefore the SE-15 verified partition.

THE DECISION THESE TESTS PIN
    Secure aggregation is keyed on the verified partition. When identity binding is unavailable —
    client auth off, or a call with no verifiable token — there IS no verified partition, so the
    handlers FAIL CLOSED with FAILED_PRECONDITION rather than falling back to the wire client_id.

    Falling back would be worse than refusing: a run would appear to be aggregating securely while
    any client could publish keys and relay shares as any other, which is exactly the attack the
    partition binding exists to stop. A privacy feature that silently degrades to no privacy is
    the one failure mode worth being loud about.
"""
from collections import OrderedDict
from unittest.mock import MagicMock

import grpc
import pytest
import torch

from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.security.key_agreement import generate_keypair
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer


class _FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def invocation_metadata(self):
        return ()


def _servicer(partition=1):
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=1, num_perturbations=2,
    )
    coordinator = FLCoordinator(strategy, min_clients_for_aggregation=1, clients_per_round=1)
    coordinator.bind_or_check_identity = MagicMock(return_value=True)
    return FederatedLearningServiceServicer(
        coordinator, partition_extractor=(lambda ctx: partition)
    )


def test_publishing_a_key_returns_the_cohort():
    s = _servicer(partition=1)
    _, pub = generate_keypair()
    resp = s.PublishPublicKey(
        pb.PublishPublicKeyRequest(client_id="a", run_id="r", round=1, public_key=pub),
        _FakeContext(),
    )
    assert resp.accepted is True
    assert resp.cohort_public_keys[1] == pub


def test_a_second_conflicting_key_is_rejected_with_a_reason_not_an_exception():
    s = _servicer(partition=1)
    _, first = generate_keypair()
    _, second = generate_keypair()
    ctx = _FakeContext()
    s.PublishPublicKey(pb.PublishPublicKeyRequest(client_id="a", run_id="r", round=1,
                                                  public_key=first), ctx)
    resp = s.PublishPublicKey(pb.PublishPublicKeyRequest(client_id="a", run_id="r", round=1,
                                                         public_key=second), ctx)
    assert resp.accepted is False
    assert "already published" in resp.rejection_reason
    assert resp.cohort_public_keys[1] == first, "the original key must survive"


def test_relayed_shares_come_back_to_their_recipient():
    s = _servicer(partition=1)
    s.SubmitSecureShares(
        pb.SubmitSecureSharesRequest(
            client_id="a", run_id="r", round=1,
            shares=[pb.SealedShare(recipient_partition=2, ciphertext=b"for-two")],
        ),
        _FakeContext(),
    )
    s2 = s  # same servicer, different caller partition
    s2._partition_extractor = lambda ctx: 2
    resp = s2.SubmitSecureShares(
        pb.SubmitSecureSharesRequest(client_id="b", run_id="r", round=1, shares=[]),
        _FakeContext(),
    )
    assert resp.accepted is True
    assert resp.inbound_ciphertexts[1] == b"for-two"


def test_secure_rpcs_fail_closed_when_identity_binding_is_unavailable():
    """The decision this module turns on: refuse rather than degrade to the untrusted client_id."""
    s = _servicer()
    s._partition_extractor = None          # auth off — no verified partition exists
    _, pub = generate_keypair()

    ctx = _FakeContext()
    resp = s.PublishPublicKey(
        pb.PublishPublicKeyRequest(client_id="a", run_id="r", round=1, public_key=pub), ctx
    )
    assert resp.accepted is False
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "identity" in (ctx.details or "").lower()


def test_a_malformed_public_key_is_rejected_with_a_reason():
    s = _servicer(partition=1)
    ctx = _FakeContext()
    resp = s.PublishPublicKey(
        pb.PublishPublicKeyRequest(client_id="a", run_id="r", round=1, public_key=b"short"), ctx
    )
    assert resp.accepted is False
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


def test_submitting_an_aggregated_share_reports_how_many_remain():
    s = _servicer(partition=1)
    resp = s.SubmitAggregatedShare(
        pb.SubmitAggregatedShareRequest(client_id="a", run_id="r", round=1,
                                        holder_index=1, summed_share=[1, 2]),
        _FakeContext(),
    )
    assert resp.received is True
    assert resp.shares_still_needed >= 0


# The masked-submission response must describe ONE state of the round. A live TLS run aggregated a wrong model: a
# holder was told the round was frozen while being handed the survivor list from before the freeze, so it summed
# shares over two of three dealers and the recovery decoded to a wrong aggregate. The handler took `survivors` when
# this request's submission was recorded and read `is_closed` again when building the response; a concurrent request
# closing the set in between produced submissions_closed=True with a partial list.

class _PartitionContext(_FakeContext):
    def __init__(self, partition):
        super().__init__()
        self.partition = partition


def _cohort_servicer(clients_per_round=3):
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=1, num_perturbations=2,
    )
    coordinator = FLCoordinator(strategy, min_clients_for_aggregation=1, clients_per_round=clients_per_round)
    coordinator.bind_or_check_identity = MagicMock(return_value=True)
    return FederatedLearningServiceServicer(coordinator, partition_extractor=lambda ctx: ctx.partition)


def _masked_request(client_id, round_num=1):
    from fedlearn.security.lightsecagg import PRIME
    return pb.SubmitGradientScalarsRequest(
        client_id=client_id, trained_on_round=round_num, num_examples=8,
        masked_gradients=pb.MaskedGradientScalars(
            elements=[5, 7], modulus=PRIME, num_local_steps=1, num_perturbations=2,
        ),
    )


def test_a_response_that_says_frozen_carries_the_whole_frozen_set_even_when_the_freeze_races_it():
    s = _cohort_servicer(clients_per_round=3)
    s.SubmitGradientScalars(_masked_request("c3"), _PartitionContext(3))
    session = s._secure_session(1)

    # Partition 2's request records its submission and closes the set in the window between partition 1's
    # submission being recorded and partition 1's response being built.
    record = session.submit_masked

    def submit_then_race(partition, elements):
        survivors = record(partition=partition, elements=elements)
        if partition == 1:
            record(partition=2, elements=elements)
            session.close_submissions()
        return survivors

    session.submit_masked = submit_then_race

    resp = s.SubmitGradientScalars(_masked_request("c1"), _PartitionContext(1))

    assert resp.received
    if resp.submissions_closed:
        assert list(resp.surviving_partitions) == [1, 2, 3], (
            f"told the round is frozen but handed {list(resp.surviving_partitions)}; a holder sums its shares over "
            f"exactly that set, so any missing dealer's mask never cancels"
        )
