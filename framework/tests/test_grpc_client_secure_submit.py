"""P2-2 — GrpcClient's masked submission path.

``SecureAggregationClient`` owns the three secure-aggregation RPCs; the masked scalars themselves
travel on ``SubmitGradientScalars``, which ``GrpcClient`` owns. This is the seam between them.
"""
from unittest.mock import MagicMock

import pytest

from fedlearn.client.grpc_client import GrpcClient
from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.security.lightsecagg import PRIME


@pytest.fixture
def client():
    c = GrpcClient.__new__(GrpcClient)   # no channel: this tests request construction, not I/O
    c.client_id = "c1"
    c.stub = MagicMock()
    c.stub.SubmitGradientScalars.return_value = pb.SubmitGradientScalarsResponse(
        received=True, surviving_partitions=[1, 2, 3]
    )
    return c


def test_a_masked_submission_carries_no_plaintext_gradients(client):
    """The whole point: the plaintext field must be ABSENT, not merely ignored by the server.

    A request carrying both would leak every scalar to anyone reading the wire, however
    faithfully the server honoured the masked half.
    """
    client.submit_masked_gradient_scalars(
        masked_elements=[7, 8, 9, 10], num_examples=32, round_num=1,
        num_local_steps=2, num_perturbations=2,
    )
    request = client.stub.SubmitGradientScalars.call_args[0][0]
    assert not request.HasField("gradients"), "plaintext gradients rode along with the masked ones"
    assert request.HasField("masked_gradients")
    assert list(request.masked_gradients.elements) == [7, 8, 9, 10]


def test_the_submission_declares_the_field_it_was_masked_under(client):
    """The server rejects a modulus mismatch, so the client must state the field it used rather
    than let the server assume one."""
    client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    )
    masked = client.stub.SubmitGradientScalars.call_args[0][0].masked_gradients
    assert masked.modulus == PRIME
    assert (masked.num_local_steps, masked.num_perturbations) == (1, 2)


def test_it_returns_the_surviving_set_the_next_phase_needs(client):
    """Phase 3b sums over the survivors, so the caller cannot proceed on a bare boolean."""
    survivors = client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    )
    assert survivors == [1, 2, 3]


def test_a_refused_masked_submission_returns_None_rather_than_an_empty_cohort(client):
    """An empty survivor list is a legitimate value (everyone dropped); a refusal is not the same
    thing and must not be mistaken for one."""
    client.stub.SubmitGradientScalars.return_value = pb.SubmitGradientScalarsResponse(
        received=False
    )
    assert client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    ) is None


def test_a_length_that_contradicts_K_times_P_is_caught_before_the_wire(client):
    """The server checks this too, but failing locally names the client's own bug instead of
    surfacing as a rejection from a remote host."""
    with pytest.raises(ValueError, match="4 masked elements"):
        client.submit_masked_gradient_scalars(
            masked_elements=[1, 2, 3, 4], num_examples=8, round_num=1,
            num_local_steps=1, num_perturbations=2,
        )
    client.stub.SubmitGradientScalars.assert_not_called()
