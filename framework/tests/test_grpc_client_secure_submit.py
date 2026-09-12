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
    result = client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    )
    assert result.survivors == [1, 2, 3]


def test_a_refusal_is_distinguishable_from_an_empty_cohort(client):
    """An empty survivor list is a legitimate value (everyone dropped); a refusal is not the same
    thing, so the two are carried on separate fields rather than collapsed into one falsy value."""
    client.stub.SubmitGradientScalars.return_value = pb.SubmitGradientScalarsResponse(
        received=False
    )
    refused = client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    )
    assert refused.accepted is False and refused.frozen is None

    client.stub.SubmitGradientScalars.return_value = pb.SubmitGradientScalarsResponse(
        received=True, surviving_partitions=[], submissions_closed=True
    )
    empty = client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    )
    assert empty.accepted is True and empty.frozen is not None and len(empty.frozen) == 0


def test_a_length_that_contradicts_K_times_P_is_caught_before_the_wire(client):
    """The server checks this too, but failing locally names the client's own bug instead of
    surfacing as a rejection from a remote host."""
    with pytest.raises(ValueError, match="4 masked elements"):
        client.submit_masked_gradient_scalars(
            masked_elements=[1, 2, 3, 4], num_examples=8, round_num=1,
            num_local_steps=1, num_perturbations=2,
        )
    client.stub.SubmitGradientScalars.assert_not_called()


# ---------------------------------------------------------------------------------------------
# The frozen-set marker — a partial survivor view must not be passable to phase 3b
# ---------------------------------------------------------------------------------------------
def test_a_partial_survivor_view_carries_no_frozen_marker(client):
    """While the round is open the view is still growing, so there is nothing safe to hand on."""
    client.stub.SubmitGradientScalars.return_value = pb.SubmitGradientScalarsResponse(
        received=True, surviving_partitions=[1], submissions_closed=False
    )
    result = client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    )
    assert result.accepted
    assert result.survivors == [1]
    assert result.frozen is None, "a growing view was handed out as final"


def test_a_closed_round_yields_a_frozen_marker(client):
    from fedlearn.client.secure_agg_client import FrozenSurvivors
    client.stub.SubmitGradientScalars.return_value = pb.SubmitGradientScalarsResponse(
        received=True, surviving_partitions=[1, 2, 3], submissions_closed=True
    )
    result = client.submit_masked_gradient_scalars(
        masked_elements=[1, 2], num_examples=8, round_num=1,
        num_local_steps=1, num_perturbations=2,
    )
    assert result.frozen == FrozenSurvivors(partitions=(1, 2, 3))


def test_finish_round_will_not_accept_a_bare_list(client):
    """The type is the guard. A plain list is exactly what an impatient caller would pass after
    reading surviving_partitions off its own submission response -- the mistake that decodes to a
    well-formed wrong aggregate."""
    from fedlearn.client.secure_agg_client import SecureAggregationClient
    c = SecureAggregationClient.__new__(SecureAggregationClient)
    c._round = 1
    with pytest.raises(TypeError, match="FrozenSurvivors"):
        c.finish_round(round_num=1, survivors=[1, 2, 3])
