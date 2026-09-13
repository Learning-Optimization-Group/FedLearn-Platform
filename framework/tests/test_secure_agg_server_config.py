"""P2-2 — switching secure aggregation on from outside Python.

Everything built so far can only be enabled by constructing the servicer by hand. A deployment
starts the server through fl_server.py -> start_server(ServerConfig), so the flag has to travel
that path or the feature is unreachable.
"""
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.server import ServerConfig, build_servicer


def _coord():
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=1, num_perturbations=2,
    )
    c = FLCoordinator(strategy, min_clients_for_aggregation=2, clients_per_round=3)
    c.bind_or_check_identity = MagicMock(return_value=True)
    return c


def test_secure_aggregation_is_off_unless_asked_for():
    """A privacy mechanism that turns itself on changes the wire format and would break every
    existing client, so the default has to be off."""
    assert ServerConfig(num_rounds=1).secure_aggregation is False
    assert build_servicer(_coord(), ServerConfig(num_rounds=1)).secure_aggregation is False


def test_the_flag_and_threshold_reach_the_servicer():
    servicer = build_servicer(
        _coord(),
        ServerConfig(num_rounds=1, secure_aggregation=True, secure_agg_threshold=3),
    )
    assert servicer.secure_aggregation is True
    assert servicer.secure_agg_threshold == 3


def test_a_threshold_below_two_is_refused():
    """A threshold of 1 admits a round with a single survivor, and a one-client "aggregate" IS
    that client's own contribution in plaintext -- the protocol would run, every check would
    pass, and the deployment would have exactly no privacy.

    Refused at construction rather than at the first round, so a misconfiguration fails before
    any client has sent anything.
    """
    with pytest.raises(ValueError, match="threshold"):
        build_servicer(
            _coord(), ServerConfig(num_rounds=1, secure_aggregation=True,
                                   secure_agg_threshold=1),
        )


def test_a_low_threshold_is_ignored_when_secure_aggregation_is_off():
    """A plaintext deployment has no reason to care what the unused threshold says."""
    assert build_servicer(
        _coord(), ServerConfig(num_rounds=1, secure_agg_threshold=1)
    ).secure_aggregation is False


def test_the_threshold_cannot_exceed_the_cohort():
    """Shamir needs `threshold` holders to return summed shares, and holders are survivors of a
    cohort of clients_per_round. A threshold above that can never be met, so every round would
    freeze and then fail its deadline."""
    with pytest.raises(ValueError, match="cohort"):
        build_servicer(
            _coord(),   # clients_per_round=3
            ServerConfig(num_rounds=1, secure_aggregation=True, secure_agg_threshold=4),
        )


# ---------------------------------------------------------------------------------------------
# Auditability: the log has to distinguish masked from plaintext
# ---------------------------------------------------------------------------------------------
def test_the_log_does_not_describe_a_masked_submission_as_plaintext_scalars(caplog):
    """A live secure run logged 'Receiving gradient scalars from c1' ten times while accepting
    only masked submissions -- the line is emitted before the masked branch.

    For a privacy mechanism that is not cosmetic: an operator auditing the logs of a secure
    deployment would read that as plaintext scalars crossing the wire and conclude the mechanism
    was not engaged, or worse, that it had failed open. The log has to say which path ran.
    """
    import logging as _logging
    from unittest.mock import MagicMock

    import grpc
    from fedlearn.communication.generated import fedlearn_pb2 as pb

    coord = _coord()
    servicer = build_servicer(
        coord, ServerConfig(num_rounds=1, secure_aggregation=True, secure_agg_threshold=2),
    )
    servicer._partition_extractor = lambda ctx: 1

    ctx = MagicMock()
    ctx.invocation_metadata.return_value = ()
    with caplog.at_level(_logging.INFO):
        servicer.SubmitGradientScalars(
            pb.SubmitGradientScalarsRequest(
                client_id="c1", trained_on_round=coord.current_round, num_examples=10,
                masked_gradients=pb.MaskedGradientScalars(
                    elements=[1, 2], modulus=2 ** 31 - 1,
                    num_local_steps=1, num_perturbations=2),
            ),
            ctx,
        )

    text = caplog.text
    assert "Masked gradients accepted" in text, "the masked path left no audit trail"
    assert "Receiving gradient scalars" not in text, (
        "a masked submission was logged as plaintext gradient scalars"
    )
