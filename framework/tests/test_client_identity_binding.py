"""SE-15: bind one connection-token partition to one wire client_id (anti-Sybil).

Covers the three layers of the fix:
  * ``FLCoordinator.bind_or_check_identity`` — the 1:1 partition<->client_id bijection.
  * ``fedlearn.security.identity`` — extracting the verified partition from an RPC's token metadata.
  * ``FederatedLearningServiceServicer._enforce_client_identity`` — the servicer gate that aborts
    PERMISSION_DENIED when a token is replayed under a second client_id.
"""
import base64
import time

import grpc
import jwt as pyjwt
import pytest
from unittest.mock import MagicMock

from fedlearn.security.identity import partition_from_metadata, partition_extractor_from_env
from fedlearn.security.interceptor import ENABLE_ENV, METADATA_KEY, SECRET_ENV
from fedlearn.security.token_verify import DEFAULT_AUDIENCE
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.server.strategy import Strategy

_SECRET = b"fedlearn-test-connection-token-secret-key!"  # 41 bytes -> HS256-capable
_SECRET_B64 = base64.b64encode(_SECRET).decode()


def _coord():
    return FLCoordinator(MagicMock(spec=Strategy), 1, 1)


def _mint(**overrides):
    payload = {"aud": DEFAULT_AUDIENCE, "exp": int(time.time()) + 300}
    payload.update(overrides)
    return pyjwt.encode(payload, _SECRET, algorithm="HS256")


def _md(token=None):
    return [(METADATA_KEY, token)] if token is not None else []


class _Abort(Exception):
    pass


class _Ctx:
    """Fake gRPC ServicerContext: invocation_metadata() + an abort() that raises like the real one."""

    def __init__(self, md=None):
        self._md = md or []
        self.aborted = None

    def invocation_metadata(self):
        return self._md

    def abort(self, code, details):
        self.aborted = (code, details)
        raise _Abort(details)


# ---- coordinator: 1:1 partition <-> client_id bijection ----

def test_first_use_binds_and_repeats_ok():
    c = _coord()
    assert c.bind_or_check_identity(5, "clientA") is True   # first use pins 5 <-> clientA
    assert c.bind_or_check_identity(5, "clientA") is True   # idempotent (re-register / heartbeat / resubmit)


def test_same_token_second_client_id_is_rejected():
    c = _coord()
    assert c.bind_or_check_identity(5, "clientA") is True
    assert c.bind_or_check_identity(5, "clientB") is False  # THE Sybil: one token replayed as a 2nd client


def test_client_id_cannot_be_claimed_by_a_second_partition():
    c = _coord()
    assert c.bind_or_check_identity(5, "shared") is True
    assert c.bind_or_check_identity(6, "shared") is False   # a different token stealing an in-use client_id


def test_distinct_partitions_and_client_ids_coexist():
    c = _coord()
    assert c.bind_or_check_identity(5, "a") is True
    assert c.bind_or_check_identity(6, "b") is True


# ---- identity extractor (token -> partition) ----

def test_partition_from_valid_token():
    assert partition_from_metadata(_md(_mint(partitionId=7, runId="r1")), _SECRET_B64) == 7


def test_partition_none_without_or_invalid_token():
    assert partition_from_metadata(_md(), _SECRET_B64) is None                       # no token
    assert partition_from_metadata(_md("garbage.not.jwt"), _SECRET_B64) is None      # unverifiable
    assert partition_from_metadata(_md(_mint(runId="r1")), _SECRET_B64) is None      # no partitionId claim
    other = base64.b64encode(b"a-different-secret-of-sufficient-length!!").decode()
    assert partition_from_metadata(_md(_mint(partitionId=7)), other) is None          # wrong signing secret


def test_extractor_gated_on_require_client_auth():
    assert partition_extractor_from_env({}) is None                        # auth off (default) -> disabled
    assert partition_extractor_from_env({ENABLE_ENV: "0"}) is None
    ext = partition_extractor_from_env({ENABLE_ENV: "1", SECRET_ENV: _SECRET_B64})
    assert ext is not None
    assert ext(_Ctx(_md(_mint(partitionId=9)))) == 9
    assert ext(_Ctx(_md())) is None                                        # no token on the call


# ---- servicer: the enforcement gate ----

def test_servicer_binds_first_client_then_rejects_a_second_on_same_token():
    servicer = FederatedLearningServiceServicer(_coord(), partition_extractor=lambda ctx: 5)
    servicer._enforce_client_identity("a", _Ctx())        # first client for partition 5 -> binds, no abort
    ctx = _Ctx()
    with pytest.raises(_Abort):                            # same token (partition 5), a different client_id
        servicer._enforce_client_identity("b", ctx)
    assert ctx.aborted[0] == grpc.StatusCode.PERMISSION_DENIED
    servicer._enforce_client_identity("a", _Ctx())        # the originally-bound client_id still passes


def test_servicer_no_binding_when_extractor_disabled():
    servicer = FederatedLearningServiceServicer(_coord(), partition_extractor=None)
    servicer._enforce_client_identity("anything", _Ctx())        # auth off: any client_id passes (dev fail-open)
    servicer._enforce_client_identity("anything-else", _Ctx())


def test_servicer_no_binding_when_call_has_no_token():
    servicer = FederatedLearningServiceServicer(_coord(), partition_extractor=lambda ctx: None)
    servicer._enforce_client_identity("a", _Ctx())        # extractor yields None -> no enforcement
    servicer._enforce_client_identity("b", _Ctx())
