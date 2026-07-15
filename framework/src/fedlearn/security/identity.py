"""SE-15: bind the FL connection token to a single client identity.

The backend-minted connection token carries a server-assigned, unforgeable ``partitionId`` — one per
run+user enrollment (``RunEnrollment``). The wire ``client_id`` on the FL RPCs is, by contrast, a
self-chosen handle the proto itself marks "NOT trusted for authz". Without binding the two, a single
valid token can be replayed with many different ``client_id`` values, so one enrolled participant can
impersonate the whole cohort and dominate FedAvg/DeComFL aggregation (each fake ``client_id`` becomes
its own averaged update).

This module extracts the verified ``partitionId`` from an RPC's metadata so the servicer can pin one
token (one partition) to one ``client_id`` (see ``FLCoordinator.bind_or_check_identity``). The
``x-connection-token`` interceptor (``interceptor.py``) already proves the token and binds it to the
server's run; it just discards the claims, so we re-read + re-verify the token here where the request's
``client_id`` and the RPC context are both in hand. Verification is a cheap HMAC check.

Enforcement is gated on the same ``FEDLEARN_REQUIRE_CLIENT_AUTH`` switch as the interceptor: when
client-auth is off (local/dev fail-open) there is no token and the extractor is absent, so binding is
skipped and behaviour is unchanged.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from fedlearn.security.interceptor import (
    ENABLE_ENV,
    METADATA_KEY,
    SECRET_ENV,
    SECRET_FALLBACK_ENV,
)
from fedlearn.security.token_verify import (
    DEFAULT_AUDIENCE,
    TokenVerificationError,
    verify_connection_token,
)

# A callable that, given a gRPC ServicerContext, returns the verified token partition (or None).
PartitionExtractor = Callable[[object], Optional[int]]


def partition_from_metadata(metadata, secret_base64: str, audience: str = DEFAULT_AUDIENCE) -> Optional[int]:
    """The verified ``partitionId`` claim from an RPC's ``x-connection-token``, or ``None``.

    Returns ``None`` when there is no token, the token fails verification, or it carries no numeric
    ``partitionId``. It never raises: the ``ConnectionTokenInterceptor`` is the auth gate that rejects
    a missing/invalid token before the servicer runs; this is only the *identity* read, so an absent
    identity simply means "don't bind" rather than an error.
    """
    token = dict(metadata or ()).get(METADATA_KEY)
    if not token:
        return None
    try:
        claims = verify_connection_token(token, secret_base64, audience)
    except TokenVerificationError:
        return None
    partition = claims.get("partitionId")
    if partition is None:
        return None
    try:
        return int(partition)
    except (TypeError, ValueError):
        return None


def partition_extractor_from_env(env=None) -> Optional[PartitionExtractor]:
    """A ``(context) -> Optional[int]`` yielding the verified token partition, or ``None`` when
    client-auth enforcement is off — mirroring ``interceptor_from_env``'s gate so the two are wired
    together (auth on => both the interceptor and identity binding; auth off => neither).

    Returns ``None`` (binding disabled) when ``FEDLEARN_REQUIRE_CLIENT_AUTH != 1`` or no secret is
    configured. The interceptor already fails the server closed on the enforce-on-but-no-secret
    misconfiguration, so we do not re-raise here.
    """
    env = os.environ if env is None else env
    if env.get(ENABLE_ENV) != "1":
        return None
    secret = env.get(SECRET_ENV) or env.get(SECRET_FALLBACK_ENV)
    if not secret:
        return None
    audience = DEFAULT_AUDIENCE

    def _extract(context) -> Optional[int]:
        return partition_from_metadata(context.invocation_metadata(), secret, audience)

    return _extract
