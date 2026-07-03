"""gRPC server-side enforcement of the FL-boundary connection token (SE-1 slice 2).

A ``grpc.ServerInterceptor`` that requires a valid ``x-connection-token`` on every FL-service RPC,
aborting ``UNAUTHENTICATED`` otherwise. Enforcement is opt-in via ``FEDLEARN_REQUIRE_CLIENT_AUTH=1`` so
local/dev runs (which don't set it) fail OPEN; deployed profiles set it and the backend provisions the
signing secret. Misconfiguration (enforce-on but no secret) fails CLOSED — the server refuses to start.
"""
from __future__ import annotations

import os
from typing import Optional

import grpc

from fedlearn.security.token_verify import (
    DEFAULT_AUDIENCE,
    TokenVerificationError,
    verify_connection_token,
)

METADATA_KEY = "x-connection-token"
ENABLE_ENV = "FEDLEARN_REQUIRE_CLIENT_AUTH"
SECRET_ENV = "FEDLEARN_FL_TOKEN_SECRET"       # dedicated FL secret (SE-7, slice 4)
SECRET_FALLBACK_ENV = "APP_JWT_SECRET"        # until slice 4 provisions the dedicated one
RUN_ID_ENV = "FEDLEARN_RUN_ID"                # FR-7: the run this server serves; token.runId must match

# Every RPC on the FL service requires auth. Matching by method NAME is package-agnostic across the
# v1/v2 proto split and auto-exempts the health-check ("Check") and server reflection.
PROTECTED_METHODS = frozenset({
    "RegisterClient", "GetGlobalModel", "GetGlobalModelStream",
    "SubmitModelUpdate", "SubmitModelUpdateStream", "GetServerStatus",
    "Heartbeat", "GetDeComFLConfig", "SubmitGradientScalars", "ReportClientMetrics",
})


class ConnectionTokenInterceptor(grpc.ServerInterceptor):
    def __init__(self, secret_base64, audience=DEFAULT_AUDIENCE, protected_methods=PROTECTED_METHODS,
                 expected_run_id=None):
        self._secret = secret_base64
        self._audience = audience
        self._protected = protected_methods
        self._expected_run_id = expected_run_id   # FR-7: bind the server to its run (None = auth-only)

    def deny_reason(self, method_name: str, metadata):
        """None if the call is allowed; else ``(grpc.StatusCode, reason)``. (Testable core.)

        UNAUTHENTICATED for a missing/invalid token (identity not proven); PERMISSION_DENIED for a
        VALID token issued for a different run (FR-7 — identity proven, but not for this federation).
        """
        if method_name not in self._protected:
            return None
        token = dict(metadata or ()).get(METADATA_KEY)
        if not token:
            return grpc.StatusCode.UNAUTHENTICATED, f"missing {METADATA_KEY} metadata"
        try:
            claims = verify_connection_token(token, self._secret, self._audience)
        except TokenVerificationError as e:
            return grpc.StatusCode.UNAUTHENTICATED, str(e)
        if self._expected_run_id is not None and claims.get("runId") != self._expected_run_id:
            return (grpc.StatusCode.PERMISSION_DENIED,
                    f"token run {claims.get('runId')!r} != server run {self._expected_run_id!r}")
        return None

    def intercept_service(self, continuation, handler_call_details):
        method_name = handler_call_details.method.rsplit("/", 1)[-1]
        denial = self.deny_reason(method_name, handler_call_details.invocation_metadata)
        if denial is None:
            return continuation(handler_call_details)
        code, reason = denial
        # A terminator matching the RPC's streaming type; abort() raises so the RPC is never processed.
        return _terminator(continuation(handler_call_details), code, reason)


def _terminator(original, code, reason: str):
    detail = f"connection token rejected: {reason}"

    def abort_unary(request, context):
        context.abort(code, detail)

    def abort_stream(request_iterator, context):
        context.abort(code, detail)

    if original is None:
        return None
    rd, rs = original.request_deserializer, original.response_serializer
    if original.unary_unary is not None:
        return grpc.unary_unary_rpc_method_handler(abort_unary, request_deserializer=rd, response_serializer=rs)
    if original.unary_stream is not None:
        return grpc.unary_stream_rpc_method_handler(abort_stream, request_deserializer=rd, response_serializer=rs)
    if original.stream_unary is not None:
        return grpc.stream_unary_rpc_method_handler(abort_stream, request_deserializer=rd, response_serializer=rs)
    return grpc.stream_stream_rpc_method_handler(abort_stream, request_deserializer=rd, response_serializer=rs)


def interceptor_from_env(env=None) -> Optional[ConnectionTokenInterceptor]:
    """Build the auth interceptor from env, or None when enforcement is off (dev fail-open).

    Fails CLOSED on misconfiguration: enforce-on with no secret raises rather than silently disabling
    auth.
    """
    env = os.environ if env is None else env
    if env.get(ENABLE_ENV) != "1":
        return None
    secret = env.get(SECRET_ENV) or env.get(SECRET_FALLBACK_ENV)
    if not secret:
        raise RuntimeError(
            f"{ENABLE_ENV}=1 but neither {SECRET_ENV} nor {SECRET_FALLBACK_ENV} is set; refusing to "
            "start the FL server without the connection-token signing secret")
    # FR-7: bind to the server's run so a token minted for another run is rejected. Absent -> auth-only.
    return ConnectionTokenInterceptor(secret, expected_run_id=env.get(RUN_ID_ENV) or None)
