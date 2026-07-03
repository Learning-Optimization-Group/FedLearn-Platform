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

# Every RPC on the FL service requires auth. Matching by method NAME is package-agnostic across the
# v1/v2 proto split and auto-exempts the health-check ("Check") and server reflection.
PROTECTED_METHODS = frozenset({
    "RegisterClient", "GetGlobalModel", "GetGlobalModelStream",
    "SubmitModelUpdate", "SubmitModelUpdateStream", "GetServerStatus",
    "Heartbeat", "GetDeComFLConfig", "SubmitGradientScalars", "ReportClientMetrics",
})


class ConnectionTokenInterceptor(grpc.ServerInterceptor):
    def __init__(self, secret_base64, audience=DEFAULT_AUDIENCE, protected_methods=PROTECTED_METHODS):
        self._secret = secret_base64
        self._audience = audience
        self._protected = protected_methods

    def deny_reason(self, method_name: str, metadata) -> Optional[str]:
        """None if the call is allowed; else a human-readable denial reason. (Testable core.)"""
        if method_name not in self._protected:
            return None
        token = dict(metadata or ()).get(METADATA_KEY)
        if not token:
            return f"missing {METADATA_KEY} metadata"
        try:
            verify_connection_token(token, self._secret, self._audience)
        except TokenVerificationError as e:
            return str(e)
        return None

    def intercept_service(self, continuation, handler_call_details):
        method_name = handler_call_details.method.rsplit("/", 1)[-1]
        reason = self.deny_reason(method_name, handler_call_details.invocation_metadata)
        if reason is None:
            return continuation(handler_call_details)
        # A terminator matching the RPC's streaming type; abort() raises so the RPC is never processed.
        return _terminator(continuation(handler_call_details), reason)


def _terminator(original, reason: str):
    detail = f"connection token rejected: {reason}"

    def abort_unary(request, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, detail)

    def abort_stream(request_iterator, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, detail)

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
    return ConnectionTokenInterceptor(secret)
