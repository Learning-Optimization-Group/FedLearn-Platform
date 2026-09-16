"""Client-side attachment of the FL-boundary connection token (SE-1 slice 3).

A gRPC client interceptor that adds the ``x-connection-token`` metadata to every outbound call, so the
server-side ``ConnectionTokenInterceptor`` accepts it. The token is minted by the backend at enrollment
and handed to the client out-of-band (ClientConnectionDto over HTTPS); the client process receives it
via the ``FEDLEARN_CONNECTION_TOKEN`` env var.
"""
from __future__ import annotations

import collections
import os
from typing import Optional

import grpc

from fedlearn.security.interceptor import METADATA_KEY

CONNECTION_TOKEN_ENV = "FEDLEARN_CONNECTION_TOKEN"


class _ClientCallDetails(
    collections.namedtuple("_ClientCallDetails", ("method", "timeout", "metadata", "credentials")),
    grpc.ClientCallDetails,
):
    pass


class ConnectionTokenClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    def __init__(self, token: str):
        self._token = token

    def _augment(self, client_call_details):
        """Return call details with the connection-token metadata appended. (Testable core.)"""
        metadata = list(client_call_details.metadata or [])
        metadata.append((METADATA_KEY, self._token))
        return _ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            metadata,
            client_call_details.credentials,
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._augment(client_call_details), request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._augment(client_call_details), request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return continuation(self._augment(client_call_details), request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return continuation(self._augment(client_call_details), request_iterator)


def maybe_wrap_channel(channel: grpc.Channel, token: Optional[str] = None) -> grpc.Channel:
    """Wrap ``channel`` so every call carries the connection token, if one is configured.

    Token precedence: explicit arg > ``FEDLEARN_CONNECTION_TOKEN`` env. No token -> the channel is
    returned unchanged (dev / unauthenticated servers).
    """
    token = token or os.environ.get(CONNECTION_TOKEN_ENV)
    if not token:
        return channel
    return grpc.intercept_channel(channel, ConnectionTokenClientInterceptor(token))
