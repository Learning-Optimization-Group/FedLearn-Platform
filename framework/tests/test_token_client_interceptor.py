"""SE-1 slice 3 — the client interceptor attaches the connection token to outbound calls."""
import json
import pathlib

import grpc

from fedlearn.security.client_interceptor import (
    CONNECTION_TOKEN_ENV,
    ConnectionTokenClientInterceptor,
    maybe_wrap_channel,
)
from fedlearn.security.interceptor import METADATA_KEY

_GOLDEN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "golden_connection_token.json").read_text())


class _Details:
    """Stand-in for grpc.ClientCallDetails."""
    def __init__(self, metadata=None):
        self.method = "/fedlearn.FederatedLearningService/RegisterClient"
        self.timeout = None
        self.metadata = metadata
        self.credentials = None


def test_augment_appends_the_token_metadata():
    itc = ConnectionTokenClientInterceptor("tok-123")
    augmented = itc._augment(_Details())
    assert (METADATA_KEY, "tok-123") in list(augmented.metadata)


def test_augment_preserves_existing_metadata():
    itc = ConnectionTokenClientInterceptor("tok-123")
    augmented = itc._augment(_Details(metadata=[("x-other", "v")]))
    md = list(augmented.metadata)
    assert ("x-other", "v") in md
    assert (METADATA_KEY, "tok-123") in md


def test_intercept_unary_unary_forwards_with_token():
    itc = ConnectionTokenClientInterceptor("tok-123")
    captured = {}

    def continuation(details, request):
        captured["metadata"] = list(details.metadata or [])
        return "response"

    assert itc.intercept_unary_unary(continuation, _Details(), "req") == "response"
    assert (METADATA_KEY, "tok-123") in captured["metadata"]


def test_maybe_wrap_channel_no_token_returns_channel_unchanged():
    sentinel = object()
    assert maybe_wrap_channel(sentinel, token=None) is sentinel


def test_maybe_wrap_channel_uses_env_token(monkeypatch):
    monkeypatch.setenv(CONNECTION_TOKEN_ENV, _GOLDEN["token"])
    # A real (lazy) channel; wrapping must return an intercepted channel, not the original.
    ch = grpc.insecure_channel("127.0.0.1:1")
    wrapped = maybe_wrap_channel(ch)
    assert wrapped is not ch
    ch.close()
