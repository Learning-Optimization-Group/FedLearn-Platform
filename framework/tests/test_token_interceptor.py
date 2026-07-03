"""SE-1 slice 2 — the gRPC connection-token interceptor's allow/deny core + env factory.

deny_reason() is the testable core (no gRPC server needed); the end-to-end gating over a real socket
is in test_token_interceptor_e2e.py.
"""
import json
import pathlib

import pytest

from fedlearn.security.interceptor import (
    ENABLE_ENV,
    METADATA_KEY,
    SECRET_ENV,
    ConnectionTokenInterceptor,
    interceptor_from_env,
)

_GOLDEN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "golden_connection_token.json").read_text())


def _interceptor():
    return ConnectionTokenInterceptor(_GOLDEN["secret_base64"])


def _md(token=None):
    return [(METADATA_KEY, token)] if token is not None else []


def test_unprotected_method_is_allowed_without_a_token():
    # Health-check / reflection / unknown methods are not gated.
    assert _interceptor().deny_reason("Check", _md()) is None


def test_protected_method_without_token_is_denied():
    assert _interceptor().deny_reason("RegisterClient", _md()) is not None


def test_protected_method_with_valid_token_is_allowed():
    assert _interceptor().deny_reason("SubmitGradientScalars", _md(_GOLDEN["token"])) is None


def test_protected_method_with_garbage_token_is_denied():
    assert _interceptor().deny_reason("SubmitModelUpdate", _md("not-a-jwt")) is not None


def test_protected_method_with_wrong_secret_is_denied():
    bad = ConnectionTokenInterceptor("d3Jvbmctc2VjcmV0LTMyLWJ5dGVzLWxvbmchISEhIQ==")  # different 32B key
    assert bad.deny_reason("RegisterClient", _md(_GOLDEN["token"])) is not None


# --- env factory: dev fail-open, deployed fail-closed-on-misconfig ---------------------------------
def test_from_env_disabled_returns_none():
    assert interceptor_from_env({}) is None
    assert interceptor_from_env({ENABLE_ENV: "0"}) is None


def test_from_env_enabled_with_secret_builds_interceptor():
    itc = interceptor_from_env({ENABLE_ENV: "1", SECRET_ENV: _GOLDEN["secret_base64"]})
    assert isinstance(itc, ConnectionTokenInterceptor)


def test_from_env_enabled_without_secret_fails_closed():
    with pytest.raises(RuntimeError):
        interceptor_from_env({ENABLE_ENV: "1"})
