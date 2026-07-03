"""SE-1 slice 2 — the gRPC connection-token interceptor's allow/deny core + env factory.

deny_reason() is the testable core (no gRPC server needed); the end-to-end gating over a real socket
is in test_token_interceptor_e2e.py.
"""
import json
import pathlib

import grpc
import pytest

from fedlearn.security.interceptor import (
    ENABLE_ENV,
    METADATA_KEY,
    RUN_ID_ENV,
    SECRET_ENV,
    ConnectionTokenInterceptor,
    interceptor_from_env,
)

_GOLDEN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "golden_connection_token.json").read_text())
_RUN_ID = _GOLDEN["claims"]["runId"]


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


# --- FR-7 run isolation: a valid token for a DIFFERENT run must be refused ---------------------------
def test_matching_run_id_is_allowed():
    itc = ConnectionTokenInterceptor(_GOLDEN["secret_base64"], expected_run_id=_RUN_ID)
    assert itc.deny_reason("SubmitGradientScalars", _md(_GOLDEN["token"])) is None


def test_wrong_run_id_is_permission_denied():
    itc = ConnectionTokenInterceptor(_GOLDEN["secret_base64"],
                                     expected_run_id="99999999-9999-9999-9999-999999999999")
    result = itc.deny_reason("SubmitGradientScalars", _md(_GOLDEN["token"]))
    assert result is not None
    assert result[0] == grpc.StatusCode.PERMISSION_DENIED   # valid token, wrong run -> authz failure


def test_no_expected_run_id_skips_the_run_check():
    # dev / older spawn without FEDLEARN_RUN_ID -> authenticate only, no run binding.
    itc = ConnectionTokenInterceptor(_GOLDEN["secret_base64"], expected_run_id=None)
    assert itc.deny_reason("RegisterClient", _md(_GOLDEN["token"])) is None


def test_missing_token_denial_carries_unauthenticated_code():
    result = _interceptor().deny_reason("RegisterClient", _md())
    assert result[0] == grpc.StatusCode.UNAUTHENTICATED


def test_from_env_enabled_binds_run_id_when_present():
    itc = interceptor_from_env({ENABLE_ENV: "1", SECRET_ENV: _GOLDEN["secret_base64"], RUN_ID_ENV: _RUN_ID})
    assert itc.deny_reason("RegisterClient", _md(_GOLDEN["token"])) is None
    # a token for another run is rejected once the server is bound to its run
    other = interceptor_from_env({ENABLE_ENV: "1", SECRET_ENV: _GOLDEN["secret_base64"],
                                  RUN_ID_ENV: "99999999-9999-9999-9999-999999999999"})
    assert other.deny_reason("RegisterClient", _md(_GOLDEN["token"]))[0] == grpc.StatusCode.PERMISSION_DENIED


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
