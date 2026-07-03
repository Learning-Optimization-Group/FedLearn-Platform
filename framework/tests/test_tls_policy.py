"""SE-2 — the server TLS policy: fail closed rather than silently serve a deployed profile in plaintext."""
import pytest

from fedlearn.security.tls import (
    REQUIRE_TLS_ENV,
    SERVER_CERT_ENV,
    SERVER_KEY_ENV,
    USE_TLS_ENV,
    TlsPolicyError,
    check_server_tls_policy,
)


def test_plaintext_when_nothing_is_set():
    assert check_server_tls_policy({}) is False


def test_require_tls_but_not_enabled_fails_closed():
    with pytest.raises(TlsPolicyError):
        check_server_tls_policy({REQUIRE_TLS_ENV: "1"})


def test_tls_enabled_with_certs_returns_true():
    assert check_server_tls_policy(
        {USE_TLS_ENV: "1", SERVER_KEY_ENV: "/k.pem", SERVER_CERT_ENV: "/c.pem"}) is True


def test_tls_enabled_missing_cert_raises():
    with pytest.raises(TlsPolicyError):
        check_server_tls_policy({USE_TLS_ENV: "1", SERVER_KEY_ENV: "/k.pem"})  # cert path missing


def test_require_and_enabled_with_certs_is_ok():
    assert check_server_tls_policy(
        {REQUIRE_TLS_ENV: "1", USE_TLS_ENV: "1", SERVER_KEY_ENV: "/k.pem", SERVER_CERT_ENV: "/c.pem"}) is True
