"""FL-boundary TLS policy (SE-2).

The TLS *mechanism* (mTLS-capable server credentials / client channel credentials) already lives in
server.py and grpc_client.py. This module enforces the *policy*: a deployed profile must not silently
fall back to plaintext. Enforcement is opt-in via ``FEDLEARN_REQUIRE_TLS=1`` — the backend sets it for
deployed spawns once server certs are provisioned — so dev/test stay plaintext and nothing breaks
before certs exist.
"""
from __future__ import annotations

import os

REQUIRE_TLS_ENV = "FEDLEARN_REQUIRE_TLS"
USE_TLS_ENV = "FEDLEARN_GRPC_USE_TLS"
SERVER_KEY_ENV = "FEDLEARN_GRPC_SERVER_KEY"
SERVER_CERT_ENV = "FEDLEARN_GRPC_SERVER_CERT"


class TlsPolicyError(RuntimeError):
    """Raised when a required-TLS deployment would run insecurely, or a TLS server is misconfigured."""


def check_server_tls_policy(env=None) -> bool:
    """Return True if the server should serve TLS, False for plaintext — or raise TlsPolicyError.

    Fail-closed: if TLS is REQUIRED (FEDLEARN_REQUIRE_TLS=1, set by the backend on deployed profiles)
    but not enabled, refuse rather than serve plaintext; if TLS is enabled, the server key + cert must
    be present.
    """
    env = os.environ if env is None else env
    require = env.get(REQUIRE_TLS_ENV) == "1"
    use = env.get(USE_TLS_ENV) == "1"
    if require and not use:
        raise TlsPolicyError(
            f"{REQUIRE_TLS_ENV}=1 but {USE_TLS_ENV} is not enabled — refusing to serve the FL boundary "
            f"in plaintext on a deployed profile. Provision server certs and set {USE_TLS_ENV}=1.")
    if use:
        missing = [n for n in (SERVER_KEY_ENV, SERVER_CERT_ENV) if not env.get(n)]
        if missing:
            raise TlsPolicyError(
                f"{USE_TLS_ENV}=1 but {', '.join(missing)} not set; cannot start a TLS server.")
    return use
