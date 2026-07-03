"""Verify the FL-boundary connection token (SE-1).

The Spring backend mints a short-lived HMAC-JWT per enrollment (``ConnectionTokenService``, JJWT) and
hands it to the client over HTTPS. The client presents it to this Python FL server on every gRPC call;
this module is the verification gate.

Design notes (see docs/superpowers/specs/2026-07-03-fl-boundary-auth-design.md):
  * PyJWT — a vetted library, not hand-rolled JWS: an auth gate must not get ``alg=none`` / alg-confusion
    wrong. The ``algorithms`` allowlist rejects ``none`` and every asymmetric alg.
  * HMAC family, not a single alg: the Java signer uses ``.signWith(key)`` with no explicit algorithm,
    so JJWT infers HS256/384/512 from the decoded key's bit length. The verifier must accept the whole
    HMAC family and pick by the token header — hardcoding HS256 would reject a >32-byte secret's tokens.
  * The signing key is the base64-DECODED secret (Java ``Decoders.BASE64.decode`` == standard base64).
"""
from __future__ import annotations

import base64

import jwt  # PyJWT

DEFAULT_AUDIENCE = "fedlearn-fl-server"
# HMAC family only. Excluding "none" and all RS*/ES*/PS* is the alg-confusion / none-alg defense.
_ALLOWED_ALGS = ["HS256", "HS384", "HS512"]


class TokenVerificationError(Exception):
    """Raised when a connection token fails verification (bad signature, alg, audience, or expiry)."""


def verify_connection_token(
    token: str,
    secret_base64: str,
    expected_audience: str = DEFAULT_AUDIENCE,
    leeway_seconds: int = 30,
) -> dict:
    """Verify a backend-minted connection token; return its claims or raise TokenVerificationError.

    Enforces: HMAC family signature over the base64-decoded secret, ``aud == expected_audience``, and a
    present, unexpired ``exp`` (with clock-skew leeway). The ``algorithms`` allowlist rejects ``none``
    and every asymmetric algorithm, closing the alg-confusion / none-alg attacks.
    """
    try:
        # validate=True so a non-standard-alphabet secret errors loudly instead of silently decoding
        # to a DIFFERENT key than Java's Decoders.BASE64 (which would reject every genuine token).
        key = base64.b64decode(secret_base64, validate=True)
    except (ValueError, TypeError) as e:
        raise TokenVerificationError(f"malformed base64 secret: {e}") from e
    # Never operate the gate under a degenerate key: an empty/short HMAC key is trivially forgeable.
    # (Java's Keys.hmacShaKeyFor already refuses <32 bytes; enforce it here too, independently.)
    if len(key) < 32:
        raise TokenVerificationError("connection-token secret too short (<32 bytes after base64 decode)")

    try:
        return jwt.decode(
            token,
            key,
            algorithms=_ALLOWED_ALGS,
            audience=expected_audience,
            leeway=leeway_seconds,
            options={"require": ["exp", "aud"]},
        )
    except jwt.PyJWTError as e:
        # Normalise every PyJWT failure (bad signature, disallowed alg, wrong/missing aud, expired,
        # malformed) to one exception the gRPC interceptor maps to UNAUTHENTICATED.
        raise TokenVerificationError(str(e) or e.__class__.__name__) from e
