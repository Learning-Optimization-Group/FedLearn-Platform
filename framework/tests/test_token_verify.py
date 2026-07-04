"""SE-1 Slice 1 — the FL-boundary connection-token verifier (PyJWT).

A vetted-library verifier that accepts the HMAC family (JJWT infers HS256/384/512 by key length),
binds to the audience, enforces expiry, and rejects the alg-confusion / none-alg / bad-signature
attacks. The Java->Python golden-fixture cross-language pin lives in test_token_verify_golden.py.
"""
import base64
import json
import time

import jwt as pyjwt  # to craft test tokens
import pytest

from fedlearn.security.token_verify import (
    DEFAULT_AUDIENCE,
    TokenVerificationError,
    verify_connection_token,
)

_SECRET_BYTES = b"fedlearn-test-connection-token-secret-key!"  # 41 bytes -> HS256-capable
_SECRET_B64 = base64.b64encode(_SECRET_BYTES).decode()


def _mint(alg: str = "HS256", key: bytes = _SECRET_BYTES, **overrides) -> str:
    payload = {"aud": DEFAULT_AUDIENCE, "exp": int(time.time()) + 300}
    payload.update(overrides)
    return pyjwt.encode(payload, key, algorithm=alg)


def _unsigned(header: dict, payload: dict) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=")
    return (seg(header) + b"." + seg(payload) + b".").decode()


def test_valid_hs256_token_verifies_and_returns_claims():
    token = _mint(runId="run-1", partitionId=3, sub="42", clientKind="SHARD")
    claims = verify_connection_token(token, _SECRET_B64)
    assert claims["runId"] == "run-1"
    assert claims["partitionId"] == 3
    assert claims["sub"] == "42"


def test_hs512_token_verifies():
    # D3: JJWT picks the HMAC variant by key length, so the verifier must accept the whole family.
    token = _mint(alg="HS512", runId="run-1")
    assert verify_connection_token(token, _SECRET_B64)["runId"] == "run-1"


def test_alg_none_is_rejected():
    token = _unsigned({"alg": "none", "typ": "JWT"},
                      {"aud": DEFAULT_AUDIENCE, "exp": int(time.time()) + 300})
    with pytest.raises(TokenVerificationError):
        verify_connection_token(token, _SECRET_B64)


def test_asymmetric_alg_is_rejected():
    # An RS256 header must be refused on the allowlist before any signature work (alg-confusion).
    token = _unsigned({"alg": "RS256", "typ": "JWT"},
                      {"aud": DEFAULT_AUDIENCE, "exp": int(time.time()) + 300}) + "ZmFrZXNpZw"
    with pytest.raises(TokenVerificationError):
        verify_connection_token(token, _SECRET_B64)


def test_wrong_signature_is_rejected():
    token = _mint(key=b"a-totally-different-signing-key-of-len-41!", runId="run-1")
    with pytest.raises(TokenVerificationError):
        verify_connection_token(token, _SECRET_B64)


def test_expired_token_is_rejected():
    token = _mint(exp=int(time.time()) - 100, runId="run-1")
    with pytest.raises(TokenVerificationError):
        verify_connection_token(token, _SECRET_B64)


def test_wrong_audience_is_rejected():
    token = _mint(aud="some-other-service", runId="run-1")
    with pytest.raises(TokenVerificationError):
        verify_connection_token(token, _SECRET_B64)


def test_degenerate_or_dirty_secret_fails_closed():
    # Defense-in-depth (review findings 1+2): a secret that decodes to an empty/short HMAC key, or a
    # non-standard-base64 secret, must be REJECTED — never operate the gate under a forgeable key.
    # The token is minted with a valid key (PyJWT >= 2.13 refuses to sign with an empty key); what's
    # under test is the VERIFY-side secret being degenerate — the verifier rejects on the secret
    # precheck, before any signature work, so the token's own signing key is irrelevant here.
    forged = _mint(runId="x")
    with pytest.raises(TokenVerificationError):
        verify_connection_token(forged, "")          # empty secret -> empty key -> reject on length
    with pytest.raises(TokenVerificationError):
        verify_connection_token(forged, "!!!!")      # non-standard base64 -> reject on validate=True
    short = base64.b64encode(b"too-short-16byte").decode()   # 16 bytes < 32
    with pytest.raises(TokenVerificationError):
        verify_connection_token(_mint(runId="r"), short)


def test_missing_exp_is_rejected():
    # A token with no expiry must not be accepted as never-expiring.
    token = pyjwt.encode({"aud": DEFAULT_AUDIENCE, "runId": "r"}, _SECRET_BYTES, algorithm="HS256")
    with pytest.raises(TokenVerificationError):
        verify_connection_token(token, _SECRET_B64)
