"""SE-1 Java->Python cross-language pin.

golden_connection_token.json holds a REAL token minted by the Java backend (JJWT) — see
backend .../run/GoldenConnectionTokenFixtureTest. This proves the Python PyJWT verifier accepts a
genuine backend token: the base64 secret decoding, the HMAC alg JJWT picked by key length, and the
JJWT array-form `aud` all interoperate. If the Java token format ever drifts, this fails.
"""
import json
import pathlib

from fedlearn.security.token_verify import TokenVerificationError, verify_connection_token

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "golden_connection_token.json"


def _load():
    return json.loads(_FIXTURE.read_text())


def test_python_verifies_a_real_java_minted_token():
    g = _load()
    claims = verify_connection_token(g["token"], g["secret_base64"])
    for key, expected in g["claims"].items():
        assert claims[key] == expected, f"claim {key}: {claims.get(key)!r} != {expected!r}"
    # aud is verified by the verifier (audience=), and JJWT emits it as an array — confirm interop.
    assert "fedlearn-fl-server" in claims["aud"]


def test_wrong_secret_rejects_the_java_token():
    # A different key must fail — proves the fixture's acceptance is a real signature check, not a pass-through.
    import base64
    other = base64.b64encode(b"a-different-32-byte-secret-key!!!").decode()
    g = _load()
    try:
        verify_connection_token(g["token"], other)
        raise AssertionError("expected verification to fail with the wrong secret")
    except TokenVerificationError:
        pass
