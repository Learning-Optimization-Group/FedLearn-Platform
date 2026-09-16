"""P2-2 slice 4 — pairwise key agreement, the gate on client-side secure aggregation.

WHY THIS EXISTS, AND WHY IT IS A PREREQUISITE RATHER THAN A REFINEMENT
    LightSecAgg requires client ``i`` to deliver a Shamir share to client ``j``. This platform's
    gRPC surface is STAR-shaped: every RPC in ``fedlearn.v2`` runs client -> server, and there is
    no client-to-client channel. Shares therefore have to be relayed by the server.

    Relay them in the clear and the server holds every share, reconstructs every mask, and learns
    every client's contribution — a scheme that provides zero privacy while passing every
    correctness test. So the shares must be encrypted to the recipient under a key the server does
    not have, and establishing that key is what this module does. Nothing client-side can ship
    before it.

CONSTRUCTION — all primitives borrowed from ``cryptography``, none implemented here
    * **X25519** for the Diffie-Hellman. Each client publishes a public key through the server;
      any pair derives a shared secret the server cannot compute from the public keys alone.
    * **HKDF-SHA256** to turn the raw DH output into a uniform 32-byte key. The raw X25519 result
      is not uniformly distributed and must not be used as a key directly.
    * **AES-GCM** for authenticated encryption, so a tampered or misrouted share is REJECTED
      rather than silently decrypting to garbage that would corrupt the reconstruction.

TWO BINDINGS THAT MATTER MORE THAN THEY LOOK
    ``context`` is mixed into the KDF, so the same client pair derives a DIFFERENT key each round.
    Without it one round's compromise would expose every other round.

    ``associated_data`` binds each ciphertext to ``(round, sender, recipient)``. Without it the
    relaying server could redirect ``a -> b``'s share to ``c``, or replay round 1's share into
    round 2. Neither breaks confidentiality, but both corrupt the aggregate — and a silent
    corruption is worse than a refusal.

THREAT MODEL, STATED PLAINLY
    This defends the shares against an honest-but-curious relaying server. It does NOT
    authenticate which client a public key belongs to — a server that substitutes its own public
    key for a client's can machine-in-the-middle the exchange. Binding public keys to the
    platform's existing client identity (``security/identity.py``, SE-15) is the next step and is
    not done here.
"""
from __future__ import annotations

import os
from typing import Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

_KEY_BYTES = 32
_NONCE_BYTES = 12
_HKDF_INFO = b"fedlearn/lightsecagg/pairwise-share-key/v1"


def generate_keypair() -> Tuple[bytes, bytes]:
    """A fresh X25519 keypair as ``(private_bytes, public_bytes)``, 32 bytes each.

    The public half is what a client publishes through the server; the private half never leaves
    the client.
    """
    private = X25519PrivateKey.generate()
    priv_bytes = private.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    pub_bytes = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv_bytes, pub_bytes


def derive_shared_key(private_bytes: bytes, peer_public_bytes: bytes, context: bytes) -> bytes:
    """The 32-byte key shared by exactly this pair, for exactly this ``context``.

    Both members of the pair reach the same value from opposite halves — that is the
    Diffie-Hellman property the relay depends on. ``context`` (round id, session id) is mixed into
    the KDF so keys do not survive across rounds.

    The raw X25519 output is passed through HKDF rather than used directly: it is a curve point,
    not a uniform string, and using it as an AES key would be a real weakness.
    """
    private = X25519PrivateKey.from_private_bytes(private_bytes)
    peer = X25519PublicKey.from_public_bytes(peer_public_bytes)
    raw = private.exchange(peer)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES,
        salt=None,
        info=_HKDF_INFO + b"|" + context,
    ).derive(raw)


def encrypt_share(key: bytes, plaintext: bytes, associated_data: bytes) -> bytes:
    """AES-GCM seal a share for its recipient. Returns ``nonce || ciphertext||tag``.

    ``associated_data`` is authenticated but not encrypted — bind it to
    ``(round, sender, recipient)`` so a relayed share cannot be redirected or replayed.

    A fresh random nonce per call: reusing a nonce under one key is catastrophic for GCM, so it is
    generated here rather than left to the caller.
    """
    if len(key) != _KEY_BYTES:
        raise ValueError(f"key must be {_KEY_BYTES} bytes, got {len(key)}")
    nonce = os.urandom(_NONCE_BYTES)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, associated_data)


def decrypt_share(key: bytes, blob: bytes, associated_data: bytes) -> bytes:
    """Open a sealed share.

    Raises:
        cryptography.exceptions.InvalidTag: if the key is wrong, the ciphertext was tampered with,
            or ``associated_data`` does not match what it was sealed under. All three are refusals
            rather than garbage plaintext, which is the point of using an AEAD here.
        ValueError: if the blob is too short to contain a nonce.
    """
    if len(key) != _KEY_BYTES:
        raise ValueError(f"key must be {_KEY_BYTES} bytes, got {len(key)}")
    if len(blob) <= _NONCE_BYTES:
        raise ValueError("sealed share is too short to contain a nonce")
    nonce, ct = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    return AESGCM(key).decrypt(nonce, ct, associated_data)
