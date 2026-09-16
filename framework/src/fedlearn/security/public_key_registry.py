"""P2-2 slice 5 — bind published X25519 public keys to the verified client identity (SE-15).

THE HOLE THIS CLOSES
    Slice 4 seals each Shamir share to its recipient under a key derived from that recipient's
    published public key. If a client can publish a key on ANOTHER client's behalf, it becomes
    that client's peer for the round: every share addressed to the victim is sealed to the
    attacker instead, and the attacker reconstructs masks it has no right to. One enrolled
    participant would be able to unmask the cohort.

    SE-15 (``security/identity.py``) already solved the identity half of this for aggregation: the
    backend-minted connection token carries a server-assigned, HMAC-verified ``partitionId``,
    while the wire ``client_id`` is explicitly marked untrusted by the proto itself. So the
    registry is keyed on the PARTITION — the thing a client cannot forge — and refuses a second,
    conflicting key for one partition.

    First write wins, deliberately. A mid-run key swap is not a recoverable event: peers have
    already sealed shares to the original key, and honouring a replacement would break
    reconstruction for the whole cohort rather than just the swapper.

WHAT THIS DOES NOT CLOSE — see ``test_binding_does_not_defend_against_a_malicious_SERVER``
    This is server-side state, so it constrains CLIENTS, not the server. A dishonest server can
    still hand a client a different key than the one registered and machine-in-the-middle the
    exchange; nothing here detects that, because a client's only root of trust IS the server.
    Closing it needs an out-of-band channel or a real PKI, neither of which this platform has.
    The threat model therefore remains honest-but-curious server, and the limitation is asserted
    in the test suite rather than left as prose someone can skim past.
"""
from __future__ import annotations

from typing import Dict, Optional

# X25519 raw public keys are exactly 32 bytes; anything else is malformed rather than merely
# unusual, and is rejected at the door so it cannot fail later inside a derive.
_PUBLIC_KEY_BYTES = 32


class PublicKeyConflict(RuntimeError):
    """Raised when a partition tries to register a key different from the one it already has."""


class PublicKeyRegistry:
    """Per-run map of verified ``partition_id`` -> published X25519 public key.

    Scoped to a run rather than a round: the keypair is long-lived and per-round separation comes
    from the ``context`` mixed into :func:`~fedlearn.security.key_agreement.derive_shared_key`,
    not from rotating keys.
    """

    def __init__(self) -> None:
        self._keys: Dict[int, bytes] = {}

    def register(self, partition_id: int, public_key: bytes) -> None:
        """Publish a public key for a VERIFIED partition.

        The caller must pass the partition extracted from the connection token
        (``identity.partition_from_metadata``), never a client-supplied identifier — the whole
        point is that the key is bound to something the client cannot choose.

        Idempotent for an identical key, so an RPC retry is not mistaken for an attack.

        Raises:
            ValueError: if the key is not a 32-byte X25519 raw public key.
            PublicKeyConflict: if this partition already registered a DIFFERENT key.
        """
        if not isinstance(public_key, (bytes, bytearray)) or len(public_key) != _PUBLIC_KEY_BYTES:
            raise ValueError(
                f"public key must be {_PUBLIC_KEY_BYTES} bytes of raw X25519, got "
                f"{len(public_key) if hasattr(public_key, '__len__') else type(public_key)}"
            )
        key = bytes(public_key)
        existing = self._keys.get(int(partition_id))
        if existing is not None and existing != key:
            raise PublicKeyConflict(
                f"partition {partition_id} already published a different public key; refusing to "
                f"replace it. Peers have already sealed shares to the original, so honouring a "
                f"swap would break reconstruction for the whole cohort."
            )
        self._keys[int(partition_id)] = key

    def get(self, partition_id: int) -> Optional[bytes]:
        """The key registered for a partition, or ``None`` if it has not published one."""
        return self._keys.get(int(partition_id))

    def cohort(self) -> Dict[int, bytes]:
        """A copy of the full ``partition_id -> public key`` map, as broadcast to the cohort.

        Returned as a copy so a caller cannot mutate the registry's state by editing the view it
        was handed.
        """
        return dict(self._keys)

    def __len__(self) -> int:
        return len(self._keys)
