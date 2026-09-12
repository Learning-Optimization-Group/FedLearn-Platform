"""P2-2 slice 5 — bind published public keys to the verified client identity (SE-15).

WHAT THIS CLOSES
    Slice 4 relays shares encrypted under a pairwise key derived from published public keys. If a
    client can publish a key on ANOTHER client's behalf, it becomes that client's peer for the
    round: shares addressed to the victim are sealed to the attacker instead, and the attacker
    reconstructs masks it has no right to.

    SE-15 already gives the server a verified, server-assigned ``partitionId`` per enrolled client
    (HMAC-verified from the connection token; the wire ``client_id`` is explicitly untrusted). So
    the fix is to key the registry on the PARTITION, not on anything the client asserts, and to
    refuse a second conflicting key for the same partition.

WHAT THIS DOES NOT CLOSE, and the test at the bottom says so out loud
    A malicious SERVER can still substitute its own public key when relaying, because a client has
    no root of trust other than the server. Closing that needs an out-of-band channel or a real
    PKI, and neither exists here. The threat model stays honest-but-curious server.
"""
import pytest

from fedlearn.security.key_agreement import generate_keypair
from fedlearn.security.public_key_registry import (
    PublicKeyConflict,
    PublicKeyRegistry,
)


def test_a_registered_key_can_be_read_back_by_partition():
    reg = PublicKeyRegistry()
    _, pub = generate_keypair()
    reg.register(partition_id=3, public_key=pub)
    assert reg.get(3) == pub


def test_registering_the_same_key_again_is_idempotent():
    """A client retrying an RPC must not be treated as an attacker."""
    reg = PublicKeyRegistry()
    _, pub = generate_keypair()
    reg.register(partition_id=3, public_key=pub)
    reg.register(partition_id=3, public_key=pub)
    assert reg.get(3) == pub


def test_a_conflicting_key_for_the_same_partition_is_refused():
    """First write wins. A mid-run key swap would break reconstruction for everyone."""
    reg = PublicKeyRegistry()
    _, first = generate_keypair()
    _, second = generate_keypair()
    reg.register(partition_id=3, public_key=first)

    with pytest.raises(PublicKeyConflict, match="partition 3"):
        reg.register(partition_id=3, public_key=second)
    assert reg.get(3) == first, "the original key must survive a rejected overwrite"


def test_distinct_partitions_hold_distinct_keys():
    reg = PublicKeyRegistry()
    _, a = generate_keypair()
    _, b = generate_keypair()
    reg.register(partition_id=1, public_key=a)
    reg.register(partition_id=2, public_key=b)
    assert reg.get(1) == a and reg.get(2) == b


def test_an_unregistered_partition_returns_none():
    assert PublicKeyRegistry().get(99) is None


def test_a_malformed_public_key_is_refused():
    reg = PublicKeyRegistry()
    with pytest.raises(ValueError, match="32 bytes"):
        reg.register(partition_id=1, public_key=b"too-short")


def test_the_cohort_view_is_what_gets_broadcast():
    reg = PublicKeyRegistry()
    keys = {}
    for pid in (1, 2, 3):
        _, pub = generate_keypair()
        keys[pid] = pub
        reg.register(partition_id=pid, public_key=pub)
    assert reg.cohort() == keys


def test_binding_does_not_defend_against_a_malicious_SERVER():
    """Documents the limit in executable form so it cannot quietly be forgotten.

    The registry is server-side state. A dishonest server can simply hand a client a different
    key than the one registered -- nothing in this module detects that, and a test asserting
    otherwise would be false comfort.
    """
    reg = PublicKeyRegistry()
    _, honest = generate_keypair()
    _, server_key = generate_keypair()
    reg.register(partition_id=1, public_key=honest)

    # The registry holds the honest key, and yet the server is free to broadcast its own.
    what_the_server_chooses_to_send = server_key
    assert reg.get(1) == honest
    assert what_the_server_chooses_to_send != reg.get(1)
