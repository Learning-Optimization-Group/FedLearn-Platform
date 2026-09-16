"""P2-2 slice 4 — key agreement, the gate on everything client-side.

WHY THIS IS A PREREQUISITE AND NOT A POLISH STEP
    LightSecAgg needs client i to hand a Shamir share to client j. The platform's gRPC service is
    STAR-shaped -- every RPC is client->server, there is no client-to-client channel -- so shares
    must be relayed through the server. If they are relayed in the clear the server reconstructs
    every mask and the scheme provides exactly zero privacy while appearing to work.

    So shares must be encrypted to the recipient, which needs a key only the two clients hold,
    which is what this module establishes.

PRIMITIVES ARE BORROWED, NOT BUILT
    X25519 for the Diffie-Hellman, HKDF to turn the raw DH output into a key, AES-GCM for
    authenticated encryption -- all from `cryptography`, already a pinned dependency. Nothing here
    implements a cipher or a curve.
"""
import pytest
from cryptography.exceptions import InvalidTag

from fedlearn.security.key_agreement import (
    decrypt_share,
    derive_shared_key,
    encrypt_share,
    generate_keypair,
)


def test_two_clients_derive_the_same_shared_key():
    """The Diffie-Hellman property everything else rests on."""
    a_priv, a_pub = generate_keypair()
    b_priv, b_pub = generate_keypair()

    a_view = derive_shared_key(a_priv, b_pub, context=b"round-1")
    b_view = derive_shared_key(b_priv, a_pub, context=b"round-1")

    assert a_view == b_view
    assert len(a_view) == 32


def test_a_third_party_derives_a_different_key():
    a_priv, a_pub = generate_keypair()
    b_priv, b_pub = generate_keypair()
    e_priv, e_pub = generate_keypair()

    ab = derive_shared_key(a_priv, b_pub, context=b"round-1")
    eavesdropper = derive_shared_key(e_priv, a_pub, context=b"round-1")
    assert ab != eavesdropper


def test_the_same_pair_derives_different_keys_in_different_rounds():
    """Context separation: a key reused across rounds would let one round's break expose another."""
    a_priv, a_pub = generate_keypair()
    b_priv, b_pub = generate_keypair()
    assert (derive_shared_key(a_priv, b_pub, context=b"round-1")
            != derive_shared_key(a_priv, b_pub, context=b"round-2"))


def test_a_share_encrypted_to_a_peer_round_trips():
    a_priv, a_pub = generate_keypair()
    b_priv, b_pub = generate_keypair()
    key_a = derive_shared_key(a_priv, b_pub, context=b"r1")
    key_b = derive_shared_key(b_priv, a_pub, context=b"r1")

    blob = encrypt_share(key_a, b"\x01\x02\x03share", associated_data=b"r1|a->b")
    assert decrypt_share(key_b, blob, associated_data=b"r1|a->b") == b"\x01\x02\x03share"


def test_the_relaying_server_cannot_read_a_share():
    """The server sees both public keys and the ciphertext, and must still learn nothing."""
    a_priv, a_pub = generate_keypair()
    b_priv, b_pub = generate_keypair()
    server_priv, _ = generate_keypair()

    blob = encrypt_share(
        derive_shared_key(a_priv, b_pub, context=b"r1"), b"secret-share", associated_data=b"r1|a->b"
    )
    # Best the server can do with what it holds: a key from its own private and a client public.
    server_key = derive_shared_key(server_priv, a_pub, context=b"r1")
    with pytest.raises(InvalidTag):
        decrypt_share(server_key, blob, associated_data=b"r1|a->b")


def test_a_tampered_ciphertext_is_rejected_rather_than_silently_decrypted():
    a_priv, a_pub = generate_keypair()
    b_priv, b_pub = generate_keypair()
    key = derive_shared_key(a_priv, b_pub, context=b"r1")

    blob = bytearray(encrypt_share(key, b"share", associated_data=b"r1|a->b"))
    blob[-1] ^= 0x01
    with pytest.raises(InvalidTag):
        decrypt_share(derive_shared_key(b_priv, a_pub, context=b"r1"),
                      bytes(blob), associated_data=b"r1|a->b")


def test_a_share_replayed_to_the_wrong_recipient_is_rejected():
    """Associated data binds the ciphertext to (round, sender, recipient).

    Without it a relaying server could redirect a->b's share to c, or replay round 1's into
    round 2, both of which corrupt the reconstruction.
    """
    a_priv, a_pub = generate_keypair()
    b_priv, b_pub = generate_keypair()
    key_a = derive_shared_key(a_priv, b_pub, context=b"r1")
    key_b = derive_shared_key(b_priv, a_pub, context=b"r1")

    blob = encrypt_share(key_a, b"share", associated_data=b"r1|a->b")
    with pytest.raises(InvalidTag):
        decrypt_share(key_b, blob, associated_data=b"r1|a->c")


# --------------------------------------------------------------------------------------------------
# Capstone: the full relay, end to end, with the server holding everything it would really hold
# --------------------------------------------------------------------------------------------------
def test_shares_relayed_through_the_server_reconstruct_while_the_server_learns_nothing():
    """The scenario the whole slice exists for.

    Clients publish public keys through the server, seal their Shamir shares to each peer, the
    server relays ciphertext it cannot read, recipients open their shares, and the aggregate of
    masks reconstructs. The server's view -- public keys plus ciphertext -- is asserted to be
    insufficient.
    """
    import torch
    from fedlearn.security.lightsecagg import (
        PRIME, aggregate_shares, client_mask, shamir_reconstruct, shamir_share,
    )

    cohort = ["a", "b", "c", "d"]
    n, t, length, rnd = len(cohort), 3, 4, 1

    keys = {c: generate_keypair() for c in cohort}
    published = {c: keys[c][1] for c in cohort}          # the server sees only these

    masks = {c: client_mask(c, round_seed=rnd, length=length) for c in cohort}
    shares = {c: shamir_share(masks[c], num_shares=n, threshold=t, seed=i)
              for i, c in enumerate(cohort)}

    # Each client seals share j to the client holding index j, and hands the ciphertext to the
    # server. The server stores and forwards; it never holds a key that opens any of them.
    relayed = {}
    for sender in cohort:
        for idx, recipient in enumerate(cohort, start=1):
            key = derive_shared_key(keys[sender][0], published[recipient],
                                    context=f"round-{rnd}".encode())
            payload = shares[sender][idx].numpy().astype("int64").tobytes()
            relayed[(sender, recipient)] = encrypt_share(
                key, payload, associated_data=f"{rnd}|{sender}->{recipient}".encode()
            )

    # Recipients open what was addressed to them and sum -- the one-shot step.
    summed = {}
    for idx, recipient in enumerate(cohort, start=1):
        opened = []
        for sender in cohort:
            key = derive_shared_key(keys[recipient][0], published[sender],
                                    context=f"round-{rnd}".encode())
            raw = decrypt_share(key, relayed[(sender, recipient)],
                                associated_data=f"{rnd}|{sender}->{recipient}".encode())
            opened.append(torch.frombuffer(bytearray(raw), dtype=torch.int64))
        summed[idx] = aggregate_shares(opened)

    recovered = shamir_reconstruct({i: summed[i] for i in range(1, t + 1)}, threshold=t)
    expected = torch.zeros(length, dtype=torch.int64)
    for c in cohort:
        expected = (expected + masks[c]) % PRIME
    assert torch.equal(recovered, expected), "relayed shares did not reconstruct the mask sum"

    # And the server, with every public key and every ciphertext, cannot open one.
    server_priv, _ = generate_keypair()
    with pytest.raises(InvalidTag):
        decrypt_share(
            derive_shared_key(server_priv, published["a"], context=f"round-{rnd}".encode()),
            relayed[("a", "b")], associated_data=f"{rnd}|a->b".encode(),
        )
