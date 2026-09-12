"""P2-2 slice 8 — the client half of LightSecAgg over the DeComFL channel.

Slices 1-7 built the primitives, the wire contract and the server's per-round session. This is the
participant: it runs the three phases against the ``fedlearn.v2`` RPCs, and it is the piece that
decides what the server is allowed to learn.

THE THREE PHASES, AND WHY THEY ARE SEPARATE CALLS
    1. ``begin_round``      -- publish this client's X25519 public key, receive the cohort's.
    2. ``distribute_shares`` then ``collect_shares`` -- Shamir-share this client's OWN mask, seal
       each share to its recipient, hand the sealed blobs to the server to relay, then collect the
       ones addressed here.
    3. ``mask`` then ``finish_round`` -- send ``quantize(x) + z`` in place of the plaintext
       scalars, then return ONE summed share over the surviving dealers.

    They cannot be collapsed into one call, and the reason is a rendezvous rather than a style
    choice. Phase 2 must seal against keys that phase 1 collects from OTHER clients; a client's
    inbound shares do not exist until every peer has relayed, so the first client to submit sees
    an empty inbox and must come back for it; and phase 3b's sum is over a surviving set the
    server can only name once every phase-3a submission has landed. Each boundary is a point
    where this client has to wait for the cohort.

TWO DECISIONS THAT ARE SECURITY, NOT STYLE
    **The mask comes from ``os.urandom``, never from ``lightsecagg.client_mask``.** That helper
    derives a mask from ``sha256(round | client_id)``, both of which are public — anyone could
    regenerate this client's mask and strip it from the masked submission, recovering the
    individual contribution. It is a legitimate test fixture (its only callers are tests) and an
    illegitimate client. See ``test_the_mask_is_not_derivable_from_public_values``.

    **Associated data is rebuilt locally on open, never taken from the wire.** The AD binds a
    ciphertext to ``(round, sender, recipient)``. Trusting a relayed copy of it would make the
    binding self-certifying and therefore worthless.

WHAT THIS CLIENT STILL TRUSTS THE SERVER FOR
    The cohort key map. A dishonest server can substitute its own public key for a peer's and
    machine-in-the-middle that pair — the limitation recorded in ``security/key_agreement.py`` and
    ``security/public_key_registry.py``. The threat model is an honest-but-curious server, and
    nothing here upgrades it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.security.key_agreement import (
    decrypt_share,
    derive_shared_key,
    encrypt_share,
    generate_keypair,
)
from fedlearn.security.lightsecagg import PRIME, aggregate_shares, mask_values, shamir_share

log = logging.getLogger(__name__)

_QUANTIZATION_SCALE = 1_000_000


class SecureAggregationNotReady(RuntimeError):
    """Raised when a phase runs before the phase it depends on."""


@dataclass(frozen=True)
class FrozenSurvivors:
    """A surviving set the server has declared FINAL for a round.

    A distinct type rather than a plain list, because the difference is not cosmetic. The
    ``surviving_partitions`` a client reads off its own submission response is a partial view
    that is still growing: the first client to submit is told ``[1]`` while the last is told
    ``[1,2,3]``. If each holder sums the shares for the set IT was handed, they produce shares of
    DIFFERENT mask-sums, and the server's single interpolation decodes to a well-formed wrong
    aggregate that nothing downstream can detect. Measured, before this existed: an expected
    ``[0.0, 1.25]`` came back as ``[-764.6, 61.4]``, with every RPC reporting success.

    Only a response carrying ``submissions_closed`` produces one of these, so the dangerous call
    is not merely discouraged — it does not typecheck.
    """

    partitions: Tuple[int, ...]

    def __iter__(self):
        return iter(self.partitions)

    def __len__(self) -> int:
        return len(self.partitions)


class SecureAggregationClient:
    """One client's participation in one run's secure aggregation.

    Holds the long-lived X25519 keypair and the per-round mask. Constructed with a stub rather
    than a channel so it can be driven against the real servicer in-process — which is how the
    protocol is tested end to end without a network.

    Args:
        stub: a ``FederatedLearningServiceStub`` (or anything exposing the three RPCs).
        client_id: the wire identifier. Carried for logging and server-side correlation only —
            **never** for authorisation, and never as mask entropy. The server binds everything
            on the SE-15 verified partition instead.
    """

    def __init__(self, stub, client_id: str) -> None:
        self._stub = stub
        self.client_id = str(client_id)

        self._private, self._public = generate_keypair()

        self.partition: int | None = None
        self._cohort: Dict[int, bytes] = {}
        self._round: int | None = None
        self._threshold: int | None = None
        self._num_scalars: int | None = None
        self._mask: torch.Tensor | None = None
        # Shares this client HOLDS, opened from the relay: dealer partition -> share vector.
        self._held: Dict[int, torch.Tensor] = {}

    # ---- phase 1 --------------------------------------------------------------------------
    def begin_round(
            self,
            round_num: int,
            threshold: int,
            num_scalars: int,
            cohort_size: int,
            run_id: str = "",
    ) -> Dict[int, bytes]:
        """Publish this client's public key and learn the cohort's.

        Also draws this round's mask. ``cohort_size`` is the expected number of participants; it
        sizes the Shamir sharing in phase 2 and is passed here so a client that is first to
        publish — and therefore sees a cohort of one — still shares to the right degree.

        Returns:
            The cohort's ``partition -> public key`` map as the server knows it *right now*,
            which for an early caller is partial. Phase 2 refreshes it.

        Raises:
            SecureAggregationNotReady: if the server refused the publish, which means this
                partition already published a different key this run.
        """
        if threshold < 1 or threshold > cohort_size:
            raise ValueError(
                f"threshold must be in [1, cohort_size={cohort_size}], got {threshold}"
            )

        response = self._stub.PublishPublicKey(
            fedlearn_pb2.PublishPublicKeyRequest(
                client_id=self.client_id,
                run_id=run_id,
                round=int(round_num),
                public_key=self._public,
            )
        )
        if not response.accepted:
            raise SecureAggregationNotReady(
                f"server refused the public key for round {round_num}: "
                f"{response.rejection_reason or 'no reason given'}"
            )

        self._round = int(round_num)
        self._threshold = int(threshold)
        self._num_scalars = int(num_scalars)
        self._cohort_size = int(cohort_size)
        self._cohort = dict(response.cohort_public_keys)
        self.partition = self._locate_self(self._cohort)

        # Real entropy, not a function of (client_id, round) -- see the module docstring.
        seed = int.from_bytes(os.urandom(8), "big") >> 1
        generator = torch.Generator().manual_seed(seed)
        self._mask = torch.randint(
            0, PRIME, (self._num_scalars,), generator=generator, dtype=torch.int64
        )
        self._held = {}
        return dict(self._cohort)

    def _locate_self(self, cohort: Dict[int, bytes]) -> int:
        """Find this client's own partition by matching its published key.

        The server never *tells* a client its partition, and that is deliberate: a client that
        accepted an assigned identifier could be told it is somebody else, and would then seal
        and index its shares under the wrong slot. Matching on the key it generated locally means
        the answer is self-evident.
        """
        for partition, key in cohort.items():
            if bytes(key) == self._public:
                return int(partition)
        raise SecureAggregationNotReady(
            "the cohort the server returned does not contain this client's own public key; it "
            "cannot determine its partition and must not guess one"
        )

    # ---- phase 2 --------------------------------------------------------------------------
    def distribute_shares(self, round_num: int, run_id: str = "") -> Dict[int, torch.Tensor]:
        """Secret-share this client's mask to the cohort, and open the shares addressed here.

        Re-publishes the (identical) public key first. That is not a redundant call: a client
        that published early saw a partial cohort, and shares must be sealed against every peer's
        real key. The registry is idempotent for an identical key precisely so this refresh is
        safe — a retry is not mistaken for a key-swap attack.

        Returns:
            The shares this client now HOLDS, keyed by dealer partition.
        """
        self._require_round(round_num)

        refreshed = self._stub.PublishPublicKey(
            fedlearn_pb2.PublishPublicKeyRequest(
                client_id=self.client_id,
                run_id=run_id,
                round=int(round_num),
                public_key=self._public,
            )
        )
        self._cohort = dict(refreshed.cohort_public_keys)
        self.partition = self._locate_self(self._cohort)

        indices = self._holder_indices()
        num_shares = max(len(indices), self._cohort_size)
        shares = shamir_share(
            secret=self._mask,
            num_shares=num_shares,
            threshold=self._threshold,
            seed=int.from_bytes(os.urandom(8), "big") >> 1,
        )

        sealed: List[fedlearn_pb2.SealedShare] = []
        for peer, peer_key in self._cohort.items():
            if int(peer) == self.partition:
                # A dealer keeps its own share rather than relaying it to itself; the server
                # rejects a self-addressed share because counting it would distort the holder set.
                self._held[self.partition] = shares[indices[self.partition]]
                continue
            associated = self._associated_data(round_num, self.partition, int(peer))
            key = derive_shared_key(
                self._private, bytes(peer_key), context=self._context(round_num)
            )
            payload = _encode_share(shares[indices[int(peer)]])
            sealed.append(
                fedlearn_pb2.SealedShare(
                    recipient_partition=int(peer),
                    ciphertext=encrypt_share(key, payload, associated),
                    associated_data=associated,
                )
            )

        response = self._stub.SubmitSecureShares(
            fedlearn_pb2.SubmitSecureSharesRequest(
                client_id=self.client_id,
                run_id=run_id,
                round=int(round_num),
                shares=sealed,
            )
        )
        if not response.accepted:
            raise SecureAggregationNotReady(
                f"server refused the sealed shares for round {round_num}"
            )

        self._absorb(round_num, response.inbound_ciphertexts)
        return dict(self._held)

    def collect_shares(self, round_num: int, run_id: str = "") -> Dict[int, torch.Tensor]:
        """Poll for shares that peers relayed AFTER this client submitted its own.

        Relay is a rendezvous: the response to ``distribute_shares`` can only carry shares that
        had already arrived, so whichever client goes first sees an empty inbox. Re-submitting
        with an EMPTY share list is a read of the inbox rather than a second relay -- the server
        stores nothing new and returns the current view.

        A deployment calls this on a backoff until it holds a share from every dealer it will be
        asked to sum over. Returning short is not an error here; ``finish_round`` is where a
        missing dealer becomes one, because that is the point at which it would corrupt the
        aggregate.
        """
        self._require_round(round_num)
        response = self._stub.SubmitSecureShares(
            fedlearn_pb2.SubmitSecureSharesRequest(
                client_id=self.client_id,
                run_id=run_id,
                round=int(round_num),
                shares=[],
            )
        )
        self._absorb(round_num, response.inbound_ciphertexts)
        return dict(self._held)

    def _absorb(self, round_num: int, inbound) -> None:
        """Open every inbound ciphertext this client has not already opened."""
        for sender, ciphertext in inbound.items():
            if int(sender) in self._held:
                continue
            self._held[int(sender)] = self._open(round_num, int(sender), bytes(ciphertext))

    def _open(self, round_num: int, sender: int, ciphertext: bytes) -> torch.Tensor:
        """Decrypt one inbound share, under an AD this client rebuilds itself."""
        peer_key = self._cohort.get(sender)
        if peer_key is None:
            raise SecureAggregationNotReady(
                f"received a share from partition {sender}, which published no public key; it "
                f"cannot be opened and must not be treated as a zero"
            )
        key = derive_shared_key(self._private, bytes(peer_key), context=self._context(round_num))
        associated = self._associated_data(round_num, sender, self.partition)
        return _decode_share(decrypt_share(key, ciphertext, associated))

    # ---- phase 3a -------------------------------------------------------------------------
    def mask(self, scalars: Sequence[float]) -> List[int]:
        """This client's wire payload: ``quantize(x) + z (mod PRIME)``, as field elements.

        Uniformly distributed to anyone without ``threshold`` shares of the mask, the server
        included. Returned as plain ``int`` so it drops straight into the proto's
        ``repeated uint64``.
        """
        if self._mask is None:
            raise SecureAggregationNotReady("mask() called before begin_round()")
        if len(scalars) != self._num_scalars:
            raise ValueError(
                f"expected {self._num_scalars} scalars (K*P), got {len(scalars)}"
            )
        return [int(v) for v in mask_values(scalars, self._mask, scale=_QUANTIZATION_SCALE)]

    # ---- phase 3b -------------------------------------------------------------------------
    def finish_round(
            self, round_num: int, survivors: "FrozenSurvivors", run_id: str = ""
    ) -> int:
        """Return ONE summed share covering the surviving dealers.

        This is the step that makes the scheme *Light*SecAgg: whatever the dropout, a holder
        replies with a single vector and the server decodes once — where classic SecAgg would
        reconstruct each dropped client's mask individually.

        A dealer that dropped is simply absent from ``survivors``, so its share is left out of
        the sum and its mask never enters the recovered total.

        Args:
            survivors: a :class:`FrozenSurvivors`, obtainable only from a response in which the
                server declared submissions closed. A bare list is refused — see that class for
                what accepting one silently produces.

        Returns:
            How many more summed shares the server still needs to reach the threshold.
        """
        if not isinstance(survivors, FrozenSurvivors):
            raise TypeError(
                f"finish_round needs a FrozenSurvivors, got {type(survivors).__name__}. The "
                f"surviving set is only safe to sum over once the server has frozen it; a view "
                f"read off this client's own submission response is still growing, and holders "
                f"acting on different views decode to a well-formed wrong aggregate."
            )
        self._require_round(round_num)

        held = [self._held[p] for p in survivors if p in self._held]
        if not held:
            raise SecureAggregationNotReady(
                f"round {round_num}: this client holds no shares from the {len(survivors)} "
                f"surviving dealers, so it cannot contribute a summed share"
            )
        if len(held) != len(survivors):
            missing = sorted(set(int(p) for p in survivors) - set(self._held))
            log.warning(
                "Round %d: holding shares for %d of %d survivors; missing dealers %s. Summing "
                "the subset would decode to the wrong aggregate.",
                round_num, len(held), len(survivors), missing,
            )
            raise SecureAggregationNotReady(
                f"round {round_num}: missing shares from surviving dealers {missing}; a partial "
                f"sum is a silently wrong aggregate, not a degraded one"
            )

        response = self._stub.SubmitAggregatedShare(
            fedlearn_pb2.SubmitAggregatedShareRequest(
                client_id=self.client_id,
                run_id=run_id,
                round=int(round_num),
                holder_index=self._holder_indices()[self.partition],
                summed_share=[int(v) for v in aggregate_shares(held)],
            )
        )
        return int(response.shares_still_needed)

    # ---- helpers --------------------------------------------------------------------------
    def _holder_indices(self) -> Dict[int, int]:
        """Partition -> Shamir x-coordinate, 1-based.

        Derived from the sorted cohort so every client computes the same mapping without the
        server assigning one. ``x = 0`` is reserved: it evaluates to the secret itself.
        """
        return {int(p): i + 1 for i, p in enumerate(sorted(int(k) for k in self._cohort))}

    def _context(self, round_num: int) -> bytes:
        """KDF context — separates keys per round so one round's compromise stays local."""
        return f"fedlearn|secagg|round={int(round_num)}".encode()

    def _associated_data(self, round_num: int, sender: int, recipient: int) -> bytes:
        """AEAD binding for one directed share. Directional on purpose: ``a -> b`` must not open
        as ``b -> a``, or the server could reflect a share back and corrupt the sum."""
        return f"round={int(round_num)}|from={int(sender)}|to={int(recipient)}".encode()

    def _require_round(self, round_num: int) -> None:
        if self._round is None:
            raise SecureAggregationNotReady("begin_round() has not run for this client")
        if int(round_num) != self._round:
            raise SecureAggregationNotReady(
                f"client is in round {self._round}, asked to act in round {round_num}; masks and "
                f"keys are round-scoped and mixing them would corrupt the aggregate"
            )


def _encode_share(share: torch.Tensor) -> bytes:
    """A share vector as fixed-width big-endian bytes for sealing.

    Fixed width rather than a text or pickle encoding: the length is then implied by the payload
    size, and there is no parser between the AEAD and the field arithmetic.
    """
    return b"".join(int(v).to_bytes(8, "big") for v in share.tolist())


def _decode_share(payload: bytes) -> torch.Tensor:
    if len(payload) % 8 != 0:
        raise ValueError(f"share payload is {len(payload)} bytes, not a multiple of 8")
    return torch.tensor(
        [int.from_bytes(payload[i:i + 8], "big") for i in range(0, len(payload), 8)],
        dtype=torch.int64,
    )
