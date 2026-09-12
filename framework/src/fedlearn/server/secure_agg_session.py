"""P2-2 — per-round server state for secure aggregation over the DeComFL channel.

The three secure-aggregation RPCs are stateful across a round: the keys published in phase 1 are
what phase 2 seals against, and the surviving set established in phase 3a decides which dealers
phase 3b must sum over. That state lives here rather than in the gRPC servicer, so it can be
tested without a channel and so the handlers stay thin adapters.

EVERYTHING SECURITY-RELEVANT IS KEYED ON THE PARTITION
    Not on the wire ``client_id``, which the proto itself marks as untrusted for authz. The
    partition comes from the SE-15 verified connection token, so a client cannot publish a key,
    relay a share, or submit a masked value as somebody else.

WHAT THE SERVER LEARNS, AND WHAT IT DOES NOT
    It holds every masked submission, every sealed share (as ciphertext it has no key for), and
    the summed shares. From those it recovers exactly one thing: the SUM of the surviving clients'
    scalars. No individual contribution is recoverable, because the individual masks are only ever
    reconstructible in aggregate — that is the LightSecAgg construction.

    This assumes an honest-but-curious server. A server that substitutes its own public key during
    phase 1 can machine-in-the-middle the exchange; see
    ``security/public_key_registry.py`` for why that is out of scope here.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import torch

from fedlearn.security.lightsecagg import PRIME, recover_aggregate
from fedlearn.security.public_key_registry import PublicKeyRegistry

log = logging.getLogger(__name__)


class SecureAggregationSession:
    """State for one secure-aggregation round.

    Args:
        round_index: the server round this session belongs to.
        threshold: Shamir reconstruction threshold — how many summed shares are needed.
        num_scalars: ``K * P``; every masked submission and summed share must have this length,
            checked on arrival so a malformed client fails loudly instead of corrupting a decode.
    """

    def __init__(self, round_index: int, threshold: int, num_scalars: int) -> None:
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")
        if num_scalars < 1:
            raise ValueError(f"num_scalars must be >= 1, got {num_scalars}")

        self.round_index = int(round_index)
        self.threshold = int(threshold)
        self.num_scalars = int(num_scalars)

        self._registry = PublicKeyRegistry()
        # (sender_partition, recipient_partition) -> sealed ciphertext the server cannot open.
        self._relayed: Dict[int, Dict[int, bytes]] = {}
        self._masked: Dict[int, torch.Tensor] = {}
        self._summed: Dict[int, torch.Tensor] = {}
        self._closed = False

    # ---- phase 1: key publication ----------------------------------------------------------
    def publish_key(self, partition: int, public_key: bytes) -> Dict[int, bytes]:
        """Record this partition's X25519 key and return the cohort's published keys.

        Raises:
            PublicKeyConflict: if the partition already published a different key. First write
                wins — peers may already have sealed shares to the original.
        """
        self._registry.register(partition_id=partition, public_key=public_key)
        return self._registry.cohort()

    def cohort_keys(self) -> Dict[int, bytes]:
        """The published keys, without publishing one. Used when a publish is REFUSED: the caller
        still needs the cohort view, and reaching into the registry from outside would couple the
        servicer to this class's internals."""
        return self._registry.cohort()

    # ---- phase 2: share relay ---------------------------------------------------------------
    def relay_shares(self, sender: int, shares: Dict[int, bytes]) -> None:
        """Accept sealed shares from ``sender``, addressed by recipient partition.

        The server stores ciphertext it has no key for. It is a post box, not a participant.

        Raises:
            ValueError: if a share is addressed to the sender itself — meaningless, and it would
                skew the holder count the threshold is measured against.
        """
        if sender in shares:
            raise ValueError(
                f"partition {sender} addressed a share to itself; a dealer does not relay to "
                f"itself and counting it would distort the holder set"
            )
        for recipient, ciphertext in shares.items():
            self._relayed.setdefault(int(recipient), {})[int(sender)] = bytes(ciphertext)

    def inbound_for(self, partition: int) -> Dict[int, bytes]:
        """Sealed shares addressed TO ``partition``, keyed by sending partition."""
        return dict(self._relayed.get(int(partition), {}))

    # ---- phase 3a: masked submissions --------------------------------------------------------
    def submit_masked(self, partition: int, elements: Sequence[int]) -> List[int]:
        """Record one client's masked scalars; return the surviving set so far, sorted.

        Raises:
            ValueError: if the submission is not exactly ``num_scalars`` long. A short or long
                vector would misalign the decode rather than fail, so it is rejected here.
        """
        if self._closed:
            raise ValueError(
                f"round {self.round_index}: submissions are closed; partition {partition} is too "
                f"late. Holders have already summed over the frozen set, so admitting a dealer "
                f"now would leave its mask in the total with no share to cancel it."
            )
        if len(elements) != self.num_scalars:
            raise ValueError(
                f"masked submission from partition {partition} has {len(elements)} elements, "
                f"expected num_scalars = {self.num_scalars}"
            )
        self._masked[int(partition)] = torch.tensor(list(elements), dtype=torch.int64) % PRIME
        return sorted(self._masked)

    @property
    def survivors(self) -> List[int]:
        """Partitions whose masked submission the server accepted this round.

        Only AUTHORITATIVE once :attr:`is_closed`. Before then it is a partial view that is still
        growing, and two holders acting on the views they were handed at different moments would
        sum over different dealer sets — producing shares of different mask-sums, which decode to
        a well-formed wrong aggregate. That is why phase 3b waits for the freeze.
        """
        return sorted(self._masked)

    @property
    def is_closed(self) -> bool:
        """True once the surviving set is frozen and safe for holders to sum over."""
        return self._closed

    def close_submissions(self) -> List[int]:
        """Freeze the surviving set. Idempotent.

        Driven by whichever comes first: the last expected client submitting, or a deadline the
        deployment enforces. Both must be able to fire, so calling this twice is not an error.

        Returns:
            The frozen surviving set.
        """
        if not self._closed:
            self._closed = True
            log.info(
                "Secure aggregation round %d: submissions closed with %d survivor(s): %s",
                self.round_index, len(self._masked), self.survivors,
            )
        return self.survivors

    # ---- phase 3b: summed shares -------------------------------------------------------------
    def submit_summed_share(self, holder_index: int, share: Sequence[int]) -> int:
        """Record one holder's summed share; return how many more are needed for the threshold."""
        if len(share) != self.num_scalars:
            raise ValueError(
                f"summed share from holder {holder_index} has {len(share)} elements, "
                f"expected num_scalars = {self.num_scalars}"
            )
        self._summed[int(holder_index)] = torch.tensor(list(share), dtype=torch.int64) % PRIME
        return max(0, self.threshold - len(self._summed))

    # ---- recovery ----------------------------------------------------------------------------
    def ready(self) -> bool:
        """True once enough summed shares are in to decode."""
        return len(self._summed) >= self.threshold and bool(self._masked)

    def recovery_inputs(self):
        """``(masked_values, summed_shares)`` in the shape the strategy's secure path expects.

        Returned as copies of the stored tensors' references in a fresh list/dict so a caller
        cannot mutate the session's state, and so the strategy never reaches into it.

        Raises:
            ValueError: if the round is not yet recoverable — the same preconditions as
                :meth:`recover`, checked here so a caller cannot build a half-valid input set.
        """
        self._assert_recoverable()
        return [self._masked[p] for p in self.survivors], dict(self._summed)

    def _assert_recoverable(self) -> None:
        if not self._closed:
            raise ValueError(
                f"round {self.round_index}: submissions have not been closed, so the surviving "
                f"set is not yet frozen; holders may have summed over different sets and the "
                f"decode would be silently wrong"
            )
        if not self._masked:
            raise ValueError(f"round {self.round_index}: no masked submissions to aggregate")
        if len(self._summed) < self.threshold:
            raise ValueError(
                f"round {self.round_index}: have {len(self._summed)} summed shares, "
                f"threshold is {self.threshold}"
            )

    def recover(self) -> torch.Tensor:
        """The plaintext sum over the surviving clients, as float32.

        One Lagrange interpolation regardless of how many clients dropped — the LightSecAgg
        property. The server never holds an individual client's mask or contribution.

        Raises:
            ValueError: if fewer than ``threshold`` summed shares have arrived, or no masked
                submissions have.
        """
        self._assert_recoverable()
        log.info(
            "Secure aggregation round %d: recovering over %d survivors from %d summed shares",
            self.round_index, len(self._masked), len(self._summed),
        )
        return recover_aggregate(
            masked_values=[self._masked[p] for p in self.survivors],
            summed_shares=self._summed,
            threshold=self.threshold,
        )
