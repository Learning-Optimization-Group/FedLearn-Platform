"""P2-2 — additive secure aggregation for the DeComFL scalar channel.

WHY THIS LIVES OVER THE ZEROTH-ORDER CHANNEL, which is the whole argument for building it here:

Secure aggregation is avoided in production because masking cost scales with the payload —
SecAgg and SecAgg+ mask ``O(d)`` values, one per model coordinate, so the overhead grows with the
model. Over the DeComFL channel there is no such vector. A client sends ``K * P`` scalar loss
differences per round (80 bytes at K=1, P=10) **regardless of model dimension**, so the masking
cost is ``O(K*P)`` and constant in ``d``.

The composition is only possible because of a property of the existing aggregation, not a change
to it: ``DeComFL.aggregate_fit`` reads exactly ``sum(grad_scalars[k][p] for ... in clients)`` and
``len(clients)``. It never touches an individual client's scalars. Additive masks that cancel over
the cohort therefore leave the aggregate *bit-identical* while the server learns nothing about any
single contribution.

SCOPE OF THIS MODULE (slice 1 of P2-2)
    Implemented: pairwise additive masking with exact cancellation in a finite field, and the
    fixed-point quantisation that makes exactness possible.

    NOT implemented, and deliberately so — these are what make a scheme *LightSecAgg* rather than
    plain pairwise masking, and each needs its own slice:
      * dropout resilience (one-shot reconstruction of the aggregate mask of the SURVIVING users)
      * secret sharing of the mask seeds, so a dropped client's mask can be removed at all
      * a key-agreement protocol — this module assumes pair seeds are already shared
    Until those land this defends against an honest-but-curious server ONLY when every client in
    the cohort completes the round. A single dropout corrupts the aggregate rather than degrading
    gracefully, which is exactly the failure LightSecAgg exists to fix.

WHY A FINITE FIELD AND NOT FLOATS
    Float masks cancel only approximately: ``(g + m) + (g' - m)`` loses the low bits of ``g``
    whenever ``m`` is much larger, and a mask that is not much larger hides little. Quantising to
    fixed-point integers and masking mod ``q`` makes cancellation exact, which is what lets the
    tests assert the masked aggregate equals the plain one rather than merely approximating it.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import torch

# 2**32 is comfortably above the quantised magnitudes a ZO scalar reaches and keeps every
# intermediate inside int64 without overflow when summed across a realistic cohort.
DEFAULT_MODULUS = 2 ** 32

# Fixed-point scale. ZO gradient scalars are O(1e-3)..O(1), so 1e6 keeps ~6 decimal places —
# well inside the noise floor of a stochastic gradient estimate.
DEFAULT_SCALE = 1_000_000


def _pair_seed(round_seed: int, a: str, b: str) -> int:
    """Deterministic 63-bit seed for an unordered client pair.

    Derived by hashing rather than arithmetic so that neither client can predict another pair's
    stream from its own, and sorted so both members of the pair derive the SAME value.
    """
    lo, hi = sorted((str(a), str(b)))
    digest = hashlib.sha256(f"{int(round_seed)}|{lo}|{hi}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def pairwise_mask(
        client_id: str,
        cohort: Sequence[str],
        round_seed: int,
        length: int,
        modulus: int = DEFAULT_MODULUS,
) -> torch.Tensor:
    """This client's additive mask, as ``int64`` of shape ``[length]``.

    For every other client in the cohort a shared pseudo-random vector is derived from the pair
    seed; the lexicographically smaller member ADDS it and the larger SUBTRACTS it. Summed over
    the whole cohort each pair contributes ``+s`` and ``-s``, so the masks cancel to exactly zero
    mod ``modulus`` — which is what :func:`sum_masked` relies on.

    The mask is a function of ``(round_seed, cohort, client_id)`` only, so the same round
    reproduces it and a different round does not. Reusing a mask across rounds would leak the
    difference of two rounds' inputs, which is why ``round_seed`` is required rather than optional.

    Raises:
        ValueError: if ``client_id`` is not in ``cohort``, the cohort has duplicates, or
            ``length`` is not positive.
    """
    peers = [str(c) for c in cohort]
    if len(set(peers)) != len(peers):
        raise ValueError(f"cohort contains duplicate client ids: {peers}")
    me = str(client_id)
    if me not in peers:
        raise ValueError(f"client {me!r} is not in the cohort {peers}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    mask = torch.zeros(length, dtype=torch.int64)
    for other in peers:
        if other == me:
            continue
        gen = torch.Generator().manual_seed(_pair_seed(round_seed, me, other))
        shared = torch.randint(0, modulus, (length,), generator=gen, dtype=torch.int64)
        # Antisymmetric by construction: the pair agrees on `shared`, and the ordering decides
        # who adds and who subtracts, so the two contributions annihilate in the sum.
        mask = mask + shared if me < other else mask - shared
    return mask % modulus


def quantize(
        values: Sequence[float],
        scale: int = DEFAULT_SCALE,
        modulus: int = DEFAULT_MODULUS,
) -> torch.Tensor:
    """Fixed-point encode floats into the field: ``round(x * scale) mod modulus``.

    Negatives wrap to the top half of the field, which :func:`dequantize` undoes. Working in
    integers is what makes mask cancellation exact rather than approximate.
    """
    q = torch.round(torch.as_tensor(values, dtype=torch.float64) * scale).to(torch.int64)
    return q % modulus


def dequantize(
        encoded: torch.Tensor,
        scale: int = DEFAULT_SCALE,
        modulus: int = DEFAULT_MODULUS,
) -> torch.Tensor:
    """Inverse of :func:`quantize`. Field elements above ``modulus/2`` decode as negative.

    Returns **float32**, matching the dtype DeComFL aggregates in, so the value flows back into
    ``aggregate_fit`` without a silent cast. The division is done in float64 first because the
    field element can exceed float32's exact-integer range (2**24) before scaling, even though
    the scaled result is small.
    """
    signed = torch.where(encoded >= modulus // 2, encoded - modulus, encoded)
    return (signed.to(torch.float64) / scale).to(torch.float32)


def mask_contribution(
        values: Sequence[float],
        client_id: str,
        cohort: Sequence[str],
        round_seed: int,
        scale: int = DEFAULT_SCALE,
        modulus: int = DEFAULT_MODULUS,
) -> torch.Tensor:
    """One client's masked contribution: ``quantize(values) + pairwise_mask`` mod ``modulus``.

    This is what the client puts on the wire in place of its raw scalars. It is indistinguishable
    from uniform noise to anyone who does not hold the pair seeds — including the server.
    """
    encoded = quantize(values, scale=scale, modulus=modulus)
    mask = pairwise_mask(
        client_id, cohort, round_seed=round_seed, length=encoded.numel(), modulus=modulus
    )
    return (encoded + mask) % modulus


def unmask_sum(
        contributions: Sequence[torch.Tensor],
        modulus: int = DEFAULT_MODULUS,
) -> torch.Tensor:
    """Sum masked contributions; the masks cancel, leaving the quantised plaintext sum.

    There is no key material here and nothing to "unmask" — summing IS the unmasking, which is
    why the server learns the aggregate and nothing else.

    .. warning::
        Correct only when EVERY client of the cohort is present. A missing contribution leaves its
        peers' halves of the shared pairs uncancelled and the result is meaningless rather than
        approximate. Dropout resilience is slice 2 (LightSecAgg's one-shot reconstruction).

    Raises:
        ValueError: if ``contributions`` is empty or the lengths disagree.
    """
    if not contributions:
        raise ValueError("unmask_sum requires at least one contribution.")
    lengths = {int(c.numel()) for c in contributions}
    if len(lengths) != 1:
        raise ValueError(f"masked contributions have differing lengths: {sorted(lengths)}")

    total = torch.zeros(contributions[0].numel(), dtype=torch.int64)
    for c in contributions:
        total = (total + c.to(torch.int64)) % modulus
    return total


# Field elements are carried as int64 on the wire; one mask element costs this many bytes.
_MASK_ELEMENT_BYTES = 8

_CHANNELS = ("fedavg", "decomfl")


def masking_cost(
        channel: str,
        model_dim: int,
        K: int = 1,
        P: int = 10,
) -> dict:
    """Masking cost for one client-round, by channel — the P2-2 headline comparison.

    Secure aggregation must mask every value a client sends, so its cost is set by the PAYLOAD
    SHAPE, not by the model:

      * ``fedavg``  — the payload is the model delta, ``d`` values, so masking is ``O(d)``. This
        is why SecAgg/SecAgg+ overhead is the reason production deployments hesitate.
      * ``decomfl`` — the payload is ``K * P`` scalar loss differences **whatever ``d`` is**, so
        masking is ``O(K*P)`` and **constant in the model dimension**.

    The ratio is the contribution: composing secure aggregation with a zeroth-order channel makes
    the masking overhead independent of model size, and the advantage grows linearly with ``d``.

    Returns:
        dict with ``mask_elements``, ``mask_bytes``, ``payload_description`` and — for the ZO
        channel — ``advantage_vs_gradient_channel``, the factor fewer elements it masks.

    Raises:
        ValueError: on an unknown channel or non-positive sizes.
    """
    ch = str(channel).lower()
    if ch not in _CHANNELS:
        raise ValueError(f"channel must be one of {_CHANNELS}, got {channel!r}")
    if model_dim <= 0:
        raise ValueError(f"model_dim must be positive, got {model_dim}")
    if K <= 0 or P <= 0:
        raise ValueError(f"K and P must be positive, got K={K}, P={P}")

    if ch == "fedavg":
        elements = int(model_dim)
        description = f"model delta, {model_dim} values"
        advantage = None
    else:
        elements = int(K) * int(P)
        description = f"{K}*{P} = {elements} ZO scalars, independent of d={model_dim}"
        advantage = model_dim / elements

    out = {
        "channel": ch,
        "model_dim": int(model_dim),
        "mask_elements": elements,
        "mask_bytes": elements * _MASK_ELEMENT_BYTES,
        "payload_description": description,
    }
    if advantage is not None:
        out["advantage_vs_gradient_channel"] = advantage
    return out
