"""P2-2 slice 2 — LightSecAgg: dropout-resilient secure aggregation for the DeComFL channel.

WHAT SLICE 1 LEFT BROKEN
    ``secure_aggregation.pairwise_mask`` cancels only over the FULL cohort. One dropout leaves the
    survivors' halves of the shared pairs uncancelled and the aggregate is meaningless rather than
    approximate — a real failure mode, since a federation of phones drops clients routinely.

THE DIFFERENCE BETWEEN CLASSIC SecAgg AND *LIGHT*SecAgg, which is the reason to build this one
    Classic SecAgg repairs a dropout by reconstructing THE DROPPED CLIENT'S mask from shares held
    by survivors — one reconstruction per dropped client, so the server's recovery cost grows with
    the number that left, exactly when the system is already under stress.

    LightSecAgg (So, Güler, Avestimehr, 2021, https://arxiv.org/abs/2109.14236) turns it around.
    Each client secret-shares its OWN mask. The survivors each sum the shares they are holding,
    and the server performs ONE decode over those summed shares, which yields the aggregate of the
    surviving masks directly. Recovery is a single interpolation no matter how many dropped.

    The whole thing rests on Shamir sharing being LINEAR: a sum of shares of different secrets is
    a valid share of the sum of those secrets. Nothing else in this module matters if that fails,
    which is why it has a test of its own.

PROTOCOL AS IMPLEMENTED HERE
    1. Each client ``i`` draws a random mask ``z_i`` and Shamir-shares it to all ``n`` clients.
    2. Client ``i`` sends the server ``y_i = quantize(x_i) + z_i  (mod PRIME)``.
    3. The server names the surviving set ``S``. Each surviving holder ``j`` returns
       ``s_j = sum_{i in S} share(z_i -> j)`` — one vector, whatever ``|S|`` is.
    4. The server interpolates ``t`` of those to recover ``sum_{i in S} z_i`` in ONE decode, and
       returns ``sum_{i in S} y_i - sum_{i in S} z_i = sum_{i in S} quantize(x_i)``.

    Step 3 is the one-shot property: a holder sends a single summed vector, not one per dealer.

FIELD
    Shamir needs a prime, so this path uses ``2**31 - 1`` rather than slice 1's power-of-two
    modulus. Quantised ZO scalars are far inside it — at scale 1e6 the field holds magnitudes up
    to ~1073, against scalars that are O(1e-3)..O(1).

STILL NOT IMPLEMENTED (slice 3)
    Key agreement. This module assumes shares reach their holders over an authenticated channel,
    and it does not defend against a malicious client submitting inconsistent shares — only
    against an honest-but-curious server, under a dropout.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Sequence

import torch

# 2**31 - 1 is prime (a Mersenne prime), which Shamir requires and slice 1's 2**32 is not.
PRIME = 2 ** 31 - 1


def _inv(a: int, p: int = PRIME) -> int:
    """Modular inverse via Fermat's little theorem (valid because ``p`` is prime)."""
    return pow(int(a) % p, p - 2, p)


def shamir_share(
        secret: torch.Tensor,
        num_shares: int,
        threshold: int,
        seed: int,
) -> Dict[int, torch.Tensor]:
    """Split ``secret`` (an int64 vector) into ``num_shares`` Shamir shares over ``PRIME``.

    Builds a degree ``threshold - 1`` polynomial per coordinate whose constant term is the secret,
    and evaluates it at ``x = 1..num_shares``. Any ``threshold`` shares reconstruct it; fewer
    reveal nothing.

    Share indices start at 1 because ``x = 0`` evaluates to the secret itself.

    Returns:
        ``{share_index: share_vector}`` for ``share_index`` in ``1..num_shares``.

    Raises:
        ValueError: if ``threshold`` is outside ``[1, num_shares]``.
    """
    if not (1 <= threshold <= num_shares):
        raise ValueError(
            f"threshold must be in [1, num_shares={num_shares}], got threshold={threshold}"
        )

    secret = secret.to(torch.int64) % PRIME
    length = secret.numel()

    gen = torch.Generator().manual_seed(int(seed))
    # coeffs[0] is the secret; the rest are uniform in the field.
    coeffs = [secret]
    for _ in range(threshold - 1):
        coeffs.append(torch.randint(0, PRIME, (length,), generator=gen, dtype=torch.int64))

    shares: Dict[int, torch.Tensor] = {}
    for x in range(1, num_shares + 1):
        # Horner from the top coefficient down keeps every intermediate inside the field.
        acc = torch.zeros(length, dtype=torch.int64)
        for c in reversed(coeffs):
            acc = (acc * x + c) % PRIME
        shares[x] = acc
    return shares


def shamir_reconstruct(shares: Dict[int, torch.Tensor], threshold: int) -> torch.Tensor:
    """Recover the secret from ``threshold`` or more shares by Lagrange interpolation at ``x=0``.

    Because interpolation is linear, feeding this SUMMED shares recovers the SUM of the underlying
    secrets — which is the property the whole LightSecAgg recovery path is built on.

    Raises:
        ValueError: if fewer than ``threshold`` shares are supplied.
    """
    if len(shares) < threshold:
        raise ValueError(
            f"reconstruction needs at least threshold={threshold} shares, got {len(shares)}"
        )

    xs = sorted(shares.keys())[:threshold]
    length = shares[xs[0]].numel()
    out = torch.zeros(length, dtype=torch.int64)

    for i in xs:
        # Lagrange basis at x = 0: prod_{j != i} (0 - x_j) / (x_i - x_j)
        num, den = 1, 1
        for j in xs:
            if j == i:
                continue
            num = (num * (-j)) % PRIME
            den = (den * (i - j)) % PRIME
        coeff = (num * _inv(den)) % PRIME
        out = (out + shares[i].to(torch.int64) % PRIME * coeff) % PRIME
    return out


# --------------------------------------------------------------------------------------------------
# The protocol, in the order the parties execute it
# --------------------------------------------------------------------------------------------------
def client_mask(client_id: str, round_seed: int, length: int) -> torch.Tensor:
    """Client ``i``'s own random mask ``z_i``, uniform over the field.

    Unlike slice 1's pairwise masks this is INDEPENDENT of the cohort — it does not have to cancel
    against anyone. Removal is handled by reconstruction from shares, which is what lets the scheme
    survive a dropout at all.

    Derived from ``(client_id, round_seed)`` so a client can regenerate it, and so reuse across
    rounds — which would leak the difference of two rounds' inputs — cannot happen by accident.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    digest = hashlib.sha256(f"{round_seed}|{client_id}".encode()).digest()
    gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") >> 1)
    return torch.randint(0, PRIME, (length,), generator=gen, dtype=torch.int64)


def mask_values(
        values: Sequence[float],
        mask: torch.Tensor,
        scale: int = 1_000_000,
) -> torch.Tensor:
    """Client ``i``'s wire payload ``y_i = quantize(x_i) + z_i (mod PRIME)``.

    Uniform to anyone without the shares, including the server.
    """
    q = torch.round(torch.as_tensor(values, dtype=torch.float64) * scale).to(torch.int64) % PRIME
    return (q + mask.to(torch.int64)) % PRIME


def aggregate_shares(shares: Sequence[torch.Tensor]) -> torch.Tensor:
    """A holder's single reply: the sum of the shares it holds from the SURVIVING dealers.

    This is the one-shot step. A holder returns ONE vector however many dealers there were, and
    the server decodes once however many dropped — where classic SecAgg would reconstruct each
    dropped client's mask separately.
    """
    if not shares:
        raise ValueError("aggregate_shares requires at least one share.")
    total = torch.zeros(shares[0].numel(), dtype=torch.int64)
    for s in shares:
        total = (total + s.to(torch.int64)) % PRIME
    return total


def recover_aggregate(
        masked_values: Sequence[torch.Tensor],
        summed_shares: Dict[int, torch.Tensor],
        threshold: int,
        scale: int = 1_000_000,
) -> torch.Tensor:
    """Server side: recover ``sum_i x_i`` over the survivors, as float32.

    ``sum(y_i) - sum(z_i)`` where the second term comes from ONE Lagrange interpolation over the
    summed shares. The server never sees an individual ``x_i`` or ``z_i``.

    Raises:
        ValueError: if there are no masked values, or fewer than ``threshold`` summed shares.
    """
    if not masked_values:
        raise ValueError("recover_aggregate requires at least one masked value.")

    length = masked_values[0].numel()
    masked_total = torch.zeros(length, dtype=torch.int64)
    for y in masked_values:
        masked_total = (masked_total + y.to(torch.int64)) % PRIME

    mask_total = shamir_reconstruct(summed_shares, threshold=threshold)
    plain = (masked_total - mask_total) % PRIME

    # Field elements above PRIME/2 are negative sums.
    signed = torch.where(plain > PRIME // 2, plain - PRIME, plain)
    return (signed.to(torch.float64) / scale).to(torch.float32)
