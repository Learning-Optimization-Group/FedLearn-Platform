"""LightSecAgg with coded mask shares (So, Guler, Avestimehr, MLSys 2022, arXiv:2109.14236).

``lightsecagg.py`` Shamir-shares every coordinate of a client's mask to every peer, so each peer
receives a full-length share and the offline traffic is ``N * L`` field elements per client. The
paper's encoding splits the mask into ``U - T`` pieces, appends ``T`` uniformly random pieces, and
sends peer ``j`` the evaluation at ``alpha_j`` of the degree ``U - 1`` polynomial through those
``U`` pieces. Each share is ``ceil(L / (U - T))`` elements long.

* Privacy: any ``T`` colluding peers see ``T`` evaluations, which the ``T`` random pieces make
  uniform and independent of the mask (the ``T x T`` block of the encoding matrix on the random
  pieces is invertible).
* Dropout resilience: any ``U`` surviving peers' summed shares determine the summed polynomial, so
  the server recovers the aggregate mask of the survivors with one decode while up to ``N - U``
  peers have dropped.

As in ``lightsecagg.py``, shares are assumed to travel over authenticated private channels, and a
malicious client submitting inconsistent shares is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["CodedParams", "client_mask", "encoding_matrix", "encode_mask", "decode_aggregate_mask",
           "secure_sum", "bytes_per_client"]


@dataclass(frozen=True)
class CodedParams:
    """``num_users`` N, privacy threshold ``privacy`` T (colluders), target survivors ``target`` U, field ``prime``."""
    num_users: int
    privacy: int
    target: int
    prime: int

    def __post_init__(self):
        if not 0 <= self.privacy < self.target <= self.num_users:
            raise ValueError(f"need 0 <= T < U <= N, got T={self.privacy} U={self.target} N={self.num_users}")
        if self.prime <= self.target + self.num_users:
            raise ValueError("the field must hold U + N distinct evaluation points")
        if self.prime >= 2 ** 31:
            raise ValueError("prime must be below 2**31 so field products fit in int64")

    @property
    def pieces(self) -> int:
        return self.target - self.privacy

    def betas(self) -> List[int]:
        return list(range(1, self.target + 1))

    def alphas(self) -> List[int]:
        return list(range(self.target + 1, self.target + self.num_users + 1))

    def piece_length(self, length: int) -> int:
        return math.ceil(length / self.pieces)


def _inv(a: int, p: int) -> int:
    return pow(int(a) % p, p - 2, p)


def _lagrange(nodes: Sequence[int], points: Sequence[int], p: int) -> np.ndarray:
    """``M[r, k]`` = the k-th Lagrange basis polynomial over ``nodes`` evaluated at ``points[r]``, mod p."""
    out = np.zeros((len(points), len(nodes)), dtype=np.int64)
    for r, x in enumerate(points):
        for k, xk in enumerate(nodes):
            num, den = 1, 1
            for m, xm in enumerate(nodes):
                if m != k:
                    num = num * (x - xm) % p
                    den = den * (xk - xm) % p
            out[r, k] = num * _inv(den, p) % p
    return out


@lru_cache(maxsize=8)
def encoding_matrix(params: CodedParams) -> np.ndarray:
    """``W[j, k]``: the weight of piece ``k`` in peer ``j``'s share (N x U), read-only and cached."""
    out = _lagrange(params.betas(), params.alphas(), params.prime)
    out.flags.writeable = False
    return out


def _matmul_mod(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """(a @ b) mod p for int64 entries in [0, p), accumulating term by term so nothing overflows."""
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    for k in range(a.shape[1]):
        out = (out + np.outer(a[:, k], b[k]) % p) % p
    return out


def client_mask(client_id: str, round_seed: int, length: int, prime: int) -> np.ndarray:
    """Client ``i``'s uniform field mask for one round, regenerable from ``(client_id, round_seed)``."""
    digest = hashlib.sha256(f"coded-lightsecagg|{round_seed}|{client_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big")).integers(0, prime, size=length, dtype=np.int64)


def encode_mask(mask: np.ndarray, params: CodedParams, rng: np.random.Generator,
                weights: np.ndarray = None) -> np.ndarray:
    """The N coded shares of ``mask`` (row j goes to peer j), each ``piece_length`` elements."""
    p, m = params.prime, params.piece_length(mask.size)
    padded = np.zeros(m * params.pieces, dtype=np.int64)
    padded[:mask.size] = np.mod(mask, p)
    pieces = np.concatenate([padded.reshape(params.pieces, m),
                             rng.integers(0, p, size=(params.privacy, m), dtype=np.int64)])
    w = encoding_matrix(params) if weights is None else weights
    return _matmul_mod(w, pieces, p)


def decode_aggregate_mask(summed_shares: Dict[int, np.ndarray], params: CodedParams, length: int) -> np.ndarray:
    """Sum of the survivors' masks from ``U`` peers' summed shares (keys are peer indices 0..N-1)."""
    if len(summed_shares) < params.target:
        raise ValueError(f"decoding needs U={params.target} summed shares, got {len(summed_shares)}")
    holders = sorted(summed_shares)[:params.target]
    alphas = params.alphas()
    basis = _lagrange([alphas[j] for j in holders], params.betas()[:params.pieces], params.prime)
    stacked = np.stack([np.mod(summed_shares[j], params.prime) for j in holders])
    return _matmul_mod(basis, stacked, params.prime).reshape(-1)[:length]


def secure_sum(messages: Dict[int, np.ndarray], params: CodedParams, round_seed: int, *,
               rng: np.random.Generator, dropped_after_upload: Iterable[int] = ()) -> Tuple[np.ndarray, Dict]:
    """Run one round of the protocol in-process over ``messages`` (client index -> field vector).

    Every client in ``messages`` uploads ``message + mask``; clients in ``dropped_after_upload`` then
    go silent, and the server decodes the aggregate mask from the remaining peers. Returns the field
    sum of the messages and a record of what was exchanged.
    """
    p = params.prime
    ids = sorted(messages)
    if len(ids) != params.num_users:
        raise ValueError(f"protocol was set up for N={params.num_users} clients, got {len(ids)}")
    length = int(next(iter(messages.values())).size)
    weights = encoding_matrix(params)
    held = {j: [] for j in range(params.num_users)}
    masked_total = np.zeros(length, dtype=np.int64)
    for i in ids:
        mask = client_mask(str(i), round_seed, length, p)
        shares = encode_mask(mask, params, rng, weights)
        for j in range(params.num_users):
            held[j].append(shares[j])
        masked_total = (masked_total + np.mod(messages[i], p) + mask) % p
    silent = {ids.index(c) for c in dropped_after_upload}
    replies = {j: np.mod(np.sum(np.stack(held[j]), axis=0), p) for j in range(params.num_users) if j not in silent}
    aggregate_mask = decode_aggregate_mask(replies, params, length)
    record = {"replies": len(replies), "piece_length": params.piece_length(length), "length": length}
    return (masked_total - aggregate_mask) % p, record


def bytes_per_client(length: int, params: CodedParams, *, bits: int, coded: bool = True) -> Dict[str, float]:
    """Analytic per-client traffic of one round, with field elements bit-packed at ``bits`` bits.

    ``coded=False`` gives the full-length Shamir sharing of ``lightsecagg.py`` under the same accounting.
    """
    share = params.piece_length(length) if coded else length
    element = bits / 8.0
    offline = (params.num_users - 1) * share * element
    upload = length * element
    recovery = share * element
    return {"offlineShares": offline, "maskedUpload": upload, "recoveryShare": recovery,
            "total": offline + upload + recovery}
