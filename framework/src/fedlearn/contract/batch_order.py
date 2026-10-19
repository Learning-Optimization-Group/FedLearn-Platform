"""The execution contract's reproducible batch order (BATCH_ORDER_SEEDED_PERMUTATION_V1).

Every runtime draws the same permutation of a participant's examples for a given run seed, round and local epoch, so
a participant's update can be replayed exactly from its data. The algorithm is specified in execution_contract.proto
(SplitMix64 stream, Fisher-Yates with rejection sampling); the native trainer implements the same and both are pinned
by framework/tests/fixtures/execution_contract_v1/batch_permutation_v1.golden.
"""
from __future__ import annotations

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def mix(z: int) -> int:
    """SplitMix64's output function applied to z + gamma."""
    z = (z + _GAMMA) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


class SplitMix64:
    """The SplitMix64 stream from ``state``: each draw is mix(state), then state advances by gamma."""

    def __init__(self, state: int) -> None:
        self.state = state & _MASK

    def next(self) -> int:
        r = mix(self.state)
        self.state = (self.state + _GAMMA) & _MASK
        return r

    def below(self, bound: int) -> int:
        """Uniform in [0, bound), rejecting the draws that would bias ``r mod bound``."""
        if bound < 1:
            raise ValueError("bound must be positive")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            r = self.next()
            if r < limit:
                return r % bound


def permutation_state(seed: int, round_: int, epoch: int) -> int:
    for value in (seed, round_, epoch):
        if not 0 <= value <= _MASK:
            raise ValueError("seed, round and epoch must be unsigned 64-bit integers")
    return mix(mix(mix(seed) ^ round_) ^ epoch)


def seeded_permutation(n: int, seed: int, round_: int, epoch: int) -> list[int]:
    """The order in which a participant visits its ``n`` examples in this round's ``epoch``."""
    if n < 0:
        raise ValueError("n must not be negative")
    rng = SplitMix64(permutation_state(seed, round_, epoch))
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def batches(order: list[int], batch_size: int, drop_last: bool = False) -> list[list[int]]:
    """Consecutive ``batch_size`` slices of ``order``; the last, shorter one is kept unless ``drop_last``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    out = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if drop_last and out and len(out[-1]) < batch_size:
        out.pop()
    return out
