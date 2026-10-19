"""Write batch_permutation_v1.golden: the seeded batch order every runtime must reproduce.

One case per line, ``n seed round epoch : i0 i1 ...``, read by the Python and native tests. The cases cover the
edge counts (0, 1, 2), a TinyNet batch (8), counts that do not divide the batch (11, 20), a large count, and seeds at
both ends of the unsigned 64-bit range.

Lines ``below state bound : result`` pin the rejection boundary of the unbiased draw, which ordinary seeds never reach
(a rejection has probability ~2^-60 at these bounds). SplitMix64's output function is invertible, so the state is
constructed to make the next draw exactly 2^64 - 1 (rejected for bound 3, since 2^64 mod 3 == 1) or 2^64 - 2 (the
largest accepted draw).

    PYTHONPATH=framework/src .venv/bin/python framework/tests/fixtures/execution_contract_v1/generate_batch_permutation.py
"""
from __future__ import annotations

import os

from fedlearn.contract.batch_order import SplitMix64, seeded_permutation

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    (0, 42, 1, 0), (1, 42, 1, 0), (2, 42, 1, 0), (2, 42, 1, 1),
    (8, 42, 1, 0), (8, 42, 1, 1), (8, 42, 2, 0), (8, 43, 1, 0),
    (11, 0, 1, 0), (20, 7, 3, 2), (257, 42, 1, 0),
    (8, 0, 0, 0), (8, (1 << 64) - 1, 1, 0), (8, 42, (1 << 32) - 1, (1 << 32) - 1),
]


_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def _unxorshift(y: int, k: int) -> int:
    x = y
    for _ in range(64 // k + 1):
        x = y ^ (x >> k)
    return x & _MASK


def _state_whose_next_draw_is(target: int) -> int:
    """Invert SplitMix64's output function: the state whose next() returns ``target``."""
    z = _unxorshift(target, 31)
    z = (z * pow(0x94D049BB133111EB, -1, 1 << 64)) & _MASK
    z = _unxorshift(z, 27)
    z = (z * pow(0xBF58476D1CE4E5B9, -1, 1 << 64)) & _MASK
    z = _unxorshift(z, 30)
    return (z - _GAMMA) & _MASK


def main() -> None:
    lines = ["# n seed round epoch : permutation (BATCH_ORDER_SEEDED_PERMUTATION_V1; generate_batch_permutation.py)"]
    for n, seed, round_, epoch in CASES:
        perm = seeded_permutation(n, seed, round_, epoch)
        lines.append(f"{n} {seed} {round_} {epoch} :" + "".join(f" {i}" for i in perm))
    for draw in (_MASK, _MASK - 1):
        state = _state_whose_next_draw_is(draw)
        if SplitMix64(state).next() != draw:
            raise SystemExit("the output-function inverse is wrong")
        lines.append(f"below {state} 3 : {SplitMix64(state).below(3)}")
    with open(os.path.join(HERE, "batch_permutation_v1.golden"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
