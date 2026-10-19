"""BATCH_ORDER_SEEDED_PERMUTATION_V1: a batch order every runtime reproduces (execution_contract.proto)."""
from __future__ import annotations

import os

import pytest

from fedlearn.contract.batch_order import SplitMix64, batches, permutation_state, seeded_permutation

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "execution_contract_v1", "batch_permutation_v1.golden")


def _cases():
    with open(GOLDEN) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("below") or not line.strip():
                continue
            head, _, tail = line.partition(":")
            n, seed, round_, epoch = (int(v) for v in head.split())
            yield n, seed, round_, epoch, [int(v) for v in tail.split()]


def test_the_stream_is_splitmix64():
    """Published SplitMix64 outputs for state 0, so the stream is the standard generator and not a look-alike."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(4)] == [
        0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC]


@pytest.mark.parametrize("case", list(_cases()), ids=lambda c: f"n{c[0]}-s{c[1]}-r{c[2]}-e{c[3]}")
def test_the_golden_permutations_are_reproduced(case):
    n, seed, round_, epoch, expected = case
    assert seeded_permutation(n, seed, round_, epoch) == expected
    assert sorted(expected) == list(range(n))


def test_the_golden_covers_the_edge_counts_and_the_seed_range():
    cases = list(_cases())
    assert {c[0] for c in cases} >= {0, 1, 2, 8, 11, 20, 257}
    assert {c[1] for c in cases} >= {0, (1 << 64) - 1}


def test_each_epoch_and_round_draws_its_own_order():
    base = seeded_permutation(20, 42, 1, 0)
    assert seeded_permutation(20, 42, 1, 1) != base
    assert seeded_permutation(20, 42, 2, 0) != base
    assert seeded_permutation(20, 43, 1, 0) != base


def test_the_state_folds_seed_round_and_epoch_in_order():
    assert permutation_state(1, 2, 3) != permutation_state(3, 2, 1)
    with pytest.raises(ValueError):
        permutation_state(-1, 1, 0)


def test_below_never_returns_the_bound_and_rejects_a_zero_bound():
    rng = SplitMix64(123)
    assert all(0 <= rng.below(3) < 3 for _ in range(1000))
    with pytest.raises(ValueError):
        rng.below(0)


def test_batches_keep_the_final_partial_batch_unless_drop_last():
    order = list(range(11))
    assert batches(order, 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    assert batches(order, 4, drop_last=True) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert batches([], 4) == []


def _below_cases():
    with open(GOLDEN) as fh:
        for line in fh:
            if line.startswith("below"):
                head, _, tail = line.partition(":")
                _, state, bound = head.split()
                yield int(state), int(bound), int(tail)


def test_the_unbiased_draw_rejects_exactly_the_draws_above_its_limit():
    """2^64 - 1 is rejected for bound 3 (2^64 mod 3 == 1) and the next draw used; 2^64 - 2 is accepted."""
    cases = list(_below_cases())
    assert len(cases) == 2
    rejected, accepted = cases
    assert SplitMix64(rejected[0]).next() == (1 << 64) - 1
    rng = SplitMix64(rejected[0])
    rng.next()
    assert SplitMix64(rejected[0]).below(3) == rejected[2] == rng.next() % 3
    assert SplitMix64(accepted[0]).next() == (1 << 64) - 2
    assert SplitMix64(accepted[0]).below(3) == accepted[2] == ((1 << 64) - 2) % 3
