from itertools import combinations
import math

import numpy as np
import pytest

from fedlearn.security.lightsecagg_coded import (
    CodedParams, bytes_per_client, client_mask, decode_aggregate_mask, encode_mask, encoding_matrix, secure_sum)

P = 65521


def _det_mod(m, p):
    m = [list(map(int, row)) for row in m]
    n, det = len(m), 1
    for c in range(n):
        pivot = next((r for r in range(c, n) if m[r][c] % p), None)
        if pivot is None:
            return 0
        if pivot != c:
            m[c], m[pivot] = m[pivot], m[c]
            det = -det
        det = det * m[c][c] % p
        inv = pow(m[c][c], p - 2, p)
        for r in range(c + 1, n):
            f = m[r][c] * inv % p
            m[r] = [(a - f * b) % p for a, b in zip(m[r], m[c])]
    return det % p


def test_parameters_are_validated():
    for bad in [(6, 4, 4, P), (6, 2, 7, P), (6, 2, 4, 7), (6, 2, 4, 2 ** 31 + 11)]:
        with pytest.raises(ValueError):
            CodedParams(*bad)
    params = CodedParams(80, 40, 64, P)
    assert params.pieces == 24 and params.piece_length(2048) == 86


def test_secure_sum_equals_the_modular_sum_with_and_without_dropouts():
    params = CodedParams(8, 3, 6, P)
    rng = np.random.default_rng(0)
    messages = {i: rng.integers(0, P, size=50, dtype=np.int64) for i in range(8)}
    direct = sum(messages.values()) % P
    for dropped in ([], [1], [0, 7]):
        total, record = secure_sum(messages, params, round_seed=9, rng=rng, dropped_after_upload=dropped)
        assert np.array_equal(total, direct)
        assert record["replies"] == 8 - len(dropped) and record["piece_length"] == math.ceil(50 / 3)
    with pytest.raises(ValueError):
        secure_sum(messages, params, round_seed=9, rng=rng, dropped_after_upload=[0, 1, 2])
    with pytest.raises(ValueError):
        secure_sum({0: messages[0]}, params, round_seed=9, rng=rng)


def test_decoding_any_target_subset_recovers_the_summed_masks():
    params = CodedParams(6, 2, 4, P)
    rng = np.random.default_rng(1)
    masks = [rng.integers(0, P, size=10, dtype=np.int64) for _ in range(6)]
    shares = [encode_mask(m, params, rng) for m in masks]
    summed = {j: sum(s[j] for s in shares) % P for j in range(6)}
    for holders in combinations(range(6), 4):
        out = decode_aggregate_mask({j: summed[j] for j in holders}, params, 10)
        assert np.array_equal(out, sum(masks) % P)


def test_any_privacy_threshold_of_peers_sees_uniform_shares():
    """For every T-subset of peers, the block of the encoding matrix on the T random pieces is
    invertible, so their shares are a bijection of uniform randomness, independent of the mask."""
    params = CodedParams(7, 3, 5, P)
    w = encoding_matrix(params)
    for peers in combinations(range(7), 3):
        block = w[np.array(peers)][:, params.pieces:]
        assert _det_mod(block, P) != 0


def test_masks_are_regenerable_and_round_specific():
    a = client_mask("3", 1, 20, P)
    assert np.array_equal(a, client_mask("3", 1, 20, P))
    assert not np.array_equal(a, client_mask("3", 2, 20, P))
    assert a.min() >= 0 and a.max() < P


def test_coded_shares_cut_offline_traffic_by_the_number_of_pieces():
    params = CodedParams(80, 40, 64, P)
    coded = bytes_per_client(2048, params, bits=16)
    plain = bytes_per_client(2048, params, bits=16, coded=False)
    assert coded["maskedUpload"] == plain["maskedUpload"] == 2048 * 2
    assert plain["offlineShares"] / coded["offlineShares"] == pytest.approx(2048 / 86)
    assert coded["total"] == pytest.approx(coded["offlineShares"] + coded["maskedUpload"] + coded["recoveryShare"])
