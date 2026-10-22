import math

import pytest

import run_effusion_canary_audit as a16


def test_matrix_and_canary_count():
    assert len(a16.cells()) == 40
    assert round(a16.behaviors()[0]['fraction'] * a16.federated.CLIENTS) == a16.CANARIES == 8
    assert a16.behaviors()[0]['norm'] == a16.RECIPE['clip_norm']


def test_theory_mu_is_half_the_claimed_replace_one_mu():
    assert a16.theory_mu(4.0, rounds=16) == pytest.approx(0.5)
    assert a16.theory_mu(4.0, rounds=16) == pytest.approx(math.sqrt(16) / 4.0 / 2)


def test_scores_are_the_inner_product_with_the_engine_direction():
    from fedlearn.simulation.behaviors import canary_direction
    import torch
    head = {'head.weight': [[1.0, 2.0], [3.0, 4.0]], 'head.bias': [0.5, -0.5]}
    like = {k: torch.tensor(v) for k, v in head.items()}
    u = canary_direction(11, 3, like)
    want = sum(float((like[k] * u[k]).sum()) for k in like)
    assert a16.scores(head, 11, [3]) == [pytest.approx(want, rel=1e-6)]
    assert a16.UNSEEN_ID_BASE > a16.federated.CLIENTS
