import math

import pytest

import run_effusion_strong_attacks as a14


def test_the_matrix_is_the_pre_registered_350_cells():
    cells = a14.cells()
    assert len(cells) == 350 and len({c['name'] for c in cells}) == 350
    assert sum(not c['private'] for c in cells) == 150 and sum(c['private'] for c in cells) == 200


@pytest.mark.parametrize('k', [1, 4, 16])
def test_trim_ratios_trim_exactly_k_per_side(k):
    assert math.floor(a14.trim_ratio(k) * a14.CLIENTS) == k


def test_attacker_fractions_give_4_8_and_16_slots():
    slots = sorted({round(b[0]['fraction'] * a14.CLIENTS) for b in a14.conditions().values() if b})
    assert slots == [4, 8, 16]


def test_replay_map_covers_exactly_the_a9_clean_cells():
    stems = {c['name'].rsplit('_', 1)[0] for c in a14.cells() if c['condition'] == 'clean'}
    assert set(a14.A9_REPLAY) == stems - {'dp4_trim1_clean', 'dp4_trim4_clean'}


def test_rule_params_appear_only_when_given():
    kw = dict(clients=80, features_path='f', report_sha256='r', rule='trimmed_mean')
    assert 'rule_params' not in a14.federated.config('n', 1, **kw)['server']
    assert a14.federated.config('n', 1, rule_params={'trim_ratio': 0.05}, **kw)['server']['rule_params'] == {'trim_ratio': 0.05}
