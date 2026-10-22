import pytest

import run_effusion_dp_rounds as a15


def test_the_matrix_is_the_pre_registered_75_cells():
    cells = a15.cells()
    assert len(cells) == 75 and len({c['name'] for c in cells}) == 75


@pytest.mark.parametrize('rounds', a15.ROUNDS)
@pytest.mark.parametrize('epsilon', [8.0, 4.0])
def test_each_round_count_gets_its_own_exact_calibration(rounds, epsilon):
    from fedlearn.privacy.dp_accountant import required_noise_multiplier
    cal = a15.calibration(epsilon, rounds=rounds)
    engine_z = required_noise_multiplier(cal['rdpEquivalentEpsilon'], 1.0, rounds, a15.DELTA)
    assert a15.check_calibration(cal, engine_z, rounds=rounds) <= epsilon + 1e-6


def test_only_t20_cells_replay_and_against_the_right_study():
    sources = {c['name']: a15.replay_source(c) for c in a15.cells()}
    assert sum(v is not None for v in sources.values()) == 15
    assert sources['T20_clean_20261101'] == ('a8', 'clean_20261101')
    assert sources['T20_eps4_20261101'] == ('a13', 'gdp_eps4_20261101')
