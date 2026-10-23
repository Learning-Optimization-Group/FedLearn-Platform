import numpy as np

import run_effusion_distributed_dp as a23


def test_matrix_records_the_infeasible_cells_as_not_run():
    todo, not_run = a23.cells()
    names = [c['name'] for c in todo]
    assert len(names) == len(set(names)) == 11 * 5 + 3 * 5 + 2 * 2 * 5
    assert [r['name'] for r in not_run] == [f'dist_b12_eps1_{s}' for s in a23.SEEDS]
    attack = next(c for c in todo if c['name'] == 'dist_b16_eps4_signflip16_20261101')
    assert attack['behaviors'] == [{'type': 'sign_flip_scale', 'scale': -10.0, 'fraction': 0.2}]
    assert attack['privacy'] == {'mode': 'distributed', 'epsilon': 4.0, 'delta': 1e-5, 'distributed': {'bits': 16}}
    assert next(c for c in todo if c['name'] == 'rdp_eps1_20261105')['privacy']['mode'] == 'central'


def test_contrasts_skip_groups_with_unusable_cells():
    todo, _ = a23.cells()
    usable = {c['name'] for c in todo}
    full = a23.contrast_spec(usable)
    assert 'distMinusRdp|b12|eps1' not in full and 'distMinusRdp|b14|eps1' in full
    assert full['distMinusGdp|b20|eps4'][0] == {'dist_b20_eps4_20261101': 1, 'a17_zero_T20_eps4_20261101': -1}
    assert full['damageDistMinusRdp|signflip4'][0] == {
        'dist_b16_eps4_20261101': 1, 'dist_b16_eps4_signflip4_20261101': -1,
        'rdp_eps4_20261101': -1, 'rdp_eps4_signflip4_20261101': 1}
    partial = a23.contrast_spec(usable - {'rdp_eps8_20261103'})
    assert 'distMinusRdp|b16|eps8' not in partial and 'rdpMinusGdp|eps8' not in partial
    assert 'distMinusGdp|b16|eps8' in partial


def _workload_inputs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    x = rng.normal(size=(400, 6)) + y[:, None] * 1.0
    patients = [str(i // 2 + 1) for i in range(400)]
    return x, y, patients, x[:100], y[:100], np.zeros(6), np.ones(6)


def test_a_distributed_cell_runs_end_to_end_with_exact_secure_sums():
    todo, _ = a23.cells()
    cell = next(c for c in todo if c['name'] == 'dist_b16_eps8_20261101')
    record, margin = a23.run_cell(cell, *_workload_inputs(), features_path='synthetic', report_sha256='a' * 64)
    check = a23.integrity(record, cell)
    assert check['secureSumExact'] and check['withinClaim'] and np.isfinite(margin).all()
    assert check['plan']['bits'] == 16 and check['plan']['d'] == 14 and check['plan']['d_pad'] == 16
    assert check['effectiveNoiseMultiplier'] > 0
    assert check['bytesPerClient']['total'] < check['bytesPerClientUncoded']['total']


def test_a_central_cell_checks_its_rdp_noise_multiplier():
    todo, _ = a23.cells()
    cell = next(c for c in todo if c['name'] == 'rdp_eps8_20261101')
    record, _ = a23.run_cell(cell, *_workload_inputs(), features_path='synthetic', report_sha256='a' * 64)
    check = a23.integrity(record, cell)
    assert check['noiseMultiplierMatches'] and check['withinClaim']
