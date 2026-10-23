import numpy as np

import run_effusion_noniid as a19


def test_matrix_is_the_pre_registered_60_cells_paired_with_a14():
    cells = a19.cells()
    assert len(cells) == 60 and len({c['name'] for c in cells}) == 60
    assert {c['iid'].rsplit('_', 1)[0] for c in cells} == {
        'nodp_mean_clean', 'nodp_trim16_clean', 'nodp_median_clean', 'nodp_mean_fang_trim_m16',
        'nodp_median_fang_trim_m16', 'dp4_mean_clean'}


def test_workload_uses_the_dirichlet_partition_and_records_it():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 6)), (rng.random(200) < 0.2).astype(int)
    patients = [str(i // 2 + 1) for i in range(200)]
    w = a19.DirichletEffusionWorkload(x, y, patients, x[:10], y[:10], np.zeros(6), np.ones(6), clients=10,
                                      partition_seed=3, info={}, alpha=0.1)
    assert sorted(np.concatenate(w.parts).tolist()) == list(range(200)) and all(len(p) for p in w.parts)
    assert w.info['partition_algorithm'] == 'patient-dirichlet-v1(alpha=0.1)'
    het = a19.heterogeneity(w)
    assert 0 <= het['positiveShare']['min'] <= het['positiveShare']['max'] <= 1
