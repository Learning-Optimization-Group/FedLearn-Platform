import numpy as np

import run_effusion_encoder_gate as a22


def _data(seed, informative):
    rng = np.random.default_rng(seed)
    y = (rng.random(600) < 0.2).astype(int)
    x = rng.normal(size=(600, 8))
    x[:, 0] += informative * y
    return x, y


def test_fit_selects_one_grid_point_and_bootstrap_detects_a_better_representation():
    xtr, ytr = _data(0, 1.0)
    xs, ys = _data(1, 1.0)
    weak_tr, weak_s = xtr.copy(), xs.copy()
    weak_tr[:, 0] = np.random.default_rng(2).normal(size=600) + 0.2 * ytr
    weak_s[:, 0] = np.random.default_rng(3).normal(size=600) + 0.2 * ys
    fits, strong = a22.fit_representation('R2', xtr, xs, ytr, ys)
    _, weak = a22.fit_representation('R0', weak_tr, weak_s, ytr, ys)
    assert sum(f['selected'] for f in fits) == 1 and len(fits) == len(a22.gate.GRID)
    patients = [str(i // 2 + 1) for i in range(600)]
    intervals, skipped = a22.bootstrap(ys, patients, {'R0': weak, 'R2': strong}, reps=200)
    assert intervals['delta_R2_R0'][0] > 0 and skipped == 0
