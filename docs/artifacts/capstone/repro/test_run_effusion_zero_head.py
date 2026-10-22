import numpy as np
import torch

import run_effusion_zero_head as a17


def _workload():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 6)), rng.integers(0, 2, 40)
    return a17.ZeroHeadWorkload(x, y, [str(i // 2 + 1) for i in range(40)], x[:10], y[:10], np.zeros(6), np.ones(6),
                                clients=4, partition_seed=1, info={})


def test_only_the_seeded_initial_head_is_zeroed():
    w = _workload()
    assert all(torch.count_nonzero(p) == 0 for p in w.build_model(7).parameters())
    assert any(torch.count_nonzero(p) > 0 for p in w.build_model().parameters())


def test_matrix_and_counterparts():
    cells = a17.cells()
    assert len(cells) == 40
    sources = {c['name']: a17.counterpart(c) for c in cells}
    assert sources['zero_T20_eps1_20261101'] == ('a13', 'gdp_eps1_20261101')
    assert sources['zero_T5_eps1_20261101'] is None
    assert sources['zero_T5_eps4_20261102'] == ('a15', 'T5_eps4_20261102')
