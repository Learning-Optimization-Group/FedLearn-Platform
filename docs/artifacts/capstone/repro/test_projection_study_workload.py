"""Public generated-only experimental design invariants."""
from collections import Counter

import numpy as np
import pytest
import torch

from projection_study_design import Cell, cells, qualification_cells, seed_for
from projection_study_workload import basis, coalition, dataset, features, partitions, validate_basis


def test_frozen_budget_has_distinct_crossed_seeds_and_no_recounted_baselines():
    rows = cells()
    assert len(rows) == len({c.name for c in rows}) == 136
    assert Counter(c.block for c in rows) == dict(full=8, controlled=32, random=40, padded=8, attack=48)
    assert {c.seed for c in rows} == set(range(24001, 24005))
    random = [c for c in rows if c.block == 'random']
    assert Counter(c.projection_seed for c in random) == {s: 8 for s in range(20261101, 20261106)}
    q = qualification_cells()
    assert len(q) == len({c.name for c in q}) == 12
    assert not {c.name for c in q} & {c.name for c in rows}
    assert {c.seed for c in q} == {25001}


def test_namespaces_are_stable_distinct_and_do_not_mutate_global_rng():
    state = torch.random.get_rng_state().clone()
    assert len({seed_for(1, n) for n in ('train', 'eval', 'bias', 'active', 'inactive')}) == 5
    assert seed_for(1, 'train', 2) == seed_for(1, 'train', 2)
    assert seed_for(1, 'train', 2) != seed_for(2, 'train', 1)
    for eta in (0., .25, .75, 1.):
        p = basis(Cell('controlled', 24001, False, eta=eta))
        assert torch.allclose(p.T @ p, torch.eye(128, dtype=torch.float64), atol=1e-12)
        assert float(p[0].square().sum()) == pytest.approx(eta)
    basis(Cell('random', 24001, False, projection_seed=20261101))
    assert torch.equal(state, torch.random.get_rng_state())


def test_generated_splits_and_partition_are_fixed_complete_and_separate():
    d = dataset()
    assert d['train_x'].shape == (4096, 512) and d['eval_x'].shape == (8192, 512)
    assert torch.equal(d['train_x'], dataset()['train_x'])
    assert not torch.equal(d['train_x'], d['eval_x'][:4096])
    parts = partitions(24001)
    assert Counter(map(len, parts)) == {205: 16, 204: 4}
    assert torch.equal(torch.sort(torch.cat(parts)).values, torch.arange(4096))
    assert not torch.equal(parts[0], partitions(24002)[0])
    assert len(set(coalition(24001))) == 4
    assert all(0 <= c < 20 for c in coalition(24001))


def test_padding_preserves_representation_and_is_not_raw_full_features():
    d = dataset()
    small = features(d, Cell('random', 24001, False, projection_seed=20261101))
    wide = features(d, Cell('padded', 24001, False, projection_seed=20261101))
    assert torch.equal(small['train_x'], wide['train_x'][:, :128])
    assert torch.count_nonzero(wide['eval_x'][:, 128:]) == 0
    assert torch.equal(d['eval_y'], wide['eval_y'])
    assert not torch.equal(wide['train_x'], d['train_x'])
    assert torch.equal(features(d, Cell('random', 24002, False, projection_seed=20261101))['train_x'], small['train_x'])


@pytest.mark.parametrize('p', [torch.zeros(512, 128), torch.eye(128), torch.full((512,128), float('nan'))])
def test_invalid_projection_refused(p):
    with pytest.raises(ValueError): validate_basis(p)
