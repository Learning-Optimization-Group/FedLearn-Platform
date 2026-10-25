"""Budget and pairing checks for the prospective fresh-seed comparison."""
from collections import Counter

import finalist_design as design


def test_finalist_budget_is_complete_unique_and_fresh():
    cells = design.cells()
    assert len(cells) == 196
    assert len({c['name'] for c in cells}) == 196
    assert design.SEEDS == tuple(range(13001, 13013))
    assert not set(design.SEEDS) & {20261211, 20261212, 20261101}
    main = [c for c in cells if c['role'] == 'main']
    assert len(main) == 192
    assert Counter(c['method'] for c in main) == {method: 48 for method in design.METHODS}


def test_every_main_condition_has_the_same_twelve_seed_pairs():
    main = [c for c in design.cells() if c['role'] == 'main']
    for method in design.METHODS:
        for private in (False, True):
            for attack in ('clean', 'owned_pgd'):
                selected = [c for c in main if c['method'] == method and c['private'] == private and c['attack'] == attack]
                assert {c['seed'] for c in selected} == set(design.SEEDS)
                assert all(c['fraction'] == (.2 if attack == 'owned_pgd' else 0.) for c in selected)


def test_finalists_preserve_recipe_and_permitted_mechanism_boundaries():
    cells = design.cells()
    assert all(c['kind'] == 'fl' and c['lr'] == .03 and c['clip_norm'] == .05 and c['alpha'] is None for c in cells)
    assert all(c['bucket'] == 1 for c in cells)
    assert {c['width'] for c in cells if c['method'] == 'projected128'} == {128}
    assert {c['tau'] for c in cells if c['method'] == 'center_quarter'} == {.0125}
    assert {c['rule'] for c in cells if c['method'] == 'trim16'} == {'trimmed_mean'}
    anchors = [c for c in cells if c['role'] == 'anchor']
    assert len(anchors) == 4
    assert all(c['backend'] == 'native' and c['tag'] == 'mean' and c['attack'] == 'clean' for c in anchors)
    assert all(c['tag'] == 'mean' for c in cells if c['method'] == 'mean512' and c['attack'] == 'clean')
