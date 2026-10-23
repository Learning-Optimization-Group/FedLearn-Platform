import run_effusion_local_ks as a21


def test_matrix_is_the_pre_registered_75_cells():
    cells = a21.cells()
    assert len(cells) == 75 and len({c['name'] for c in cells}) == 75
    assert sum(c['condition'] == 'clean' for c in cells) == 15 and all(
        c['filter'] == 'normks' for c in cells if c['condition'] == 'clean')


def test_norm_only_clean_comparator_is_a20():
    assert a21.clean_name('norm', 32.0, 20261101) == 'a20_filter_eps32_mean_clean_20261101'
    assert a21.clean_name('normks', 32.0, 20261101) == 'normks_eps32_mean_clean_20261101'


def test_matched_attacks_are_a9_attacks_with_match_norm():
    assert a21.CONDITIONS['label_flip_matched'] == [{**a21.A9_BEHAVIORS['label_flip'][0], 'match_norm': True}]
