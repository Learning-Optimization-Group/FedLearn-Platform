import run_effusion_local_filter as a20


def test_matrix_is_the_pre_registered_105_cells():
    cells = a20.cells()
    assert len(cells) == 105 and len({c['name'] for c in cells}) == 105
    assert sum(c['filter'] for c in cells) == 75


def test_noise_matching_attacks_are_a9_attacks_with_privatize():
    assert a20.CONDITIONS['sign_flip_noisy'] == [{**a20.A9_BEHAVIORS['sign_flip'][0], 'privatize': True}]
    assert 'privatize' not in a20.CONDITIONS['sign_flip'][0]


def test_filter_off_comparators_come_from_a12_or_this_study():
    assert a20.off_name(8.0, 'label_flip', 20261101) == 'a12_local_eps8_mean_label_flip_20261101'
    assert a20.off_name(32.0, 'sign_flip_noisy', 20261101) == 'nofilter_eps32_mean_sign_flip_noisy_20261101'


def test_norm_filter_enters_the_config_only_when_given():
    kw = dict(clients=80, features_path='f', report_sha256='r')
    assert 'norm_filter' not in a20.federated.config('n', 1, **kw)['server']
    assert a20.federated.config('n', 1, norm_filter={'k': 4.0}, **kw)['server']['norm_filter'] == {'k': 4.0}
