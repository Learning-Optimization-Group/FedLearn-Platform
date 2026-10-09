import hashlib
import json

import numpy as np
import pytest
from sklearn.datasets import load_iris

from benchmark import group_partitions, prepare
from fedlearn.tree.data import load_csv


def test_duplicate_measurements_never_cross_client_or_heldout_boundaries():
    iris = load_iris()
    x = iris.data[50:].astype('float32')
    y = (iris.target[50:] - 1).astype('float32')
    parts = group_partitions(x, y, seed=42)
    ids = np.concatenate(list(parts.values()))
    assert sorted(ids.tolist()) == list(range(100))
    seen = set()
    for indices in parts.values():
        assert set(y[indices]) == {0, 1}
        keys = {hashlib.sha256(row.tobytes()).hexdigest() for row in x[indices]}
        assert not seen & keys
        seen |= keys
    again = group_partitions(x, y, seed=42)
    assert all(np.array_equal(parts[k], again[k]) for k in parts)


def test_conflicting_labels_for_identical_measurements_are_refused():
    x = np.repeat(np.arange(20, dtype='float32')[:, None], 4, axis=1)
    y = np.arange(20) % 2
    with pytest.raises(ValueError, match='conflicting'):
        group_partitions(np.vstack([x, x[0]]), np.append(y, 1), seed=42)


def test_preparation_preserves_named_meaning_and_canonical_csv_float_bits(tmp_path):
    output = tmp_path / 'iris'
    report = prepare(output, seed=42)
    iris = load_iris()
    assert report['feature_mapping']['f0'] == 'sepal length (cm)'
    assert report['feature_mapping']['f3'] == 'petal width (cm)'
    assert report['class_mapping'] == {'0': 'versicolor', '1': 'virginica'}
    for name, split in report['partitions'].items():
        x, y, data_hash = load_csv(output / (name + '.csv'))
        np.testing.assert_array_equal(x.view(np.uint32), iris.data[split['source_ids']].astype('float32').view(np.uint32))
        np.testing.assert_array_equal(y, iris.target[split['source_ids']] - 1)
        assert split['dataset_sha256'] == data_hash
    assert json.loads((output / 'provenance.json').read_text()) == report
    with pytest.raises(FileExistsError):
        prepare(output, seed=42)


@pytest.mark.parametrize('x,y', [(np.ones((20, 3)), np.arange(20)%2),
                                (np.full((20,4),np.nan),np.arange(20)%2),
                                (np.ones((20,4)),np.full(20,2))])
def test_schema_and_nonfinite_data_are_refused(x, y):
    with pytest.raises(ValueError):
        group_partitions(x, y, seed=42)


def test_centralized_comparison_trains_eight_trees_and_preserves_predictions(tmp_path):
    from benchmark import centralized_comparison
    data = tmp_path / 'data'
    prepare(data)
    results = centralized_comparison(data, tmp_path / 'central', seed=42)
    assert len(results['controls']) == 2
    assert {c['learning_rate'] for c in results['controls']} == {.15, .3}
    assert all(c['trees'] == 8 and c['reload_predictions_exact'] for c in results['controls'])
    assert all(c['validation']['log_loss'] < np.log(2) for c in results['controls'])
    assert results['test_used_for_training'] is False
    with pytest.raises(FileExistsError):
        centralized_comparison(data, tmp_path / 'central', seed=42)
