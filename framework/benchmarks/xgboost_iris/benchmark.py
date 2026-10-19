"""Reproducible real-data preparation for the closed binary XGBoost contract.

Iris species versicolor/virginica use the platform's four numeric features.
Identical feature rows stay together across both clients and held-out splits.
The small benchmark qualifies execution; it does not establish clinical quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedGroupKFold
import sklearn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'framework/src'))
from fedlearn.tree.data import load_csv
from fedlearn.tree.model import FEATURES, VERSION

SOURCE = 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html'


def group_partitions(features, labels, *, seed):
    x, y = np.asarray(features, dtype='<f4'), np.asarray(labels)
    if (x.ndim != 2 or x.shape[1] != 4 or y.shape != (len(x),) or not np.isfinite(x).all()
            or not np.isin(y, [0, 1]).all() or set(y.tolist()) != {0, 1}):
        raise ValueError('expected finite four-feature binary data with both classes')
    groups = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in x])
    for key in np.unique(groups):
        if len(set(y[groups == key].tolist())) != 1:
            raise ValueError('identical measurements have conflicting labels')
    if any(len(np.unique(groups[y == label])) < 5 for label in (0, 1)):
        raise ValueError('each class needs at least five distinct feature groups')
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, groups))
    test, validation = folds[0][1], folds[1][1]
    training = np.setdiff1d(np.arange(len(x)), np.concatenate([test, validation]))
    clients = list(StratifiedGroupKFold(2, shuffle=True, random_state=seed).split(
        x[training], y[training], groups[training]))
    result = {'client-0': training[clients[0][1]], 'client-1': training[clients[1][1]],
              'validation': validation, 'test': test}
    seen = set()
    for name, indices in result.items():
        if set(y[indices].tolist()) != {0, 1}:
            raise ValueError(f'{name} does not contain both classes')
        keys = set(groups[indices].tolist())
        if seen & keys:
            raise ValueError('feature-group split leaked measurements')
        seen |= keys
    return {name: np.sort(indices) for name, indices in result.items()}


def prepare(output, *, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    iris = load_iris()
    all_x = iris.data.astype('<f4')
    all_y = iris.target.astype('<i8')
    source_ids = np.flatnonzero(iris.target != 0)
    x, y = all_x[source_ids], all_y[source_ids] - 1
    partitions = group_partitions(x, y, seed=seed)
    report = dict(dataset='scikit-learn bundled Iris', source=SOURCE,
                  sklearn_version=sklearn.__version__, numpy_version=np.__version__,
                  source_data_sha256=hashlib.sha256(all_x.tobytes() + all_y.tobytes()).hexdigest(),
                  selection='versicolor and virginica; setosa excluded',
                  feature_mapping=dict(zip(FEATURES, iris.feature_names)),
                  class_mapping={'0': 'versicolor', '1': 'virginica'},
                  seed=seed, split_method='StratifiedGroupKFold(5), fold0 test, fold1 validation; remaining folds split into two clients by StratifiedGroupKFold(2)',
                  group_definition='SHA256 of four ordered little-endian float32 measurements; labels excluded',
                  official_test_split_exists=False, feature_group_overlap=0, partitions={})
    output.mkdir(parents=True)
    try:
        for name, indices in partitions.items():
            path = output / (name + '.csv')
            with path.open('x') as file:
                file.write(','.join([*FEATURES, 'label']) + '\n')
                for row, label in zip(x[indices], y[indices]):
                    file.write(','.join(format(float(v), '.9g') for v in row) + f',{int(label)}\n')
            actual_x, actual_y, data_hash = load_csv(path)
            np.testing.assert_array_equal(actual_x.view(np.uint32), x[indices].view(np.uint32))
            np.testing.assert_array_equal(actual_y, y[indices])
            report['partitions'][name] = dict(source_ids=source_ids[indices].tolist(), count=len(indices),
                label_counts=np.bincount(y[indices], minlength=2).tolist(), dataset_sha256=data_hash,
                csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (output / 'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except BaseException:
        shutil.rmtree(output)
        raise
    return report


def metrics(labels, probabilities):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, log_loss, confusion_matrix
    y, p = np.asarray(labels), np.asarray(probabilities)
    if (y.ndim != 1 or p.shape != y.shape or set(y.tolist()) != {0, 1}
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()):
        raise ValueError('invalid binary probabilities or labels')
    predicted = p >= .5
    return dict(accuracy=float(accuracy_score(y, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                auroc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
                majority_accuracy=float(max(np.mean(y == 0), np.mean(y == 1))))


def centralized_comparison(data_directory, output, *, seed):
    import xgboost as xgb
    from fedlearn.tree.model import initial_model, load_model
    if xgb.__version__ != VERSION:
        raise ValueError(f'XGBoost {VERSION} is required')
    output, data_directory = Path(output), Path(data_directory)
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((data_directory / 'provenance.json').read_text())
    iris = load_iris()
    data, groups = {}, set()
    for name in ('client-0', 'client-1', 'validation', 'test'):
        path = data_directory / (name + '.csv')
        x, y, sha = load_csv(path)
        receipt = provenance['partitions'][name]
        if (sha != receipt['dataset_sha256'] or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['csv_sha256']):
            raise ValueError('CSV differs from recorded Iris preparation')
        ids = receipt['source_ids']
        if any(type(i) is not int or not 50 <= i < 150 for i in ids) or len(ids) != len(x):
            raise ValueError('invalid binary Iris source identities')
        np.testing.assert_array_equal(x, iris.data[ids].astype('float32'))
        np.testing.assert_array_equal(y, iris.target[ids] - 1)
        keys = {hashlib.sha256(row.tobytes()).hexdigest() for row in x}
        if groups & keys:
            raise ValueError('cross-partition measurement leakage')
        groups |= keys
        data[name] = x, y
    training_x = np.concatenate([data[n][0] for n in ('client-0', 'client-1')])
    training_y = np.concatenate([data[n][1] for n in ('client-0', 'client-1')])
    matrix = xgb.DMatrix(training_x, label=training_y, feature_names=list(FEATURES), feature_types=['float'] * 4)
    report = dict(xgboost_version=xgb.__version__, seed=seed, training_rows=len(training_y),
                  test_used_for_training=False, validation_used_for_training=False,
                  comparison='Eight centralized sequential trees versus eight federated trees from four parallel two-client rounds; eta .15 matches per-client shrinkage, eta .3 is the unscaled control.',
                  provenance_sha256=hashlib.sha256((data_directory / 'provenance.json').read_bytes()).hexdigest(), controls=[])
    output.mkdir(parents=True)
    for rate in (.15, .3):
        model = xgb.train(dict(objective='binary:logistic', base_score=.5, max_depth=3, eta=rate,
                              tree_method='hist', device='cpu', nthread=1, seed=seed % (2**31),
                              subsample=1, colsample_bytree=1), matrix, num_boost_round=8,
                          xgb_model=load_model(initial_model()))
        path = output / f'central-eta-{rate}.ubj'
        model.save_model(path)
        restored = xgb.Booster(model_file=path)
        result = dict(learning_rate=rate, trees=model.num_boosted_rounds(),
                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), reload_predictions_exact=True)
        for name in ('validation', 'test'):
            x, y = data[name]
            predicted = model.inplace_predict(x)
            np.testing.assert_array_equal(predicted, restored.inplace_predict(x))
            result[name] = metrics(y, predicted)
        report['controls'].append(result)
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = prepare(args.out, seed=args.seed)
    print(json.dumps({name: split['count'] for name, split in report['partitions'].items()}))


if __name__ == '__main__':
    main()
