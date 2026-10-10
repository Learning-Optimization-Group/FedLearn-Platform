import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fedlearn.tree import model
from fedlearn.tree.data import load_csv
from fedlearn.tree.contract import validate_contract
from fedlearn.tree.coordinator import TreeCoordinator
from test_rounds import contract

NAMES = tuple(prefix + "_" + measure for prefix in ("mean", "error", "worst") for measure in
              ("radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"))
ROOT = Path(__file__).resolve().parents[2]


def breast_parent():
    return model.initial_model(feature_names=NAMES)


def test_thirty_feature_training_reloads_and_rejects_cross_schema_delta():
    parent = breast_parent()
    x = np.random.default_rng(42).normal(size=(96, 30)).astype("float32")
    y = (x[:, 29] > 0).astype("float32")
    delta = model.train_delta(parent, x, y, cohort_size=2, seed=42)
    merged = model.merge_trees(parent, [("a", delta)])
    booster = model.load_model(merged)
    assert booster.feature_names == list(NAMES)
    assert np.mean((booster.inplace_predict(x) > .5) == y) > .9
    with pytest.raises(ValueError):
        model.merge_trees(model.initial_model(), [("a", delta)])
    with pytest.raises(ValueError):
        model.validate_model(merged, feature_names=model.FEATURES)
    with pytest.raises(ValueError):
        model.train_delta(parent, x[:, :4], y, cohort_size=2, seed=42)


@pytest.mark.parametrize("names", [NAMES[::-1], NAMES[:-1], (*NAMES[:-1], NAMES[0]), tuple("f"+str(i) for i in range(30))])
def test_unknown_reordered_missing_and_duplicate_schemas_refused(names):
    with pytest.raises(ValueError):
        model.initial_model(feature_names=names)


def test_contract_and_initial_model_must_share_the_same_schema():
    value, legacy = contract()
    value.tree_training.ClearField("feature_names")
    value.tree_training.feature_names.extend(NAMES)
    validate_contract(value)
    with pytest.raises(ValueError):
        TreeCoordinator(value, legacy)
    initial = bytes(model.load_model(breast_parent()).save_raw(raw_format="ubj"))
    value.tree_training.initial_model_sha256 = hashlib.sha256(initial).hexdigest()
    coordinator = TreeCoordinator(value, initial)
    assert model.load_model(coordinator.parent).feature_names == list(NAMES)


def test_named_csv_preserves_order_and_refuses_reordered_header_and_nonfinite(tmp_path):
    path = tmp_path / "local.csv"
    header = ",".join((*NAMES, "label"))
    path.write_text(header + "\n" + ",".join(["1"]*30 + ["0"]) + "\n")
    x, y, _ = load_csv(path, feature_names=NAMES)
    assert x.shape == (1, 30) and y.tolist() == [0]
    with pytest.raises(ValueError): load_csv(path)
    path.write_text(",".join((*NAMES[::-1], "label")) + "\n" + ",".join(["1"]*30 + ["0"]) + "\n")
    with pytest.raises(ValueError): load_csv(path, feature_names=NAMES)
    path.write_text(header + "\n" + ",".join(["nan"] + ["1"]*29 + ["0"]) + "\n")
    with pytest.raises(ValueError): load_csv(path, feature_names=NAMES)


def test_named_model_initialization_and_inference_pin_the_width(tmp_path):
    checkpoint = tmp_path / "initial.ubj"
    init = subprocess.run([sys.executable, str(ROOT/'fl-runtime/init_model.py'), '--model-type', 'XGBOOST',
        '--model-name', 'xgboost_breast_cancer', '--optimizer', 'TreeBoosting', '--out', str(checkpoint)], capture_output=True, text=True)
    assert init.returncode == 0, init.stderr
    payload, output = tmp_path/'input.json', tmp_path/'result.json'
    for values, expected in [([0]*30, True), ([0]*4, False), ([0]*29+[float('inf')], False)]:
        payload.write_text(json.dumps(dict(kind='vector', values=values)))
        result = subprocess.run([sys.executable, str(ROOT/'fl-runtime/infer.py'), '--model-type', 'XGBOOST',
            '--model-name', 'xgboost_breast_cancer', '--model-path', str(checkpoint), '--in', str(payload), '--out', str(output)], capture_output=True, text=True)
        actual = json.loads(output.read_text())
        assert actual['ok'] == expected
        assert (result.returncode == 0) == expected
        if not expected: assert actual['errorKind'] == 'input'
    payload.write_text(json.dumps(dict(kind='vector',values=[0]*4)))
    result = subprocess.run([sys.executable, str(ROOT/'fl-runtime/infer.py'), '--model-type', 'XGBOOST',
        '--model-name', 'xgboost_binary', '--model-path', str(checkpoint), '--in', str(payload), '--out', str(output)], capture_output=True,text=True)
    assert result.returncode != 0 and not json.loads(output.read_text())['ok']


def test_legacy_zero_tree_artifact_bytes_remain_compatible():
    artifact=bytes(model.load_model(model.initial_model()).save_raw(raw_format='ubj'))
    assert hashlib.sha256(artifact).hexdigest()=='dded417268dcaed8f155d63bb4bfcc428e8d2659a5fe894a5abe92aa25a0cee1'


def test_thirty_feature_coordinator_rejects_four_feature_contribution_before_acceptance():
    from fedlearn.communication.generated import tree_pb2 as p
    value, _ = contract()
    value.tree_training.ClearField('feature_names'); value.tree_training.feature_names.extend(NAMES)
    initial = bytes(model.load_model(breast_parent()).save_raw(raw_format='ubj'))
    value.tree_training.initial_model_sha256 = hashlib.sha256(initial).hexdigest()
    coordinator = TreeCoordinator(value, initial)
    coordinator.register('a', partition=0); coordinator.register('b', partition=1)
    x = np.ones((32,4),dtype='float32')
    delta = model.train_delta(model.initial_model(),x,np.zeros(32,dtype='float32'),cohort_size=2,seed=42)
    request = p.TreeContribution(client_id='a',run_id=value.run_id,contract_id=coordinator.contract_id,
        server_round=1,parent_sha256=hashlib.sha256(coordinator.parent).hexdigest(),
        delta_sha256=hashlib.sha256(delta).hexdigest(),new_trees_json=delta,num_examples=32)
    before = coordinator.parent
    with pytest.raises(ValueError): coordinator.submit(request)
    assert coordinator.parent == before and coordinator.accepted_count == 0 and coordinator.pending == {}


@pytest.mark.parametrize('kind',['width','split','cycle'])
def test_thirty_feature_tree_graph_and_shape_limits_are_checked_before_loading(kind):
    x = np.random.default_rng(42).normal(size=(96,30)).astype('float32')
    value = json.loads(model.train_delta(breast_parent(),x,(x[:,29]>0).astype('float32'),cohort_size=2,seed=42))
    tree = value['learner']['gradient_booster']['model']['trees'][0]
    if kind=='width': tree['tree_param']['num_feature']='31'
    elif kind=='split': tree['split_indices'][0]=30
    else: tree['left_children'][0]=0
    with pytest.raises(ValueError): model.validate_model(json.dumps(value).encode(),expected_trees=1)
