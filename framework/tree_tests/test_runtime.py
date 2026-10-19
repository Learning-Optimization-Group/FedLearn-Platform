import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fedlearn.tree.data import load_csv

ROOT=Path(__file__).resolve().parents[2]


def test_canonical_init_and_inference_use_tree_artifacts(tmp_path):
    model=tmp_path/'initial.ubj'
    init=subprocess.run([sys.executable,str(ROOT/'fl-runtime/init_model.py'),
        '--model-type','XGBOOST','--model-name','xgboost_binary','--optimizer','TreeBoosting',
        '--out',str(model),'--pretrain-epochs','0'],capture_output=True,text=True)
    assert init.returncode==0,init.stderr
    payload=tmp_path/'input.json';payload.write_text(json.dumps({'kind':'vector','values':[0,0,0,0]}))
    out=tmp_path/'out.json'
    infer=subprocess.run([sys.executable,str(ROOT/'fl-runtime/infer.py'),
        '--model-type=XGBOOST','--model-name','xgboost_binary','--model-path',str(model),
        '--in',str(payload),'--out',str(out)],capture_output=True,text=True)
    assert infer.returncode==0,infer.stderr
    result=json.loads(out.read_text())
    assert result['ok'] and result['classes']==['0','1']
    assert result['probabilities']==[.5,.5]
    payload.write_text(json.dumps({'kind':'vector','values':[1,2,3]}))
    infer=subprocess.run([sys.executable,str(ROOT/'fl-runtime/infer.py'),
        '--model-type','XGBOOST','--model-name','xgboost_binary','--model-path',str(model),
        '--in',str(payload),'--out',str(out)],capture_output=True,text=True)
    assert infer.returncode!=0 and json.loads(out.read_text())['errorKind']=='input'


def test_csv_preserves_all_local_rows_and_pins_order(tmp_path):
    path=tmp_path/'local.csv';path.write_text('f0,f1,f2,f3,label\n1,2,3,4,0\n5,6,7,8,1\n')
    x,y,digest=load_csv(path)
    np.testing.assert_array_equal(x,[[1,2,3,4],[5,6,7,8]])
    np.testing.assert_array_equal(y,[0,1]); assert len(digest)==64
    path.write_text('f1,f0,f2,f3,label\n1,2,3,4,0\n')
    with pytest.raises(ValueError):load_csv(path)


@pytest.mark.parametrize('row',['nan,2,3,4,0','1,2,3,4,2','1,2,3,4','1e100,2,3,4,0'])
def test_csv_refuses_invalid_features_labels_and_width(tmp_path,row):
    path=tmp_path/'local.csv';path.write_text('f0,f1,f2,f3,label\n'+row+'\n')
    with pytest.raises(ValueError):load_csv(path)


def test_tree_server_exits_when_rpc_already_failed_cohort(tmp_path, monkeypatch):
    import base64
    from test_rounds import contract
    from fedlearn.tree.coordinator import TreeCoordinator
    from fedlearn.communication.generated import tree_pb2 as p

    monkeypatch.syspath_prepend(str(ROOT/'fl-runtime'))
    import tree_runtime

    value, initial = contract()
    now = [100.0]
    coordinator = TreeCoordinator(value, initial, clock=lambda: now[0])
    now[0] += 2
    # A client RPC can observe the deadline before the server's polling thread.
    with pytest.raises(TimeoutError):
        coordinator.check_deadline()
    assert coordinator.state == p.TreeModel.FAILED
    monkeypatch.setattr(tree_runtime, 'TreeCoordinator', lambda *_: coordinator)
    monkeypatch.setattr(coordinator.condition, 'wait',
                        lambda *_: pytest.fail('server kept waiting after the cohort failed'))
    monkeypatch.setattr(tree_runtime.requests, 'post',
                        lambda *_, **__: pytest.fail('failed run registered a checkpoint or completion'))
    for name in ('FEDLEARN_REQUIRE_TLS', 'FEDLEARN_GRPC_USE_TLS',
                 'FEDLEARN_REQUIRE_CLIENT_AUTH', 'FEDLEARN_CONNECTION_TOKEN_SECRET'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('FEDLEARN_INTERNAL_API_KEY', 'test-key')
    monkeypatch.setenv('FEDLEARN_INTERNAL_RUN_TOKEN', 'test-scoped-token')
    monkeypatch.setenv('FEDLEARN_BACKEND_URL', 'http://127.0.0.1:1')
    monkeypatch.setenv('FEDLEARN_TREE_CONTRACT_B64',
                       base64.b64encode(value.SerializeToString()).decode())
    initial_path = tmp_path/'initial.ubj'
    initial_path.write_bytes(initial)
    final_path = tmp_path/'final.ubj'
    monkeypatch.setattr(sys, 'argv', ['server', '--model-type', 'XGBOOST',
        '--model-name', 'xgboost_binary', '--project-id', value.project_id,
        '--init-model-path', str(initial_path), '--model-path', str(final_path),
        '--port', '0', '--strategy', 'XgbBagging', '--num-rounds', '2', '--min-clients', '2'])
    with pytest.raises(RuntimeError, match='tree run failed'):
        tree_runtime.server_main()
    assert not final_path.exists()
    assert coordinator.state == p.TreeModel.FAILED
