import json
from pathlib import Path
import subprocess
import sys

from google.protobuf.json_format import MessageToJson
from test_rounds import contract

ROOT=Path(__file__).resolve().parents[2]


def test_canonical_client_refuses_mismatched_contract_before_opening_data_or_network(tmp_path):
    value,_=contract();value.tree_training.feature_names[0]='wrong'
    path=tmp_path/'contract.json';path.write_text(MessageToJson(value))
    result=subprocess.run([sys.executable,str(ROOT/'fl-runtime/client.py'),'--model-type','XGBOOST',
        '--model-name','xgboost_binary','--project-id',value.project_id,'--run-id',value.run_id,
        '--server-address','127.0.0.1:1','--partition-id','0','--execution-contract',str(path),
        '--dataset-path',str(tmp_path/'nonexistent.csv')],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'features/classes' in result.stderr
    assert 'FileNotFoundError' not in result.stderr and 'UNAVAILABLE' not in result.stderr


def test_tree_client_import_does_not_initialize_torch():
    result=subprocess.run([sys.executable,'-c',
        "import fedlearn.tree.client,sys; assert 'torch' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
