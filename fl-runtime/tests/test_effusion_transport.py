"""Generated data traverse the real canonical gRPC server and executable client."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch
import pytest

from test_effusion_runtime import pack as host_pack, load, reference
from test_effusion_server import pack as server_pack


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_canonical_transport_matches_simulator_heads(tmp_path):
    torch.set_num_threads(1)
    data_dir, eval_dir = tmp_path/'data', tmp_path/'selection'
    data_dir.mkdir();eval_dir.mkdir()
    host_path, _, _, _ = host_pack(data_dir,width=128,counts=(259,263,267))
    server_path, plan = server_pack(eval_dir)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    plan['bind_address']=f'127.0.0.1:{port}'
    plan['client_hosts']={cid:'mac' for cid in plan['client_ids']}
    server_path.write_text(json.dumps(plan))
    root=Path(__file__).resolve().parents[2]
    env=dict(os.environ,PYTHONPATH=f'{root}/framework/src:{root}/fl-runtime',
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    output=tmp_path/'server-output';host_output=tmp_path/'host-output'
    with (tmp_path/'server.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(root/'fl-runtime/effusion_server.py'),
            '--server-plan',str(server_path),'--server-plan-sha256',sha(server_path),
            '--output',str(output)],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise AssertionError((tmp_path/'server.log').read_text())
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:time.sleep(.05)
            else:raise AssertionError('canonical server did not bind')
            client=subprocess.run([sys.executable,str(root/'fl-runtime/client.py'),
                '--effusion-host-plan',str(host_path),'--host-plan-sha256',sha(host_path),
                '--server-address',plan['bind_address'],'--output',str(host_output)],
                env=env,capture_output=True,text=True,timeout=60)
            assert client.returncode==0,client.stdout+'\n'+client.stderr
            assert process.wait(timeout=15)==0,(tmp_path/'server.log').read_text()
        finally:
            if process.poll() is None:
                process.terminate();process.wait(timeout=10)
    result=json.loads((output/'summary.json').read_text())
    assert result['status']=='complete' and result['completed_rounds']==5
    receipts=json.loads((output/'receipts.json').read_text())
    assert all(row['client_ids']==plan['client_ids'] for row in receipts)
    host=json.loads((host_output/'summary.json').read_text())
    assert host['status']=='complete' and host['completed_client_rounds']==15
    assert host['peak_active_fits']<=2
    assert any(e['event']=='upload' and e['accepted'] for e in host['events'])
    from fedlearn.simulation.rng import ClientRng,torch_rng_scope
    from fedlearn.server.composed_strategy import ComposedStrategy
    data=load(host_path);clients=reference(data)
    parameters=OrderedDict([('head.weight',torch.zeros((2,128))),('head.bias',torch.zeros(2))])
    strategy=ComposedStrategy(parameters,rule='mean',clip_norm=.05,min_fit_clients=3,clients_per_round=3)
    for current in range(1,6):
        updates=[]
        for cid,client in enumerate(clients):
            with torch_rng_scope(ClientRng(data.manifest['seed'],cid).torch_seed(current)):
                head,count=client.fit(parameters,{})
            updates.append((cid,head,count))
        parameters=strategy.aggregate_fit(current,updates)
    saved=np.load(output/'head.npz',allow_pickle=False)
    assert max(float(np.max(np.abs(saved[k]-v.numpy()))) for k,v in parameters.items())<=1e-5


def test_real_transport_rejects_duplicate_stale_updates_and_missing_client_cannot_publish_head(tmp_path,monkeypatch):
    import effusion_server
    from fedlearn.client.grpc_client import GrpcClient
    torch.set_num_threads(1)
    server_path,plan=server_pack(tmp_path)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    plan['bind_address']=f'127.0.0.1:{port}';server_path.write_text(json.dumps(plan))
    @contextmanager
    def bounded_generated_deadline(n):
        with monkeypatch.context() as changes:
            changes.setenv('MAX_CLIENTS',str(n))
            changes.setenv('FEDLEARN_ROUND_TIMEOUT_S','6')
            yield
    monkeypatch.setattr(effusion_server,'_server_environment',bounded_generated_deadline)
    out=tmp_path/'failed-server'
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(effusion_server.execute,server_path,sha(server_path),out)
        clients=[GrpcClient(cid,plan['bind_address']) for cid in plan['client_ids']]
        try:
            deadline=time.monotonic()+5
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                except OSError:
                    if time.monotonic()>deadline:raise AssertionError('server readiness failed')
                    time.sleep(.05)
            assert all(client.register() for client in clients)
            outsider=GrpcClient('unknown-logical-client',plan['bind_address'])
            try:
                assert not outsider.register()
            finally:outsider.close()
            parameters=OrderedDict([('head.weight',torch.zeros((2,128))),('head.bias',torch.zeros(2))])
            assert not clients[0].submit_update(parameters,5,0)
            assert not clients[0].submit_update(parameters,5,2)
            assert clients[0].submit_update(parameters,5,1)
            assert not clients[0].submit_update(parameters,5,1)
            assert clients[1].submit_update(parameters,9,1)
            clients[2].close()  # Simulate a registered client exiting without an upload.
            with pytest.raises(RuntimeError,match='failed'):
                future.result(timeout=15)
        finally:
            for client in clients:client.close()
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='failed' and summary['completed_rounds']==0
    assert not (out/'head.npz').exists() and not (out/'receipts.json').exists()
    assert json.loads((out/'failed-receipts.json').read_text())==[]
