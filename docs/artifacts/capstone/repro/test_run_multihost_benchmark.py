"""Supervisor and public exports must fail closed on incomplete transport evidence."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def module():
    return importlib.import_module('run_multihost_benchmark')


def test_private_launch_config_requires_both_distinct_remote_targets(tmp_path):
    config=dict(local_python='/local/python', hosts={host:dict(
        ssh=['ssh',host], python='/remote/python', root='/remote/study') for host in ('g14','orin')})
    path=tmp_path/'launch.json';path.write_text(json.dumps(config))
    assert module().load_launch_config(path)==config
    config['hosts']['orin']['ssh']=['ssh','g14']
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='distinct'):
        module().load_launch_config(path)


def test_owned_cleanup_verifies_exact_plan_and_executable_before_signalling():
    check=module().owned_process_matches
    assert check(['python','/s/fl-runtime/client.py','--effusion-host-plan','/d/host.json'],
                 '/s/fl-runtime/client.py','/d/host.json')
    assert not check(['python','/s/fl-runtime/client.py','--effusion-host-plan','/other/host.json'],
                     '/s/fl-runtime/client.py','/d/host.json')
    assert not check(['python','/other/client.py','--effusion-host-plan','/d/host.json'],
                     '/s/fl-runtime/client.py','/d/host.json')


def test_public_cell_export_requires_full_server_and_host_receipts(tmp_path):
    cell=dict(name='generated',num_clients=3,rounds=5,placement='three_hosts',width=128,
              method='projected128',seed=14001,stage='systems')
    server=tmp_path/'server';server.mkdir()
    from effusion_server import head_sha256
    pin=head_sha256({'head.weight':torch.zeros((2,128)),'head.bias':torch.zeros(2)})
    (server/'summary.json').write_text(json.dumps(dict(status='complete',completed_rounds=5,head_sha256=pin)))
    (server/'rounds.json').write_text(json.dumps([dict(round=i,auc=.5,round_completed_elapsed_s=float(i))
                                               for i in range(1,6)]))
    ids=[f'generated:client-{i:05d}' for i in range(3)]
    (server/'receipts.json').write_text(json.dumps([dict(round=i,client_ids=ids,client_count=3,head_sha256=pin)
                                                 for i in range(1,6)]))
    np.savez(server/'head.npz',**{'head.weight':np.zeros((2,128),np.float32),'head.bias':np.zeros(2,np.float32)})
    for cid,host in enumerate(('mac','g14','orin')):
        d=tmp_path/host;d.mkdir()
        (d/'summary.json').write_text(json.dumps(dict(status='complete',run_id='generated',host_label=host,
            client_ids=[cid],completed_client_rounds=5,planned_client_rounds=5,duplicate_fit_calls=0,
            peak_active_fits=1,peak_process_rss_bytes=123,runtime=dict(architecture='test'),
            events=[dict(event='fit',queue_s=.2)])))
    out=module().summarize_cell(tmp_path,cell,elapsed_s=8)
    assert out['status']=='complete' and out['steady_round_mean_s']==1
    assert out['selection_auc']==.5 and out['completed_client_rounds']==15
    assert out['hosts']['orin']['queue_wait_sum_s']==pytest.approx(.2)
    assert 'client_ids' not in json.dumps(out) and str(tmp_path) not in json.dumps(out)
    np.savez(server/'head.npz',**{'head.weight':np.ones((2,128),np.float32),'head.bias':np.zeros(2,np.float32)})
    with pytest.raises(ValueError,match='hash'):
        module().summarize_cell(tmp_path,cell,elapsed_s=8)
    np.savez(server/'head.npz',**{'head.weight':np.zeros((2,128),np.float32),'head.bias':np.zeros(2,np.float32)})
    (tmp_path/'orin/summary.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        module().summarize_cell(tmp_path,cell,elapsed_s=8)


def test_head_comparison_checks_shapes_and_finite_values(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    np.savez(a,**{'head.weight':np.zeros((2,128),np.float32),'head.bias':np.zeros(2,np.float32)})
    np.savez(b,**{'head.weight':np.full((2,128),1e-6,np.float32),'head.bias':np.zeros(2,np.float32)})
    assert module().compare_heads(a,b)==pytest.approx(1e-6)
    np.savez(b,**{'head.weight':np.zeros((2,512),np.float32),'head.bias':np.zeros(2,np.float32)})
    with pytest.raises(ValueError):module().compare_heads(a,b)


def test_remote_cleanup_checks_owned_process_even_after_ssh_exits(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(module(),'_stop_remote',lambda *args:calls.append(args))
    monkeypatch.setattr(module(),'_fetch',lambda *args: (_ for _ in ()).throw(FileNotFoundError()))
    class ExitedSSH:
        def poll(self):return 255
    errors=module().cleanup_remote_jobs([('orin',Path('/r'),Path('/s/client.py'),Path('/r/data/host.json'),ExitedSSH())],
                                      {'hosts':{'orin':{'test':True}}},tmp_path)
    assert errors==[] and len(calls)==1
