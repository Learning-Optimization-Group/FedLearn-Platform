"""Independent audit must detect plausible but inconsistent completion artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from audit_multihost_results import audit_cell


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('damage',['batch_budget','example_count','head_value'])
def test_audit_rejects_inconsistent_batch_count_examples_or_final_head(tmp_path,damage):
    prepared=tmp_path/'prepared';execution=tmp_path/'execution';name='fixture'
    cell=dict(name=name,rounds=1,num_clients=1,width=128,placement='mac_only',server_plan_sha256='a'*64,
              hosts={'mac':dict(manifest_path='host.json')})
    write(prepared/'host.json',dict(clients=[dict(client_id=0,start=0,stop=259)]))
    events=[dict(event='fit',client_id=0,round=1,batch_count=14,fit_elapsed_s=.1,queue_s=.01),
            dict(event='upload',client_id=0,round=1,accepted=True)]
    host=execution/name/'mac';server=execution/name/'server'
    summary=dict(events=events,status='complete',device='cpu')
    write(host/'summary.json',summary)
    (host/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    head={'head.weight':np.zeros((2,128),np.float32),'head.bias':np.zeros(2,np.float32)}
    digest=hashlib.sha256()
    for key in ('head.weight','head.bias'):
        digest.update(key.encode());digest.update(str(head[key].shape).encode());digest.update(head[key].tobytes())
    pin=digest.hexdigest()
    receipt=dict(round=1,client_ids=['fixture:client-00000'],client_count=1,num_examples=[259],assigned_hosts=['mac'],
                 head_sha256=pin,stats=dict(failed=False,aggregated=1))
    write(server/'summary.json',dict(status='complete',server_plan_sha256='a'*64,head_sha256=pin))
    write(server/'receipts.json',[receipt]);write(server/'rounds.json',[dict(round=1,head_sha256=pin)])
    np.savez(server/'head.npz',**head)
    assert audit_cell(prepared,execution,cell)['completed_client_rounds']==1
    if damage=='batch_budget':
        events[0]['batch_count']=13
        write(host/'summary.json',summary)
        (host/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    elif damage=='example_count':
        receipt['num_examples']=[260];write(server/'receipts.json',[receipt])
    else:
        head['head.weight'][0,0]=.01;np.savez(server/'head.npz',**head)
    with pytest.raises(ValueError):audit_cell(prepared,execution,cell)


@pytest.mark.parametrize('damage',['auc','speedup','tolerance','duplicate'])
def test_audit_rejects_public_export_that_differs_from_regenerated_artifacts(tmp_path,monkeypatch,damage):
    import copy
    import audit_multihost_results as module
    cells=[dict(name=f'{method}_{placement}',rounds=5,num_clients=3,seed=14001,stage='systems')
           for method in ('mean512','projected128') for placement in ('mac_only','three_hosts')]
    plan=dict(cells=cells,synthetic=True,public_manifest_sha256='b'*64)
    ledger={c['name']:dict(status='complete',elapsed_s=1) for c in cells}
    execution=tmp_path/'execution';write(execution/'ledger.json',ledger)
    expected=dict(public_manifest_sha256='b'*64,
        cells=[dict(name=c['name'],selection_rescore_verified=True,selection_auc=.7) for c in cells],
        placement_pairs=[dict(three_host_cell=f'{m}_three_hosts',max_abs_head_difference=1e-8,
                             within_prospective_1e_5_tolerance=True,elapsed_speedup=2) for m in ('mean512','projected128')],
        historical_replays=[])
    changed=copy.deepcopy(expected)
    if damage=='auc':changed['cells'][0]['selection_auc']=.9
    elif damage=='speedup':changed['placement_pairs'][0]['elapsed_speedup']=99
    elif damage=='tolerance':changed['placement_pairs'][0]['max_abs_head_difference']=.1
    else:changed['placement_pairs'][1]=copy.deepcopy(changed['placement_pairs'][0])
    analysis=tmp_path/'analysis.json';write(analysis,changed)
    monkeypatch.setattr(module,'verify_prepared',lambda *a:plan)
    monkeypatch.setattr(module,'audit_cell',lambda *a:dict(completed_client_rounds=15,training_examples=4,output_commitments={}))
    monkeypatch.setattr(module,'analyze',lambda *a,**k:expected,raising=False)
    with pytest.raises(ValueError,match='recomputed|regenerated'):
        module.audit(tmp_path/'prepared',execution,analysis,tmp_path/'verification.json',checkpoint='fixture',qualification=True)
