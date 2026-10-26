"""Independently reconcile frozen cohorts, client journals and aggregate exports."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from multihost_manifest import _digest, _json, verify_prepared, assignments
from run_multihost_benchmark import analyze


def read(path):
    return json.loads(Path(path).read_text())


def audit_cell(prepared,execution,cell):
    name=cell['name'];directory=execution/name
    summary=read(directory/'server/summary.json')
    receipts=read(directory/'server/receipts.json');rounds=read(directory/'server/rounds.json')
    if summary['status']!='complete' or summary['server_plan_sha256']!=cell['server_plan_sha256']:
        raise ValueError('server completion or frozen plan binding differs')
    if [r['round'] for r in receipts]!=list(range(1,cell['rounds']+1)) or len(rounds)!=len(receipts):
        raise ValueError('round accounting differs')
    expected_ids=[f'{name}:client-{cid:05d}' for cid in range(cell['num_clients'])]
    mapping=assignments(cell['num_clients'],cell['placement']);sizes={};host_pins={};total=0
    for host,ref in cell['hosts'].items():
        manifest=read(prepared/ref['manifest_path'])
        counts={member['client_id']:member['stop']-member['start'] for member in manifest['clients']}
        sizes.update(counts)
        row=read(directory/host/'summary.json')
        events=[json.loads(line) for line in (directory/host/'events.jsonl').read_text().splitlines()]
        if row['events']!=events or row['status']!='complete' or row['device']!='cpu':
            raise ValueError('raw host journal differs from its summary')
        expected={(cid,r) for cid in mapping[host] for r in range(1,cell['rounds']+1)}
        fits=[e for e in events if e['event']=='fit'];uploads=[e for e in events if e['event']=='upload']
        for collection in (fits,uploads):
            observed=[(e['client_id'],e['round']) for e in collection]
            if len(observed)!=len(expected) or set(observed)!=expected:
                raise ValueError('duplicate, missing or unexpected client-round event')
        if any(e.get('accepted') is not True for e in uploads) or any(e['event']=='error' for e in events):
            raise ValueError('host has an unaccepted upload or error')
        for event in fits:
            if event['batch_count']!=7*math.ceil(counts[event['client_id']]/256):
                raise ValueError('fit did not execute the frozen seven-epoch batch budget')
            if event['fit_elapsed_s']<0 or event['queue_s']<0:
                raise ValueError('invalid fit/queue timing')
        total+=len(fits)
        host_pins[host]={f'{key}_sha256':_digest(directory/host/file) for key,file in
                         (('summary','summary.json'),('events','events.jsonl'))}
    for receipt,evaluation in zip(receipts,rounds):
        if (receipt['client_ids']!=expected_ids or receipt['client_count']!=cell['num_clients']
                or receipt['num_examples']!=[sizes[cid] for cid in range(cell['num_clients'])]
                or receipt['assigned_hosts']!=sorted(cell['hosts'])
                or receipt['head_sha256']!=evaluation['head_sha256']
                or receipt['stats']['failed'] or receipt['stats']['aggregated']!=cell['num_clients']):
            raise ValueError('aggregate receipt differs from assigned clients, examples or evaluated head')
    with np.load(directory/'server/head.npz',allow_pickle=False) as head:
        digest=hashlib.sha256()
        for key,shape in (('head.weight',(2,cell['width'])),('head.bias',(2,))):
            array=head[key]
            if array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():
                raise ValueError('invalid final head')
            digest.update(key.encode());digest.update(str(shape).encode());digest.update(array.tobytes())
        if digest.hexdigest()!=summary['head_sha256'] or digest.hexdigest()!=receipts[-1]['head_sha256']:
            raise ValueError('independent final-head digest mismatch')
    pins={f'{key}_sha256':_digest(directory/'server'/file) for key,file in (
        ('summary','summary.json'),('receipts','receipts.json'),('rounds','rounds.json'),('head_npz','head.npz'))}
    return dict(completed_client_rounds=total,training_examples=sum(sizes.values()),
                output_commitments=dict(server=pins,hosts=host_pins))


def audit(prepared,execution,analysis,output,*,checkpoint,qualification=False,historical=None):
    torch.set_num_threads(1)
    prepared,execution=Path(prepared).absolute(),Path(execution).absolute()
    plan=verify_prepared(prepared);public=read(analysis);ledger=read(execution/'ledger.json')
    cells=plan['cells'] if not qualification else [c for c in plan['cells']
        if c['stage']=='systems' and c['num_clients']==3 and c['seed']==14001]
    if set(ledger)!={c['name'] for c in cells} or any(s['status']!='complete' for s in ledger.values()):
        raise ValueError('not every prospectively declared cell completed')
    if any(isinstance(s['elapsed_s'],bool) or not math.isfinite(s['elapsed_s']) or s['elapsed_s']<=0 for s in ledger.values()):
        raise ValueError('invalid recorded supervisor elapsed time')
    if public['public_manifest_sha256']!=plan['public_manifest_sha256']:
        raise ValueError('analysis manifest binding differs')
    if {c['name'] for c in public['cells']}!=set(ledger) or any(not c.get('selection_rescore_verified') for c in public['cells']):
        raise ValueError('not every final head passed independent AUROC/loss rescoring')
    observations={c['name']:audit_cell(prepared,execution,c) for c in cells}
    with tempfile.TemporaryDirectory(prefix='.audit-regenerated-',dir=execution) as temporary:
        regenerated=analyze(prepared,execution,Path(temporary)/'analysis.json',historical=historical)
    if json.dumps(public,sort_keys=True,allow_nan=False)!=json.dumps(regenerated,sort_keys=True,allow_nan=False):
        raise ValueError('public analysis differs from recomputed execution artifacts')
    if not qualification and any(c['training_examples']!=68650 for c in observations.values()):
        raise ValueError('medical training cohort size changed')
    pairs=public['placement_pairs'];replays=public['historical_replays']
    if len(pairs)!=len(cells)//2 or (not qualification and len(replays)!=4):
        raise ValueError('placement/reference comparison budget differs')
    passed=all(p['within_prospective_1e_5_tolerance'] for p in pairs+replays)
    result=dict(schema='capstone.multihost_verification.v1',date='2026-10-26',
        status='passed' if passed else 'numerical_acceptance_failed',synthetic=plan['synthetic'],
        source_checkpoint=checkpoint,public_manifest_sha256=plan['public_manifest_sha256'],
        public_analysis_sha256=_digest(analysis),private_ledger_sha256=_digest(execution/'ledger.json'),
        completed_cells=len(cells),completed_rounds=sum(c['rounds'] for c in cells),
        completed_client_rounds=sum(c['completed_client_rounds'] for c in observations.values()),
        selection_rescored_heads=len(cells),placement_pairs=len(pairs),historical_replays=len(replays),
        max_abs_placement_head_difference=max(p['max_abs_head_difference'] for p in pairs),
        max_abs_historical_head_difference=max((p['max_abs_head_difference'] for p in replays),default=None),
        independent_checks=['raw JSONL equals host summary','exact one fit and acknowledged upload per client round',
            'seven epochs including partial batches','cohort identities and exact example counts',
            'every aggregate head equals evaluated head','independent final float32 head digest',
            'all saved heads independently rescored for loss and AUROC',
            'entire public analysis regenerated from artifacts: metrics, unique pairs, ratios, head differences and acceptance flags'],
        observations=observations,postprocessing_source_sha256={Path(__file__).name:_digest(__file__)})
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    _json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','execution','analysis','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--qualification',action='store_true')
    p.add_argument('--historical',type=Path)
    a=p.parse_args();v=audit(a.prepared,a.execution,a.analysis,a.output,checkpoint=a.checkpoint,qualification=a.qualification,historical=a.historical)
    print(json.dumps({k:v[k] for k in ('status','completed_cells','completed_rounds','completed_client_rounds')}))
