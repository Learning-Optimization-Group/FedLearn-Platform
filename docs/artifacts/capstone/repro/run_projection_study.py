"""One-attempt generated projection experiment with source-bound manifests."""
import argparse
from collections import OrderedDict
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import ndtr
import torch
from torch.nn import functional as F

from fedlearn.simulation.owned_data_attack import ascent_upload
from projection_study_design import Cell, SETTINGS, cells, qualification_cells, seed_for
from projection_study_metrics import metrics
from projection_study_strategy import StudyStrategy, initial_parameters, local_fit, score
from projection_study_workload import basis, coalition, dataset, features, partitions

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SPEC=ROOT/'docs/superpowers/specs/2026-10-27-projection-mechanism-design.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def read_cell(raw):
    return Cell(**{f.name:raw[f.name] for f in fields(Cell)})


def sources():
    paths=list((ROOT/'framework/src/fedlearn').rglob('*.py'))
    for pattern in ('projection_study*.py','test_projection_study*.py','*run_projection_study.py',
                    '*analyze_projection_study.py','plot_projection_study.py'):
        paths.extend(HERE.glob(pattern))
    paths.extend([HERE/'capstone_pilot_mechanisms.py',SPEC])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def calibration_check():
    strategy=StudyStrategy(initial_parameters(128),seed=1,private=True)
    z=strategy.z;mu=math.sqrt(20)/z
    delta=float(ndtr(-4/mu+mu/2)-math.exp(4)*ndtr(-4/mu-mu/2))
    if abs(delta-1e-5)>1e-12:raise ValueError('independent GDP conversion failed')
    return dict(z=z,mu=mu,delta=delta,sensitivity=.005,sigma=z*.005)


def preflight(output,stage,qualification=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if stage not in ('qualification','main'):raise ValueError('unknown stage')
    rows=qualification_cells() if stage=='qualification' else cells()
    q=None
    if stage=='main':
        if qualification is None:raise ValueError('qualification directory required')
        q=validate_qualification(Path(qualification))
        if q['projected_main_s']>43200:raise ValueError('main runtime ceiling exceeded')
        old=json.loads((Path(qualification)/'frozen-plan.json').read_text())['sources']
        now=sources()
        if any(now.get(p)!=v for p,v in old.items()):raise ValueError('qualified source changed')
    data=dataset()
    plan=dict(schema='capstone.projection.plan.v1',stage=stage,settings=SETTINGS,
              synthetic=True,cells=[c.record() for c in rows],sources=sources(),
              data={k:tensor_sha(t) for k,t in data.items()},
              bases={c.name:tensor_sha(basis(c)) for c in rows},
              calibration=calibration_check(),qualification=q,
              runtime={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy')})
    write_json(output/'plan.json',plan);(output/'plan.sha256').write_text(sha(output/'plan.json')+'\n')
    return plan


def verify_plan(path):
    path=Path(path);p=path/'plan.json';plan=json.loads(p.read_text())
    if sha(p)!=(path/'plan.sha256').read_text().strip():raise ValueError('manifest digest mismatch')
    expected=qualification_cells() if plan['stage']=='qualification' else cells()
    if plan['cells']!=[c.record() for c in expected] or plan['settings']!=SETTINGS \
            or plan['sources']!=sources() or plan.get('synthetic') is not True:
        raise ValueError('resolved design or source drift')
    data=dataset()
    if plan['data']!={k:tensor_sha(t) for k,t in data.items()}:
        raise ValueError('generated dataset drift')
    if plan['bases']!={c.name:tensor_sha(basis(c)) for c in expected}:
        raise ValueError('projection basis drift')
    if plan['calibration']!=calibration_check():raise ValueError('calibration drift')
    return plan


def attack_check():
    args=dict(weight=torch.zeros((2,128)),bias=torch.zeros(2),
              owned_x=torch.cat((-torch.ones((8,1)),torch.zeros((8,127))),1),
              owned_y=torch.zeros(8,dtype=torch.int64),initial_delta=torch.zeros(258),
              clip_norm=.05,coalition_fraction=.2,seed=31415,restarts=3,step_ratio=.25)
    values=[];previous=-math.inf
    for steps in (20,80,160):
        r=ascent_upload(**args,steps=steps)
        assert r['gradient_evaluations']==steps*3 and r['loss_evaluations']==(steps+1)*3
        v=r['delta'];logits=args['owned_x'].double()@(.2*v[:256].reshape(2,128)).T+.2*v[256:]
        loss=float(F.cross_entropy(logits,args['owned_y']))
        if abs(loss-r['objective_after'])>1e-12 or loss<previous-1e-12:
            raise ValueError('attack prefix or independently recomputed objective failed')
        previous=loss;values.append(loss)
    # A separate known classifier checks actual damaging direction on generated examples.
    args.update(weight=torch.tensor([[-.02]+[0.]*127,[.02]+[0.]*127]),
                initial_delta=torch.tensor([.05]+[0.]*127+[-.05]+[0.]*129),clip_norm=.5)
    r=ascent_upload(**args,steps=20)
    clean=float(F.cross_entropy(args['owned_x']@args['weight'].T,args['owned_y']))
    v=r['delta'];candidate=torch.cat((args['weight'].flatten(),args['bias'])).double()+.2*v
    damaged=float(F.cross_entropy(args['owned_x'].double()@candidate[:256].reshape(2,128).T+candidate[256:],args['owned_y']))
    if damaged<=clean:raise ValueError('generated attack positive control failed')
    return dict(qualified=True,budgets=[60,240,480],objectives=values,
                clean_loss=clean,attacked_loss=damaged)


def _flat(p):
    return torch.cat([p[k].flatten() for k in ('head.weight','head.bias')])


def _attack(parameters,uploads,data,parts,cell,round,history):
    owned=coalition(cell.seed);indices=torch.cat([parts[i] for i in owned])
    own_delta=torch.stack([_flat(uploads[i][1])-_flat(parameters) for i in owned]).mean(0).double()
    initial=-own_delta*.05/torch.linalg.vector_norm(own_delta).clamp_min(1e-300)
    result=ascent_upload(weight=parameters['head.weight'],bias=parameters['head.bias'],
              owned_x=data['train_x'][indices],owned_y=data['train_y'][indices],initial_delta=initial,
              clip_norm=.05,coalition_fraction=.2,seed=seed_for(cell.seed,'attack',round),
              released_delta=history,steps=cell.steps,restarts=3,step_ratio=.25)
    delta=result.pop('delta');width=cell.width
    malicious=OrderedDict([('head.weight',parameters['head.weight']+delta[:2*width].float().reshape(2,width)),
                           ('head.bias',parameters['head.bias']+delta[2*width:].float())])
    result.update(coalition=list(owned),owned_count=len(indices),owned_indices_sha256=tensor_sha(indices))
    return [(cid,malicious if int(cid) in owned else p,n) for cid,p,n in uploads],result


def run_cell(cell,data,out,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'rounds').mkdir()
    start=time.monotonic();rounds=[];parameters=initial_parameters(cell.width)
    strategy=StudyStrategy(parameters,seed=cell.seed,private=cell.private)
    parts=partitions(cell.seed);history=None
    try:
        for round in range(1,21):
            uploads=[]
            for cid,index in enumerate(parts):
                if time.monotonic()>deadline:raise TimeoutError('declared execution budget exhausted')
                p,n=local_fit(parameters,data['train_x'][index],data['train_y'][index],cell.seed,round,cid)
                uploads.append((str(cid),p,n))
            clean= strategy.preview(uploads)
            attack=None
            if cell.steps:uploads,attack=_attack(parameters,uploads,data,parts,cell,round,history)
            if time.monotonic()>deadline:raise TimeoutError('declared execution budget exhausted')
            previous=parameters;parameters=strategy.aggregate_fit(round,uploads)
            if parameters is None:raise RuntimeError('fixed cohort refused')
            history=_flat(parameters)-_flat(previous)
            delta=strategy.last_pre_noise['head.weight']-previous['head.weight']
            effective=(delta[1]-delta[0]).double()
            if cell.family=='full':lift=effective
            else:lift=basis(cell)@effective[:128]
            norm=float(torch.linalg.vector_norm(lift))
            std=strategy.last_round_stats['noise_std'];x=data['eval_x']
            noise=strategy.last_noise;ns=score(noise,x)
            row=dict(round=round,**strategy.last_round_stats,attack=attack,
                     sample_counts=[n for _,_,n in uploads],teacher_alignment=float(lift[0])/norm if norm else None,
                     parameter_noise_energy=float(_flat(noise).double().square().sum()),
                     active_noise_energy=float(noise['head.weight'][:,:128].double().square().sum()+noise['head.bias'].double().square().sum()),
                     expected_parameter_noise_energy=(2*cell.width+2)*std**2,
                     predicted_score_noise_variance=float(2*std**2*(x.double().square().sum(1)+1).mean()),
                     realized_score_noise_mse=float(ns.square().mean()),
                     clean_aggregate_norm=float(torch.linalg.vector_norm(_flat(clean)-_flat(previous))))
            raww=torch.stack([p['head.weight']-previous['head.weight'] for _,p,_ in uploads])
            rawb=torch.stack([p['head.bias']-previous['head.bias'] for _,p,_ in uploads])
            np.savez_compressed(out/'rounds'/f'{round:02}.npz',weight=parameters['head.weight'].numpy(),
                bias=parameters['head.bias'].numpy(),pre_weight=strategy.last_pre_noise['head.weight'].numpy(),
                pre_bias=strategy.last_pre_noise['head.bias'].numpy(),noise_weight=noise['head.weight'].numpy(),
                noise_bias=noise['head.bias'].numpy(),raw_weight=raww.numpy(),raw_bias=rawb.numpy())
            row['arrays_sha256']=sha(out/'rounds'/f'{round:02}.npz');rounds.append(row)
            write_json(out/'rounds.json',rounds)
        scores=score(parameters,data['eval_x']).numpy()
        result=dict(**cell.record(),status='completed',rounds=20,client_fits=400,
                    metrics=metrics(data['eval_y'].numpy(),scores),elapsed_s=time.monotonic()-start)
        np.savez_compressed(out/'head.npz',weight=parameters['head.weight'].numpy(),bias=parameters['head.bias'].numpy())
        np.savez_compressed(out/'predictions.npz',scores=scores,y=data['eval_y'].numpy())
        result['artifacts']={p.name:sha(p) for p in (out/'head.npz',out/'predictions.npz',out/'rounds.json')}
        write_json(out/'result.json',result)
        return result
    except BaseException as error:
        (out/'head.npz').unlink(missing_ok=True);(out/'result.json').unlink(missing_ok=True)
        write_json(out/'failure.json',dict(status='failed',error=f'{type(error).__name__}: {error}',completed_rounds=len(rounds)))
        raise


def validate_qualification(out):
    out=Path(out);summary=json.loads((out/'summary.json').read_text())
    expected=qualification_cells()
    if summary['status']!='completed' or summary['stage']!='qualification' or len(summary['runs'])!=12:
        raise ValueError('complete qualification required')
    for c,r in zip(expected,summary['runs'],strict=True):
        if c.record()!={k:r[k] for k in c.record()}:raise ValueError('qualification cell drift')
        path=out/'runs'/c.name
        if sha(path/'result.json')!=summary['result_hashes'][c.name]:raise ValueError('qualification result changed')
        for file,digest in r['artifacts'].items():
            if sha(path/file)!=digest:raise ValueError('qualification artifact changed')
    checks={}
    for left,right in ((0,1),(3,4),(5,6)):
        a=out/'runs'/expected[left].name;b=out/'runs'/expected[right].name
        for round in range(1,21):
            x=np.load(a/'rounds'/f'{round:02}.npz');y=np.load(b/'rounds'/f'{round:02}.npz')
            for key in ('weight','pre_weight','noise_weight'):
                difference=float(np.max(np.abs(x[key][:,:128]-y[key][:,:128])))
                if difference>1e-5:raise ValueError('qualification padding/replay discrepancy')
            if float(np.max(np.abs(x['bias']-y['bias'])))>1e-5:raise ValueError('qualification bias discrepancy')
        ax=np.load(a/'predictions.npz')['scores'];bx=np.load(b/'predictions.npz')['scores']
        difference=float(np.max(np.abs(ax-bx)))
        if difference>1e-5:raise ValueError('qualification prediction discrepancy')
        checks[f'{left+1}-{right+1}']=difference
    return dict(status='qualified',summary_sha256=sha(out/'summary.json'),padding_replay_max_score_discrepancies=checks,
                projected_main_s=max(r['elapsed_s'] for r in summary['runs'])*136,
                elapsed_s=summary['elapsed_s'],cells=12)


def execute(plan_dir,output,checkpoint):
    plan_dir,output=Path(plan_dir),Path(output);plan=verify_plan(plan_dir)
    for path,digest in plan['sources'].items():
        content=subprocess.check_output(['git','show',f'{checkpoint}:{path}'],cwd=ROOT,stderr=subprocess.PIPE)
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('source differs from checkpoint')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'frozen-plan.json',plan)
    budget=1800 if plan['stage']=='qualification' else 43200
    start=time.monotonic();deadline=start+budget;rows=[];hashes={};cache={};data=dataset()
    summary=dict(status='running',stage=plan['stage'],checkpoint=checkpoint,plan_sha256=sha(plan_dir/'plan.json'),runs=rows,result_hashes=hashes)
    write_json(output/'summary.json',summary)
    try:
        if plan['stage']=='qualification':write_json(output/'attack-check.json',attack_check())
        for raw in plan['cells']:
            c=read_cell(raw);key=(c.family,c.eta,c.projection_seed)
            if key not in cache:cache[key]=features(data,c)
            result=run_cell(c,cache[key],output/'runs'/c.name,deadline)
            rows.append(result);hashes[c.name]=sha(output/'runs'/c.name/'result.json')
            summary['elapsed_s']=time.monotonic()-start;write_json(output/'summary.json',summary)
            print(json.dumps(dict(completed=len(rows),total=len(plan['cells']),name=c.name,elapsed_s=result['elapsed_s'])),flush=True)
        summary['status']='completed';summary['elapsed_s']=time.monotonic()-start
        write_json(output/'summary.json',summary)
        if plan['stage']=='qualification':write_json(output/'qualification.json',validate_qualification(output))
        return summary
    except BaseException as error:
        summary.update(status='failed',elapsed_s=time.monotonic()-start,error=f'{type(error).__name__}: {error}')
        write_json(output/'summary.json',summary);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    pre=sub.add_parser('preflight');pre.add_argument('--stage',choices=('qualification','main'),required=True)
    pre.add_argument('--qualification',type=Path);pre.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('execute');run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True);run.add_argument('--checkpoint',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    if args.mode=='preflight':
        result=preflight(args.output,args.stage,args.qualification);print(json.dumps(dict(cells=len(result['cells']),model_fits=0)))
    else:execute(args.plan,args.output,args.checkpoint)


if __name__=='__main__':main()
