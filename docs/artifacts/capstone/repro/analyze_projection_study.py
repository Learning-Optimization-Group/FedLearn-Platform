"""Independent rank, clipping and saved-head checks for generated projection runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

from projection_study_design import cells,qualification_cells
from projection_study_workload import basis,dataset,features
from projection_study_strategy import noise_draw
from run_projection_study import calibration_check,read_cell,sha,validate_plan,verify_checkpoint,write_json


def independent_metrics(y,s):
    y=np.asarray(y);s=np.asarray(s,dtype=float)
    if y.shape!=s.shape or y.ndim!=1 or set(np.unique(y))!={0,1} or not np.isfinite(s).all():
        raise ValueError('invalid independent score inputs')
    n1=int(y.sum());n0=len(y)-n1
    auc=(rankdata(s,method='average')[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
    return dict(auroc=float(auc),log_loss=float(np.mean(np.logaddexp(0,s)-y*s)),
                brier=float(np.mean((expit(s)-y)**2)))


def check_round(a,previous_weight,previous_bias):
    arrays={k:np.asarray(v,dtype=float) for k,v in a.items()}
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite round arrays')
    rw,rb=arrays['raw_weight'],arrays['raw_bias']
    if rw.shape!=(20,*previous_weight.shape) or rb.shape!=(20,2):raise ValueError('missing client arrays')
    norms=np.sqrt(np.sum(rw**2,axis=(1,2))+np.sum(rb**2,axis=1))
    shrink=np.minimum(1.,.05/(norms+1e-12))
    prew=previous_weight+np.mean(rw*shrink[:,None,None],axis=0)
    preb=previous_bias+np.mean(rb*shrink[:,None],axis=0)
    clipping_error=max(float(np.max(np.abs(prew-arrays['pre_weight']))),float(np.max(np.abs(preb-arrays['pre_bias']))))
    release_error=max(float(np.max(np.abs(arrays['pre_weight']+arrays['noise_weight']-arrays['weight']))),
                      float(np.max(np.abs(arrays['pre_bias']+arrays['noise_bias']-arrays['bias']))))
    if max(clipping_error,release_error)>1e-6:raise ValueError('clipping or noise release mismatch')
    return dict(clipping_error=clipping_error,release_error=release_error,norms=norms,shrink=shrink)


def check_noise_geometry(cell,round,a,info,x,previous_weight,previous_bias):
    std=calibration_check()['sigma'] if cell.private else 0.
    if not np.isfinite(info.get('noise_std',np.nan)) or abs(info['noise_std']-std)>1e-12:
        raise ValueError('declared Gaussian scale mismatch')
    expected=noise_draw(cell.seed,round,cell.width,std)
    for name,key in (('noise_weight','head.weight'),('noise_bias','head.bias')):
        if not np.array_equal(a[name],expected[key].numpy()):raise ValueError('seeded Gaussian draw mismatch')
    nw,nb=a['noise_weight'].astype(float),a['noise_bias'].astype(float)
    ns=np.einsum('ij,j->i',x,nw[1]-nw[0],optimize=False)+nb[1]-nb[0]
    values=dict(parameter_noise_energy=float(np.sum(nw*nw)+np.sum(nb*nb)),
                active_noise_energy=float(np.sum(nw[:,:128]**2)+np.sum(nb*nb)),
                expected_parameter_noise_energy=(2*cell.width+2)*std**2,
                predicted_score_noise_variance=float(np.mean(2*std**2*(np.sum(x*x,axis=1)+1))),
                realized_score_noise_mse=float(np.mean(ns*ns)))
    delta=a['pre_weight'].astype(float)-previous_weight
    effective=delta[1]-delta[0]
    lift=effective if cell.family=='full' else np.einsum('ij,j->i',basis(cell).numpy(),effective[:128],optimize=False)
    norm=float(np.sqrt(np.sum(lift**2)))
    alignment=float(lift[0])/norm if norm else None
    if alignment is None:
        if info.get('teacher_alignment') is not None:raise ValueError('zero-norm teacher alignment mismatch')
    elif not np.isfinite(info.get('teacher_alignment',np.nan)) or abs(info['teacher_alignment']-alignment)>1e-6:
        raise ValueError('teacher alignment mismatch')
    crw,crb=a['clean_raw_weight'].astype(float),a['clean_raw_bias'].astype(float)
    cnorm=np.sqrt(np.sum(crw**2,axis=(1,2))+np.sum(crb**2,axis=1))
    shrink=np.minimum(1.,.05/(cnorm+1e-12))
    meanw=np.mean(crw*shrink[:,None,None],axis=0);meanb=np.mean(crb*shrink[:,None],axis=0)
    values['clean_aggregate_norm']=float(np.sqrt(np.sum(meanw*meanw)+np.sum(meanb*meanb)))
    errors=[]
    for name,expected_value in values.items():
        actual=info.get(name,np.nan)
        if not np.isfinite(actual) or abs(actual-expected_value)>1e-6*max(1.,abs(expected_value)):
            raise ValueError(f'{name} diagnostic mismatch')
        errors.append(abs(actual-expected_value))
    return max(errors)


def summarize(rows):
    expected={c.name:c.record() for c in cells()}
    if len(rows)!=136 or {r['name'] for r in rows}!=set(expected):raise ValueError('exact main matrix required')
    if any(any(r[k]!=v for k,v in expected[r['name']].items()) for r in rows):raise ValueError('resolved cell mismatch')
    full={(r['seed'],r['private']):r for r in rows if r['block']=='full'}
    retention=[];random=[];attacks=[];padding=[]
    for private in (False,True):
        for eta in (0.,.25,.75,1.):
            points=[dict(seed=r['seed'],auroc=r['metrics']['auroc'],gain=r['metrics']['auroc']-full[r['seed'],private]['metrics']['auroc'])
                    for r in rows if r['block']=='controlled' and r['private']==private and r['eta']==eta]
            retention.append(dict(private=private,eta=eta,points=points,mean_gain=float(np.mean([p['gain'] for p in points]))))
        for b in range(20261101,20261106):
            points=[dict(seed=r['seed'],auroc=r['metrics']['auroc'],gain=r['metrics']['auroc']-full[r['seed'],private]['metrics']['auroc'])
                    for r in rows if r['block']=='random' and r['private']==private and r['projection_seed']==b]
            random.append(dict(private=private,projection_seed=b,points=points,
                               mean_gain=float(np.mean([p['gain'] for p in points]))))
        for r in rows:
            if r['block']=='padded' and r['private']==private:
                ref=next(s for s in rows if s['block']=='random' and s['seed']==r['seed'] and s['private']==private and s['projection_seed']==20261101)
                padding.append(dict(seed=r['seed'],private=private,auroc_difference=r['metrics']['auroc']-ref['metrics']['auroc']))
    gates={}
    for steps in (20,80,160):
        damage=[full[r['seed'],False]['metrics']['auroc']-r['metrics']['auroc'] for r in rows
                if r['block']=='attack' and r['kind']=='full' and not r['private'] and r['steps']==steps]
        gates[steps]=dict(mean_auroc_damage=float(np.mean(damage)),qualified=bool(np.mean(damage)>=.02))
        for private in (False,True):
            for kind in ('full','random'):
                selected=[r for r in rows if r['block']=='attack' and r['kind']==kind and r['steps']==steps and r['private']==private]
                attacks.append(dict(steps=steps,private=private,kind=kind,gate=gates[steps],
                                    points=[dict(seed=r['seed'],auroc=r['metrics']['auroc']) for r in selected],
                                    mean_auroc=float(np.mean([r['metrics']['auroc'] for r in selected]))))
    return dict(retention=retention,random_bases=random,attack_budgets=attacks,attack_damage_gates=gates,padding=padding)


def analyze(run_dir):
    run_dir=Path(run_dir);summary=json.loads((run_dir/'summary.json').read_text())
    plan=json.loads((run_dir/'frozen-plan.json').read_text());stage=plan['stage']
    if sha(run_dir/'frozen-plan.json')!=summary.get('plan_sha256'):
        raise ValueError('frozen manifest digest mismatch')
    validate_plan(plan);verify_checkpoint(plan,summary.get('checkpoint'))
    expected=cells() if stage=='main' else qualification_cells()
    if summary['status']!='completed' or summary['stage']!=stage or plan['cells']!=[c.record() for c in expected] \
            or [r['name'] for r in summary['runs']]!=[c.name for c in expected]:
        raise ValueError('incomplete or altered run set')
    data=dataset();cache={};maximum=dict(clipping_error=0.,release_error=0.,score_error=0.,metric_error=0.,geometry_error=0.)
    all_rows=[];padding={};count=0
    for cell,row in zip(expected,summary['runs'],strict=True):
        folder=run_dir/'runs'/cell.name
        if sha(folder/'result.json')!=summary['result_hashes'][cell.name] or json.loads((folder/'result.json').read_text())!=row:
            raise ValueError('result digest or summary mismatch')
        if row['status']!='completed' or row['rounds']!=20 or row['client_fits']!=400:raise ValueError('incomplete cell')
        if set(row['artifacts'])!={'head.npz','predictions.npz','rounds.json'} or not np.isfinite(row['elapsed_s']) or row['elapsed_s']<=0:
            raise ValueError('invalid artifact inventory or runtime')
        for name,digest in row['artifacts'].items():
            if sha(folder/name)!=digest:raise ValueError('artifact digest mismatch')
        key=(cell.family,cell.eta,cell.projection_seed)
        if key not in cache:cache[key]=features(data,cell)
        d=cache[key];x=d['eval_x'].numpy().astype(float)
        rounds=json.loads((folder/'rounds.json').read_text())
        if len(rounds)!=20:raise ValueError('incomplete round journal')
        previous_weight=np.zeros((2,cell.width));previous_bias=np.zeros(2)
        sequence=[]
        for i,info in enumerate(rounds,1):
            path=folder/'rounds'/f'{i:02}.npz'
            if sha(path)!=info['arrays_sha256'] or info['round']!=i or info['aggregated']!=20 \
                    or info['sample_counts']!=[205]*16+[204]*4:raise ValueError('round admission mismatch')
            with np.load(path,allow_pickle=False) as loaded:a={k:loaded[k] for k in loaded.files}
            audit=check_round(a,previous_weight,previous_bias)
            maximum['geometry_error']=max(maximum['geometry_error'],check_noise_geometry(cell,i,a,info,x,previous_weight,previous_bias))
            for key in ('clipping_error','release_error'):maximum[key]=max(maximum[key],audit[key])
            if abs(info['sensitivity']-.005)>1e-12:raise ValueError('sensitivity mismatch')
            recorded_norms=np.array([info['orig_norms'][str(c)] for c in range(20)])
            if np.max(np.abs(recorded_norms-audit['norms']))>1e-7:raise ValueError('raw norm diagnostic mismatch')
            recorded_shrink=np.minimum(1.,.05/(recorded_norms+1e-12))
            if abs(info['clip_rate']-float(np.mean(recorded_shrink<1)))>1e-7 \
                    or abs(info['mean_shrink']-float(recorded_shrink.mean()))>1e-7:
                raise ValueError('clipping diagnostic mismatch')
            if cell.steps:
                attack=info['attack']
                if attack['gradient_evaluations']!=cell.steps*3 or attack['loss_evaluations']!=(cell.steps+1)*3:
                    raise ValueError('attack budget mismatch')
            if cell.family=='padded' and np.max(np.abs(a['raw_weight'][:,:,128:]))!=0:
                raise ValueError('inactive padded local delta')
            previous_weight,previous_bias=a['weight'].astype(float),a['bias'].astype(float)
            sequence.append((a['weight'][:,:128],a['bias'],info['clip_rate']))
            count+=1
        head=np.load(folder/'head.npz',allow_pickle=False)
        if not np.array_equal(head['weight'],previous_weight) or not np.array_equal(head['bias'],previous_bias):
            raise ValueError('final head differs from last release')
        independently_scored=np.einsum('ij,j->i',x,head['weight'][1].astype(float)-head['weight'][0].astype(float),optimize=False) \
                              +float(head['bias'][1])-float(head['bias'][0])
        predictions=np.load(folder/'predictions.npz',allow_pickle=False)
        if not np.array_equal(predictions['y'],d['eval_y'].numpy()):raise ValueError('prediction labels differ')
        error=float(np.max(np.abs(independently_scored-predictions['scores'])))
        if error>1e-5:raise ValueError('independent score mismatch')
        maximum['score_error']=max(maximum['score_error'],error)
        values=independent_metrics(predictions['y'],predictions['scores'])
        for name,value in values.items():
            error=abs(value-row['metrics'][name]);maximum['metric_error']=max(maximum['metric_error'],error)
            if error>1e-10:raise ValueError('independent metric mismatch')
        if cell.family in ('random','padded') and cell.steps==0:
            padding[(cell.seed,cell.private,cell.projection_seed,cell.family)]=(sequence,predictions['scores'].copy())
        all_rows.append(row)
    padding_error=0.
    for key,(seq,preds) in padding.items():
        if key[-1]!='padded':continue
        other,other_predictions=padding[(*key[:-1],'random')]
        for (w,b,clip),(ow,ob,oclip) in zip(seq,other,strict=True):
            padding_error=max(padding_error,float(np.max(np.abs(w-ow))),float(np.max(np.abs(b-ob))))
            if clip!=oclip:raise ValueError('padding clipping decision mismatch')
        padding_error=max(padding_error,float(np.max(np.abs(preds-other_predictions))))
    if padding_error>1e-5:raise ValueError('padding training invariance failed')
    return dict(schema='capstone.projection.audit.v1',stage=stage,status='verified',cells=len(all_rows),
                rounds=count,client_fits=count*20,maximum_errors=maximum,padding_max_error=padding_error,
                summary_sha256=sha(run_dir/'summary.json'),rows=all_rows,
                comparisons=summarize(all_rows) if stage=='main' else None,
                scope='One fixed generated dataset; descriptive crossed seeds; no clinical or originality claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('audit output exists')
    result=analyze(args.runs);write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','cells','rounds','client_fits','maximum_errors','padding_max_error')}))


if __name__=='__main__':main()
