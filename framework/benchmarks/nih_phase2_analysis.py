"""Independent score arithmetic and predeclared matched-seed inference (no engine metrics)."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

SEEDS=tuple(range(20261101,20261111))
T975_DF9=2.2621571628540993


def _binary(scores,labels,weights=None):
    scores,labels=np.asarray(scores),np.asarray(labels)
    if (scores.ndim!=1 or not scores.size or scores.dtype.kind not in 'fiu' or not np.isfinite(scores).all()
            or labels.shape!=scores.shape or labels.dtype.kind not in 'iu' or not np.isin(labels,[0,1]).all()):
        raise ValueError('invalid binary scores or integer labels')
    weights=np.ones(scores.size) if weights is None else np.asarray(weights,dtype=np.float64)
    if (weights.shape!=scores.shape or not np.isfinite(weights).all() or (weights<0).any()
            or not math.isfinite(float(weights.sum())) or not weights[labels==1].sum()>0 or not weights[labels==0].sum()>0):
        raise ValueError('invalid weights or missing weighted class')
    return scores.astype(np.float64),labels,weights


def auc(scores,labels,weights=None):
    scores,labels,weights=_binary(scores,labels,weights)
    weights=weights/weights.sum()
    order=np.argsort(scores,kind='stable');s,y,w=scores[order],labels[order],weights[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    pos=np.add.reduceat(w*(y==1),starts);neg=np.add.reduceat(w*(y==0),starts)
    previous=np.cumsum(neg)-neg
    return float(np.sum(pos*(previous+.5*neg))/(pos.sum()*neg.sum()))


def metrics(logits,labels,weights=None,*,threshold=None):
    logits=np.asarray(logits)
    if logits.ndim!=2 or logits.shape[1]!=2 or logits.dtype.kind!='f' or not np.isfinite(logits).all():
        raise ValueError('invalid binary logits')
    margin=logits[:,1].astype(np.float64)-logits[:,0].astype(np.float64)
    margin,y,w=_binary(margin,labels,weights)
    if threshold is not None and (type(threshold) not in (int,float) or not math.isfinite(threshold)):
        raise ValueError('invalid operating threshold')
    predicted=(margin>0) if threshold is None else (margin>=threshold)
    count=lambda mask:float(w[mask].sum()) if weights is not None else int(mask.sum())
    tp,tn,fp,fn=(count(mask) for mask in (predicted&(y==1),~predicted&(y==0),predicted&(y==0),~predicted&(y==1)))
    sens,spec=tp/(tp+fn),tn/(tn+fp)
    # Calibration always describes argmax confidence, even for an additional
    # threshold's confusion metrics. Stable float64 arithmetic is independent of torch.
    confidence=1/(1+np.exp(-np.abs(margin)));correct=(margin>0)==y
    bins=np.minimum((confidence*10).astype(np.int64),9)
    ece=0.
    for b in range(10):
        mask=bins==b;mass=w[mask].sum()
        if mass:ece+=abs(float(np.average(correct[mask],weights=w[mask]))-float(np.average(confidence[mask],weights=w[mask])))*mass/w.sum()
    loss=np.where(y==1,np.logaddexp(0.,-margin),np.logaddexp(0.,margin))
    return {'auc':auc(margin,y,w),'loss':float(np.average(loss,weights=w)),'ece':float(ece),
            'accuracy':float((tp+tn)/w.sum()),'balanced_accuracy':(sens+spec)/2,'sensitivity':sens,'specificity':spec,
            'tp':tp,'tn':tn,'fp':fp,'fn':fn,'n_images':len(y),'weight_sum':float(w.sum())}


def recall_threshold(scores,labels):
    scores,labels,_=_binary(scores,labels)
    positive=np.sort(scores[labels==1])[::-1]
    return float(positive[math.ceil(.8*len(positive))-1])


def patient_weights(patient_ids):
    ids=np.asarray(patient_ids)
    if ids.ndim!=1 or not ids.size or ids.dtype.kind not in 'US' or (ids=='').any():
        raise ValueError('invalid patient identities')
    _,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True)
    return 1./counts[inverse]


def _split(data):
    logits,labels,margin=np.asarray(data['logits']),np.asarray(data['labels']),np.asarray(data['margin'])
    _binary(margin,labels)
    if (margin.dtype!=np.float64 or logits.dtype!=np.float32 or logits.shape!=(len(labels),2)
            or not np.isfinite(logits).all() or not np.array_equal(margin,logits[:,1].astype('float64')-logits[:,0].astype('float64'))):
        raise ValueError('invalid aligned raw logits/margin')
    for key in ('image_ids','patient_ids'):
        ids=np.asarray(data[key])
        if ids.shape!=labels.shape or ids.dtype.kind not in 'US' or (ids=='').any():raise ValueError('invalid aligned identities')
    if len(set(data['image_ids']))!=len(labels):raise ValueError('duplicate image identity')
    return data


def cell_analysis(bundle,excluded_test_image_ids):
    val,test=(_split(bundle['splits'][s]) for s in ('validation','test'))
    excluded=list(excluded_test_image_ids)
    if len(set(excluded))!=len(excluded) or not set(excluded)<=set(test['image_ids']):
        raise ValueError('ambiguous-content mask identities differ')
    threshold=recall_threshold(val['margin'],val['labels'])
    keep=~np.isin(test['image_ids'],excluded)
    proof=bundle['manifest']
    return {'evaluation_split':'test',**{k:proof[k] for k in ('config_hash','final_digest','features_report_sha256')},
        'validation':metrics(val['logits'],val['labels']),'test':metrics(test['logits'],test['labels']),
        'validation_threshold':threshold,
        'threshold':{s:metrics(data['logits'],data['labels'],threshold=threshold) for s,data in (('validation',val),('test',test))},
        'patient_weighted':{s:metrics(data['logits'],data['labels'],patient_weights(data['patient_ids'])) for s,data in (('validation',val),('test',test))},
        'test_ambiguous_excluded':metrics(test['logits'][keep],test['labels'][keep]),
        'excluded_test_images':len(excluded)}


def study_analysis(cases,expected_configs):
    """Summarize parent-requalified cells; the CLI first reloads raw models/scores."""
    expected={c['name']:c for c in expected_configs}
    if len(expected_configs)!=360 or len(expected)!=360 or len(cases)!=360:
        raise ValueError('study requires all360planned entries')
    actual={c['name']:c for c in cases}
    if len(actual)!=360 or set(actual)!=set(expected):raise ValueError('duplicate/missing/foreign case')
    for name,cfg in expected.items():
        c=actual[name];h=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if c.get('config_hash')!=h or c['status'] not in ('qualified','failed','timeout','not_run'):
            raise ValueError('case configuration/status mismatch')
        if c['status']=='qualified':
            a=c['analysis']
            if (a['evaluation_split']!='test' or a['config_hash']!=h or a['final_digest']!=c['record_final_digest']
                    or a['features_report_sha256']!=cfg['workload']['features_report_sha256']):
                raise ValueError('unbound final score summary')
            for key in ('auc','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity'):
                value=a['test'][key]
                if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (key!='loss' and value>1):
                    raise ValueError('invalid final metric summary')
    per_seed=[];missing=[]
    for seed in SEEDS:
        components={};absent=[];planned=set()
        for name,cfg in expected.items():
            if not name.startswith('primary_') or cfg['seed']!=seed:continue
            p=cfg['privacy'];mode='none' if p['mode']=='none' else 'eps4' if p['epsilon']==4 else None
            condition='clean' if not cfg['behaviors'] else 'sign_flip' if cfg['behaviors'][0]['type']=='sign_flip_scale' else None
            if mode is None or condition is None:continue
            key=f'{condition}/{cfg["server"]["rule"]}/{mode}'
            if key in planned:raise ValueError('duplicate primary component recipe')
            planned.add(key)
            if actual[name]['status']=='qualified':components[key]=actual[name]['analysis']['test']['auc']
            else:absent.append(name)
        if planned!={f'{a}/{r}/{p}' for a in ('clean','sign_flip') for r in ('mean','trimmed_mean') for p in ('none','eps4')}:
            raise ValueError('frozen primary component coverage differs')
        effect=None
        if not absent:
            def benefit(mode):return (components[f'sign_flip/trimmed_mean/{mode}']-components[f'sign_flip/mean/{mode}'])-(components[f'clean/trimmed_mean/{mode}']-components[f'clean/mean/{mode}'])
            effect=benefit('eps4')-benefit('none')
        per_seed.append({'seed':seed,'components':components,'effect':effect,'missing':absent});missing.extend(absent)
    effects=[s['effect'] for s in per_seed if s['effect'] is not None]
    complete=len(effects)==10
    mean=float(np.mean(effects)) if effects else None
    sd=float(np.std(effects,ddof=1)) if len(effects)>1 else None
    ci=[mean-T975_DF9*sd/math.sqrt(10),mean+T975_DF9*sd/math.sqrt(10)] if complete else None
    descriptive={}
    for name,cfg in expected.items():
        key=name.replace(f's{cfg["seed"]}','matched_seeds',1)
        group=descriptive.setdefault(key,{'planned':0,'qualified':0,'values':[]})
        group['planned']+=1
        if actual[name]['status']=='qualified':group['values'].append({'seed':cfg['seed'],**actual[name]['analysis']['test']});group['qualified']+=1
    for group in descriptive.values():
        values=[v['auc'] for v in group['values']]
        group['auc_mean']=float(np.mean(values)) if values else None
        group['auc_range']=[min(values),max(values)] if values else None
    return {'analysis_version':'nih-phase2-v1','complete':all(c['status']=='qualified' for c in cases),
        'primary':{'complete':complete,'n_complete':len(effects),'requested_seeds':10,'per_seed':per_seed,
            'missing':missing,'mean':mean,'sd':sd,'range':[min(effects),max(effects)] if effects else None,
            'ci95':ci,'scope':'conditional_simulation_seed_variation','interval_method':'paired_t_df9_if_all10_complete'},
        'descriptive':descriptive,'statuses':{s:sum(c['status']==s for c in cases) for s in ('qualified','failed','timeout','not_run')}}


def main():
    import argparse
    from pathlib import Path
    from benchmarks.nih_phase2_study import analyze_saved_study
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():parser.exit(2,'refusing existing analysis output\n')
    try:
        result=analyze_saved_study(args.study)
        with args.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    except (OSError,ValueError) as error:parser.exit(2,f'error: {error}\n')
    print(json.dumps({'complete':result['complete'],'primary_complete':result['primary']['complete']}))


if __name__=='__main__':main()
