"""Independent fixed-study reconciliation; imports no project runner/model/metrics."""

if not __debug__:
    raise RuntimeError("Verification requires Python assertions; rerun without -O or PYTHONOPTIMIZE.")

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess

import numpy as np
import torch
import torch.nn.functional as functional

PIN = '6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd'
DOC = 'docs/artifacts/capstone/nih-phase2-study-v1-protocol.md'
DOC_PIN = '4245a8446d3d88a93cd921d24f7a4f565273a7285663e446e43f1fcb123012a0'
MASK_PIN = '352b10c16342e4160d06a1e536ea82d201ea739eec17ed67fc9801141e60470d'
SEEDS = list(range(20261101, 20261111))
ORDERS = [1+x/10 for x in range(1,100)] + list(range(12,64)) + [128,256,512]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(2**20), b''): value.update(block)
    return value.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def close(actual, expected, tolerance=1e-12):
    assert math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0), (actual, expected)


def verify_primary(primary, all_scores):
    """Reconstruct all ten planned rows before accepting any saved inference."""
    rows = []
    missing = []
    effects = []
    for seed in SEEDS:
        components = {}
        absent = []
        terms = []
        for condition, rule, mode in itertools.product(
            ['clean', 'sign_flip'], ['mean', 'trimmed_mean'], ['none', 'eps4']
        ):
            name = f'primary_s{seed}_linear_n80_c005_{condition}_{rule}_{mode}'
            if name not in all_scores:
                absent.append(name)
                continue
            value = float(all_scores[name])
            assert math.isfinite(value) and 0 <= value <= 1
            components[f'{condition}/{rule}/{mode}'] = value
            sign = ((1 if condition == 'sign_flip' else -1)
                    * (1 if rule == 'trimmed_mean' else -1)
                    * (1 if mode == 'eps4' else -1))
            terms.append(sign * value)
        effect = None if absent else math.fsum(terms)
        rows.append(dict(seed=seed, components=components, missing=absent, effect=effect))
        missing.extend(absent)
        if effect is not None:
            effects.append(effect)
    mean = statistics.mean(effects) if effects else None
    sd = statistics.stdev(effects) if len(effects) > 1 else None
    ci95 = None
    if len(effects) == 10:
        half_width = 2.2621571628540993 * sd / math.sqrt(10)
        ci95 = [mean - half_width, mean + half_width]
    expected = dict(
        complete=len(effects) == 10, n_complete=len(effects), requested_seeds=10,
        per_seed=rows, missing=missing, mean=mean, sd=sd,
        range=[min(effects), max(effects)] if effects else None, ci95=ci95,
        scope='conditional_simulation_seed_variation',
        interval_method='paired_t_df9_if_all10_complete',
    )

    def compare(actual, wanted, path):
        if isinstance(wanted, dict):
            assert type(actual) is dict and set(actual) == set(wanted), path
            for key, value in wanted.items():
                compare(actual[key], value, f'{path}.{key}')
        elif isinstance(wanted, list):
            assert type(actual) is list and len(actual) == len(wanted), path
            for index in range(len(wanted)):
                compare(actual[index], wanted[index], f'{path}[{index}]')
        elif type(wanted) is float:
            assert type(actual) in (int, float) and math.isfinite(actual), path
            assert math.isclose(actual, wanted, abs_tol=1e-12, rel_tol=0), path
        else:
            assert type(actual) is type(wanted) and actual == wanted, path

    compare(primary, expected, 'primary')
    # Preserve the saved arithmetic's last-bit rounding after independent agreement.
    return primary


def rank_auc(scores, labels, weights=None):
    """Mann-Whitney midranks; weighted analogue uses centered mass ranks."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    assert scores.shape == labels.shape and scores.ndim == 1 and np.isfinite(scores).all()
    assert set(labels) == {0, 1}
    ordered = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    if weights is None:
        positive_ranks = 0.; start = 0
        while start < len(ordered):
            end = start+1
            while end < len(ordered) and scores[ordered[end]] == scores[ordered[start]]: end += 1
            midrank = (start+1+end)/2
            positive_ranks += midrank*sum(int(labels[ordered[j]]) for j in range(start,end))
            start = end
        positives = int(labels.sum()); negatives = len(labels)-positives
        return (positive_ranks-positives*(positives+1)/2)/(positives*negatives)
    weights = np.asarray(weights, dtype=float)
    assert np.isfinite(weights).all() and (weights > 0).all()
    total_positive = math.fsum(weights[labels == 1]); total_negative = math.fsum(weights[labels == 0])
    before = 0.; positive_ranks = 0.; start = 0
    while start < len(ordered):
        end = start+1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[start]]: end += 1
        members = ordered[start:end]
        mass = math.fsum(weights[members])
        positive_ranks += math.fsum(weights[i] for i in members if labels[i])*(before+mass/2)
        before += mass; start = end
    return (positive_ranks-total_positive**2/2)/(total_positive*total_negative)


assert rank_auc([0.,0.,1.,2.], np.array([0,1,0,1])) == .625
assert rank_auc([0.,0.,1.,2.], np.array([0,1,0,1]), [1.,1.,1.,1.]) == .625
assert rank_auc([0.,1.], np.array([1,0])) == 0.


def score_metrics(logits, labels, weights=None, threshold=None):
    logits = np.asarray(logits); y = np.asarray(labels)
    margin = logits[:,1].astype(float)-logits[:,0].astype(float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    predicted = margin > 0 if threshold is None else margin >= threshold
    counts = [math.fsum(w[mask]) for mask in (predicted & (y==1), ~predicted & (y==0), predicted & (y==0), ~predicted & (y==1))]
    tp, tn, fp, fn = counts; total = math.fsum(w)
    recall, specificity = tp/(tp+fn), tn/(tn+fp)
    losses = np.logaddexp(0, np.where(y == 1, -margin, margin))
    confidence = 1/(1+np.exp(-np.abs(margin))); correct = (margin > 0) == y
    bins = np.minimum(np.floor(confidence*10).astype(int),9)
    calibration = 0.
    for index in range(10):
        selected = bins == index
        if selected.any():calibration += abs(math.fsum(w[selected]*(correct[selected]-confidence[selected])))/total
    return dict(auc=rank_auc(margin,y,weights), loss=math.fsum(w*losses)/total, ece=calibration,
        accuracy=(tp+tn)/total, balanced_accuracy=(recall+specificity)/2,
        sensitivity=recall,specificity=specificity,tp=tp,tn=tn,fp=fp,fn=fn,n_images=len(y),weight_sum=total)


def assert_metrics(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():close(actual[key],value,1e-10)


def expected_cells(recorded_cache_path):
    cells=[]
    definitions=[('primary',SEEDS,['linear'],[80],.05,['clean','sign_flip','label_flip'],['none',1,4,8]),
        ('head',SEEDS[:3],['mlp8','mlp128','mlp512'],[80],.05,['clean','sign_flip'],['none',4]),
        ('cohort',SEEDS[:3],['linear'],[10,1000],.05,['clean'],['none',4]),
        ('clipping',SEEDS[:3],['linear'],[80],.4,['clean','sign_flip'],['none',4])]
    for family,seeds,heads,cohorts,clip,conditions,privacy in definitions:
        for seed,head,n,condition,rule,p in itertools.product(seeds,heads,cohorts,conditions,['mean','trimmed_mean'],privacy):
            mode='none' if p=='none' else f'eps{p}'
            name=f'{family}_s{seed}_{head}_n{n}_c{"005" if clip==.05 else "04"}_{condition}_{rule}_{mode}'
            hidden=None if head=='linear' else int(head[3:])
            behavior=[] if condition=='clean' else [{'type':'sign_flip_scale','fraction':.2,'scale':-4.}] if condition=='sign_flip' else [{'type':'label_flip','fraction':.2}]
            cfg={'schema_version':1,'name':name,'seed':seed,
                'seeds':{'partition':seed,'init':seed,'client':seed,'attack':seed+1000,'noise':seed+2000},
                'workload':{'dataset':'nih','features_path':recorded_cache_path,'balance':False,'head':{'type':'linear','hidden':128} if hidden is None else {'type':'mlp','hidden':hidden},
                    'features_report_sha256':PIN,'evaluation_split':'test','allow_random_features':False,'auc_score':'margin_v2'},
                'federation':{'num_clients':n,'clients_per_round':n,'rounds':20,'partition':{'type':'patient_iid','alpha':.5},
                    'network':{'uplink_mbps':10.,'downlink_mbps':50.,'latency_ms':50.,'device_gflops':10.,'round_deadline_s':None}},
                'client':{'algorithm':'fedavg','proximal_mu':0.,'local_epochs':7,'lr':.003,'batch_size':256},
                'server':{'clip_norm':clip,'rule':rule,'rule_params':{'trim_ratio':.2,'byzantine_fraction':None,'tau':1.,'iterations':1,'multi_krum_m':None},'optimizer':{'type':'none','lr':.01}},
                'privacy':{'mode':'none' if p=='none' else 'central','epsilon':4. if p=='none' else float(p),'delta':1e-5,'adjacency':'replace_one'},'behaviors':behavior,'eval':{'per_client_every':0}}
            cells.append((cfg,hidden,condition))
    assert len(cells)==360
    return cells


def allocation(rows,seed,n):
    groups={}
    for index,row in enumerate(rows):groups.setdefault(row['patient_id'],[]).append(index)
    shuffled=np.random.Generator(np.random.PCG64(seed)).permutation(sorted(groups))
    q,extra=divmod(len(shuffled),n);start=0;parts=[];counts={};assigned=[]
    for cid in range(n):
        patients=shuffled[start:start+q+(cid<extra)];start+=len(patients)
        indices=sorted(i for p in patients for i in groups[p]);assigned.extend(indices)
        positive=sum(rows[i]['label']=='1' for i in indices)
        counts[str(cid)]={'images':len(indices),'patients':len(patients),'positive_images':positive,'negative_images':len(indices)-positive}
        parts.append({'client_id':cid,'image_ids':[rows[i]['image_index'] for i in indices]})
    assert start==len(shuffled) and sorted(assigned)==list(range(len(rows)))
    return counts,hashlib.sha256(canonical({'algorithm':'nih-client-assignment-v1','clients':parts})).hexdigest()


def checkpoint(path,hidden):
    blob=path.read_bytes();length=struct.unpack('<Q',blob[:8])[0]
    header=json.loads(blob[8:8+length]);offset=8+length
    expected={'head.weight':[2,512],'head.bias':[2]} if hidden is None else {'hidden.weight':[hidden,512],'hidden.bias':[hidden],'head.weight':[2,hidden],'head.bias':[2]}
    assert set(header)==set(expected);parameters={};cursor=0
    for key,entry in header.items():
        assert entry['dtype']=='F32' and entry['shape']==expected[key]
        begin,end=entry['data_offsets'];assert begin==cursor and end-begin==4*math.prod(entry['shape']);cursor=end
        array=np.frombuffer(blob[offset+begin:offset+end],dtype='<f4').reshape(entry['shape']).copy()
        assert np.isfinite(array).all();parameters[key]=torch.from_numpy(array)
    assert offset+cursor==len(blob)
    digest=hashlib.sha256()
    for key in sorted(parameters):digest.update(key.encode());digest.update(parameters[key].numpy().tobytes())
    return parameters,digest.hexdigest()[:16],len(blob)


def forward(parameters,x,hidden):
    with torch.no_grad():
        inputs=torch.from_numpy(np.array(x,copy=True))
        if hidden is not None:inputs=functional.relu(functional.linear(inputs,parameters['hidden.weight'],parameters['hidden.bias']))
        return functional.linear(inputs,parameters['head.weight'],parameters['head.bias']).numpy()


def main(study,cache,source_root,audit_output):
    OUTPUT=Path(study).resolve();CACHE=Path(cache).resolve();ROOT=Path(source_root).resolve();target=Path(audit_output)
    if target.exists() or target.is_symlink():raise FileExistsError('refusing existing audit output')
    if any(target.resolve().is_relative_to(p) for p in (OUTPUT,CACHE)):raise ValueError('audit output must be outside immutable study/cache')
    checker_hash=sha(Path(__file__))
    torch.set_num_threads(4)
    result=json.loads((OUTPUT/'summary.json').read_text());protocol=json.loads((OUTPUT/'protocol.json').read_text())
    assert result['study_version']==protocol['study_version']=='nih-phase2-v1'
    assert result['source_identity']==protocol['source_identity']
    identity=protocol['source_identity'];commit=identity['git_commit']
    subprocess.run(['git','merge-base','--is-ancestor','aa7ae53',commit],cwd=ROOT,check=True)
    expected_paths=set(subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--','framework/src','framework/benchmarks','docs/artifacts/capstone/nih-optimization-pilot-v2-protocol.md'],cwd=ROOT,text=True).splitlines())|{DOC}
    assert set(identity['files'])==expected_paths
    for path,pin in identity['files'].items():
        raw=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==pin,path
    frozen=subprocess.check_output(['git','show',f'aa7ae53:{DOC}'],cwd=ROOT)
    assert hashlib.sha256(frozen).hexdigest()==DOC_PIN==protocol['protocol_document_sha256']==result['protocol_document_sha256']
    assert protocol['per_cell_seconds_cap']==180 and protocol['total_child_seconds_cap']==10800 and protocol['max_child_rss_bytes']==2**31
    assert result['evaluation_split']==protocol['evaluation_split']=='test'
    assert result['features_report_sha256']==protocol['features_report_sha256']==PIN
    assert result['test_mask_sha256']==protocol['test_mask_sha256']==sha(OUTPUT/'test-mask.json')==MASK_PIN
    mask=json.loads((OUTPUT/'test-mask.json').read_text());excluded=mask['excluded_test_image_ids']
    assert len(excluded)==len(set(excluded))==9
    assert sha(CACHE/'report.json')==PIN
    report=json.loads((CACHE/'report.json').read_text())
    assert report['pretrained'] and report['feat_dim']==512
    for path,pin in report['payload_hashes'].items():assert sha(CACHE/path)==pin,path
    rows={}
    for split,size,patients,positives in [('train',68650,22399,28515),('validation',17763,5600,7470),('test',25596,2797,15735)]:
        with (CACHE/f'{split}.csv').open(newline='') as file:rows[split]=list(csv.DictReader(file))
        r=rows[split];assert len(r)==size and len({x['patient_id'] for x in r})==patients and sum(x['label']=='1' for x in r)==positives
    for a,b in itertools.combinations(rows,2):assert not {r['patient_id'] for r in rows[a]}&{r['patient_id'] for r in rows[b]}
    masked=[r for r in rows['test'] if r['image_index'] in excluded]
    assert len(masked)==9 and len({r['patient_id'] for r in masked})==7 and sum(r['label']=='1' for r in masked)==3
    cells=expected_cells(protocol['cache']);recipes=[cfg for cfg,_,_ in cells]
    assert protocol['configs']==recipes
    hashes={c['name']:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for c in recipes}
    assert protocol['config_hashes']==hashes
    cases=result['cases'];assert len(cases)==360 and [c['name'] for c in cases]==[c['name'] for c in recipes]
    assert json.loads((OUTPUT/'index.json').read_text())['cases']==cases
    x={split:np.load(CACHE/f'{split}-x.npy',allow_pickle=False,mmap_mode='r') for split in ('validation','test')}
    assignments={};round0={};verified=[];failed=[];all_scores={};all_metrics={}
    for case,(cfg,hidden,condition) in zip(cases,cells):
        name=cfg['name'];n=cfg['federation']['num_clients'];seed=cfg['seed'];clip=cfg['server']['clip_norm'];rule=cfg['server']['rule']
        assert case['config_hash']==hashes[name] and json.loads((OUTPUT/'configs'/f'{name}.json').read_text())==cfg
        directory=OUTPUT/'runs'/name;log=OUTPUT/'logs'/f'{name}.log'
        files=([p for p in directory.rglob('*') if p.is_file()] if directory.exists() else [])+([log] if log.exists() else [])
        assert all(not p.is_symlink() and p.resolve().is_relative_to(OUTPUT.resolve()) for p in files)
        registry={p.relative_to(directory).as_posix() if p.is_relative_to(directory) else 'child.log':{'file':p.relative_to(OUTPUT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
        assert case.get('artifacts',{})==registry
        if case['status']!='qualified':
            assert case['status'] in ('failed','timeout','not_run')
            failed.append({'name':name,'status':case['status'],'error':case.get('error',case.get('reason'))});continue
        assert 0<=case['child_execution_seconds']<=181 and case['parent_control_seconds']>=0
        close(case['elapsed_seconds'],case['child_execution_seconds']+case['parent_control_seconds'],1e-8)
        for filename,phase in [('executed-source-start.json','before_engine'),('executed-source.json','after_engine_return')]:
            proof=json.loads((directory/filename).read_text());assert proof['phase']==phase
            assert all(identity['files'].get(k)==v for k,v in proof['files'].items())
            for required in ['framework/src/fedlearn/simulation/engine.py','framework/src/fedlearn/simulation/core_qualification.py','framework/benchmarks/nih_final_artifacts.py','framework/benchmarks/nih_phase2_study.py','framework/benchmarks/nih_phase2_analysis.py']:
                assert proof['files'][required]==identity['files'][required]
        execution=json.loads((directory/'execution.json').read_text());assert execution['status']=='ok' and execution['record_encoding']=='json'
        record=json.loads((directory/'record.json').read_text());meta=record['meta'];lines=record['rounds'];wl=meta['workload']
        assert meta['config']==cfg and meta['config_hash']==hashes[name] and meta['provenance']['git_commit']==commit
        assert meta['participation_schema_version']==2 and not meta['stopped_early'] and meta['stop_reason'] is None and meta['refusals']==[]
        key=(seed,n)
        if key not in assignments:assignments[key]=allocation(rows['train'],seed,n)
        counts,assignment=assignments[key];assert wl['clients']==counts and wl['assignment_sha256']==assignment
        behavior=cfg['behaviors'];attackers=sorted(int(i) for i in np.random.Generator(np.random.PCG64(seed+1000)).choice(n,int(.2*n),replace=False)) if behavior else []
        assert meta['behavior_assignment']=={str(i):[behavior[0]['type']] for i in attackers}
        assert meta['core_qualification']=={'version':1,'scope':'simulation_execution_only','accepted_clients':n,'attack_knowledge':'own_data_or_own_update','privacy_mode':cfg['privacy']['mode'],'evaluation_split':'test','feature_control':False,'research_record_dp_safe':False}
        parameters=1026 if hidden is None else 515*hidden+2
        assert wl['parameters']==parameters and wl['training_device']=='cpu' and wl['torch_num_threads']==4
        assert wl['features_report_sha256']==PIN and wl['pretrained'] and wl['auc_score']=='margin_v2' and wl['evaluation_split']=='test'
        assert wl['n_train']==68650 and wl['n_test']==25596
        assert all(meta['provenance'][k]==protocol['runtime'][k] for k in ('python','torch','numpy'))
        assert 0<meta['peak_rss_bytes']<=2**31
        assert [line['round'] for line in lines]==list(range(1,21))
        central=cfg['privacy']['mode']=='central';epsilon=cfg['privacy']['epsilon'];z=meta['noise_multiplier']
        expected_z=min(math.sqrt(20*a/(2*(epsilon-math.log(1e5)/(a-1)))) for a in ORDERS if epsilon>math.log(1e5)/(a-1)) if central else 0.
        assert math.isclose(z,expected_z,rel_tol=1e-6,abs_tol=1e-12)
        spent=lambda r:min(r*a/(2*z*z)+math.log(1e5)/(a-1) for a in ORDERS)
        if central:close(meta['accounted_epsilon'],spent(20),1e-10)
        if not central:assert meta['accounted_epsilon'] is None
        else:assert meta['accounted_epsilon']<=epsilon+1e-6
        final=directory/'final';manifest=json.loads((final/'manifest.json').read_text())
        assert manifest['config_hash']==hashes[name] and manifest['source_commit']==commit and manifest['features_report_sha256']==PIN and manifest['feature_control'] is False and manifest['research_record_dp_safe'] is False
        assert manifest['record_sha256']==hashlib.sha256(canonical(record)).hexdigest()
        params,final_digest,transfer=checkpoint(final/'model.safetensors',hidden)
        assert final_digest==meta['final_digest']==manifest['final_digest']==case['record_final_digest']
        assert sum(t.numel() for t in params.values())==parameters and manifest['parameter_shapes']=={k:list(v.shape) for k,v in params.items()}
        assert manifest['prediction_recipe']=={'device':'cpu','batch':'complete_split','logits_dtype':'float32','margin_dtype':'float64_subtraction','torch_num_threads':4,'torch':torch.__version__,'numpy':np.__version__}
        assert set(manifest['payloads'])=={'model.safetensors','validation.npz','test.npz'}
        for path,entry in manifest['payloads'].items():assert sha(final/path)==entry['sha256'] and (final/path).stat().st_size==entry['bytes']
        data={}
        for split in ('validation','test'):
            with np.load(final/f'{split}.npz',allow_pickle=False) as arrays:data[split]={k:arrays[k] for k in arrays.files}
            d=data[split];r=rows[split];assert set(d)=={'logits','margin','labels','image_ids','patient_ids'}
            assert d['logits'].dtype==np.float32 and d['margin'].dtype==np.float64 and d['labels'].dtype==np.int64
            assert d['logits'].shape==(len(r),2) and d['margin'].shape==d['labels'].shape==(len(r),)
            assert all(d[k].dtype.kind in 'US' and d[k].shape==(len(r),) for k in ('image_ids','patient_ids'))
            assert np.array_equal(d['labels'],[int(row['label']) for row in r])
            assert np.array_equal(d['image_ids'],[row['image_index'] for row in r]) and np.array_equal(d['patient_ids'],[row['patient_id'] for row in r])
            assert np.array_equal(d['logits'],forward(params,x[split],hidden))
            assert np.array_equal(d['margin'],d['logits'][:,1].astype(float)-d['logits'][:,0].astype(float))
            assert manifest['splits'][split]=={'rows':len(r),'manifest_sha256':sha(CACHE/f'{split}.csv')}
        analysis=json.loads((directory/'score-analysis.json').read_text());assert analysis==case['analysis']
        assert analysis['evaluation_split']=='test' and analysis['config_hash']==hashes[name] and analysis['final_digest']==final_digest and analysis['features_report_sha256']==PIN
        positives=sorted(data['validation']['margin'][data['validation']['labels']==1],reverse=True)
        threshold=float(positives[math.ceil(.8*len(positives))-1]);close(analysis['validation_threshold'],threshold)
        metrics={}
        for split,d in data.items():
            metrics[split]=score_metrics(d['logits'],d['labels']);assert_metrics(analysis[split],metrics[split])
            assert_metrics(analysis['threshold'][split],score_metrics(d['logits'],d['labels'],threshold=threshold))
            counts_by_patient={}
            for patient in d['patient_ids']:counts_by_patient[patient]=counts_by_patient.get(patient,0)+1
            weights=np.array([1/counts_by_patient[p] for p in d['patient_ids']])
            assert_metrics(analysis['patient_weighted'][split],score_metrics(d['logits'],d['labels'],weights))
        d=data['test'];keep=np.array([image not in excluded for image in d['image_ids']])
        assert int((~keep).sum())==analysis['excluded_test_images']==9
        assert_metrics(analysis['test_ambiguous_excluded'],score_metrics(d['logits'][keep],d['labels'][keep]))
        all_scores[name]=metrics['test']['auc'];all_metrics[name]=metrics['test']
        for metric in ('auc','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity'):
            close(lines[-1][metric],metrics['test'][metric],1e-4 if metric=='ece' else 1e-12 if metric=='auc' else 1e-6)
        per_image_flops=4096 if hidden is None else 4*512*hidden+6*hidden*2
        round_flops=68650*7*per_image_flops;stages={}
        sensitivity=2*clip/(n-2*int(.2*n) if rule=='trimmed_mean' else n)
        clip_rates=[];shrink_factors=[];signal_ratios=[]
        for line in lines:
            clip_rate,shrink=line['privacy']['clip_rate'],line['privacy']['mean_shrink']
            assert isinstance(clip_rate,(int,float)) and math.isfinite(clip_rate) and 0<=clip_rate<=1
            assert isinstance(shrink,(int,float)) and math.isfinite(shrink) and 0<=shrink<=1
            close(clip_rate*n,round(clip_rate*n),1e-6)
            clip_rates.append(clip_rate);shrink_factors.append(shrink)
            ratio=line['privacy']['noise_to_signal']
            if ratio is not None:
                assert math.isfinite(ratio) and ratio>=0;signal_ratios.append(ratio)
            for metric in ('auc','auc_probability','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity'):
                value=line[metric];assert isinstance(value,(int,float)) and math.isfinite(value) and value>=0 and (metric=='loss' or value<=1)
            tp=round(line['sensitivity']*15735);tn=round(line['specificity']*9861)
            close(line['accuracy'],(tp+tn)/25596,1e-6)
            close(line['balanced_accuracy'],(tp/15735+tn/9861)/2,1e-6)
            assert line['participants']=={'sampled':n,'dropped':0,'straggled':0,'submitted':n,'admitted':n,'accepted':n,'aggregated':n}
            assert not line['forced'] and line['refusal'] is None
            close(line['privacy']['sensitivity'],sensitivity);close(line['privacy']['noise_std'],z*sensitivity)
            if central:close(line['privacy']['epsilon_spent'],spent(line['round']),1e-10)
            else:assert line['privacy']['epsilon_spent'] is None
            assert line['comm']['up_bytes']==line['comm']['down_bytes']==n*transfer
            assert line['compute']['client_flops']==line['compute']['client_flops_analytic']==round_flops
            for field,value in line['time']['wall_s'].items():assert math.isfinite(value) and value>=0;stages[field]=stages.get(field,0)+value
        totals=meta['totals'];assert totals['up_bytes']==totals['down_bytes']==20*n*transfer
        assert totals['client_flops']==totals['client_flops_analytic']==20*round_flops
        assert sum(totals['per_client_bytes'].values())==totals['up_bytes']+totals['down_bytes']
        summary=case['summary'];assert summary['stage_wall_seconds']==stages and summary['totals']==totals and summary['parameters']==parameters
        assert summary['peak_rss_bytes']==meta['peak_rss_bytes'] and summary['peak_rss_scope']==meta['peak_rss_scope']
        assert summary['final_engine_metrics']=={k:lines[-1][k] for k in ('auc','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity')}
        assert summary['federation_loop_seconds']==meta['wall_seconds'] and summary['core_execution_seconds']==execution['core_execution_seconds'] and summary['artifact_execution_seconds']==execution['artifact_execution_seconds']
        assert meta['wall_seconds']+1e-8>=stages.get('train',0)+stages.get('eval',0)
        assert meta['wall_seconds']<=execution['core_execution_seconds'] and execution['core_execution_seconds']+execution['artifact_execution_seconds']<=case['child_execution_seconds']
        matching=(seed,hidden,n,condition);round0.setdefault(matching,set()).add(meta['round0_upload_digest'])
        verified.append({'name':name,'final':metrics['test'],'child_seconds':case['child_execution_seconds'],'peak_rss_bytes':meta['peak_rss_bytes'],'noise_multiplier':z,'sensitivity':sensitivity,'noise_std':z*sensitivity,'per_transfer_bytes':transfer,'parameters':parameters,'source_bound_update_diagnostics':{'clip_rate_mean':statistics.mean(clip_rates),'mean_shrink_mean':statistics.mean(shrink_factors),'noise_to_signal_mean':statistics.mean(signal_ratios) if signal_ratios else None,'noise_to_signal_rounds':len(signal_ratios),'noise_rms_per_round':z*sensitivity*math.sqrt(parameters),'scope':'source_bound_reported_update_summaries_not_raw_update_reconstruction'}})
        print(json.dumps({'verified':len(verified),'name':name}),flush=True)
    assert all(len(digests)==1 for digests in round0.values())
    groups={}
    for cfg in recipes:
        name=cfg['name'];key=name.replace(f's{cfg["seed"]}','matched_seeds',1)
        group=groups.setdefault(key,{'planned':0,'values':[]})
        group['planned']+=1
        if name in all_metrics:group['values'].append({'seed':cfg['seed'],**all_metrics[name]})
    assert len(groups)==64 and set(result['analysis']['descriptive'])==set(groups)
    for key,group in groups.items():
        saved=result['analysis']['descriptive'][key]
        assert saved['planned']==group['planned'] and saved['qualified']==len(group['values'])
        assert len(saved['values'])==len(group['values'])
        for actual,expected in zip(saved['values'],group['values']):
            assert actual['seed']==expected['seed']
            assert_metrics({k:v for k,v in actual.items() if k!='seed'},{k:v for k,v in expected.items() if k!='seed'})
        values=[v['auc'] for v in group['values']]
        if values:
            close(saved['auc_mean'],statistics.mean(values))
            for actual,expected in zip(saved['auc_range'],[min(values),max(values)]):close(actual,expected)
        else:assert saved['auc_mean'] is None and saved['auc_range'] is None
    assert result['analysis']['complete']==(len(verified)==360)
    assert result['analysis']['statuses']=={status:sum(c['status']==status for c in cases) for status in ('qualified','failed','timeout','not_run')}
    if len(verified)==360:assert len(assignments)==16
    primary=verify_primary(result['analysis']['primary'],all_scores)
    assert result['complete']==(len(verified)==360 and not result['source_integrity_failed'])
    close(result['child_execution_seconds'],math.fsum(c.get('child_execution_seconds',0) for c in cases),1e-8)
    close(result['elapsed_seconds'],result['child_execution_seconds']+result['parent_control_seconds'],1e-8)
    assert result['child_execution_seconds']<=10801
    assert sha(Path(__file__))==checker_hash
    audit={'checker_sha256':checker_hash,'raw_summary_sha256':sha(OUTPUT/'summary.json'),'raw_index_sha256':sha(OUTPUT/'index.json'),'raw_protocol_sha256':sha(OUTPUT/'protocol.json'),'source_commit':commit,'source_files_verified':len(identity['files']),'qualified':len(verified),'verified_rounds':20*len(verified),'failed_or_unrun':failed,
        'allocation_groups_verified':len(assignments),'final_models_and_both_split_scores_reforwarded':len(verified),'primary':primary,'cases':verified,
        'child_seconds':result['child_execution_seconds'],'parent_control_seconds':result['parent_control_seconds'],'official_test_evaluated':bool(verified)}
    with target.open('xb') as file:file.write(canonical(audit))
    print(json.dumps({'qualified':len(verified),'failed_or_unrun':len(failed),'primary_complete':primary['complete'],'source_commit':commit}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.study,args.cache,args.source_root,args.output)
