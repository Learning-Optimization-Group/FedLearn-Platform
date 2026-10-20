"""The prospectively fixed NIH phase-2 study; no outcome-dependent cells or retries."""
from __future__ import annotations

from benchmarks.nih_optimization_pilot import FIXED_REPORT_SHA256, _config as _base_config
from fedlearn.simulation.config import resolve

DOCUMENT='docs/artifacts/capstone/nih-phase2-study-v1-protocol.md'
PROTOCOL_SHA256='4245a8446d3d88a93cd921d24f7a4f565273a7285663e446e43f1fcb123012a0'
MASK_SHA256='352b10c16342e4160d06a1e536ea82d201ea739eec17ed67fc9801141e60470d'
SEEDS=tuple(range(20261101,20261111))
PER_CELL_SECONDS,TOTAL_SECONDS,MAX_RSS_BYTES=180,10800,2**31


def configs(cache,pin):
    recipes=[]
    families=(('primary',SEEDS,('linear',),(80,),.05,('clean','sign_flip','label_flip'),('none',1,4,8)),
              ('head',SEEDS[:3],('mlp8','mlp128','mlp512'),(80,),.05,('clean','sign_flip'),('none',4)),
              ('cohort',SEEDS[:3],('linear',),(10,1000),.05,('clean',),('none',4)),
              ('clipping',SEEDS[:3],('linear',),(80,),.4,('clean','sign_flip'),('none',4)))
    for family,seeds,heads,cohorts,clip,conditions,modes in families:
        for seed in seeds:
            for head in heads:
                for n in cohorts:
                    for condition in conditions:
                        for rule in ('mean','trimmed_mean'):
                            for mode in modes:
                                privacy='none' if mode=='none' else f'eps{mode}'
                                name=f'{family}_s{seed}_{head}_n{n}_c{"005" if clip==.05 else "04"}_{condition}_{rule}_{privacy}'
                                cfg=_base_config(cache,pin,n,.003,name=name,seed=seed,clip=clip,rule=rule)
                                cfg['client']['local_epochs']=7
                                cfg['workload']['evaluation_split']='test'
                                cfg['workload']['head']={'type':'linear','hidden':128} if head=='linear' else {'type':'mlp','hidden':int(head[3:])}
                                cfg['privacy']={'mode':'none' if mode=='none' else 'central','epsilon':4. if mode=='none' else float(mode),'delta':1e-5,'adjacency':'replace_one'}
                                cfg['behaviors']=([] if condition=='clean' else
                                    [{'type':'sign_flip_scale','fraction':.2,'scale':-4.}] if condition=='sign_flip' else
                                    [{'type':'label_flip','fraction':.2}])
                                recipes.append(resolve(cfg))
    return recipes


import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from benchmarks.nih_optimization_pilot import source_identity as _source_identity, verify_imports
from benchmarks.nih_validation_pilot import _checkpoint
from benchmarks.nih_head_cost_probe import _save_record
from benchmarks.nih_final_artifacts import write_final_artifacts, verify_final_artifacts
from benchmarks.nih_phase2_analysis import cell_analysis, study_analysis
from fedlearn.simulation.config import config_hash
from fedlearn.simulation.core_qualification import CoreQualificationError, qualify_core_record, run_core
from fedlearn.simulation.nih_cache import load_nih_cache

PROOF_REQUIRED=('framework/src/fedlearn/simulation/engine.py',
    'framework/src/fedlearn/simulation/core_qualification.py','framework/benchmarks/nih_final_artifacts.py',
    'framework/benchmarks/nih_phase2_study.py','framework/benchmarks/nih_phase2_analysis.py')
CHILD_BOOTSTRAP='''
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'framework/src'),str(root/'framework')]
from benchmarks.nih_phase2_study import run_child
raise SystemExit(run_child(root,Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4])))
'''


class SourceIntegrityError(ValueError):
    pass


def source_identity(root):
    identity=_source_identity(root)
    raw=(Path(root)/DOCUMENT).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise SourceIntegrityError('frozen protocol bytes differ')
    identity['files'][DOCUMENT]=PROTOCOL_SHA256
    return identity


def _verify(root,identity):
    try:return verify_imports(root,identity['files'])
    except ValueError as error:raise SourceIntegrityError(str(error)) from error


def _cache(path):
    try:
        cache=load_nih_cache(Path(path),expected_report_sha256=FIXED_REPORT_SHA256)
        if not cache['provenance']['pretrained']:raise ValueError('random/control cache refused')
        return cache
    except (OSError,ValueError) as error:raise SourceIntegrityError(f'cache integrity: {error}') from error


def read_mask(path,cache):
    try:
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=MASK_SHA256:raise ValueError('frozen mask digest differs')
        mask=json.loads(raw);test=cache['splits']['test'];ids=list(test['image_ids'])
        excluded=mask['excluded_test_image_ids']
        selected=[i for i,image in enumerate(ids) if image in set(excluded)]
        if (mask['test_manifest_sha256']!=cache['provenance']['selected_manifest_hashes']['test.csv']
                or mask['cohort_report_sha256']!=cache['provenance']['curated_cohort_sha256']
                or mask['audit_report_sha256']!=cache['provenance']['image_audit_sha256']
                or len(excluded)!=len(set(excluded)) or len(selected)!=9 or len(excluded)!=9
                or len({test['patient_ids'][i] for i in selected})!=7 or int(test['y'][selected].sum())!=3
                or mask['performance_examined'] is not False or mask['training_or_primary_test_membership_changed'] is not False):
            raise ValueError('mask/cache semantics differ')
        return mask
    except (OSError,KeyError,TypeError,ValueError) as error:raise SourceIntegrityError(f'mask integrity: {error}') from error


def _protocol_recipes(protocol):
    recipes=configs(protocol['cache'],FIXED_REPORT_SHA256)
    header={'per_cell_seconds_cap':PER_CELL_SECONDS,'total_child_seconds_cap':TOTAL_SECONDS,
        'max_child_rss_bytes':MAX_RSS_BYTES,'evaluation_split':'test',
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':4}}
    if any(json.dumps(protocol.get(k),sort_keys=True,allow_nan=False)!=json.dumps(v,sort_keys=True,allow_nan=False) for k,v in header.items()):
        raise SourceIntegrityError('declared runtime/resource/evaluation protocol differs')
    if (protocol['study_version']!='nih-phase2-v1' or protocol['protocol_document_sha256']!=PROTOCOL_SHA256
            or protocol['features_report_sha256']!=FIXED_REPORT_SHA256 or protocol['test_mask_sha256']!=MASK_SHA256
            or protocol['configs']!=recipes or protocol['config_hashes']!={c['name']:config_hash(c) for c in recipes}):
        raise SourceIntegrityError('study protocol/config identities differ')
    return recipes


def _read_evidence(path):
    try:
        value=json.loads(Path(path).read_text())
        if not isinstance(value,dict):raise ValueError('expected JSON object')
        return value
    except (OSError,TypeError,ValueError) as error:
        raise SourceIntegrityError(f'invalid protocol/config evidence: {error}') from error


def _validate_protocol(protocol):
    try:return _protocol_recipes(protocol)
    except (KeyError,TypeError,ValueError) as error:
        raise SourceIntegrityError(f'invalid study protocol: {error}') from error


def _check_fixed_evidence(output,protocol,cfg):
    if (_read_evidence(output/'protocol.json')!=protocol
            or _read_evidence(output/'configs'/f'{cfg["name"]}.json')!=cfg):
        raise SourceIntegrityError('generated protocol/config evidence differs')


def child_command(root,config,output,protocol):
    return [sys.executable,'-I','-c',CHILD_BOOTSTRAP,str(root),str(config),str(output),str(protocol)]


def run_child(root,config_path,output,protocol_path):
    output.mkdir(parents=True)
    protocol=identity=None
    outcome={'status':'failed'};record=None;core_seconds=artifact_seconds=None
    try:
        protocol=_read_evidence(protocol_path)
        try:identity=protocol['source_identity']
        except KeyError as error:raise SourceIntegrityError('missing protocol source identity') from error
        proof=_verify(root,identity)
        _checkpoint(output/'executed-source-start.json',{**proof,'phase':'before_engine'})
        recipes=_validate_protocol(protocol)
        if source_identity(root)!=identity:raise SourceIntegrityError('child source identity changed')
        cfg=_read_evidence(config_path)
        if cfg not in recipes or cfg.get('name')!=config_path.stem:raise SourceIntegrityError('child recipe differs from fixed cell')
        cache=_cache(protocol['cache']);mask=read_mask(protocol_path.parent/'test-mask.json',cache)
        del cache
        torch.set_num_threads(4);captured=[]
        core_started=time.monotonic()
        try:record=run_core(cfg,feature_report_sha256=FIXED_REPORT_SHA256,clients=cfg['federation']['num_clients'],evaluation_split='test',on_final_parameters=captured.append)
        finally:core_seconds=time.monotonic()-core_started
        encoding=_save_record(output/'record.json',record)
        if len(captured)!=1:raise ValueError('missing qualified final parameters')
        artifact_started=time.monotonic()
        try:
            write_final_artifacts(cfg,record,captured[0],output/'final',feature_report_sha256=FIXED_REPORT_SHA256)
            bundle=verify_final_artifacts(cfg,record,output/'final',feature_report_sha256=FIXED_REPORT_SHA256)
            _checkpoint(output/'score-analysis.json',cell_analysis(bundle,mask['excluded_test_image_ids']))
        finally:artifact_seconds=time.monotonic()-artifact_started
        outcome={'status':'ok','record_encoding':encoding}
    except Exception as error:
        outcome={'status':'failed','error':str(error),'error_kind':type(error).__name__}
        if isinstance(error,CoreQualificationError) and error.record is not None:record=error.record
        if record is not None:outcome['record_encoding']=_save_record(output/'record.json',record)
        traceback.print_exc()
    finally:
        outcome.update(core_execution_seconds=core_seconds,artifact_execution_seconds=artifact_seconds)
        try:
            if protocol is None or identity is None:raise SourceIntegrityError('unreadable protocol/source identity')
            _validate_protocol(protocol)
            if _read_evidence(protocol_path)!=protocol:raise SourceIntegrityError('child protocol evidence changed')
            read_mask(protocol_path.parent/'test-mask.json',_cache(protocol['cache']))
            proof=_verify(root,identity)
            if source_identity(root)!=identity:raise SourceIntegrityError('child source identity changed')
            _checkpoint(output/'executed-source.json',{**proof,'phase':'after_engine_return' if outcome['status']=='ok' else 'after_engine_error'})
        except Exception as error:
            outcome.update(status='failed',error=str(error),error_kind='SourceIntegrityError');traceback.print_exc()
        _checkpoint(output/'execution.json',outcome)
    return 0 if outcome['status']=='ok' else 2


def _check_proofs(directory,identity):
    try:
        for name,phase in [('executed-source-start.json','before_engine'),('executed-source.json','after_engine_return')]:
            proof=json.loads((directory/name).read_text());files=proof['files']
            if (proof['phase']!=phase or any(files.get(k)!=identity['files'][k] for k in PROOF_REQUIRED)
                    or any(identity['files'].get(k)!=v for k,v in files.items())):
                raise ValueError('loaded source fingerprints differ')
    except (OSError,KeyError,TypeError,ValueError) as error:raise SourceIntegrityError(f'invalid source proof: {error}') from error


def qualify_saved_case(directory,cfg,identity,protocol,child_seconds):
    _check_proofs(directory,identity)
    execution=json.loads((directory/'execution.json').read_text())
    if execution['status']!='ok' or execution['record_encoding']!='json':raise ValueError('child did not return complete JSON evidence')
    record=json.loads((directory/'record.json').read_text());meta=record['meta'];wl=meta['workload']
    bare={**record,'meta':{k:v for k,v in meta.items() if k!='core_qualification'}}
    marker=qualify_core_record(bare,cfg,feature_report_sha256=FIXED_REPORT_SHA256,clients=cfg['federation']['num_clients'],evaluation_split='test')
    head=cfg['workload']['head'];parameters=1026 if head['type']=='linear' else 515*head['hidden']+2
    if (meta.get('core_qualification')!=marker or marker['feature_control'] or meta['provenance']['git_commit']!=identity['git_commit']
            or wl['parameters']!=parameters or wl['training_device']!='cpu' or wl['torch_num_threads']!=4):
        raise ValueError('record source/core/model/device identity differs')
    runtime=protocol['runtime']
    if any(meta['provenance'][k]!=runtime[k] for k in ('python','torch','numpy')):raise SourceIntegrityError('runtime versions differ')
    rss=meta['peak_rss_bytes']
    if type(rss) is not int or not 0<rss<=MAX_RSS_BYTES or meta['peak_rss_scope'] not in ('run','process'):raise ValueError('invalid/excessive observed RSS')
    loop,core,artifact=meta['wall_seconds'],execution['core_execution_seconds'],execution['artifact_execution_seconds']
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (loop,core,artifact,child_seconds))
            or not loop<=core or core+artifact>child_seconds):raise ValueError('invalid child/core/loop/artifact timing')
    mask=read_mask(directory.parent.parent/'test-mask.json',_cache(protocol['cache']))
    bundle=verify_final_artifacts(cfg,record,directory/'final',feature_report_sha256=FIXED_REPORT_SHA256)
    analysis=cell_analysis(bundle,mask['excluded_test_image_ids'])
    if json.loads((directory/'score-analysis.json').read_text())!=analysis:raise ValueError('saved analysis differs from actual final scores')
    for key in ('auc','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity'):
        if not math.isclose(analysis['test'][key],record['rounds'][-1][key],rel_tol=0. if key=='auc' else 1e-6,abs_tol=1e-4 if key=='ece' else 1e-6 if key!='auc' else 1e-12):
            raise ValueError(f'final score metric differs from engine: {key}')
    stage={}
    for line in record['rounds']:
        for key,value in line['time']['wall_s'].items():
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('invalid stage timing')
            stage[key]=stage.get(key,0.)+value
    for key,section in (('up_bytes','comm'),('down_bytes','comm'),('client_flops','compute'),('client_flops_analytic','compute')):
        values=[line[section][key] for line in record['rounds']]
        total=meta['totals'][key]
        if type(total) is not int or total<0 or any(type(v) is not int or v<0 for v in values) or sum(values)!=total:
            raise ValueError('invalid/inconsistent cost totals')
    if any(l['compute']['client_flops']!=l['compute']['client_flops_analytic'] for l in record['rounds']):raise ValueError('measured/analytic FLOPs differ')
    return {'record_final_digest':meta['final_digest'],'analysis':analysis,
        'summary':{'parameters':parameters,'federation_loop_seconds':loop,'core_execution_seconds':core,
            'artifact_execution_seconds':artifact,'stage_wall_seconds':stage,'peak_rss_bytes':rss,'peak_rss_scope':meta['peak_rss_scope'],
            'totals':meta['totals'],'final_engine_metrics':{k:record['rounds'][-1][k] for k in ('auc','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity')}}}


def _artifacts(output,directory,log):
    paths=[p for p in directory.rglob('*') if p.is_file()] if directory.exists() else []
    if log.is_file():paths.append(log)
    return {p.relative_to(directory).as_posix() if p.is_relative_to(directory) else 'child.log':
        {'file':p.relative_to(output).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths}


def run_study(cache,pin,mask_path,output):
    torch.set_num_threads(4)
    started=time.monotonic();root=Path(__file__).resolve().parents[2]
    cache,output=Path(cache).resolve(),Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('refusing existing study output')
    if output.resolve().is_relative_to(cache):raise ValueError('output must be outside immutable cache')
    if pin!=FIXED_REPORT_SHA256:raise ValueError('study requires frozen cache pin')
    identity=source_identity(root);_verify(root,identity)
    if hashlib.sha256((root/DOCUMENT).read_bytes()).hexdigest()!=PROTOCOL_SHA256:raise SourceIntegrityError('frozen protocol differs')
    read_mask(mask_path,_cache(cache))
    output=output.resolve();output.mkdir(parents=True)
    (output/'test-mask.json').write_bytes(Path(mask_path).read_bytes())
    for name in ('configs','logs','runs'):(output/name).mkdir()
    recipes=configs(cache,pin)
    protocol={'study_version':'nih-phase2-v1','source_identity':identity,'protocol_document_sha256':PROTOCOL_SHA256,
        'cache':str(cache),'features_report_sha256':pin,'test_mask_sha256':MASK_SHA256,
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':4},
        'configs':recipes,'config_hashes':{c['name']:config_hash(c) for c in recipes},
        'per_cell_seconds_cap':PER_CELL_SECONDS,'total_child_seconds_cap':TOTAL_SECONDS,'max_child_rss_bytes':MAX_RSS_BYTES,'evaluation_split':'test'}
    _checkpoint(output/'protocol.json',protocol)
    cases=[{'name':c['name'],'config_hash':config_hash(c),'status':'not_run','reason':'pending'} for c in recipes]
    for cfg in recipes:_checkpoint(output/'configs'/f'{cfg["name"]}.json',cfg)
    _checkpoint(output/'index.json',{'cases':cases})
    environment=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    spent=0.;integrity_failed=False
    for i,cfg in enumerate(recipes):
        name=cfg['name'];base={'name':name,'config_hash':config_hash(cfg)}
        if integrity_failed or spent>=TOTAL_SECONDS:
            cases[i]={**base,'status':'not_run','reason':'source_integrity' if integrity_failed else 'global_budget_exhausted'}
            _checkpoint(output/'index.json',{'cases':cases});continue
        before=time.monotonic();child_seconds=0.;directory=output/'runs'/name;log=output/'logs'/f'{name}.log'
        try:
            _check_fixed_evidence(output,protocol,cfg)
            if source_identity(root)!=identity:raise SourceIntegrityError('source identity changed')
            read_mask(output/'test-mask.json',_cache(cache))
            with log.open('xb') as stream:
                child_started=time.monotonic()
                try:subprocess.run(child_command(root,output/'configs'/f'{name}.json',directory,output/'protocol.json'),cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=min(PER_CELL_SECONDS,TOTAL_SECONDS-spent),check=True)
                finally:child_seconds=time.monotonic()-child_started;spent+=child_seconds
            _check_fixed_evidence(output,protocol,cfg)
            if source_identity(root)!=identity:raise SourceIntegrityError('source identity changed')
            _check_proofs(directory,identity)
            case={**base,'status':'qualified',**qualify_saved_case(directory,cfg,identity,protocol,child_seconds)}
        except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
            integrity_failed=integrity_failed or isinstance(error,SourceIntegrityError)
            try:
                _check_fixed_evidence(output,protocol,cfg)
                integrity_failed=integrity_failed or source_identity(root)!=identity
                read_mask(output/'test-mask.json',_cache(cache))
                if (directory/'execution.json').exists():integrity_failed=integrity_failed or json.loads((directory/'execution.json').read_text()).get('error_kind')=='SourceIntegrityError'
            except (OSError,ValueError,subprocess.SubprocessError):integrity_failed=True
            case={**base,'status':'timeout' if isinstance(error,subprocess.TimeoutExpired) else 'failed','error':str(error)}
        elapsed=time.monotonic()-before
        cases[i]={**case,'elapsed_seconds':elapsed,'child_execution_seconds':child_seconds,'parent_control_seconds':elapsed-child_seconds,'artifacts':_artifacts(output,directory,log)}
        _checkpoint(output/'index.json',{'cases':cases})
        print(json.dumps({'name':name,'status':case['status'],'child_seconds':child_seconds}),flush=True)
    aggregate=study_analysis(cases,recipes)
    elapsed=time.monotonic()-started
    result={'study_version':'nih-phase2-v1','source_identity':identity,'protocol_document_sha256':PROTOCOL_SHA256,
        'features_report_sha256':pin,'test_mask_sha256':MASK_SHA256,'evaluation_split':'test',
        'complete':not integrity_failed and all(c['status']=='qualified' for c in cases),'source_integrity_failed':integrity_failed,
        'elapsed_seconds':elapsed,'child_execution_seconds':spent,'parent_control_seconds':elapsed-spent,
        'cases':cases,'analysis':aggregate}
    _checkpoint(output/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--expected-report-sha256',required=True)
    parser.add_argument('--test-mask',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:result=run_study(args.cache,args.expected_report_sha256,args.test_mask,args.output)
    except (OSError,ValueError,subprocess.SubprocessError) as error:parser.exit(2,f'error: {error}\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_identity','analysis')}))
    if not result['complete']:parser.exit(2,'study incomplete; retain failed/unrun cases\n')



def analyze_saved_study(output):
    """Rebuild analysis from verified raw records/checkpoints, never trusted index metrics."""
    torch.set_num_threads(4)
    output=Path(output).resolve();root=Path(__file__).resolve().parents[2]
    try:
        protocol=json.loads((output/'protocol.json').read_text())
        recipes=_validate_protocol(protocol);identity=protocol['source_identity']
        _verify(root,identity)
        read_mask(output/'test-mask.json',_cache(protocol['cache']))
        cases=json.loads((output/'index.json').read_text())['cases']
        expected={c['name']:c for c in recipes}
        if len(cases)!=360 or len({c['name'] for c in cases})!=360 or {c['name'] for c in cases}!=set(expected):raise ValueError('saved case coverage differs')
        checked=[]
        for case in cases:
            cfg=expected[case['name']]
            if case['config_hash']!=config_hash(cfg):raise ValueError('saved configuration hash differs')
            if json.loads((output/'configs'/f'{cfg["name"]}.json').read_text())!=cfg:raise ValueError('saved configuration differs')
            fresh_artifacts=_artifacts(output,output/'runs'/cfg['name'],output/'logs'/f'{cfg["name"]}.log')
            if case.get('artifacts',{})!=fresh_artifacts:raise ValueError('saved artifact registry differs')
            for artifact in case.get('artifacts',{}).values():
                path=output/artifact['file']
                if (not path.resolve().is_relative_to(output) or path.is_symlink() or not path.is_file()
                        or path.stat().st_size!=artifact['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']):raise ValueError('saved artifact hash/path differs')
            if case['status']=='qualified':
                fresh=qualify_saved_case(output/'runs'/cfg['name'],cfg,identity,protocol,case['child_execution_seconds'])
                if any(case[key]!=fresh[key] for key in ('summary','analysis','record_final_digest')):raise ValueError('saved summary differs from raw evidence')
                checked.append({**case,**fresh})
            else:checked.append(case)
        return study_analysis(checked,recipes)
    except (OSError,KeyError,TypeError) as error:raise ValueError(f'incomplete saved study: {error}') from error


if __name__=='__main__':main()
