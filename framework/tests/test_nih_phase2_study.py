"""The frozen study's exact bounded cells and source-bound execution contract."""
import json
from collections import Counter

from benchmarks import nih_phase2_study as study
from tests.test_capstone_nih_features import template

PIN='6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd'


def test_exact_360_cells_and_family_order_with_reused_controls():
    recipes=study.configs('cache',PIN)
    assert len(recipes)==len({c['name'] for c in recipes})==360
    assert Counter(c['name'].split('_',1)[0] for c in recipes)=={'primary':240,'head':72,'cohort':24,'clipping':24}
    assert [recipes[i]['name'] for i in (0,1,2,3,239,240,311,312,335,336,359)]==[
        'primary_s20261101_linear_n80_c005_clean_mean_none',
        'primary_s20261101_linear_n80_c005_clean_mean_eps1',
        'primary_s20261101_linear_n80_c005_clean_mean_eps4',
        'primary_s20261101_linear_n80_c005_clean_mean_eps8',
        'primary_s20261110_linear_n80_c005_label_flip_trimmed_mean_eps8',
        'head_s20261101_mlp8_n80_c005_clean_mean_none',
        'head_s20261103_mlp512_n80_c005_sign_flip_trimmed_mean_eps4',
        'cohort_s20261101_linear_n10_c005_clean_mean_none',
        'cohort_s20261103_linear_n1000_c005_clean_trimmed_mean_eps4',
        'clipping_s20261101_linear_n80_c04_clean_mean_none',
        'clipping_s20261103_linear_n80_c04_sign_flip_trimmed_mean_eps4']
    scientific=[]
    for c in recipes:
        cfg={k:v for k,v in c.items() if k!='name'}
        scientific.append(json.dumps(cfg,sort_keys=True))
    assert len(set(scientific))==360  # Shared controls are referenced, not rerun.


def test_exact_primary_recipe_and_matched_streams():
    recipes=study.configs('cache',PIN)
    for c in recipes:
        seed=c['seed']
        assert c['seeds']=={'partition':seed,'init':seed,'client':seed,'attack':seed+1000,'noise':seed+2000}
        assert c['federation']['rounds']==20 and c['federation']['clients_per_round']==c['federation']['num_clients']
        assert c['federation']['partition']['type']=='patient_iid'
        assert c['client']['local_epochs']==7 and c['client']['lr']==.003 and c['client']['batch_size']==256
        assert c['server']['rule_params']['trim_ratio']==.2 and c['server']['optimizer']['type']=='none'
        assert c['workload']['evaluation_split']=='test' and c['workload']['auc_score']=='margin_v2'
        assert not c['workload']['balance'] and not c['workload']['allow_random_features']
        assert c['privacy']['delta']==1e-5 and c['privacy']['adjacency']=='replace_one'
    assert {c['seed'] for c in recipes[:240]}==set(range(20261101,20261111))
    assert {c['seed'] for c in recipes[240:]}=={20261101,20261102,20261103}
    assert all(c['server']['clip_norm']==.05 for c in recipes[:336])
    assert all(c['server']['clip_norm']==.4 for c in recipes[336:])
    assert recipes[0]['behaviors']==[]
    assert recipes[8]['behaviors']==[{'type':'sign_flip_scale','fraction':.2,'scale':-4.}]
    assert recipes[16]['behaviors']==[{'type':'label_flip','fraction':.2}]


import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import pytest


def _orchestrate(tmp_path,monkeypatch,outcome,verification=None):
    root=tmp_path/'repo';doc=root/study.DOCUMENT;doc.parent.mkdir(parents=True);doc.write_text('frozen')
    monkeypatch.setattr(study,'__file__',str(root/'framework/benchmarks/nih_phase2_study.py'))
    monkeypatch.setattr(study,'PROTOCOL_SHA256',hashlib.sha256(doc.read_bytes()).hexdigest())
    mask=tmp_path/'mask.json';mask.write_text('mask')
    monkeypatch.setattr(study,'MASK_SHA256',hashlib.sha256(mask.read_bytes()).hexdigest())
    identity={'git_commit':'test','files':{k:k+'-hash' for k in study.PROOF_REQUIRED}}
    state={'calls':0,'now':0.,'changed':False}
    monkeypatch.setattr(study,'source_identity',lambda root:{**identity,'git_commit':'changed'} if state['changed'] else identity)
    monkeypatch.setattr(study,'verify_imports',lambda *a:{'files':identity['files']})
    monkeypatch.setattr(study,'_cache',lambda *a:{'provenance':{'pretrained':True}})
    monkeypatch.setattr(study,'read_mask',lambda *a:{'excluded_test_image_ids':['bound']})
    monkeypatch.setattr(study.time,'monotonic',lambda:state['now'])
    if outcome=='budget':monkeypatch.setattr(study,'TOTAL_SECONDS',0)
    def child(command,**kw):
        state['calls']+=1
        protocol=json.loads((tmp_path/'out/protocol.json').read_text())
        index=json.loads((tmp_path/'out/index.json').read_text())
        assert len(index['cases'])==len(protocol['configs'])==len(protocol['config_hashes'])==360
        assert all(protocol['config_hashes'][c['name']]==hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for c in protocol['configs'])
        directory=Path(command[-2]);directory.mkdir()
        for filename,phase in [('executed-source-start.json','before_engine'),('executed-source.json','after_engine_return')]:
            if outcome=='proof' and state['calls']==1 and phase=='after_engine_return':continue
            (directory/filename).write_text(json.dumps({'phase':phase,'files':identity['files']}))
        (directory/'execution.json').write_text(json.dumps({'status':'ok','record_encoding':'json','core_execution_seconds':1.,'artifact_execution_seconds':.5}))
        state['now']+=180. if outcome=='exhaustion' else 3.
        if outcome=='exhaustion':
            assert kw['timeout']==180
            raise subprocess.TimeoutExpired(command,kw['timeout'])
        if state['calls']==1:
            if outcome=='error':raise subprocess.CalledProcessError(2,command)
            if outcome=='timeout':raise subprocess.TimeoutExpired(command,kw['timeout'])
            if outcome=='mutation':state['changed']=True
            if outcome in ('protocol_json','protocol_field','config_json'):
                path=tmp_path/'out/protocol.json' if outcome!='config_json' else Path(command[-3])
                if outcome=='protocol_field':
                    corrupted=json.loads(path.read_text());corrupted.pop('configs');path.write_text(json.dumps(corrupted))
                else:path.write_text('{broken')
                raise subprocess.CalledProcessError(2,command)
            if outcome=='cache_mutation':
                def changed(*a):raise study.SourceIntegrityError('cache changed')
                monkeypatch.setattr(study,'_cache',changed)
                raise subprocess.CalledProcessError(2,command)
    monkeypatch.setattr(study.subprocess,'run',child)
    def qualify(directory,cfg,identity,protocol,child_seconds):
        if verification is not None:verification()
        h=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        return {'summary':{'federation_loop_seconds':.5,'core_execution_seconds':1.,'artifact_execution_seconds':.5},
            'record_final_digest':'d'*16,'analysis':{'evaluation_split':'test','config_hash':h,'final_digest':'d'*16,
                'features_report_sha256':PIN,'test':{'auc':.6,'loss':.6,'ece':.1,'accuracy':.6,'balanced_accuracy':.6,'sensitivity':.5,'specificity':.7}}}
    monkeypatch.setattr(study,'qualify_saved_case',qualify)
    result=study.run_study(tmp_path/'cache',PIN,mask,tmp_path/'out')
    assert len(result['cases'])==360
    assert json.loads((tmp_path/'out/index.json').read_text())['cases']==result['cases']
    assert result['elapsed_seconds']==pytest.approx(result['child_execution_seconds']+result['parent_control_seconds'])
    if outcome=='success':assert result['complete'] and result['analysis']['primary']['complete'] and state['calls']==360
    elif outcome=='budget':assert state['calls']==0 and all(c['status']=='not_run' for c in result['cases'])
    elif outcome=='exhaustion':
        assert state['calls']==60 and result['child_execution_seconds']==10800.
        assert [c['status'] for c in result['cases']]==['timeout']*60+['not_run']*300
    elif outcome in ('mutation','proof','cache_mutation','protocol_json','protocol_field','config_json'):
        assert state['calls']==1 and result['source_integrity_failed']
        assert all(c['status']=='not_run' for c in result['cases'][1:])
    else:
        assert state['calls']==360 and not result['complete'] and not result['analysis']['primary']['complete']
        assert result['cases'][0]['status']==('timeout' if outcome=='timeout' else 'failed')
        assert all(c['status']=='qualified' for c in result['cases'][1:])
    return result


@pytest.mark.parametrize('outcome',['success','error','timeout','budget','exhaustion','mutation','proof','cache_mutation','protocol_json','protocol_field','config_json'])
def test_initial_all_case_persistence_and_failure_integrity_budgets(tmp_path,monkeypatch,outcome):
    _orchestrate(tmp_path,monkeypatch,outcome)


def test_invalid_pin_and_existing_output_refuse_without_cache_access(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'_cache',lambda *a:pytest.fail('cache reached'))
    with pytest.raises(ValueError):study.run_study(tmp_path/'cache','0'*64,tmp_path/'mask',tmp_path/'out')
    out=tmp_path/'out';out.mkdir();(out/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):study.run_study(tmp_path/'cache',PIN,tmp_path/'mask',out)
    assert (out/'sentinel').read_text()=='keep'


@pytest.fixture
def saved_case(tmp_path,monkeypatch):
    import numpy as np
    from benchmarks.nih_phase2_analysis import cell_analysis
    from tests.test_nih_mechanism_qualification import _record
    cfg=study.configs('cache',PIN)[0];record=_record(cfg)
    record['meta']['workload']['evaluation_split']='test'
    record['meta']['core_qualification']['evaluation_split']='test'
    record['meta']['final_digest']='d'*16
    record['meta']['wall_seconds']=15.
    record['meta']['provenance']={'git_commit':'test','python':'py','torch':'torch','numpy':'numpy'}
    record['meta']['totals'].update(up_bytes=6800000,down_bytes=6800000,client_flops=1000,client_flops_analytic=1000)
    for line in record['rounds']:
        line.update(auc=.5,auc_probability=.5,loss=math.log(2),accuracy=.5,balanced_accuracy=.5,sensitivity=0.,specificity=1.,ece=0.)
        line['comm']={'up_bytes':340000,'down_bytes':340000}
    directory=tmp_path/'out/runs'/cfg['name'];directory.mkdir(parents=True)
    identity={'git_commit':'test','files':{k:k+'-hash' for k in study.PROOF_REQUIRED}}
    for filename,phase in [('executed-source-start.json','before_engine'),('executed-source.json','after_engine_return')]:
        (directory/filename).write_text(json.dumps({'phase':phase,'files':identity['files']}))
    (directory/'execution.json').write_text(json.dumps({'status':'ok','record_encoding':'json','core_execution_seconds':16.,'artifact_execution_seconds':2.}))
    def split(prefix):return {'logits':np.zeros((4,2),dtype=np.float32),'margin':np.zeros(4,dtype=np.float64),
        'labels':np.array([0,1,0,1]),'image_ids':np.array([f'{prefix}{i}' for i in range(4)]),'patient_ids':np.array(['1','1','2','2'])}
    bundle={'manifest':{'config_hash':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        'final_digest':'d'*16,'features_report_sha256':PIN},'splits':{'validation':split('v'),'test':split('t')}}
    (directory/'score-analysis.json').write_text(json.dumps(cell_analysis(bundle,['t0'])))
    monkeypatch.setattr(study,'_cache',lambda *a:{})
    monkeypatch.setattr(study,'read_mask',lambda *a:{'excluded_test_image_ids':['t0']})
    monkeypatch.setattr(study,'verify_final_artifacts',lambda *a,**k:bundle)
    protocol={'cache':'cache','runtime':{'python':'py','torch':'torch','numpy':'numpy'}}
    return directory,cfg,identity,protocol,record


import math


def test_parent_requalifies_raw_record_against_independent_scores(saved_case):
    directory,cfg,identity,protocol,record=saved_case
    (directory/'record.json').write_text(json.dumps(record))
    result=study.qualify_saved_case(directory,cfg,identity,protocol,20.)
    assert result['analysis']['test']['auc']==.5 and result['record_final_digest']=='d'*16


@pytest.mark.parametrize('bad',['quorum','marker','source','runtime','rss','metric','metric_tiny','cost','cost_sum','stage'])
def test_raw_record_cannot_bypass_parent_qualification(saved_case,bad):
    directory,cfg,identity,protocol,record=saved_case
    if bad=='quorum':record['rounds'][0]['participants']['accepted']=79
    elif bad=='marker':record['meta']['core_qualification']['research_record_dp_safe']=True
    elif bad=='source':record['meta']['provenance']['git_commit']='foreign'
    elif bad=='runtime':record['meta']['provenance']['torch']='foreign'
    elif bad=='rss':record['meta']['peak_rss_bytes']=2**31+1
    elif bad=='metric':record['rounds'][-1]['auc']=.9
    elif bad=='metric_tiny':record['rounds'][-1]['auc']+=5e-8
    elif bad=='cost':record['meta']['totals']['up_bytes']=-1
    elif bad=='cost_sum':record['meta']['totals']['up_bytes']+=1
    else:record['rounds'][0]['time']['wall_s']['train']=-1.
    (directory/'record.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):study.qualify_saved_case(directory,cfg,identity,protocol,20.)


def test_saved_reanalysis_reloads_artifacts_instead_of_trusting_index(tmp_path,monkeypatch):
    original=_orchestrate(tmp_path,monkeypatch,'success')
    reloaded=study.analyze_saved_study(tmp_path/'out')
    assert reloaded==original['analysis']
    index=json.loads((tmp_path/'out/index.json').read_text())
    index['cases'][0]['analysis']['test']['auc']=.99
    (tmp_path/'out/index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):study.analyze_saved_study(tmp_path/'out')


def _protocol(cache,identity):
    recipes=study.configs(cache,PIN)
    return {'study_version':'nih-phase2-v1','source_identity':identity,'cache':str(cache),
        'protocol_document_sha256':study.PROTOCOL_SHA256,'features_report_sha256':PIN,'test_mask_sha256':study.MASK_SHA256,
        'configs':recipes,'config_hashes':{c['name']:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for c in recipes},
        'per_cell_seconds_cap':180,'total_child_seconds_cap':10800,'max_child_rss_bytes':2**31,'evaluation_split':'test',
        'runtime':{'python':study.platform.python_version(),'torch':study.torch.__version__,'numpy':study.np.__version__,'cpu_threads':4}}


@pytest.mark.parametrize('bad',['oracle','validation','epsilon','foreign_name'])
def test_main_child_refuses_nonprotocol_recipe_before_core(tmp_path,monkeypatch,bad):
    root=Path(study.__file__).resolve().parents[2];identity={'git_commit':'control','files':{}}
    protocol=_protocol('missing-cache',identity);cfg=copy.deepcopy(protocol['configs'][0])
    original_name=cfg['name']
    if bad=='oracle':cfg['behaviors']=[{'type':'ipm','fraction':.2,'scale':4.}]
    elif bad=='validation':cfg['workload']['evaluation_split']='validation'
    elif bad=='epsilon':cfg['privacy']={'mode':'central','epsilon':2.,'delta':1e-5,'adjacency':'replace_one'}
    else:cfg['name']=protocol['configs'][1]['name']
    pp=tmp_path/'protocol.json';pp.write_text(json.dumps(protocol))
    cp=tmp_path/f'{original_name}.json';cp.write_text(json.dumps(cfg))
    monkeypatch.setattr(study,'source_identity',lambda *_:identity)
    monkeypatch.setattr(study,'verify_imports',lambda *a:{'files':{}})
    monkeypatch.setattr(study,'_cache',lambda *a:{})
    monkeypatch.setattr(study,'read_mask',lambda *a:{'excluded_test_image_ids':[]})
    monkeypatch.setattr(study,'run_core',lambda *a,**kw:pytest.fail('core reached'))
    out=tmp_path/'child';assert study.run_child(root,cp,out,pp)==2
    execution=json.loads((out/'execution.json').read_text())
    assert execution['error_kind']=='SourceIntegrityError' and 'recipe' in execution['error']
    assert not (out/'record.json').exists() and not (out/'final').exists()


def test_mask_checks_hash_and_semantics_against_bound_cache(tmp_path,monkeypatch):
    import numpy as np
    ids=[f'i{i}' for i in range(9)]
    cache={'splits':{'test':{'image_ids':ids,'patient_ids':['1','1','2','2','3','4','5','6','7'],'y':np.array([1,1,1,0,0,0,0,0,0])}},
        'provenance':{'selected_manifest_hashes':{'test.csv':'t'},'curated_cohort_sha256':'c','image_audit_sha256':'a'}}
    mask={'excluded_test_image_ids':ids,'test_manifest_sha256':'t','cohort_report_sha256':'c','audit_report_sha256':'a',
        'performance_examined':False,'training_or_primary_test_membership_changed':False}
    p=tmp_path/'mask.json';p.write_text(json.dumps(mask));pin=hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(study,'MASK_SHA256',pin)
    assert study.read_mask(p,cache)==mask
    p.write_text('changed')
    with pytest.raises(study.SourceIntegrityError):study.read_mask(p,cache)
    mask['test_manifest_sha256']='foreign';p.write_text(json.dumps(mask))
    monkeypatch.setattr(study,'MASK_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(study.SourceIntegrityError):study.read_mask(p,cache)


def test_actual_interrupted_child_keeps_real_loaded_source_start_proof(tmp_path):
    root=Path(study.__file__).resolve().parents[2]
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ('framework/src','framework/benchmarks') for p in (root/folder).rglob('*.py')}
    files[study.DOCUMENT]=study.PROTOCOL_SHA256
    protocol=_protocol('missing-cache',{'git_commit':'synthetic-interrupt-control','files':files})
    pp=tmp_path/'protocol.json';pp.write_text(json.dumps(protocol))
    cfg=protocol['configs'][0];cp=tmp_path/f'{cfg["name"]}.json';cp.write_text(json.dumps(cfg));out=tmp_path/'child'
    bootstrap='''
import sys,time,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'framework/src'),str(root/'framework')]
from benchmarks import nih_phase2_study as study
from tests.test_capstone_nih_features import template
identity=json.loads(Path(sys.argv[4]).read_text())['source_identity']
study.source_identity=lambda root:identity
study._cache=lambda path:{}
study.read_mask=lambda *args:{'excluded_test_image_ids':[]}
def controlled_engine(*args,**kwargs):
    Path(sys.argv[3],'engine-entered').write_text('synthetic slow boundary')
    time.sleep(30)
study.run_core=controlled_engine
raise SystemExit(study.run_child(root,Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4])))
'''
    with (tmp_path/'child.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-I','-c',bootstrap,str(root),str(cp),str(out),str(pp)],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+20
            while not (out/'engine-entered').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
            assert (out/'engine-entered').exists(),(tmp_path/'child.log').read_text()
        finally:
            if child.poll() is None:child.kill()
            child.wait(timeout=5)
    proof=json.loads((out/'executed-source-start.json').read_text())
    assert proof['phase']=='before_engine' and all(proof['files'][k]==files[k] for k in study.PROOF_REQUIRED)
    assert not (out/'executed-source.json').exists() and not (out/'execution.json').exists()


def test_actual_clis_refuse_invalid_input_without_overwriting_evidence(tmp_path):
    import os
    root=Path(study.__file__).resolve().parents[2];env=dict(os.environ,PYTHONPATH='framework/src:framework')
    result=subprocess.run([sys.executable,'-m','benchmarks.nih_phase2_study','--cache',str(tmp_path/'cache'),
        '--expected-report-sha256','0'*64,'--test-mask',str(tmp_path/'mask'),'--output',str(tmp_path/'out')],cwd=root,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and not (tmp_path/'out').exists()
    preserved=tmp_path/'analysis.json';preserved.write_text('keep')
    result=subprocess.run([sys.executable,'-m','benchmarks.nih_phase2_analysis',str(tmp_path/'missing'),
        '--output',str(preserved)],cwd=root,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and preserved.read_text()=='keep'


def test_reanalysis_refuses_a_dropped_artifact_hash_entry(tmp_path,monkeypatch):
    _orchestrate(tmp_path,monkeypatch,'success')
    p=tmp_path/'out/index.json';index=json.loads(p.read_text());index['cases'][0]['artifacts'].pop('execution.json')
    p.write_text(json.dumps(index))
    with pytest.raises(ValueError):study.analyze_saved_study(tmp_path/'out')


@pytest.mark.parametrize('entrypoint',['run','reanalysis'])
def test_real_small_core_child_writes_verified_scores_with_synthetic_features(template,tmp_path,monkeypatch,entrypoint):
    from fedlearn.simulation.config import resolve
    from fedlearn.simulation.nih_cache import load_nih_cache
    cache=template[1];pin=hashlib.sha256((cache/'report.json').read_bytes()).hexdigest()
    cfg=resolve({'name':'synthetic_control','seed':42,'workload':{'dataset':'nih','features_path':str(cache),'balance':False,
        'features_report_sha256':pin,'evaluation_split':'test','allow_random_features':True},
        'federation':{'num_clients':3,'rounds':2,'partition':{'type':'patient_iid'}},
        'client':{'local_epochs':1,'lr':.001},'server':{'rule':'trimmed_mean','clip_norm':.05},
        'privacy':{'mode':'central','epsilon':4.}})
    original_configs=study.configs
    identity={'git_commit':'synthetic_control','files':{}}
    monkeypatch.setattr(study,'FIXED_REPORT_SHA256',pin)
    monkeypatch.setattr(study,'configs',lambda *a:[cfg])
    monkeypatch.setattr(study,'source_identity',lambda *a:identity)
    monkeypatch.setattr(study,'verify_imports',lambda *a:{'files':{}})
    monkeypatch.setattr(study,'_cache',lambda *a:load_nih_cache(cache,expected_report_sha256=pin))
    monkeypatch.setattr(study,'read_mask',lambda *a:{'excluded_test_image_ids':[]})
    protocol=_protocol(cache,identity);protocol['features_report_sha256']=pin
    pp=tmp_path/'protocol.json';pp.write_text(json.dumps(protocol))
    cp=tmp_path/'synthetic_control.json';cp.write_text(json.dumps(cfg));out=tmp_path/'child'
    assert study.run_child(Path(study.__file__).resolve().parents[2],cp,out,pp)==0
    execution=json.loads((out/'execution.json').read_text())
    record=json.loads((out/'record.json').read_text());manifest=json.loads((out/'final/manifest.json').read_text())
    assert record['meta']['core_qualification']['feature_control'] is True
    assert manifest['feature_control'] is True and manifest['final_digest']==record['meta']['final_digest']
    assert execution['core_execution_seconds']>=record['meta']['wall_seconds'] and execution['artifact_execution_seconds']>0
    score=json.loads((out/'score-analysis.json').read_text())
    assert score['test']['auc']==pytest.approx(record['rounds'][-1]['auc'],abs=1e-12)
    monkeypatch.setattr(study,'configs',original_configs)
    monkeypatch.setattr(study,'FIXED_REPORT_SHA256',PIN)
    calls=0
    def verify_real():
        nonlocal calls
        if calls%360==0:
            study.verify_final_artifacts(cfg,record,out/'final',feature_report_sha256=pin)
        calls+=1
    original_threads=study.torch.get_num_threads()
    try:
        study.torch.set_num_threads(1 if entrypoint=='run' else 4)
        result=_orchestrate(tmp_path/'parent',monkeypatch,'success',verification=verify_real)
        if entrypoint=='reanalysis':
            study.torch.set_num_threads(1)
            assert study.analyze_saved_study(tmp_path/'parent/out')==result['analysis']
    finally:study.torch.set_num_threads(original_threads)


@pytest.mark.parametrize('field,value',[('per_cell_seconds_cap',181),('total_child_seconds_cap',10801),
    ('max_child_rss_bytes',2**31+1),('evaluation_split','validation'),('runtime',{})])
def test_saved_protocol_cannot_change_declared_resources_split_or_runtime(field,value):
    protocol=_protocol('cache',{'git_commit':'control','files':{}})
    protocol[field]=value
    with pytest.raises(ValueError):study._validate_protocol(protocol)


@pytest.mark.parametrize('bad',['protocol_json','protocol_field','config_json'])
def test_child_retains_malformed_input_as_integrity_failure(tmp_path,monkeypatch,bad):
    identity={'git_commit':'control','files':{}}
    protocol=_protocol('missing-cache',identity);cfg=protocol['configs'][0]
    if bad=='protocol_field':protocol.pop('configs')
    pp=tmp_path/'protocol.json';pp.write_text('{broken' if bad=='protocol_json' else json.dumps(protocol))
    cp=tmp_path/f'{cfg["name"]}.json';cp.write_text('{broken' if bad=='config_json' else json.dumps(cfg))
    monkeypatch.setattr(study,'source_identity',lambda *_:identity)
    monkeypatch.setattr(study,'verify_imports',lambda *a:{'files':{}})
    monkeypatch.setattr(study,'_cache',lambda *a:{})
    monkeypatch.setattr(study,'read_mask',lambda *a:{'excluded_test_image_ids':[]})
    monkeypatch.setattr(study,'run_core',lambda *a,**kw:pytest.fail('core reached'))
    out=tmp_path/'child'
    assert study.run_child(Path(study.__file__).resolve().parents[2],cp,out,pp)==2
    execution=json.loads((out/'execution.json').read_text())
    assert execution['status']=='failed' and execution['error_kind']=='SourceIntegrityError'
    assert not (out/'record.json').exists()
