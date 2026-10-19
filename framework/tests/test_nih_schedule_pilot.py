"""Prospective matched-exposure schedule budget; no outcome-dependent extensions."""
import pytest
from benchmarks.nih_schedule_pilot import initial_configs, followup_configs, select_schedule
from tests.test_nih_validation_pilot import PIN, _record


def records():
    values=[_record(c) for c in initial_configs('cache',PIN)]
    for value in values:
        value['meta']['workload']['auc_score']='margin_v2'
        for line in value['rounds']: line['auc_probability']=line['auc']
    return values


def test_exact_two_initial_and_four_followon_schedules_have_equal_image_exposure():
    initial=initial_configs('cache',PIN)
    assert [c['name'] for c in initial]==['local7_round20','local1_round140']
    assert [(c['client']['local_epochs'],c['federation']['rounds']) for c in initial]==[(7,20),(1,140)]
    for c in initial:
        assert c['client']['local_epochs']*c['federation']['rounds']==140
        assert c['federation']['num_clients']==80 and c['client']['lr']==.003
        assert c['workload']['auc_score']=='margin_v2' and c['workload']['evaluation_split']=='validation'
        assert not c['workload']['balance'] and c['privacy']['mode']=='none' and not c['behaviors']
    for schedule in ['local7_round20','local1_round140']:
        follow=followup_configs('cache',PIN,schedule)
        assert [c['seed'] for c in follow]==[20261020,20261021,20261019,20261019]
        assert [c['server']['clip_norm'] for c in follow]==[None,None,.4,.4]
        assert follow[-1]['server']['rule']=='trimmed_mean'
        assert sum(c['federation']['rounds'] for c in initial+follow)<=720
    with pytest.raises(ValueError): followup_configs('cache',PIN,'extra_schedule')


def test_selects_fewer_releases_if_eligible_without_chasing_better_loss():
    values=records()
    values[1]['rounds'][-1]['loss']=.5
    assert select_schedule(values)=='local7_round20'
    values[0]['rounds'][-1]['ece']=.11
    assert select_schedule(values)=='local1_round140'
    values[1]['rounds'][-1]['auc']=.67
    assert select_schedule(values) is None


def test_no_incomplete_duplicate_or_changed_recipe_selection():
    values=records()
    for invalid in [values[:1],values+values[:1],[values[0],values[0]]]:
        with pytest.raises(ValueError): select_schedule(invalid)
    values[0]['meta']['config']['client']['lr']=.004
    with pytest.raises(ValueError): select_schedule(values)


def test_exact_source_protocol_is_fingerprinted(tmp_path, monkeypatch):
    import hashlib
    from benchmarks import nih_schedule_pilot as pilot
    doc=tmp_path/pilot.DOCUMENT
    doc.parent.mkdir(parents=True)
    doc.write_text('fixed protocol')
    monkeypatch.setattr(pilot,'_source_identity',lambda _: {'git_commit':'test','files':{}})
    first=pilot.source_identity(tmp_path)
    assert first['files'][pilot.DOCUMENT]==hashlib.sha256(doc.read_bytes()).hexdigest()
    doc.write_text('changed protocol')
    assert pilot.source_identity(tmp_path)!=first


def test_probe_refuses_wrong_cache_and_existing_output_before_loading(tmp_path, monkeypatch):
    from benchmarks import nih_schedule_pilot as pilot
    monkeypatch.setattr(pilot,'load_nih_cache',lambda *a,**k: pytest.fail('cache must not load'))
    with pytest.raises(ValueError,match='fixed protocol cache'):
        pilot.run_probe(tmp_path/'cache','1'*64,tmp_path/'out')
    out=tmp_path/'out'
    out.mkdir()
    (out/'sentinel').write_text('preserve')
    with pytest.raises(FileExistsError): pilot.run_probe(tmp_path/'cache',PIN,out)
    assert (out/'sentinel').read_text()=='preserve'


def _exercise_orchestration(tmp_path,monkeypatch,outcome,control_overhead=0):
    import json
    import subprocess
    from pathlib import Path
    from benchmarks import nih_schedule_pilot as pilot
    root=tmp_path/'repo'
    document=root/pilot.DOCUMENT
    document.parent.mkdir(parents=True)
    document.write_text('fixed test protocol')
    monkeypatch.setattr(pilot,'__file__',str(root/'framework/benchmarks/nih_schedule_pilot.py'))
    identity={'git_commit':'test-commit','files':{'framework/src/fedlearn/simulation/engine.py':'test-engine-hash'}}
    state={'mutated':False,'calls':0,'clock_calls':0}
    virtual={'now':0.0}
    def identity_with_overhead(_):
        virtual['now']+=control_overhead
        return {**identity,'git_commit':'changed'} if state['mutated'] else identity
    monkeypatch.setattr(pilot,'source_identity',identity_with_overhead)
    if control_overhead: monkeypatch.setattr(pilot.time,'monotonic',lambda:virtual['now'])
    monkeypatch.setattr(pilot,'verify_imports',lambda *a: {'files':{}})
    monkeypatch.setattr(pilot,'load_nih_cache',lambda *a,**k: {'provenance':{'pretrained':True}})
    if outcome=='budget': monkeypatch.setattr(pilot,'TOTAL_SECONDS',0)
    def child(command,**kwargs):
        state['calls']+=1
        virtual['now']+=.1
        if state['calls']==1 and outcome=='error': raise subprocess.CalledProcessError(1,command)
        if state['calls']==1 and outcome=='timeout': raise subprocess.TimeoutExpired(command,300)
        if state['calls']==1 and outcome=='mutation':
            state['mutated']=True
            return
        config=json.loads(Path(command[-3]).read_text())
        record=_record(config,auc=.67 if outcome=='ineligible' else .7)
        record['meta']['config_hash']=pilot.config_hash(config)
        record['meta']['provenance']={'git_commit':'test-commit'}
        record['meta']['workload'].update(auc_score='margin_v2',features_report_sha256=pilot.FIXED_REPORT_SHA256)
        for line in record['rounds']: line['auc_probability']=line['auc']
        directory=Path(command[-2]); directory.mkdir()
        (directory/'record.json').write_text(json.dumps(record))
        (directory/'index.json').write_text(json.dumps({'runs':[{'status':'ok','file':'record.json'}]}))
        for filename,phase in [('executed-source-start.json','before_engine'),('executed-source.json','after_engine_return')]:
            if outcome=='missing_proof' and phase=='after_engine_return': continue
            files=identity['files'] if outcome!='wrong_proof' else {'framework/src/fedlearn/simulation/engine.py':'tampered'}
            (directory/filename).write_text(json.dumps({'phase':phase,'files':files}))
    monkeypatch.setattr(pilot.subprocess,'run',child)
    result=pilot.run_probe(tmp_path/'cache',pilot.FIXED_REPORT_SHA256,tmp_path/'out')
    assert len(result['cases'])==6
    assert json.loads((tmp_path/'out/index.json').read_text())['cases']==result['cases']
    if outcome=='success':
        assert result['complete'] and result['core_optimization_gate_passed']
        assert result['selected_schedule']=='local7_round20' and state['calls']==6
    else:
        assert not result['complete'] and not result['core_optimization_gate_passed']
        assert result['selected_schedule'] is None and all(c['status']=='not_run' for c in result['cases'][2:])
        if outcome=='budget': assert state['calls']==0
        elif outcome=='mutation': assert state['calls']==1 and result['source_integrity_failed']
        else: assert state['calls']==2
    return result


@pytest.mark.parametrize('outcome',['success','ineligible','error','timeout','mutation','budget','missing_proof','wrong_proof'])
def test_schedule_orchestration_preserves_six_entries_and_correct_core_gate(tmp_path,monkeypatch,outcome):
    _exercise_orchestration(tmp_path,monkeypatch,outcome)


def test_verification_overhead_is_separate_from_bounded_child_budget(tmp_path,monkeypatch):
    result=_exercise_orchestration(tmp_path,monkeypatch,'success',control_overhead=1000)
    assert result['child_execution_seconds']==pytest.approx(.6)
    assert result['parent_control_seconds']>=13000
    assert result['elapsed_seconds']==pytest.approx(result['child_execution_seconds']+result['parent_control_seconds'])
    assert all(c['child_execution_seconds']==pytest.approx(.1) for c in result['cases'])


def test_interrupted_child_preserves_initial_proof_without_completion_claim(tmp_path):
    import hashlib,json,os,subprocess,time
    from pathlib import Path
    from benchmarks import nih_schedule_pilot as pilot
    from tests.test_simulation_engine_cli import _features
    root=Path(pilot.__file__).resolve().parents[2]
    manifest={'source_identity':{'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('framework/src','framework/benchmarks') for p in (root/folder).rglob('*.py')}}}
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(manifest))
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'workload':{'features_path':_features(tmp_path)},
        'federation':{'num_clients':2,'rounds':10000},'client':{'local_epochs':1}}))
    out=tmp_path/'out'
    with (tmp_path/'child.log').open('w') as log:
        child=subprocess.Popen(pilot.child_command(root,config,out,protocol),env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+15
            while not (out/'executed-source-start.json').exists() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            assert (out/'executed-source-start.json').exists(), (tmp_path/'child.log').read_text()
            proof=json.loads((out/'executed-source-start.json').read_text())
            assert proof['phase']=='before_engine'
            assert proof['files']['framework/src/fedlearn/simulation/engine.py']==manifest['source_identity']['files']['framework/src/fedlearn/simulation/engine.py']
        finally:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
    assert not (out/'executed-source.json').exists()
