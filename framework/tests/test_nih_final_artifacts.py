"""Qualified final parameters and independently reloadable NIH research scores."""
import hashlib

import pytest
import torch

from fedlearn.simulation import core_qualification as core
from tests.test_capstone_nih_features import template as feature_template
from tests.test_simulation_core_qualification import cfg


def _digest(params):
    h=hashlib.sha256()
    for key in sorted(params):
        h.update(key.encode());h.update(params[key].detach().numpy().tobytes())
    return h.hexdigest()[:16]


def _policy(cfg):
    return {'clients':3,'feature_report_sha256':cfg['workload']['features_report_sha256']}


@pytest.mark.parametrize('privacy,head', [('none',{'type':'linear'}),('central',{'type':'mlp','hidden':8})])
def test_capture_actual_final_state_preserves_default_training(cfg,privacy,head):
    cfg['workload']['head']=head;cfg['privacy']={'mode':privacy,'epsilon':4.}
    baseline=core.run_core(cfg,**_policy(cfg));captured=[]
    record=core.run_core(cfg,**_policy(cfg),on_final_parameters=captured.append)
    assert len(captured)==1 and _digest(captured[0])==record['meta']['final_digest']==baseline['meta']['final_digest']
    assert record['meta']['round0_upload_digest']==baseline['meta']['round0_upload_digest']
    for left,right in zip(record['rounds'],baseline['rounds']):
        assert {k:left[k] for k in ('auc','auc_probability','loss','accuracy','ece')}=={k:right[k] for k in ('auc','auc_probability','loss','accuracy','ece')}
    assert all(v.device.type=='cpu' and not v.requires_grad for v in captured[0].values())
    for v in captured[0].values():v.zero_()
    assert _digest(captured[0])!=record['meta']['final_digest']
    assert core.run_core(cfg,**_policy(cfg))['meta']['final_digest']==baseline['meta']['final_digest']


def test_failed_postflight_never_exports_final_parameters(cfg):
    seen=[];rounds=[]
    with pytest.raises(core.CoreQualificationError) as error:
        core.run_core(cfg,**_policy(cfg),on_round=rounds.append,should_stop=lambda:len(rounds)==1,
                      on_final_parameters=seen.append)
    assert seen==[] and len(error.value.record['rounds'])==1
    assert 'core_qualification' not in error.value.record['meta']


def test_artifact_callback_error_preserves_qualified_execution_record(cfg):
    def failed_export(params):raise OSError('disk full')
    with pytest.raises(core.CoreQualificationError,match='final parameter callback') as error:
        core.run_core(cfg,**_policy(cfg),on_final_parameters=failed_export)
    assert len(error.value.record['rounds'])==2
    assert error.value.record['meta']['core_qualification']['scope']=='simulation_execution_only'
    assert isinstance(error.value.__cause__,OSError)


@pytest.fixture(params=['linear','mlp'])
def export_case(cfg,request):
    from fedlearn.simulation.config import resolve
    cfg['workload']['head']={'type':request.param,'hidden':8}
    cfg=resolve(cfg);states=[]
    rec=core.run_core(cfg,**_policy(cfg),on_final_parameters=states.append)
    return cfg,rec,states[0]


def test_artifact_roundtrip_preserves_checkpoint_and_aligned_scores(export_case,tmp_path):
    import numpy as np
    from benchmarks.nih_final_artifacts import write_final_artifacts,verify_final_artifacts
    cfg,rec,params=export_case;out=tmp_path/'artifacts'
    proof=write_final_artifacts(cfg,rec,params,out,feature_report_sha256=_policy(cfg)['feature_report_sha256'])
    loaded=verify_final_artifacts(cfg,rec,out,feature_report_sha256=_policy(cfg)['feature_report_sha256'])
    assert proof==loaded['manifest'] and proof['research_record_dp_safe'] is False
    assert proof['feature_control'] is True and proof['final_digest']==rec['meta']['final_digest']
    for split in ('validation','test'):
        scores=loaded['splits'][split]
        assert scores['logits'].dtype==np.float32 and scores['margin'].dtype==np.float64
        np.testing.assert_array_equal(scores['margin'],scores['logits'][:,1].astype('float64')-scores['logits'][:,0].astype('float64'))
        assert scores['labels'].dtype==np.int64 and scores['patient_ids'].dtype.kind=='U'
        assert len(scores['image_ids'])==len(scores['labels'])
    assert set(loaded['parameters'])==set(params)
    for key in params:np.testing.assert_array_equal(loaded['parameters'][key],params[key].numpy())


@pytest.mark.parametrize('bad',['pin','config','record','model','nonfinite','shape','key'])
def test_wrong_artifact_inputs_refuse_without_completion(export_case,tmp_path,bad):
    import copy
    from benchmarks.nih_final_artifacts import write_final_artifacts
    cfg,rec,params=copy.deepcopy(export_case);pin=_policy(cfg)['feature_report_sha256']
    if bad=='pin':pin='0'*64
    elif bad=='config':cfg['seed']+=1
    elif bad=='record':rec['meta']['core_qualification']['research_record_dp_safe']=True
    elif bad=='model':params[next(iter(params))].add_(1)
    elif bad=='nonfinite':params[next(iter(params))].flatten()[0]=float('nan')
    elif bad=='shape':params[next(iter(params))]=torch.zeros(1)
    elif bad=='key':params['unexpected']=torch.zeros(1)
    with pytest.raises(ValueError):write_final_artifacts(cfg,rec,params,tmp_path/'bad',feature_report_sha256=pin)
    assert not (tmp_path/'bad/manifest.json').exists()


@pytest.mark.parametrize('where',['existing','inside_cache'])
def test_artifact_output_cannot_overwrite_sources(export_case,tmp_path,where):
    from pathlib import Path
    from benchmarks.nih_final_artifacts import write_final_artifacts
    cfg,rec,params=export_case
    out=tmp_path/'existing' if where=='existing' else Path(cfg['workload']['features_path'])/'new'
    if where=='existing':out.mkdir();(out/'sentinel').write_text('preserve')
    with pytest.raises((ValueError,FileExistsError)):write_final_artifacts(cfg,rec,params,out,feature_report_sha256=_policy(cfg)['feature_report_sha256'])
    if where=='existing':assert (out/'sentinel').read_text()=='preserve'
    else:assert not out.exists()


@pytest.mark.parametrize('bad',['bytes','score','label','image','model','partial'])
def test_artifact_verifier_rejects_tampering_even_rehashed_payloads(export_case,tmp_path,bad):
    import json
    import numpy as np
    from benchmarks.nih_final_artifacts import write_final_artifacts,verify_final_artifacts
    cfg,rec,params=export_case;out=tmp_path/'artifacts';pin=_policy(cfg)['feature_report_sha256']
    write_final_artifacts(cfg,rec,params,out,feature_report_sha256=pin)
    manifest=json.loads((out/'manifest.json').read_text())
    if bad=='partial':(out/'manifest.json').unlink()
    elif bad=='bytes':(out/'validation.npz').write_bytes(b'broken')
    else:
        if bad=='model':
            path=out/'model.safetensors';blob=bytearray(path.read_bytes());blob[-1]^=1;path.write_bytes(blob)
        else:
            path=out/'validation.npz'
            with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
            key={'score':'margin','label':'labels','image':'image_ids'}[bad]
            if bad=='image':arrays[key][0]='99999999_999.png'
            else:arrays[key][0]+=1
            np.savez(path,**arrays)
        manifest['payloads'][path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    with pytest.raises(ValueError):verify_final_artifacts(cfg,rec,out,feature_report_sha256=pin)


def test_failed_score_write_leaves_no_completion_manifest(export_case,tmp_path,monkeypatch):
    import numpy as np
    from benchmarks import nih_final_artifacts as artifacts
    cfg,rec,params=export_case;original=np.savez;calls=[]
    def interrupted(path,**arrays):
        calls.append(path.name)
        if path.name=='test.npz':raise OSError('write interrupted')
        original(path,**arrays)
    monkeypatch.setattr(np,'savez',interrupted)
    out=tmp_path/'incomplete'
    with pytest.raises(OSError,match='interrupted'):
        artifacts.write_final_artifacts(cfg,rec,params,out,feature_report_sha256=_policy(cfg)['feature_report_sha256'])
    assert (out/'model.safetensors').is_file() and (out/'validation.npz').is_file()
    assert not (out/'manifest.json').exists()
    with pytest.raises(ValueError):artifacts.verify_final_artifacts(cfg,rec,out,feature_report_sha256=_policy(cfg)['feature_report_sha256'])


def test_finite_inputs_with_overflowed_logits_cannot_be_exported():
    import numpy as np
    from benchmarks.nih_final_artifacts import _predict
    from fedlearn.simulation.workloads import LinearHead
    model=LinearHead(512,2)
    with torch.no_grad():
        model.head.weight.fill_(1e30);model.head.bias.zero_()
    data={'x':np.full((2,512),1e30,dtype=np.float32),'y':np.array([0,1]),
          'image_ids':['00000001_000.png','00000002_000.png'],'patient_ids':['1','2']}
    with pytest.raises(ValueError,match='logits'):_predict(model,data)


@pytest.mark.parametrize('field',['evaluation_manifest_sha256','train_manifest_sha256','curated_cohort_sha256',
    'model_state_sha256','source_hashes','split_counts','n_train','n_test','parameters','backbone','source_dataset'])
def test_inconsistent_record_cache_identity_refuses_before_completion(export_case,tmp_path,field):
    import copy
    from benchmarks.nih_final_artifacts import write_final_artifacts
    cfg,rec,params=copy.deepcopy(export_case)
    original=rec['meta']['workload'][field]
    rec['meta']['workload'][field]=original+1 if type(original) is int else {} if isinstance(original,dict) else 'incorrect'
    with pytest.raises(ValueError):write_final_artifacts(cfg,rec,params,tmp_path/'bad',feature_report_sha256=_policy(cfg)['feature_report_sha256'])
    assert not (tmp_path/'bad/manifest.json').exists()


def test_verifier_rejects_inconsistent_record_even_when_record_hash_is_rewritten(export_case,tmp_path):
    import copy,json
    from benchmarks.nih_final_artifacts import write_final_artifacts,verify_final_artifacts
    cfg,rec,params=export_case;out=tmp_path/'artifacts';pin=_policy(cfg)['feature_report_sha256']
    write_final_artifacts(cfg,rec,params,out,feature_report_sha256=pin)
    rec=copy.deepcopy(rec);rec['meta']['workload']['n_test']+=1
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['record_sha256']=hashlib.sha256((json.dumps(rec,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    with pytest.raises(ValueError):verify_final_artifacts(cfg,rec,out,feature_report_sha256=pin)


def test_margin_subtraction_keeps_ranks_collapsed_by_float32():
    import numpy as np
    from benchmarks.nih_final_artifacts import _predict
    class Fixed(torch.nn.Module):
        def forward(self,x):return torch.tensor([[-1.,1e8],[0.,1e8]],dtype=torch.float32)
    split={'x':np.zeros((2,512),dtype=np.float32),'y':np.array([1,0]),
           'image_ids':['00000001_000.png','00000002_000.png'],'patient_ids':['1','2']}
    actual=_predict(Fixed(),split)
    np.testing.assert_array_equal(actual['margin'],[100000001.,100000000.])
    assert actual['margin'][0]>actual['margin'][1]
    assert (actual['logits'][:,1]-actual['logits'][:,0]).tolist()==[100000000.,100000000.]
