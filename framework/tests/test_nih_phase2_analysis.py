"""Hand-checked score, threshold, patient weighting and matched-seed inference."""
import copy
import hashlib
import json
import math

import numpy as np
import pytest

from benchmarks import nih_phase2_analysis as analysis
from benchmarks.nih_phase2_study import configs

PIN='6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd'


def test_exact_weighted_tie_aware_auc():
    assert analysis.auc([0.,0.,1.,2.],[0,1,1,0])==.375
    assert analysis.auc([0.,0.,1.,2.],[0,1,1,0],[1.,2.,1.,1.])==pytest.approx(1/3)
    assert analysis.auc([1.,1.],[0,1])==.5
    assert analysis.auc([1.,2.],[0,1])==1.
    assert analysis.auc([2.,1.],[0,1])==0.


@pytest.mark.parametrize('scores,labels,weights', [([],[],None),([1.],[1],None),([1.,2.],[0,0],None),
    ([1.,np.nan],[0,1],None),([1.,2.],[0,2],None),([1.,2.],[0.,1.],None),
    ([1.,2.],[0,1],[1.,-1.]),([1.,2.],[0,1],[1.,0.]),([1.,2.],[0,1],[1.]),
    ([[1.,2.]],[0,1],None)])
def test_invalid_or_single_class_score_inputs_refuse(scores,labels,weights):
    with pytest.raises(ValueError):analysis.auc(scores,labels,weights)


def test_default_argmax_tie_and_weighted_confusion_have_literal_counts():
    logits=np.array([[0.,0.],[0.,1.],[1.,0.],[0.,2.]],dtype=np.float32)
    result=analysis.metrics(logits,np.array([1,1,0,0]))
    assert {k:result[k] for k in ('tp','tn','fp','fn')}=={'tp':1,'tn':1,'fp':1,'fn':1}
    assert result['accuracy']==result['balanced_accuracy']==.5
    weighted=analysis.metrics(logits,np.array([1,1,0,0]),[2.,1.,3.,1.])
    assert {k:weighted[k] for k in ('tp','tn','fp','fn')}=={'tp':1.,'tn':3.,'fp':1.,'fn':2.}
    assert weighted['sensitivity']==pytest.approx(1/3) and weighted['specificity']==.75
    assert weighted['accuracy']==pytest.approx(4/7)


def test_threshold_ties_and_patient_weights_are_defined_without_test_tuning():
    threshold=analysis.recall_threshold([.9,.8,.8,.2,.1,.5,.2,-.1],[1,1,1,1,1,0,0,0])
    assert threshold==.2
    np.testing.assert_array_equal(analysis.patient_weights(['1','1','2']),[.5,.5,1.])
    logits=np.array([[0.,.15],[0.,.3],[0.,.2],[0.,-.3]],dtype=np.float64)
    result=analysis.metrics(logits,np.array([1,1,0,0]),threshold=threshold)
    assert result['sensitivity']==result['specificity']==.5
    assert result['tp']==result['fp']==result['tn']==result['fn']==1


def _split(margins,labels,prefix):
    margins=np.array(margins,dtype=np.float32)
    return {'logits':np.column_stack([np.zeros(len(margins),dtype=np.float32),margins]),
        'margin':margins.astype('float64'),'labels':np.array(labels,dtype=np.int64),
        'image_ids':np.array([f'{prefix}{i}' for i in range(len(margins))]),
        'patient_ids':np.array([str(i//2) for i in range(len(margins))])}


def test_complete_cell_analysis_uses_validation_threshold_and_bound_mask():
    bundle={'manifest':{'config_hash':'c'*64,'final_digest':'d'*16,'features_report_sha256':PIN},'splits':{
        'validation':_split([.9,.8,.8,.2,.1,.5,.2,-.1],[1,1,1,1,1,0,0,0],'v'),
        'test':_split([.15,.3,.2,-.3],[1,1,0,0],'t')}}
    result=analysis.cell_analysis(bundle,['t3'])
    assert result['evaluation_split']=='test' and result['threshold']['validation']['sensitivity']==.8
    assert result['threshold']['test']['sensitivity']==.5 and result['threshold']['test']['specificity']==.5
    assert result['test_ambiguous_excluded']['n_images']==3
    assert result['patient_weighted']['validation']['weight_sum']==4.
    assert result['config_hash']=='c'*64 and result['final_digest']=='d'*16
    with pytest.raises(ValueError):analysis.cell_analysis(bundle,['unknown'])
    with pytest.raises(ValueError):analysis.cell_analysis(bundle,['t3','t3'])


def _cases():
    recipes=configs('cache',PIN);result=[]
    components={('clean','mean','none'):.8,('clean','trimmed_mean','none'):.79,
        ('clean','mean','central'):.75,('clean','trimmed_mean','central'):.7,
        ('sign_flip','mean','none'):.6,('sign_flip','trimmed_mean','none'):.7,
        ('sign_flip','mean','central'):.55,('sign_flip','trimmed_mean','central'):.62}
    for c in recipes:
        mode=c['privacy']['mode'];cond='clean' if not c['behaviors'] else 'sign_flip' if c['behaviors'][0]['type']=='sign_flip_scale' else 'label_flip'
        value=components.get((cond,c['server']['rule'],mode),.65)
        if cond=='sign_flip' and c['server']['rule']=='trimmed_mean' and mode=='central':value+=.001*(c['seed']-20261101)
        h=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        stats={'auc':value,'loss':.6,'ece':.1,'accuracy':.6,'balanced_accuracy':.6,'sensitivity':.5,'specificity':.7}
        result.append({'name':c['name'],'config_hash':h,'status':'qualified','record_final_digest':'d'*16,
            'analysis':{'config_hash':h,'final_digest':'d'*16,'features_report_sha256':PIN,'evaluation_split':'test','test':stats}})
    return recipes,result


def test_ten_seed_primary_contrast_and_interval_are_paired_before_averaging():
    recipes,cases=_cases();result=analysis.study_analysis(cases,recipes)
    p=result['primary'];assert result['complete'] and p['complete'] and p['n_complete']==10
    assert [s['effect'] for s in p['per_seed']]==pytest.approx([.01+i*.001 for i in range(10)])
    assert p['mean']==pytest.approx(.0145)
    sd=math.sqrt(.0000825/9)
    assert p['sd']==pytest.approx(sd)
    assert p['ci95']==pytest.approx([.0145-2.2621571628540993*sd/math.sqrt(10),.0145+2.2621571628540993*sd/math.sqrt(10)])
    assert p['scope']=='conditional_simulation_seed_variation'
    assert len(p['per_seed'][0]['components'])==8


def test_missing_primary_component_has_no_formal_interval_or_replacement_seed():
    recipes,cases=_cases();cases[0]={'name':cases[0]['name'],'config_hash':cases[0]['config_hash'],'status':'not_run'}
    result=analysis.study_analysis(cases,recipes)
    assert not result['primary']['complete'] and result['primary']['n_complete']==9
    assert result['primary']['ci95'] is None and len(result['primary']['missing'])==1
    assert result['primary']['per_seed'][0]['effect'] is None
    assert not result['complete']


@pytest.mark.parametrize('bad',['duplicate','missing_row','hash','split','state','nonfinite','foreign_name'])
def test_unbound_or_malformed_summaries_cannot_produce_primary_inference(bad):
    recipes,cases=_cases()
    if bad=='duplicate':cases[-1]=copy.deepcopy(cases[0])
    elif bad=='missing_row':cases.pop()
    elif bad=='hash':cases[0]['config_hash']='0'*64
    elif bad=='split':cases[0]['analysis']['evaluation_split']='validation'
    elif bad=='state':cases[0]['analysis']['final_digest']='0'*16
    elif bad=='nonfinite':cases[0]['analysis']['test']['auc']=float('nan')
    else:cases[0]['name']='not_planned'
    with pytest.raises(ValueError):analysis.study_analysis(cases,recipes)


def test_actual_numeric_metrics_feed_the_primary_analysis_without_coercion_failure():
    recipes,cases=_cases()
    cases[0]['analysis']['test']=analysis.metrics(np.array([[0.,1.],[1.,0.]],dtype=np.float32),np.array([1,0]))
    result=analysis.study_analysis(cases,recipes)
    assert result['primary']['complete']
    json.dumps(result,allow_nan=False)
