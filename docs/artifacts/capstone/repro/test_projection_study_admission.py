"""Non-fitting regression fixtures for qualification and audit admission."""
import json
import numpy as np
import pytest
import torch

import analyze_projection_study as audit
import run_projection_study as runner
from projection_study_design import Cell
from projection_study_workload import dataset
from projection_study_strategy import noise_draw


def fabricated_run(root):
    expected=runner.qualification_cells();labels=dataset()['eval_y'].numpy();rows=[];hashes={}
    for cell in expected:
        folder=root/'runs'/cell.name;(folder/'rounds').mkdir(parents=True)
        w=np.zeros((2,cell.width),dtype=np.float32);b=np.zeros(2,dtype=np.float32);journal=[]
        for r in range(1,21):
            p=folder/'rounds'/f'{r:02}.npz'
            np.savez_compressed(p,raw_weight=np.zeros((20,2,cell.width)),raw_bias=np.zeros((20,2)),
                                pre_weight=w,pre_bias=b,weight=w,bias=b,noise_weight=w,noise_bias=b)
            journal.append(dict(round=r,arrays_sha256=runner.sha(p),aggregated=20,
                sample_counts=[205]*16+[204]*4,sensitivity=.005,clip_rate=0.,noise_std=-1.,
                parameter_noise_energy=-999.,predicted_score_noise_variance=-999.,realized_score_noise_mse=-999.,
                attack=dict(gradient_evaluations=cell.steps*3,loss_evaluations=(cell.steps+1)*3)))
        runner.write_json(folder/'rounds.json',journal)
        np.savez_compressed(folder/'head.npz',weight=w,bias=b)
        np.savez_compressed(folder/'predictions.npz',scores=np.zeros(len(labels)),y=labels)
        row=dict(**cell.record(),status='completed',rounds=20,client_fits=400,elapsed_s=0.,
                 metrics=audit.independent_metrics(labels,np.zeros(len(labels))),
                 artifacts={n:runner.sha(folder/n) for n in ('head.npz','predictions.npz','rounds.json')})
        runner.write_json(folder/'result.json',row);rows.append(row);hashes[cell.name]=runner.sha(folder/'result.json')
    runner.write_json(root/'frozen-plan.json',dict(stage='qualification',cells=[c.record() for c in expected],
                     settings={},sources={},data={},bases={},calibration={}))
    runner.write_json(root/'summary.json',dict(status='completed',stage='qualification',runs=rows,
                     result_hashes=hashes,elapsed_s=0.,plan_sha256='wrong'))
    return rows


def test_hash_consistent_zero_fit_qualification_is_refused(tmp_path):
    rows=fabricated_run(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text())
    for row in rows:
        row.update(status='failed',rounds=0,client_fits=0,artifacts={})
        p=tmp_path/'runs'/row['name']/'result.json';runner.write_json(p,row)
        summary['result_hashes'][row['name']]=runner.sha(p)
    summary['runs']=rows;runner.write_json(tmp_path/'summary.json',summary)
    with pytest.raises(ValueError):runner.validate_qualification(tmp_path)


def test_hash_consistent_audit_with_invalid_provenance_is_refused(tmp_path):
    fabricated_run(tmp_path)
    with pytest.raises(ValueError):audit.analyze(tmp_path)


def test_zero_private_noise_and_negative_diagnostics_are_refused():
    cell=Cell('full',25001,True)
    arrays=dict(noise_weight=np.zeros((2,512)),noise_bias=np.zeros(2),
                pre_weight=np.zeros((2,512)),pre_bias=np.zeros(2))
    info=dict(noise_std=-1.,parameter_noise_energy=-999.)
    with pytest.raises(ValueError):
        audit.check_noise_geometry(cell,1,arrays,info,np.ones((2,512)),np.zeros((2,512)),np.zeros(2))


@pytest.mark.parametrize('field',['settings','sources','data','bases','calibration','runtime'])
def test_each_frozen_manifest_binding_is_required(tmp_path,field):
    plan=runner.preflight(tmp_path/'plan','qualification');plan[field]={}
    with pytest.raises(ValueError):runner.validate_plan(plan)


@pytest.mark.parametrize('field',['noise_std','parameter_noise_energy','active_noise_energy',
                                'expected_parameter_noise_energy','predicted_score_noise_variance',
                                'realized_score_noise_mse','teacher_alignment','clean_aggregate_norm'])
def test_each_geometry_diagnostic_is_checked_individually(field):
    cell=Cell('full',25001,True);std=runner.calibration_check()['sigma']
    n=noise_draw(cell.seed,1,512,std);w=n['head.weight'].numpy();b=n['head.bias'].numpy();x=np.ones((2,512))
    scores=np.sum(w[1].astype(float)-w[0].astype(float))+float(b[1])-float(b[0])
    a=dict(noise_weight=w,noise_bias=b,pre_weight=np.zeros((2,512)),pre_bias=np.zeros(2),
           clean_raw_weight=np.zeros((20,2,512)),clean_raw_bias=np.zeros((20,2)))
    energy=float(np.sum(w.astype(float)**2)+np.sum(b.astype(float)**2))
    info=dict(noise_std=std,parameter_noise_energy=energy,
              active_noise_energy=float(np.sum(w[:,:128].astype(float)**2)+np.sum(b.astype(float)**2)),
              expected_parameter_noise_energy=1026*std**2,predicted_score_noise_variance=2*std**2*513,
              realized_score_noise_mse=scores**2,teacher_alignment=None,clean_aggregate_norm=0.)
    audit.check_noise_geometry(cell,1,a,info,x,np.zeros((2,512)),np.zeros(2))
    info[field]=-999.
    with pytest.raises(ValueError):audit.check_noise_geometry(cell,1,a,info,x,np.zeros((2,512)),np.zeros(2))
    if field=='noise_std':
        info[field]=std;a['noise_weight']=w[::-1].copy()
        with pytest.raises(ValueError):audit.check_noise_geometry(cell,1,a,info,x,np.zeros((2,512)),np.zeros(2))
