"""Generated runner and metric failure boundaries, without medical inputs."""
from dataclasses import replace
import json
import time

import numpy as np
import pytest
import torch

from projection_study_design import Cell
from projection_study_metrics import metrics
from projection_study_workload import dataset, features
from run_projection_study import attack_check, preflight, run_cell, verify_plan


def test_tied_auroc_agrees_with_pairwise_definition_and_refuses_nonfinite():
    y=np.array([0,1,0,1,1]);s=np.array([0.,0.,1.,2.,2.])
    pair=s[y==1,None]-s[y==0][None,:]
    expected=float(np.mean((pair>0)+.5*(pair==0)))
    assert metrics(y,s)['auroc']==pytest.approx(expected)
    for labels,scores in ((y,s*np.nan),(np.ones(5),s),(y,s[:-1])):
        with pytest.raises(ValueError):metrics(labels,scores)


def test_preflight_refuses_reuse_and_detects_manifest_tampering(tmp_path):
    out=tmp_path/'plan';preflight(out,'qualification')
    with pytest.raises(FileExistsError):preflight(out,'qualification')
    p=out/'plan.json';raw=json.loads(p.read_text());raw['cells'][0]['seed']=1;p.write_text(json.dumps(raw))
    with pytest.raises(ValueError):verify_plan(out)


def test_deadline_cannot_produce_a_successful_head(tmp_path):
    data=features(dataset(),Cell('full',25001,False))
    with pytest.raises(TimeoutError):
        run_cell(Cell('full',25001,False),data,tmp_path/'failed',time.monotonic()-1)
    assert not (tmp_path/'failed'/'head.npz').exists()
    assert json.loads((tmp_path/'failed'/'failure.json').read_text())['status']=='failed'


def test_owned_attack_retains_prefix_candidates_and_budget_counters():
    result=attack_check()
    assert result['qualified'] and result['budgets']==[60,240,480]
