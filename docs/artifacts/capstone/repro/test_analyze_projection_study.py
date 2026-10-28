"""Independent result auditing refuses altered or incomplete scientific evidence."""
import numpy as np
import pytest

from analyze_projection_study import independent_metrics, check_round, summarize


def test_independent_rank_metric_handles_ties_and_matches_explicit_oracle():
    y=np.array([0,1,0,1]);s=np.array([0.,0.,1.,2.])
    pair=s[y==1,None]-s[y==0][None,:]
    assert independent_metrics(y,s)['auroc']==pytest.approx(np.mean((pair>0)+.5*(pair==0)))


def test_round_audit_detects_changed_clipping_and_noise_releases():
    raw=np.ones((20,2,128),dtype=np.float32)*.01;bias=np.zeros((20,2),dtype=np.float32)
    norms=np.sqrt(np.sum(raw.astype(float)**2,axis=(1,2)))
    pre=(raw.astype(float)*np.minimum(1.,.05/(norms+1e-12))[:,None,None]).mean(0)
    arrays=dict(raw_weight=raw,raw_bias=bias,pre_weight=pre,pre_bias=np.zeros(2),
                weight=pre, bias=np.zeros(2),noise_weight=np.zeros((2,128)),noise_bias=np.zeros(2))
    assert check_round(arrays,np.zeros((2,128)),np.zeros(2))['release_error']<1e-6
    arrays['weight']=pre+.001
    with pytest.raises(ValueError):check_round(arrays,np.zeros((2,128)),np.zeros(2))


def test_incomplete_matrix_and_duplicate_baseline_cannot_be_summarized():
    with pytest.raises(ValueError):summarize([])
    with pytest.raises(ValueError):summarize([dict(name='duplicate'),dict(name='duplicate')])
