"""Canonical training, admitted cohorts and coupled noise controls."""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.composed_strategy import ComposedStrategy
from projection_study_strategy import StudyStrategy, initial_parameters, local_fit, noise_draw, score


def test_coupling_maps_both_weight_rows_biases_and_independent_inactive_entries():
    a=noise_draw(11,2,128,.1);b=noise_draw(11,2,512,.1)
    assert torch.equal(a['head.weight'],b['head.weight'][:,:128])
    assert torch.equal(a['head.bias'],b['head.bias'])
    assert not torch.equal(b['head.weight'][0,128:256],a['head.weight'][0])
    assert not torch.equal(a['head.weight'],noise_draw(11,3,128,.1)['head.weight'])


def test_native_no_noise_parity_and_exact_cohort_admission():
    p=initial_parameters(128)
    native=ComposedStrategy(p,clip_norm=.05,clients_per_round=20,min_fit_clients=20)
    study=StudyStrategy(p,seed=11,private=False)
    updates=[(str(i),OrderedDict((k,v+.03) for k,v in p.items()),205) for i in range(20)]
    expected=native.aggregate_fit(1,updates);actual=study.aggregate_fit(1,updates)
    assert all(torch.equal(expected[k],actual[k]) for k in p)
    for bad in (updates[:-1],updates[:-1]+[updates[0]],updates[:-1]+[('bad',p,205)]):
        assert StudyStrategy(p,seed=11,private=True).aggregate_fit(1,bad) is None
    for private in (False,True):
        invalid=updates[:-1]+[('19',p,0)]
        assert StudyStrategy(p,seed=11,private=private).aggregate_fit(1,invalid) is None


def test_gaussian_marginals_covariance_and_score_variance():
    samples=torch.stack([torch.cat((noise_draw(19,r,128,.1)['head.weight'].flatten(),
                                   noise_draw(19,r,128,.1)['head.bias'])) for r in range(1,4001)]).double()
    assert torch.max(torch.abs(samples.mean(0))) < .007
    assert torch.allclose(samples.var(0),torch.full((258,),.01,dtype=torch.float64),rtol=.10,atol=0)
    cov=torch.cov(samples[:,:16].T);cov.fill_diagonal_(0)
    assert float(cov.abs().max()) < .0007
    x=torch.zeros((1,128));x[0,:2]=torch.tensor([1.,2.])
    v=(samples[:,128:256]-samples[:,:128])@x.double().flatten()+samples[:,-1]-samples[:,-2]
    assert float(v.var()) == pytest.approx(2*.01*6,rel=.06)


def test_local_trainer_padding_preserves_active_updates_and_global_rng():
    rng=torch.Generator().manual_seed(3)
    x=torch.randn((205,128),generator=rng);y=(x[:,0]>0).long()
    state=torch.random.get_rng_state().clone()
    a,_=local_fit(initial_parameters(128),x,y,11,1,0)
    wide=torch.cat((x,torch.zeros((205,384))),1)
    b,_=local_fit(initial_parameters(512),wide,y,11,1,0)
    assert torch.equal(state,torch.random.get_rng_state())
    assert torch.allclose(a['head.weight'],b['head.weight'][:,:128],atol=1e-7,rtol=0)
    assert torch.allclose(a['head.bias'],b['head.bias'],atol=1e-7,rtol=0)
    assert torch.count_nonzero(b['head.weight'][:,128:])==0
    assert torch.allclose(score(a,x),score(b,wide),atol=1e-6,rtol=0)


def test_noise_is_added_once_and_preview_does_not_consume_it():
    p=initial_parameters(128);s=StudyStrategy(p,seed=11,private=True)
    updates=[(str(i),p,205) for i in range(20)]
    preview=s.preview(updates)
    assert all(torch.equal(preview[k],p[k]) for k in p)
    actual=s.aggregate_fit(1,updates)
    expected=noise_draw(11,1,128,s.last_round_stats['noise_std'])
    assert all(torch.equal(actual[k],expected[k]) for k in p)
    assert s.last_round_stats['sensitivity']==pytest.approx(.005)
