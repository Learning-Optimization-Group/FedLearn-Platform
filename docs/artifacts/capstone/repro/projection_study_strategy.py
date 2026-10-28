"""Study-only Gaussian coupling around canonical clipping and SGD training."""
from collections import OrderedDict
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedlearn.client.local_trainer import LocalTrainer
from fedlearn.privacy.gdp_accountant import gdp_required_noise_multiplier
from fedlearn.server.composed_strategy import ComposedStrategy
from projection_study_design import SETTINGS, seed_for


def initial_parameters(width):
    if width not in (128,512):
        raise ValueError('study supports widths 128 and 512')
    return OrderedDict([('head.weight',torch.zeros((2,width))),('head.bias',torch.zeros(2))])


def noise_draw(seed,round,width,std):
    if width not in (128,512) or round < 1 or not math.isfinite(std) or std < 0:
        raise ValueError('invalid Gaussian draw settings')
    out=initial_parameters(width)
    for row in range(2):
        generator=torch.Generator().manual_seed(seed_for(seed,'noise',round,'active',row))
        out['head.weight'][row,:128]=torch.randn(128,generator=generator)*std
        if width==512:
            generator=torch.Generator().manual_seed(seed_for(seed,'noise',round,'inactive',row))
            out['head.weight'][row,128:]=torch.randn(384,generator=generator)*std
    generator=torch.Generator().manual_seed(seed_for(seed,'noise',round,'bias'))
    out['head.bias']=torch.randn(2,generator=generator)*std
    return out


class StudyStrategy(ComposedStrategy):
    def __init__(self,parameters,*,seed,private):
        z=gdp_required_noise_multiplier(4.,20,1e-5) if private else 0.
        super().__init__(parameters,rule='mean',clip_norm=.05,clients_per_round=20,
                         min_fit_clients=20,central_noise_multiplier=z)
        self.study_seed=seed
        self.last_noise=initial_parameters(parameters['head.weight'].shape[1])
        self.last_pre_noise=None

    def _pipeline(self,server_round,results,*,apply_noise):
        if len(results)!=20 or {str(c) for c,_,_ in results}!={str(i) for i in range(20)} \
                or any(n<=0 or not self._valid(p) for _,p,n in results):
            return None,dict(failed=True,aggregated=len(results),message='complete fixed study cohort required')
        out,stats=super()._pipeline(server_round,results,apply_noise=False)
        if out is None:
            return out,stats
        self.last_pre_noise=OrderedDict((k,v.clone()) for k,v in out.items())
        if apply_noise:
            self.last_noise=noise_draw(self.study_seed,server_round,out['head.weight'].shape[1],stats['noise_std'])
            out=OrderedDict((k,v+self.last_noise[k]) for k,v in out.items())
        return out,stats


def local_fit(parameters,x,y,seed,round,client):
    width=x.shape[1]
    with torch.random.fork_rng(devices=[]):
        model=nn.Sequential(OrderedDict([('head',nn.Linear(width,2))]))
    generator=torch.Generator().manual_seed(seed_for(seed,'minibatch',round,client))
    loader=DataLoader(TensorDataset(x,y),batch_size=128,shuffle=True,generator=generator,num_workers=0)
    trainer=LocalTrainer(model,loader)
    return trainer.fit(parameters,dict(learning_rate=.03,local_epochs=1))


def score(parameters,x):
    w=parameters['head.weight'].double();b=parameters['head.bias'].double()
    return torch.nn.functional.linear(x.double(),(w[1]-w[0]).unsqueeze(0),
                                      (b[1]-b[0]).reshape(1)).squeeze(1)
