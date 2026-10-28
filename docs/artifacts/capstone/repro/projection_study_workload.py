"""Public synthetic features; no filesystem, network or medical-data inputs."""
import numpy as np
from scipy.special import expit
import torch

from capstone_pilot_mechanisms import public_projection
from projection_study_design import SETTINGS, seed_for


def dataset(seed=20261027):
    out = {}
    for split,n in (('train',4096),('eval',8192)):
        rng = np.random.Generator(np.random.PCG64(seed_for(seed,split,'features')))
        x = rng.standard_normal((n,512)).astype(np.float32)
        labels = np.random.Generator(np.random.PCG64(seed_for(seed,split,'labels')))
        y = (labels.random(n) < expit(2*x[:,0].astype(np.float64))).astype(np.int64)
        out[split+'_x'],out[split+'_y'] = torch.from_numpy(x),torch.from_numpy(y)
    return out


def validate_basis(p):
    if p.shape != (512,128) or not torch.isfinite(p).all() or not torch.allclose(
            p.double().T @ p.double(),torch.eye(128,dtype=torch.float64),atol=1e-10,rtol=0):
        raise ValueError('finite orthonormal 512 by 128 basis required')


def basis(cell):
    if cell.family == 'full':
        return torch.eye(512,dtype=torch.float64)
    if cell.family == 'controlled':
        p = torch.zeros((512,128),dtype=torch.float64)
        p[0,0],p[128,0] = cell.eta**.5,(1-cell.eta)**.5
        p[torch.arange(1,128),torch.arange(1,128)] = 1.
    else:
        p = public_projection(512,128,seed=cell.projection_seed)
    validate_basis(p)
    return p


def features(data,cell):
    p = basis(cell)
    out = {k:data[k] for k in ('train_y','eval_y')}
    for split in ('train','eval'):
        x = data[split+'_x']
        transformed = x if cell.family == 'full' else x @ p.float()
        if cell.family == 'padded':
            transformed = torch.cat((transformed,torch.zeros((len(x),384))),dim=1)
        out[split+'_x'] = transformed.contiguous()
    return out


def partitions(seed):
    rng = np.random.Generator(np.random.PCG64(seed_for(seed,'partition')))
    return [torch.from_numpy(x.copy()) for x in np.array_split(rng.permutation(SETTINGS['train']),20)]


def coalition(seed):
    rng = np.random.Generator(np.random.PCG64(seed_for(seed,'coalition')))
    return tuple(sorted(map(int,rng.choice(20,4,replace=False))))
