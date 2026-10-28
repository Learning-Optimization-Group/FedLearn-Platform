"""Sorted-threshold ROC integration, independent of the later rank-sum audit."""
import numpy as np
from scipy.special import expit


def metrics(y,s):
    y,s=np.asarray(y),np.asarray(s,dtype=np.float64)
    if y.ndim!=1 or s.shape!=y.shape or not len(y) or not np.isfinite(s).all() \
            or set(np.unique(y))!={0,1}:
        raise ValueError('finite matching scores and both binary classes required')
    order=np.argsort(-s,kind='stable');sorted_y=y[order];sorted_s=s[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_s)),len(y)-1]
    tp=np.cumsum(sorted_y)[ends];fp=ends+1-tp
    auroc=np.trapezoid(np.r_[0,tp]/sum(y),np.r_[0,fp]/sum(1-y))
    p=expit(s)
    return dict(auroc=float(auroc),log_loss=float(np.mean(np.logaddexp(0,s)-y*s)),
                brier=float(np.mean((p-y)**2)))
