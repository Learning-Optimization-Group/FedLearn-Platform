"""BATCH_NORM_RUNNING_STATS_V1: how a participant updates a BatchNorm layer's running statistics in training.

After each local step, for each BatchNorm layer, with the batch mean and the biased batch variance the step normalised
with, and ``count`` = examples x height x width values per channel:

    running_mean <- (1 - momentum) * running_mean + momentum * batch_mean
    running_var  <- (1 - momentum) * running_var  + momentum * batch_var * count / (count - 1)

computed in float64 and rounded to float32 once, as torch's CPU BatchNorm does (its accumulate type for float32 is
double). ``num_batches_tracked`` is a counter, not federated. A device's training program returns the batch statistics;
its native trainer applies this rule; laptops get the same result from torch's own BatchNorm.
"""
from __future__ import annotations

import numpy as np


def running_stats_update(running_mean, running_var, batch_mean, batch_var, count: int, momentum: float):
    """(new running_mean, new running_var) as float32, from the step's batch mean and biased batch variance."""
    if count < 2:
        raise ValueError("the unbiased variance needs at least two values per channel")
    m = float(momentum)
    mean = (1.0 - m) * np.asarray(running_mean, np.float64) + m * np.asarray(batch_mean, np.float64)
    unbiased = np.asarray(batch_var, np.float64) * (count / (count - 1.0))
    var = (1.0 - m) * np.asarray(running_var, np.float64) + m * unbiased
    return mean.astype(np.float32), var.astype(np.float32)
