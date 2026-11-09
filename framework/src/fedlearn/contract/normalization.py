"""Normalization layers contract and privacy preservation utilities.

Provides mechanisms to mitigate batch normalization feature and distribution leakage in federated learning:
1. Converting BatchNorm layers to GroupNorm (Wu & He, ECCV 2018; Hsieh et al., 2020), eliminating
   running_mean and running_var buffers entirely.
2. Freezing BatchNorm buffers (SiloBN / Local BN, Li et al., ICLR 2021) to prevent client feature
   distribution leakage across non-IID shards.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

DEFAULT_MAX_GROUPS = 32


def has_batch_norm(module: nn.Module) -> bool:
    """True if ``module`` contains any 1D, 2D, or 3D BatchNorm layer."""
    return any(isinstance(m, _BatchNorm) for m in module.modules())


def freeze_batch_norm_buffers(module: nn.Module) -> nn.Module:
    """Put every BatchNorm module into eval mode, in place, preserving parameter grad requirements.

    This prevents local forward passes from mutating running_mean and running_var buffers,
    closing feature distribution side-channels on private client data.
    """
    for m in module.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
    return module


def convert_bn_to_gn(
    module: nn.Module,
    max_groups: int = DEFAULT_MAX_GROUPS,
    copy_affine: bool = False,
) -> nn.Module:
    """Replace every ``_BatchNorm`` with a ``GroupNorm`` over the same channels, in place.

    - Eliminates running_mean and running_var state entirely, preventing feature moment leakage.
    - Prevents federated divergence under non-IID data where averaging local moments distorts features.
    - Enables mobile export (ExecuTorch Core ATen compatibility).

    Args:
        module: PyTorch module to convert.
        max_groups: Upper bound on the number of groups for GroupNorm. The actual group count
            is the largest divisor of ``num_features`` that is <= ``max_groups``.
        copy_affine: If True and the source BatchNorm has affine parameters, copies weight and bias
            into the newly instantiated GroupNorm.
    """
    if max_groups < 1:
        raise ValueError(f"max_groups must be >= 1, got {max_groups}")

    for name, child in module.named_children():
        if isinstance(child, _BatchNorm):
            c = child.num_features
            groups = min(max_groups, c)
            while c % groups:
                groups -= 1
            gn = nn.GroupNorm(groups, c)
            if copy_affine and child.affine and child.weight is not None and child.bias is not None:
                with torch.no_grad():
                    gn.weight.copy_(child.weight.detach())
                    gn.bias.copy_(child.bias.detach())
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, max_groups=max_groups, copy_affine=copy_affine)
    return module
