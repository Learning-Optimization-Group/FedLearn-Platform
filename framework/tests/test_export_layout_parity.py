"""MO-15 done-when 2: the phone's exported flat-trainable layout must match the server's aggregation
order, or DeComFL gradient scalars would land on the wrong coordinates.

The mobile ExecuTorch export (mobile_client/scripts/pte_export.py) flattens trainable params in
named_parameters() requires_grad order; the server's ZerothOrderEstimator._get_flat_params flattens in
parameters() requires_grad order. These are the same iteration order, but nothing enforced it — this
pins it. Pure torch, no ExecuTorch runtime needed, so it runs in the framework pytest CI gate.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# pte_export lives in the mobile unit; import it by path (torch-only, no RN deps).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mobile_client", "scripts"))
import pte_export  # noqa: E402

from fedlearn.estimators.zeroth_order import ZerothOrderEstimator  # noqa: E402


class CnnNet(nn.Module):
    """Structural mirror of init_model.CnnNet (CIFAR-10). Copied inline so this test needs neither
    torchvision nor transformers (init_model.py imports both at module load). done-when 2 names CnnNet
    explicitly as the validation target."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TinyFrozen(nn.Module):
    """Linear net with a FROZEN second layer — mirrors the golden TinyNet (fc2 frozen), so the parity
    check also exercises requires_grad filtering: the flat vector must SKIP frozen params on both sides."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)
        for p in self.fc2.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _models() -> dict:
    torch.manual_seed(0)
    return {"CnnNet": CnnNet().eval(), "TinyFrozen": TinyFrozen().eval()}


def test_mobile_export_flat_layout_matches_server_zo_order():
    """The exported flat-trainable vector equals the server's _get_flat_params vector — same dim, same
    order, same values (the manifest paramLayout describes exactly this vector)."""
    for name, model in _models().items():
        server = ZerothOrderEstimator._get_flat_params(model)
        mobile = pte_export.trainable_flat(model)
        assert mobile.numel() == server.numel(), f"{name}: flat dim {mobile.numel()} != server {server.numel()}"
        assert torch.equal(mobile, server), f"{name}: flat order/values diverge from server _get_flat_params"


def test_param_layout_names_are_trainable_only_in_named_parameter_order():
    """The manifest paramLayout names are exactly the trainable params in named_parameters() order, with
    no frozen param leaking in (a leaked frozen name would desync the phone's unflatten from the server)."""
    for name, model in _models().items():
        layout_names = pte_export.trainable_names(model)
        expected = [n for n, p in model.named_parameters() if p.requires_grad]
        assert layout_names == expected, f"{name}: paramLayout not named_parameters requires_grad order"
        frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        assert not (set(layout_names) & frozen), f"{name}: frozen param leaked into paramLayout"


def test_roundtrip_set_then_get_is_identity_under_shared_order():
    """A vector written by the server's _set_flat_params reads back identically — confirms both sides
    agree on the unflatten order too (not just the flatten)."""
    for name, model in _models().items():
        flat = pte_export.trainable_flat(model)
        perturbed = flat + 1.0
        ZerothOrderEstimator._set_flat_params(model, perturbed)
        assert torch.equal(pte_export.trainable_flat(model), perturbed), f"{name}: set/get order mismatch"
