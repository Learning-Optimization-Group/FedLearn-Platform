"""Freeze the ZerothOrderEstimator g-scalar + flat-param golden reference for the C++ core.

The native C++ mobile core (mobile_client/shared) must reproduce the Python framework's
zeroth-order gradient scalar within tolerance, and must filter frozen (requires_grad=False)
layers out of the flat parameter vector exactly as Python does. This script freezes a tiny
scripted model (one trainable + one FROZEN layer), a fixed batch, and the reference g for a
few seeds, all computed with the *real* Python reference
(fedlearn.estimators.zeroth_order.ZerothOrderEstimator) using the canonical perturbation.

Consumed by:
  * mobile_client/shared/tests/g_scalar_parity_test.cpp   (g within tolerance)
  * mobile_client/shared/tests/flatparam_filter_test.cpp  (trainable count == Python)

Run ONLY on an intentional torch bump:
    cd framework && PYTHONPATH=src python tests/fixtures/decomfl_golden/generate_zo.py
"""
from __future__ import annotations

import hashlib
import json
import os

import torch
import torch.nn as nn

from fedlearn.estimators.perturbation import canonical_perturbation
from fedlearn.estimators.zeroth_order import ZerothOrderEstimator

HERE = os.path.dirname(os.path.abspath(__file__))
MU = 0.001
SEEDS = [11, 22, 33, 4242]


class TinyNet(nn.Module):
    """Linear(4,5) -> ReLU -> Linear(5,3); fc2 is FROZEN (requires_grad=False).

    Trainable flat dim = fc1 weights+bias = 4*5 + 5 = 25. Total params = 25 + (5*3+3) = 43.
    The flat-param filter must include only the 25 trainable values.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)
        for p in self.fc2.parameters():
            p.requires_grad_(False)

    def forward(self, x):  # noqa: D401
        return self.fc2(torch.relu(self.fc1(x)))


def main() -> None:
    torch.manual_seed(0)
    net = TinyNet().eval()
    zo = ZerothOrderEstimator(smoothing_param=MU, device="cpu")

    flat = zo._get_flat_params(net)  # requires_grad-filtered
    flat_dim = int(flat.numel())
    total = int(sum(p.numel() for p in net.parameters()))
    trainable = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    gin = torch.Generator().manual_seed(123)
    inputs = torch.randn(8, 4, generator=gin)
    targets = torch.randint(0, 3, (8,), generator=gin)

    golden_g = []
    for s in SEEDS:
        z = canonical_perturbation(s, flat_dim)  # the v2 canonical z the C++ also uses
        golden_g.append(float(zo.compute_gradient_scalar(net, flat, z, inputs, targets)))

    scripted = torch.jit.script(net)
    model_path = os.path.join(HERE, "zo_model_tiny.pt")
    scripted.save(model_path)
    with open(model_path, "rb") as fh:
        model_sha = hashlib.sha256(fh.read()).hexdigest()

    inputs.numpy().astype("<f4").tofile(os.path.join(HERE, "zo_inputs.f32"))
    targets.numpy().astype("<i8").tofile(os.path.join(HERE, "zo_targets.i64"))

    manifest = {
        "description": "ZerothOrderEstimator g-scalar + flat-param-filter golden reference (C++ mobile core).",
        "torch_version": torch.__version__,
        "architecture": "Linear(4,5)->ReLU->Linear(5,3); fc2 FROZEN (requires_grad=False)",
        "model_file": "zo_model_tiny.pt",
        "model_sha256": model_sha,
        "total_params": total,
        "trainable_params": trainable,
        "flat_dim": flat_dim,
        "mu": MU,
        "method": "forward",
        "loss": "cross_entropy",
        "inputs_file": "zo_inputs.f32",
        "inputs_shape": [8, 4],
        "targets_file": "zo_targets.i64",
        "targets_shape": [8],
        "seeds": SEEDS,
        "golden_g": golden_g,
    }
    with open(os.path.join(HERE, "zo_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"trainable={trainable} total={total} flat_dim={flat_dim}")
    print("golden_g =", golden_g)


if __name__ == "__main__":
    main()
