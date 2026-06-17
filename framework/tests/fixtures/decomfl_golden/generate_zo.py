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
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--refreeze",
        action="store_true",
        help="recompute the canonical golden_g + model .pt + input batch freeze "
        "(reference-platform / intentional torch-bump ONLY; default PRESERVES them, since the "
        "g-scalar is platform-sensitive at ~1e-4 and the C++ tests are tolerance-based).",
    )
    args = ap.parse_args()

    # The functional .pte exporter lives in the sibling mobile_client unit.
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "mobile_client", "scripts"))
    from pte_export import export_functional_pte  # noqa: E402

    torch.manual_seed(0)
    net = TinyNet().eval()
    zo = ZerothOrderEstimator(smoothing_param=MU, device="cpu")

    flat = zo._get_flat_params(net)  # requires_grad-filtered
    flat_dim = int(flat.numel())
    total = int(sum(p.numel() for p in net.parameters()))
    trainable = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    manifest_path = os.path.join(HERE, "zo_manifest.json")
    inputs_path = os.path.join(HERE, "zo_inputs.f32")
    targets_path = os.path.join(HERE, "zo_targets.i64")
    model_path = os.path.join(HERE, "zo_model_tiny.pt")

    existing = None
    if os.path.exists(manifest_path) and not args.refreeze:
        with open(manifest_path) as fh:
            existing = json.load(fh)

    if existing is None:
        # Full (re)freeze — run on the canonical reference platform only.
        gin = torch.Generator().manual_seed(123)
        inputs = torch.randn(8, 4, generator=gin)
        targets = torch.randint(0, 3, (8,), generator=gin)
        golden_g = [
            float(zo.compute_gradient_scalar(net, flat, canonical_perturbation(s, flat_dim), inputs, targets))
            for s in SEEDS
        ]
        torch.jit.script(net).save(model_path)
        with open(model_path, "rb") as fh:
            model_sha = hashlib.sha256(fh.read()).hexdigest()
        inputs.numpy().astype("<f4").tofile(inputs_path)
        targets.numpy().astype("<i8").tofile(targets_path)
    else:
        # Preserve the canonical g-scalar freeze; read the COMMITTED batch so the new .pte
        # fixture is consistent with the existing g-scalar fixture (same inputs the C++ reads).
        import numpy as np

        golden_g = existing["golden_g"]
        model_sha = existing["model_sha256"]
        inputs = torch.from_numpy(np.fromfile(inputs_path, dtype="<f4").reshape(8, 4).copy())
        targets = torch.from_numpy(np.fromfile(targets_path, dtype="<i8").reshape(8).copy())

    # Phase 2 — functional .pte (weights-as-inputs) + the trainable flat + a single-forward
    # loss reference, all from THIS net + the committed batch: a self-consistent fixture the
    # C++ ExecuTorch forward must reproduce within tolerance (golden_loss == .pte forward).
    pte_bytes = export_functional_pte(net, (inputs, targets))
    with open(os.path.join(HERE, "zo_model_tiny.pte"), "wb") as fh:
        fh.write(pte_bytes)
    pte_sha = hashlib.sha256(pte_bytes).hexdigest()
    flat.detach().numpy().astype("<f4").tofile(os.path.join(HERE, "zo_flat.f32"))
    golden_loss = float(torch.nn.functional.cross_entropy(net(inputs), targets))

    manifest = {
        "description": "ZerothOrderEstimator g-scalar + flat-param-filter golden reference (C++ mobile core).",
        # Base version (strip +cpu/+cuXXX); the CPU kernel is identical across build variants.
        "torch_version": torch.__version__.split("+")[0],
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
        # Phase 2 — functional .pte (weights-as-inputs) + ExecuTorch forward reference.
        "pte_file": "zo_model_tiny.pte",
        "pte_sha256": pte_sha,
        "flat_file": "zo_flat.f32",
        "golden_loss": golden_loss,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    mode = "REFROZE canonical golden_g" if existing is None else "preserved canonical golden_g"
    print(f"trainable={trainable} total={total} flat_dim={flat_dim} ({mode})")
    print("golden_g =", golden_g)
    print("golden_loss =", golden_loss, "| pte_sha256 =", pte_sha[:12])


if __name__ == "__main__":
    main()
