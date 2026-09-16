#!/usr/bin/env python3
"""Freeze the trainable TINYNET .pte the C++ FedAvg parity gtest replays (Phase B M1c).

Exports the joint forward+backward graph (pte_export.export_trainable_pte) for the SAME seed-0
TinyNet the ZO/FedAvg goldens use (fc2 frozen), so ET's TrainingModule can do real backprop on it.
The frozen fc2 is baked into the graph, so it MUST match the framework's fc2 — this asserts the
seed-0 init reproduces the committed zo_flat.f32 (fc1) before exporting, catching any torch-version
init drift that would silently break parity (the baked fc2 would differ from the framework's).

Runs in an ExecuTorch-enabled env (executorch pulls its own torch); TinyNet is inlined so no
framework import is needed. Writes the .pte + a small sidecar manifest (path + sha256 + param names).

Usage: python generate_fedavg_pte.py <golden_dir>
    (golden_dir = framework/tests/fixtures/decomfl_golden)
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this dir -> pte_export
from pte_export import export_trainable_pte, training_trainable_names


class TinyNet(nn.Module):
    """EXACT mirror of framework generate_zo.py TinyNet: Linear(4,5) -> ReLU -> Linear(5,3), fc2
    FROZEN. Construction order fixes the seed-0 init stream, so manual_seed(0) reproduces zo_flat."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)
        for p in self.fc2.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main(golden_dir: str) -> None:
    torch.manual_seed(0)
    net = TinyNet()

    # init-drift guard: baked fc2 matches the framework only if the seed-0 init matches. Verify fc1.
    trainable = torch.cat([p.detach().reshape(-1) for _, p in net.named_parameters() if p.requires_grad])
    zo_flat = np.fromfile(os.path.join(golden_dir, "zo_flat.f32"), dtype="<f4")
    got = trainable.numpy().astype("<f4")
    if not np.array_equal(got, zo_flat):
        raise SystemExit(
            f"seed-0 TinyNet fc1 != committed zo_flat.f32 (torch={torch.__version__}); the baked fc2 "
            "would diverge from the framework and break parity. Export under a torch whose init matches."
        )

    # example inputs pin the graph's shapes to the committed batch ({8,4} float, {8} int64).
    x = torch.from_numpy(np.fromfile(os.path.join(golden_dir, "zo_inputs.f32"), dtype="<f4").reshape(8, 4).copy())
    y = torch.from_numpy(np.fromfile(os.path.join(golden_dir, "zo_targets.i64"), dtype="<i8").reshape(8).copy())

    pte = export_trainable_pte(net, (x, y))
    out_pte = os.path.join(golden_dir, "tinynet_trainable.pte")
    with open(out_pte, "wb") as fh:
        fh.write(pte)
    sha = hashlib.sha256(pte).hexdigest()

    manifest = {
        "description": "Trainable (forward+backward) TinyNet .pte for the C++ FedAvg parity gtest. "
                       "fc2 frozen (baked); only fc1 (25 params) trainable via the ET training extension.",
        "torch_version": torch.__version__.split("+")[0],
        "pte_file": "tinynet_trainable.pte",
        "pte_sha256": sha,
        # fully-qualified ET trainable names in canonical (framework named_parameters) flat order.
        "param_names_flat_order": training_trainable_names(net),
    }
    with open(os.path.join(golden_dir, "fedavg_pte_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print("WROTE", out_pte, len(pte), "bytes")
    print("pte_sha256 =", sha)
    print("param_names_flat_order =", manifest["param_names_flat_order"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
