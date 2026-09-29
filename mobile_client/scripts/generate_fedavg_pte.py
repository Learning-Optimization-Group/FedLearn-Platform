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
from pte_export import (export_functional_infer_pte, export_functional_pte, export_trainable_pte,
                        training_trainable_names)


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


PROBE_LEARNING_RATE = 0.1


def probe_batch(step: int, rows: int, width: int, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The qualification probe's synthetic batch (Stage 3 D2): zeros at step 1, then the fixed pattern
    x[i][j] = ((i * width + j) mod 7 - 3) / 4; labels i mod classes. The native probe builds the same batch."""
    if step == 1:
        x = torch.zeros(rows, width, dtype=torch.float32)
    else:
        x = torch.tensor([[((i * width + j) % 7 - 3) / 4 for j in range(width)] for i in range(rows)],
                         dtype=torch.float32)
    return x, torch.tensor([i % classes for i in range(rows)], dtype=torch.int64)


def probe_reference(net: nn.Module, rows: int, classes: int) -> dict:
    """What the trainable program must report on the probe: two SGD steps from its embedded (export-time) weights.

    The probe is a property of the artifact, not of a run, so a device can cache its result per program digest.
    """
    import copy
    model = copy.deepcopy(net)
    width = model.fc1.in_features
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=PROBE_LEARNING_RATE)
    losses = []
    for step in (1, 2):
        x, y = probe_batch(step, rows, width, classes)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return {"rows": rows, "width": width, "classes": classes, "learning_rate": PROBE_LEARNING_RATE,
            "loss_step1": losses[0], "loss_step2": losses[1],
            # ExecuTorch's portable kernels agree with torch to ~1e-7 on this graph; the tolerance leaves room for
            # other CPUs while rejecting a program that computes something else.
            "loss_tolerance": 1e-4}


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

    # The programs a TinyNet run stages (scripts/stage_model_bundle.py): the same graphs with a dynamic example count
    # (1..8), so a device can train its own dataset rather than only a batch of exactly 8, which the static
    # programs refuse at runtime (ExecuTorch NotSupported). The static ones stay as the parity goldens.
    dynbatch = {"max_batch": 8}
    for key, filename, program in (
        ("loss", "tinynet_loss_dynbatch.pte", export_functional_pte(net.eval(), (x, y), max_batch=8)),
        ("infer", "tinynet_infer_dynbatch.pte", export_functional_infer_pte(net.eval(), x, max_batch=8)),
        ("trainable", "tinynet_trainable_dynbatch.pte", export_trainable_pte(net, (x, y), max_batch=8)),
    ):
        with open(os.path.join(golden_dir, filename), "wb") as fh:
            fh.write(program)
        dynbatch[f"{key}_file"] = filename
        dynbatch[f"{key}_sha256"] = hashlib.sha256(program).hexdigest()
    dynbatch["probe"] = probe_reference(net, rows=8, classes=3)

    manifest = {
        "description": "Trainable (forward+backward) TinyNet .pte for the C++ FedAvg parity gtest. "
                       "fc2 frozen (baked); only fc1 (25 params) trainable via the ET training extension.",
        "torch_version": torch.__version__.split("+")[0],
        "pte_file": "tinynet_trainable.pte",
        "pte_sha256": sha,
        # fully-qualified ET trainable names in canonical (framework named_parameters) flat order.
        "param_names_flat_order": training_trainable_names(net),
        "dynbatch": dynbatch,
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
