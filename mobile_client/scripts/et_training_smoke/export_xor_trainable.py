#!/usr/bin/env python3
"""export_xor_trainable.py — export a TRAINABLE ExecuTorch ``.pte`` (forward graph + captured
backward graph) so an on-device C++ trainer can run real backprop.

This is the export half of the Phase-A de-risk for "mobile supports all algorithms" (first-order
FedAvg/FedProx on device). ExecuTorch is an inference runtime with no autograd, so a normal ``.pte``
cannot be trained. The training extension's trick (mirrored from ExecuTorch's own
``extension/training/examples/XOR/export_model.py``) is to capture the backward pass AS a graph at
export time via ``torch.export.experimental._export_forward_backward`` — the resulting ``.pte`` then
exposes gradients through ``training_module.named_gradients`` at runtime.

Must run inside a venv that has ``executorch`` installed (it pulls its own pinned torch). See
``run_training_smoke_macos.sh``, which provisions that venv and calls this.

Usage: python export_xor_trainable.py <out.pte>
"""
import sys

import torch
from torch.export import export
from torch.export.experimental import _export_forward_backward
from executorch.exir import to_edge


class Net(torch.nn.Module):
    """Tiny 2->10->2 MLP — the classic XOR net (non-linear, needs a hidden layer)."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(2, 10)
        self.act = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.act(self.l1(x)))


class TrainingNet(torch.nn.Module):
    """Wraps the net so ``forward(x, y)`` returns (loss, prediction) — the shape the backward-graph
    capture and the C++ ``execute_forward_backward("forward", {x, y})`` call both expect."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.net(x)
        return self.loss(out, y), out.detach().argmax(1)


def main(out_path: str) -> None:
    net = TrainingNet(Net())
    # Example inputs pin the shapes: x:[1,2] f32, y:[1] int64.
    x = torch.randn(1, 2)
    y = torch.ones(1, dtype=torch.int64)

    exported = export(net, (x, y), strict=True)
    exported = _export_forward_backward(exported)  # capture the backward pass as a graph
    edge = to_edge(exported)
    program = edge.to_executorch()

    with open(out_path, "wb") as f:
        f.write(program.buffer)
    print(f"WROTE_PTE {out_path} {len(program.buffer)} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
