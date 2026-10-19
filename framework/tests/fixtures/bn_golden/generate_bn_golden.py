"""BatchNorm fixtures for Stage 4 S7: a device trains BatchNorm in training mode as a laptop does.

A small Conv-BatchNorm network trains with torch's own nn.BatchNorm2d in training mode (the laptop's semantics): Adam,
BATCH_ORDER_SEEDED_PERMUTATION_V1 minibatches with the final partial one kept. The golden is the final parameters and
the final running statistics. A device trains the same network through pte_export.export_bn_trainable_pte (BatchNorm
rebuilt from primitive ops, returning the batch statistics) and updates the running statistics by
BATCH_NORM_RUNNING_STATS_V1; it must land on both. The controls are what plausible mistakes would produce.

    PYTHONPATH=framework/src:fl-runtime:mobile_client/scripts .venv/bin/python \\
        framework/tests/fixtures/bn_golden/generate_bn_golden.py
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn

import pte_export
from fedlearn.contract.batch_order import batches, seeded_permutation

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES, BATCH, EPOCHS, LR, SEED, ROUND = 20, 8, 2, 1e-3, 42, 1
ATOL = 1e-4
# The wide fixture: the same network on 64 x 64 inputs, so its first BatchNorm reduces 8 x 62 x 62 = 30,752 values per
# channel, as a real image model's does. Its BatchNorm biases start at zero, so one SGD step at WIDE_LR leaves each at
# exactly -WIDE_LR times its gradient, which a device reads back to float32 precision.
WIDE, WIDE_LR, WIDE_BIAS_RTOL = 64, 0.1, 1e-6


class BnNet(nn.Module):
    """conv-BN-relu-conv-BN-relu, global average pool, linear: two BatchNorm layers at different spatial sizes.

    The convolutions have no bias, as in ResNet: BatchNorm subtracts the mean, so a bias before it has a true gradient
    of exactly zero. Its computed gradient is rounding noise whose sign differs between implementations, and Adam
    turns that sign into a full learning-rate step, so such a parameter is not replayable across runtimes.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.fc(x.mean(dim=(2, 3)))


def build_net():
    torch.manual_seed(0)
    net = BnNet()
    with torch.no_grad():                       # non-trivial starting statistics, as a pretrained model has
        for bn in (net.bn1, net.bn2):
            bn.running_mean.normal_(0.0, 0.5)
            bn.running_var.uniform_(0.5, 1.5)
    return net.train()


def dataset():
    g = torch.Generator().manual_seed(5)
    return torch.randn(EXAMPLES, 3, 8, 8, generator=g), torch.randint(0, 10, (EXAMPLES,), generator=g)


def state_flat(net) -> np.ndarray:
    """Trainable parameters, then each BatchNorm layer's running mean and variance (pte_export.federated_flat)."""
    return pte_export.federated_flat(net).numpy().astype("<f4")


def train(order_for=None, bn_mode="train", momentum=None, batch_size=BATCH) -> np.ndarray:
    net = build_net()
    if momentum is not None:
        for bn in (net.bn1, net.bn2):
            bn.momentum = momentum
    x, y = dataset()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    frozen = {n: b.clone() for n, b in net.named_buffers()}
    for epoch in range(EPOCHS):
        order = (order_for or (lambda e: seeded_permutation(EXAMPLES, SEED, ROUND, e)))(epoch)
        for idx in batches(order, batch_size):
            if bn_mode == "eval":
                net.bn1.eval(), net.bn2.eval()
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(x[idx]), y[idx]).backward()
            opt.step()
            if bn_mode == "stats_never_updated":
                with torch.no_grad():
                    for n, b in net.named_buffers():
                        b.copy_(frozen[n])
    return state_flat(net)


def biased_variance_control() -> np.ndarray:
    """The running variance updated with the biased batch variance (no n / (n - 1))."""
    net = build_net()
    x, y = dataset()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    stats = {}
    hooks = [m.register_forward_hook(lambda m, i, o, key=k: stats.__setitem__(key, i[0]))
             for k, m in (("bn1", net.bn1), ("bn2", net.bn2))]
    for epoch in range(EPOCHS):
        for idx in batches(seeded_permutation(EXAMPLES, SEED, ROUND, epoch), BATCH):
            before = {k: (getattr(net, k).running_mean.clone(), getattr(net, k).running_var.clone()) for k in stats or ("bn1", "bn2")}
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(x[idx]), y[idx]).backward()
            opt.step()
            with torch.no_grad():
                for k in ("bn1", "bn2"):
                    bn, inp = getattr(net, k), stats[k].double()
                    var = inp.var(dim=(0, 2, 3), unbiased=False)
                    bn.running_var.copy_(((1 - bn.momentum) * before[k][1].double() + bn.momentum * var).float())
    for h in hooks:
        h.remove()
    return state_flat(net)


def wide_fixture() -> tuple[bytes, np.ndarray]:
    """The wide program, and torch's float64 gradient of the probe batch's loss in its trainable order."""
    x, y = pte_export.probe_batch(1, BATCH, 3 * WIDE * WIDE, 10)
    x = x.view(BATCH, 3, WIDE, WIDE)
    pte = pte_export.export_bn_trainable_pte(build_net(), (x, y), max_batch=BATCH)
    exact = build_net().double()
    torch.nn.functional.cross_entropy(exact(x.double()), y).backward()
    params = dict(exact.named_parameters())
    grad = torch.cat([params[n.removeprefix("base.")].grad.reshape(-1) for n in pte_export.training_trainable_names(
        build_net())])
    return pte, grad.numpy()


def main() -> None:
    net = build_net()
    state_flat(net).tofile(os.path.join(HERE, "bn_init.f32"))
    x, y = dataset()
    x.numpy().astype("<f4").tofile(os.path.join(HERE, "bn_inputs.f32"))
    y.numpy().astype("<i8").tofile(os.path.join(HERE, "bn_targets.i64"))
    golden = train()
    golden.tofile(os.path.join(HERE, "bn_adam_final.f32"))
    controls = {
        "stats_never_updated": train(bn_mode="stats_never_updated"),
        "batch_norm_in_eval_mode": train(bn_mode="eval"),
        "momentum_0_01": train(momentum=0.01),
        "biased_running_variance": biased_variance_control(),
        "sequential_order": train(order_for=lambda e: list(range(EXAMPLES))),
    }
    separation = {k: float(np.abs(v - golden).max()) for k, v in controls.items()}
    if min(separation.values()) < 10 * ATOL:
        raise SystemExit(f"a control lands within 10x the tolerance: {separation}")

    trainable = pte_export.export_bn_trainable_pte(net, (x[:BATCH], y[:BATCH]), max_batch=BATCH)
    loss_pte = pte_export.export_functional_pte(build_net().eval(), (x[:BATCH], y[:BATCH]), max_batch=BATCH,
                                                include_buffers=True)
    for name, data in (("bn_trainable_dynbatch.pte", trainable), ("bn_loss_dynbatch.pte", loss_pte)):
        with open(os.path.join(HERE, name), "wb") as fh:
            fh.write(data)
    # Each layer's values per example: the height x width of its input.
    sizes = {}
    hooks = [m.register_forward_hook(lambda m, i, o, key=k: sizes.__setitem__(key, i[0].shape[2] * i[0].shape[3]))
             for k, m in (("bn1", net.bn1), ("bn2", net.bn2))]
    with torch.no_grad():
        build_net().eval()(x[:1])
        net.eval()(x[:1])
    for h in hooks:
        h.remove()
    wide_pte, wide_grad = wide_fixture()
    with open(os.path.join(HERE, "bn_wide_trainable.pte"), "wb") as fh:
        fh.write(wide_pte)
    wide_grad.astype("<f8").tofile(os.path.join(HERE, "bn_wide_grad.f64"))
    params = [(n, list(p.shape)) for n, p in net.named_parameters()]
    buffers = [(n, list(dict(net.named_buffers())[n].shape)) for n in pte_export.federated_buffer_names(net)]
    manifest = {
        "description": "Training-mode BatchNorm golden (Stage 4 S7): torch nn.BatchNorm2d in training mode, Adam.",
        "torch_version": torch.__version__.split("+")[0],
        "examples": EXAMPLES, "input_shape": [3, 8, 8], "batch_size": BATCH, "local_epochs": EPOCHS,
        "learning_rate": LR, "seed": SEED, "round": ROUND,
        "state_layout": params + buffers, "trainable_param_count": sum(int(np.prod(s)) for _, s in params),
        "param_names_flat_order": pte_export.training_trainable_names(net),
        "batch_norm": [{"module": n, "num_features": f, "momentum": m, "eps": e, "values_per_example": sizes[n]}
                       for n, f, m, e in pte_export.batch_norm_layers(net)],
        "pte_file": "bn_trainable_dynbatch.pte", "pte_sha256": hashlib.sha256(trainable).hexdigest(),
        "loss_pte_file": "bn_loss_dynbatch.pte", "loss_pte_sha256": hashlib.sha256(loss_pte).hexdigest(),
        "final_state_file": "bn_adam_final.f32", "endpoint_atol": ATOL, "control_separation": separation,
        "wide": {
            "description": "Gradient precision of a BatchNorm over many values per channel: one SGD step on the "
                           "probe batch (pte_export.probe_batch, step 1), gradient from torch in float64.",
            "pte_file": "bn_wide_trainable.pte", "pte_sha256": hashlib.sha256(wide_pte).hexdigest(),
            "input_shape": [3, WIDE, WIDE], "rows": BATCH, "classes": 10, "learning_rate": WIDE_LR,
            "gradient_file": "bn_wide_grad.f64", "bias_rtol": WIDE_BIAS_RTOL,
        },
    }
    with open(os.path.join(HERE, "bn_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("separations", separation, "batch_norm", manifest["batch_norm"])


if __name__ == "__main__":
    main()
