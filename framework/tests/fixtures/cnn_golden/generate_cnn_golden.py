"""CNN fixtures for Stage 4 S6: the trainable program and the torch golden the native trainer must reproduce.

The CNN (CIFAR-10 LeNet: conv-pool-conv-pool-fc-fc-fc, no dropout) trains on a device from images prepared by the
contract's ImageToUnitTensor and NormalizeChannels(0.5, 0.5). The golden is torch training the same model eagerly:
Adam (the laptop's), BATCH_ORDER_SEEDED_PERMUTATION_V1 minibatches with the final partial one kept. The controls are
what plausible mistakes would produce; the native test's tolerance sits well below all of them.

    PYTHONPATH=framework/src:fl-runtime:mobile_client/scripts .venv/bin/python \\
        framework/tests/fixtures/cnn_golden/generate_cnn_golden.py
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

import pte_export
import recipes
from fedlearn.contract.batch_order import batches, seeded_permutation
from fedlearn.contract.image_transforms import image_to_unit_tensor, normalize_channels

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES, BATCH, EPOCHS, LR, SEED, ROUND = 20, 8, 2, 1e-3, 42, 1
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
# The native trainer's tolerance. Not the MLP's 1e-5: this golden's own float32 arithmetic differs from the same
# training in float64 by up to 1.26e-5 (one element; 99.9% agree within 1e-7), because Adam normalises the step of a
# parameter whose gradient is near zero, so a rounding-level gradient difference moves it by a visible amount. The
# native round measured 1.1e-5. 1e-4 is about 8x float32's own deviation and still 60x below the nearest control.
ATOL = 1e-4


def build_net():
    torch.manual_seed(0)
    return recipes.get_recipe("CNN").build_model("cpu").train()


def raw_images() -> np.ndarray:
    return np.random.default_rng(17).integers(0, 256, size=(EXAMPLES, 32, 32, 3), dtype=np.uint8)


def dataset(normalize=True):
    images = [image_to_unit_tensor(p) for p in raw_images()]
    if normalize:
        images = [normalize_channels(t, MEAN, STD) for t in images]
    x = torch.from_numpy(np.stack(images))
    y = torch.from_numpy(np.random.default_rng(18).integers(0, 10, size=EXAMPLES).astype(np.int64))
    return x, y


def flat(net) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters() if p.requires_grad]).numpy().astype("<f4")


def train(order_for=None, optimizer="adam", batch_size=BATCH, normalize=True) -> np.ndarray:
    net = build_net()
    x, y = dataset(normalize)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR) if optimizer == "adam" else torch.optim.SGD(params, lr=LR)
    for epoch in range(EPOCHS):
        order = (order_for or (lambda e: seeded_permutation(EXAMPLES, SEED, ROUND, e)))(epoch)
        for idx in batches(order, batch_size):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(x[idx]), y[idx]).backward()
            opt.step()
    return flat(net)


def main() -> None:
    net = build_net()
    flat(net).tofile(os.path.join(HERE, "cnn_init.f32"))
    x, y = dataset()
    x.numpy().astype("<f4").tofile(os.path.join(HERE, "cnn_inputs.f32"))
    y.numpy().astype("<i8").tofile(os.path.join(HERE, "cnn_targets.i64"))

    golden = train()
    golden.tofile(os.path.join(HERE, "cnn_adam_final.f32"))
    controls = {
        "sequential_order": train(order_for=lambda e: list(range(EXAMPLES))),
        "round_zero_order": train(order_for=lambda e: seeded_permutation(EXAMPLES, SEED, ROUND - 1, e)),
        "one_full_batch": train(batch_size=EXAMPLES),
        "sgd_not_adam": train(optimizer="sgd"),
        "unnormalized_images": train(normalize=False),
    }
    separation = {k: float(np.abs(v - golden).max()) for k, v in controls.items()}
    if min(separation.values()) < 10 * ATOL:
        raise SystemExit(f"a control lands within 10x the tolerance: {separation}")

    trainable = pte_export.export_trainable_pte(net, (x[:BATCH], y[:BATCH]), max_batch=BATCH)
    with open(os.path.join(HERE, "cnn_trainable_dynbatch.pte"), "wb") as fh:
        fh.write(trainable)
    loss_pte = pte_export.export_functional_pte(build_net().eval(), (x[:BATCH], y[:BATCH]), max_batch=BATCH)
    with open(os.path.join(HERE, "cnn_loss_dynbatch.pte"), "wb") as fh:
        fh.write(loss_pte)
    manifest = {
        "description": "CNN image training golden (Stage 4 S6).",
        "torch_version": torch.__version__.split("+")[0],
        "examples": EXAMPLES, "input_shape": [3, 32, 32], "batch_size": BATCH, "local_epochs": EPOCHS,
        "learning_rate": LR, "seed": SEED, "round": ROUND, "normalize": {"mean": MEAN, "std": STD},
        "param_layout": [[n, list(p.shape)] for n, p in build_net().named_parameters() if p.requires_grad],
        "param_names_flat_order": pte_export.training_trainable_names(net),
        "flat_dim": int(flat(net).size), "max_batch": BATCH,
        "pte_file": "cnn_trainable_dynbatch.pte", "pte_sha256": hashlib.sha256(trainable).hexdigest(),
        "loss_pte_file": "cnn_loss_dynbatch.pte", "loss_pte_sha256": hashlib.sha256(loss_pte).hexdigest(),
        "final_flat_file": "cnn_adam_final.f32", "endpoint_atol": ATOL, "control_separation": separation,
        "probe": pte_export.probe_reference(build_net(), rows=BATCH, width=3072, classes=10, input_shape=(3, 32, 32)),
    }
    with open(os.path.join(HERE, "cnn_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("trainable pte", len(trainable), manifest["pte_sha256"][:12], "separations", separation)


if __name__ == "__main__":
    main()
