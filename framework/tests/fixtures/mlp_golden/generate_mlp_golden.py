"""MLP fixtures for Stage 4 S3b: the masked trainable program and the torch golden the native trainer must reproduce.

The MLP (ECG, 140 -> 64 -> 64 -> 2, two Dropout(0.3)) trains on a device with its dropout masks drawn from the
contract's DROPOUT_MASKS_SEEDED_V1 stream and fed to the program as inputs. The golden is torch training the same
masked graph eagerly: Adam (the laptop's), seeded minibatches (BATCH_ORDER_SEEDED_PERMUTATION_V1), each step's mask per
layer from (seed, round, step, layer). The controls are what plausible mistakes would produce; the native test's
tolerance sits well below all of them.

    PYTHONPATH=framework/src:fl-runtime:mobile_client/scripts .venv/bin/python \
        framework/tests/fixtures/mlp_golden/generate_mlp_golden.py
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
from fedlearn.contract.dropout_masks import dropout_mask

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES, WIDTH, BATCH, EPOCHS, LR, SEED, ROUND = 20, 140, 8, 2, 1e-3, 42, 1
ATOL = 1e-5


def build_net():
    torch.manual_seed(0)
    return recipes.get_recipe("MLP").build_model("cpu").train()


def dataset():
    g = torch.Generator().manual_seed(11)
    x = torch.randn(EXAMPLES, WIDTH, generator=g, dtype=torch.float32)
    y = torch.randint(0, 2, (EXAMPLES,), generator=g, dtype=torch.int64)
    return x, y


def flat(net) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters() if p.requires_grad]).numpy().astype("<f4")


def train(mask_for, order_for=None) -> np.ndarray:
    """Torch Adam over the masked graph; mask_for(step, layer, rows, width) gives each layer's mask."""
    net = build_net()
    wrapper = pte_export._MaskedTrainingGraph(net)
    x, y = dataset()
    params = [p for p in wrapper.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    widths = [64 for _ in wrapper.slots]
    step = 0
    for epoch in range(EPOCHS):
        order = (order_for or (lambda e: seeded_permutation(EXAMPLES, SEED, ROUND, e)))(epoch)
        for idx in batches(order, BATCH):
            rows = len(idx)
            masks = tuple(torch.from_numpy(mask_for(step, layer, rows, widths[layer]).reshape(rows, widths[layer]))
                          for layer in range(len(widths)))
            opt.zero_grad()
            loss, _ = wrapper(x[idx], y[idx], masks)
            loss.backward()
            opt.step()
            step += 1
    return torch.cat([p.detach().reshape(-1) for p in wrapper.base.parameters()]).numpy().astype("<f4")


def main() -> None:
    net = build_net()
    rates = pte_export.dropout_layers(net)
    assert [r for _, r in rates] == [0.3, 0.3], rates
    flat(net).tofile(os.path.join(HERE, "mlp_init.f32"))
    x, y = dataset()
    x.numpy().astype("<f4").tofile(os.path.join(HERE, "mlp_inputs.f32"))
    y.numpy().astype("<i8").tofile(os.path.join(HERE, "mlp_targets.i64"))

    seeded = lambda step, layer, rows, width: dropout_mask(rows * width, rates[layer][1], SEED, ROUND, step, layer)
    golden = train(seeded)
    golden.tofile(os.path.join(HERE, "mlp_masked_adam_final.f32"))
    controls = {
        "dropout_off": train(lambda s, l, rows, width: np.ones(rows * width, dtype=np.float32)),
        "step_counter_reset_each_epoch": None,
        "layers_swapped": train(lambda s, l, rows, width: dropout_mask(rows * width, 0.3, SEED, ROUND, s, 1 - l)),
        "wrong_seed": train(lambda s, l, rows, width: dropout_mask(rows * width, 0.3, SEED + 1, ROUND, s, l)),
    }
    # Step counter restarting at 0 each epoch (3 steps per epoch here).
    controls["step_counter_reset_each_epoch"] = train(
        lambda s, l, rows, width: dropout_mask(rows * width, 0.3, SEED, ROUND, s % 3, l))
    separation = {k: float(np.abs(v - golden).max()) for k, v in controls.items()}
    if min(separation.values()) < 10 * ATOL:
        raise SystemExit(f"a control lands within 10x the tolerance: {separation}")

    pte = pte_export.export_masked_trainable_pte(net, (x[:BATCH], y[:BATCH]), mask_shapes=[(64,), (64,)],
                                                 max_batch=BATCH)
    with open(os.path.join(HERE, "mlp_trainable_masked_dynbatch.pte"), "wb") as fh:
        fh.write(pte)
    # The functional loss program (weights as inputs, eval mode, so dropout is the identity): what ModelManager loads
    # to hold the flat parameters, and what a device evaluates its dataset with.
    loss_pte = pte_export.export_functional_pte(build_net().eval(), (x[:BATCH], y[:BATCH]), max_batch=BATCH)
    with open(os.path.join(HERE, "mlp_loss_dynbatch.pte"), "wb") as fh:
        fh.write(loss_pte)
    manifest = {
        "description": "MLP masked-dropout training golden (Stage 4 S3b).",
        "loss_pte_file": "mlp_loss_dynbatch.pte", "loss_pte_sha256": hashlib.sha256(loss_pte).hexdigest(),
        "param_layout": [[n, list(p.shape)] for n, p in build_net().named_parameters() if p.requires_grad],
        "torch_version": torch.__version__.split("+")[0],
        "examples": EXAMPLES, "width": WIDTH, "batch_size": BATCH, "local_epochs": EPOCHS, "learning_rate": LR,
        "seed": SEED, "round": ROUND, "dropout": [{"module": n, "rate": r} for n, r in rates],
        "mask_shapes": [[64], [64]], "param_names_flat_order": pte_export.training_trainable_names(net),
        "flat_dim": int(flat(net).size), "pte_file": "mlp_trainable_masked_dynbatch.pte",
        "pte_sha256": hashlib.sha256(pte).hexdigest(), "max_batch": BATCH,
        "final_flat_file": "mlp_masked_adam_final.f32", "endpoint_atol": ATOL, "control_separation": separation,
    }
    with open(os.path.join(HERE, "mlp_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("pte", len(pte), manifest["pte_sha256"][:12], "separations", separation)


if __name__ == "__main__":
    main()
