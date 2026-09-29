"""Freeze a FedAvg (first-order) local-update golden — Python<->C++ endpoint parity.

The mobile first-order path (Phase B) must reproduce the framework's REAL FedAvg client update
within tolerance, exactly as the DeComFL multiround golden pins the zeroth-order path. FedAvg =
``LocalTrainer.fit`` with ``mu=0`` (local_trainer.py module docstring + :78-84): plain minibatch
``torch.optim.SGD(model.parameters(), lr)`` over ``local_epochs`` passes, CrossEntropyLoss, returns
the updated state_dict. This script runs it on the SAME committed TinyNet + batch the ZO goldens use
and freezes the endpoint (trainable fc1 flat) so the native ``TrainableExecutorchModel`` can replay
K SGD steps and assert a tolerance-bounded match.

Consumed by:
  * framework/tests/test_fedavg_local_golden.py        (Python self-consistency, CI-gated, pure torch)
  * mobile_client/shared/tests/fedavg_parity_test.cpp  (C++ ET first-order endpoint, added in M1c)
  * mobile_client/shared/tests/fedavg_firstorder_round_test.cpp  (the FedProx endpoint, fedprox_local_*)

Pure torch (NO executorch) — runs in the framework pytest gate. Freeze ONLY on an intentional torch
bump (torch pinned 2.12.0, matching zo_manifest.json):
    cd framework && PYTHONPATH=src python tests/fixtures/decomfl_golden/generate_fedavg_golden.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform

import numpy as np
import torch

from fedlearn.client.local_trainer import LocalTrainer
from fedlearn.estimators.params import flat_params, param_layout

from generate_zo import TinyNet  # the SAME seed-0 net the ZO goldens freeze (fc2 frozen, fc1 25 params)

HERE = os.path.dirname(os.path.abspath(__file__))

# --- first-order local-update config. Full-batch (one batch/epoch) so the trajectory is fully
#     deterministic and trivially replayable in C++: local_epochs == number of SGD steps. Kept
#     small so cross-runtime drift (ET backward vs torch autograd) stays inside endpoint_atol. ---
LR = 0.1
LOCAL_EPOCHS = 5  # == number of full-batch SGD steps
SMALL_BATCH = 6  # a device dataset smaller than the 8-example batch the programs are exported with


class _OneBatchLoader:
    """Yields the whole committed batch once per epoch; ``.dataset`` len == n (num_examples).

    LocalTrainer does one optimiser step per yielded batch and reads ``len(self.train_loader.dataset)``
    for num_examples (local_trainer.py:97,126), so one batch/epoch == one full-batch SGD step/epoch.
    """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self._batch = (inputs, targets)
        self.dataset = list(range(int(inputs.shape[0])))  # len == n

    def __iter__(self):
        yield self._batch


class _SeededMinibatchLoader:
    """Yields BATCH_ORDER_SEEDED_PERMUTATION_V1 minibatches: each iteration (one LocalTrainer epoch) draws the next
    epoch's permutation and slices it into ``batch_size`` batches, keeping the last. ``epoch_offset=None`` pins every
    epoch to epoch 0's order (a plausible implementation bug the golden must be able to reject)."""

    def __init__(self, inputs, targets, batch_size, seed, round_, epoch_offset=0, order=None):
        self._x, self._y = inputs, targets
        self._batch_size, self._seed, self._round = batch_size, seed, round_
        self._epoch, self._pin = 0, epoch_offset is None
        self._order = order  # a fixed order for every epoch (the sequential control), or None for the seeded one
        self.dataset = list(range(int(inputs.shape[0])))

    def __iter__(self):
        from fedlearn.contract.batch_order import batches, seeded_permutation
        n = int(self._x.shape[0])
        epoch = 0 if self._pin else self._epoch
        order = self._order if self._order is not None else seeded_permutation(n, self._seed, self._round, epoch)
        self._epoch += 1
        for idx in batches(order, self._batch_size):
            yield self._x[idx], self._y[idx]


# Stage 3 slice C2: a device dataset larger than one batch. 20 examples in batches of 8 (8, 8, then a kept 4) for
# MINIBATCH_EPOCHS epochs, in the contract's seeded order for (MINIBATCH_SEED, MINIBATCH_ROUND).
MINIBATCH_EXAMPLES, MINIBATCH_SIZE, MINIBATCH_EPOCHS = 20, 8, 2
MINIBATCH_SEED, MINIBATCH_ROUND = 42, 1


def minibatch_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    x = torch.randn(MINIBATCH_EXAMPLES, 4, generator=g, dtype=torch.float32)
    y = torch.randint(0, 3, (MINIBATCH_EXAMPLES,), generator=g, dtype=torch.int64)
    return x, y


def compute_minibatch_endpoint(loader) -> np.ndarray:
    net = build_initial_net()
    LocalTrainer(net, loader, device="cpu").fit(
        None, {"learning_rate": str(LR), "local_epochs": str(MINIBATCH_EPOCHS), "proximal_mu": "0"})
    return flat_params(net).detach().cpu().numpy().astype("<f4")


def write_minibatch_endpoint() -> None:
    """The native minibatch loop must land here; the controls are what a wrong batch order would produce."""
    x, y = minibatch_dataset()
    x.numpy().astype("<f4").tofile(os.path.join(HERE, "minibatch_inputs.f32"))
    y.numpy().astype("<i8").tofile(os.path.join(HERE, "minibatch_targets.i64"))
    seeded = compute_minibatch_endpoint(
        _SeededMinibatchLoader(x, y, MINIBATCH_SIZE, MINIBATCH_SEED, MINIBATCH_ROUND))
    controls = {
        "sequential_order": compute_minibatch_endpoint(_SeededMinibatchLoader(
            x, y, MINIBATCH_SIZE, MINIBATCH_SEED, MINIBATCH_ROUND, order=list(range(MINIBATCH_EXAMPLES)))),
        "one_full_batch": compute_minibatch_endpoint(_SeededMinibatchLoader(
            x, y, MINIBATCH_EXAMPLES, MINIBATCH_SEED, MINIBATCH_ROUND)),
        "epoch_zero_order_every_epoch": compute_minibatch_endpoint(_SeededMinibatchLoader(
            x, y, MINIBATCH_SIZE, MINIBATCH_SEED, MINIBATCH_ROUND, epoch_offset=None)),
    }
    separations = {k: float(np.abs(v - seeded).max()) for k, v in controls.items()}
    seeded.tofile(os.path.join(HERE, "fedavg_minibatch_final.f32"))
    manifest = {
        "description": "First-order minibatch golden (Stage 3 C2): LocalTrainer.fit on minibatch_inputs/targets in "
                       "BATCH_ORDER_SEEDED_PERMUTATION_V1 order, one SGD step per minibatch, final batch kept.",
        "torch_version": torch.__version__.split("+")[0],
        "examples": MINIBATCH_EXAMPLES, "batch_size": MINIBATCH_SIZE, "local_epochs": MINIBATCH_EPOCHS,
        "learning_rate": LR, "seed": MINIBATCH_SEED, "round": MINIBATCH_ROUND,
        "initial_flat_file": "zo_flat.f32", "inputs_file": "minibatch_inputs.f32",
        "targets_file": "minibatch_targets.i64", "final_flat_file": "fedavg_minibatch_final.f32",
        "final_flat_sha256": hashlib.sha256(seeded.tobytes()).hexdigest(),
        # How far each wrong batch order lands from the golden; the native test's tolerance must sit far below all.
        "control_separation": separations,
    }
    with open(os.path.join(HERE, "fedavg_minibatch_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("minibatch endpoint: control separations", separations)


def build_initial_net() -> "TinyNet":
    """manual_seed(0) TinyNet — fc1 == committed zo_flat.f32, fc2 frozen + deterministic."""
    torch.manual_seed(0)
    return TinyNet().eval()


def load_committed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """The SAME batch the ZO goldens + the C++ tests read (zo_inputs / zo_targets)."""
    inputs = torch.from_numpy(
        np.fromfile(os.path.join(HERE, "zo_inputs.f32"), dtype="<f4").reshape(8, 4).copy()
    )
    targets = torch.from_numpy(
        np.fromfile(os.path.join(HERE, "zo_targets.i64"), dtype="<i8").reshape(8).copy()
    )
    return inputs, targets


def compute_fedavg_endpoint(*, lr: float = LR, local_epochs: int = LOCAL_EPOCHS,
                            proximal_mu: float = 0.0, examples: int = 8) -> np.ndarray:
    """Run the REAL framework FedAvg client update and return the final trainable flat (<f4).

    Uses LocalTrainer.fit — the actual FedAvg client code path, not a reimplementation. mu=0 is FedAvg; a
    positive proximal_mu is the FedProx client. ``examples`` trains the first that many of the committed batch,
    as one full batch.
    """
    net = build_initial_net()
    inputs, targets = load_committed_batch()
    inputs, targets = inputs[:examples], targets[:examples]
    trainer = LocalTrainer(net, _OneBatchLoader(inputs, targets), device="cpu")
    # config values flow through a protobuf map<string,string> in production — pass them as strings
    # so this exercises the same str->float coercion the wire path does.
    trainer.fit(None, {"learning_rate": str(lr), "local_epochs": str(local_epochs),
                       "proximal_mu": str(proximal_mu) if proximal_mu else "0"})
    return flat_params(net).detach().cpu().numpy().astype("<f4")


# Endpoint tolerance for the contract-numbers replay. The contract's single step at lr=1e-3 moves the
# parameters by only ~1.4e-4, so the 2e-3 family above would pass on a trainer that did nothing at all;
# this is set from the measured ET-vs-torch deviation instead, and the C++ test also asserts the step
# actually moved the parameters.
CONTRACT_ENDPOINT_ATOL = 1e-6


def contract_training() -> tuple[float, int]:
    """The lr and step count the execution contract v1 golden states, read from its projection fixture.

    Single source: ../execution_contract_v1/generate.py emits that projection from the contract itself, so
    this golden cannot be generated for training the contract does not state. Run that generator first.
    """
    path = os.path.join(HERE, "..", "execution_contract_v1", "projection_tinynet_fedavg.json")
    with open(path) as fh:
        projection = json.load(fh)
    return float(projection["learningRate"]), int(projection["numLocalSteps"])


def write_contract_endpoint(layout) -> None:
    """The same real framework update, run at the numbers the execution contract states.

    The C++ execution-contract test replays the contract's own training and must land here, which pins the
    chain contract -> projection -> trained endpoint across the reader, the projection and the native trainer.
    """
    lr, local_epochs = contract_training()
    final_flat = compute_fedavg_endpoint(lr=lr, local_epochs=local_epochs)
    final_flat.tofile(os.path.join(HERE, "contract_local_final.f32"))
    initial = np.fromfile(os.path.join(HERE, "zo_flat.f32"), dtype="<f4")
    moved = float(np.abs(final_flat - initial).max())
    if moved <= CONTRACT_ENDPOINT_ATOL:
        raise SystemExit(
            f"the contract's training moves the parameters by {moved:g}, at or below the endpoint "
            f"tolerance {CONTRACT_ENDPOINT_ATOL:g} — this golden could not tell training from a no-op")
    manifest = {
        "description": "FedAvg local-update golden at the numbers execution contract v1 states for this run "
                       "(see ../execution_contract_v1/projection_tinynet_fedavg.json). Same real "
                       "LocalTrainer.fit(mu=0) as fedavg_local_manifest.json, different training.",
        "torch_version": torch.__version__.split("+")[0],
        "platform_machine": platform.machine(),
        "learning_rate": lr,
        "local_epochs": local_epochs,
        "flat_dim": int(final_flat.shape[0]),
        "initial_flat_file": "zo_flat.f32",
        "inputs_file": "zo_inputs.f32",
        "targets_file": "zo_targets.i64",
        "param_layout": [[name, list(shape), k] for name, shape, k in layout],
        "final_flat_file": "contract_local_final.f32",
        "final_flat_sha256": hashlib.sha256(final_flat.tobytes()).hexdigest(),
        "endpoint_atol": CONTRACT_ENDPOINT_ATOL,
        # How far this training moves the parameters. The tolerance must stay well under it, or the test
        # cannot distinguish the contract's training from no training.
        "max_param_movement": moved,
    }
    with open(os.path.join(HERE, "contract_local_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"contract endpoint: lr={lr} local_epochs={local_epochs} moved={moved:g}")


# FedProx: the coefficient the FedProx FL server sends (fl-runtime/strategy_client_settings.py; a fl-runtime test
# pins the two equal). Run at the FedAvg golden's lr 0.1 x 5 steps: at one step w == w_global and the proximal
# gradient is zero, so the contract's own single-step training cannot exercise it.
FEDPROX_MU = 0.1
# The proximal term moves this endpoint by only ~1.4e-3 from FedAvg's, inside the 2e-3 FedAvg tolerance. So this
# golden has its own, and the generator refuses to write one whose separation is not well outside it.
FEDPROX_ENDPOINT_ATOL = 1e-4
FEDPROX_MIN_SEPARATION = 10 * FEDPROX_ENDPOINT_ATOL


def write_fedprox_endpoint(layout, fedavg_final: np.ndarray) -> None:
    """The real FedProx client update (LocalTrainer.fit with mu > 0) over the FedAvg golden's training."""
    final_flat = compute_fedavg_endpoint(proximal_mu=FEDPROX_MU)
    separation = float(np.abs(final_flat - fedavg_final).max())
    if separation < FEDPROX_MIN_SEPARATION:
        raise SystemExit(
            f"the proximal term moves the endpoint by {separation:g} from FedAvg's, under "
            f"{FEDPROX_MIN_SEPARATION:g} — this golden could not tell FedProx from FedAvg")
    final_flat.tofile(os.path.join(HERE, "fedprox_local_final.f32"))
    manifest = {
        "description": "FedProx local-update golden. LocalTrainer.fit with proximal_mu > 0: the FedAvg golden's "
                       "local_epochs full-batch SGD steps at lr, each adding mu * (w - w_global) to the gradient.",
        "torch_version": torch.__version__.split("+")[0],
        "platform_machine": platform.machine(),
        "learning_rate": LR,
        "local_epochs": LOCAL_EPOCHS,
        "proximal_mu": FEDPROX_MU,
        "flat_dim": int(final_flat.shape[0]),
        "initial_flat_file": "zo_flat.f32",
        "inputs_file": "zo_inputs.f32",
        "targets_file": "zo_targets.i64",
        "param_layout": [[name, list(shape), k] for name, shape, k in layout],
        "final_flat_file": "fedprox_local_final.f32",
        "final_flat_sha256": hashlib.sha256(final_flat.tobytes()).hexdigest(),
        "endpoint_atol": FEDPROX_ENDPOINT_ATOL,
        # How far the proximal term moves the endpoint from fedavg_local_final.f32. The tolerance must stay well
        # under it, or the test cannot distinguish FedProx from FedAvg.
        "separation_from_fedavg": separation,
    }
    with open(os.path.join(HERE, "fedprox_local_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"fedprox endpoint: mu={FEDPROX_MU} separation_from_fedavg={separation:g}")


def main() -> None:
    from fedlearn.communication.safetensors_codec import save_safetensors

    # integrity: the FedAvg golden must start from the byte-identical committed init the ZO goldens use.
    init_net = build_initial_net()
    init_flat = flat_params(init_net).detach().numpy().astype("<f4")
    zo_flat = np.fromfile(os.path.join(HERE, "zo_flat.f32"), dtype="<f4")
    if not np.array_equal(init_flat, zo_flat):
        raise SystemExit(
            "initial trainable flat != committed zo_flat.f32 — torch version drift? "
            f"(this torch={torch.__version__}); regenerate under the pinned torch 2.12.0."
        )

    final_flat = compute_fedavg_endpoint()
    d = int(final_flat.shape[0])
    final_flat.tofile(os.path.join(HERE, "fedavg_local_final.f32"))
    final_sha = hashlib.sha256(final_flat.tobytes()).hexdigest()

    # A device dataset smaller than the example batch: the first 6 examples as one full batch. A program exported
    # with a static batch of 8 cannot train it; one with a dynamic batch must land here. The golden must sit well
    # outside the tolerance of the 8-example endpoint, or a trainer that ignored the batch size would pass.
    final_6 = compute_fedavg_endpoint(examples=SMALL_BATCH)
    separation = float(np.abs(final_6 - final_flat).max())
    if separation < 10 * 2e-3:
        raise SystemExit(f"the {SMALL_BATCH}-example endpoint is only {separation} from the 8-example one")
    final_6.tofile(os.path.join(HERE, "fedavg_local_final_6.f32"))

    # safetensors state-dict of the final trainable flat (byte-exact codec contract, ZO-golden layout).
    layout = param_layout(build_initial_net())  # [(name, shape, numel)] canonical named_parameters order
    named_tensors, off = [], 0
    for name, shape, k in layout:
        named_tensors.append((name, final_flat[off:off + k].reshape(list(shape))))
        off += k
    state_blob = save_safetensors(named_tensors, {"num_examples": "8", "local_epochs": str(LOCAL_EPOCHS)})
    with open(os.path.join(HERE, "fedavg_local_state.safetensors"), "wb") as fh:
        fh.write(state_blob)
    state_sha = hashlib.sha256(state_blob).hexdigest()

    manifest = {
        "description": "FedAvg (first-order) local-update golden. LocalTrainer.fit(mu=0): "
                       "local_epochs full-batch SGD steps, lr, CrossEntropy, on the committed TinyNet.",
        "torch_version": torch.__version__.split("+")[0],
        "platform_machine": platform.machine(),
        "learning_rate": LR,
        "local_epochs": LOCAL_EPOCHS,
        "flat_dim": d,
        "initial_flat_file": "zo_flat.f32",
        "inputs_file": "zo_inputs.f32",
        "targets_file": "zo_targets.i64",
        # canonical flat order (named_parameters(), trainable-only) — the C++ side MUST re-map ET's
        # alphabetical named_parameters() std::map into THIS order or the flat vector transposes.
        "param_layout": [[name, list(shape), k] for name, shape, k in layout],
        "final_flat_file": "fedavg_local_final.f32",
        "final_flat_sha256": final_sha,
        "state_file": "fedavg_local_state.safetensors",
        "state_sha256": state_sha,
        # endpoint tolerance for the cross-runtime (ET backward vs torch autograd) C++ replay; same
        # 2e-3 family as the DeComFL endpoint golden. Never assert bit-exact cross-arch/cross-runtime.
        "endpoint_atol": 2e-3,
        # The same update on the first SMALL_BATCH examples (a dynamic-batch program's endpoint).
        "small_batch_examples": SMALL_BATCH,
        "small_batch_final_flat_file": "fedavg_local_final_6.f32",
        "small_batch_final_flat_sha256": hashlib.sha256(final_6.tobytes()).hexdigest(),
        "small_batch_separation_from_full_batch": separation,
    }
    with open(os.path.join(HERE, "fedavg_local_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    write_contract_endpoint(layout)
    write_fedprox_endpoint(layout, final_flat)
    write_minibatch_endpoint()

    print(f"lr={LR} local_epochs={LOCAL_EPOCHS} d={d} torch={torch.__version__}")
    print("final_flat[:5] =", final_flat[:5].tolist())
    print("final_flat_sha256 =", final_sha[:12], "| state_sha256 =", state_sha[:12])


if __name__ == "__main__":
    main()
