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


def compute_fedavg_endpoint(*, lr: float = LR, local_epochs: int = LOCAL_EPOCHS) -> np.ndarray:
    """Run the REAL framework FedAvg client update and return the final trainable flat (<f4).

    Uses LocalTrainer.fit(mu=0) — the actual FedAvg client code path, not a reimplementation.
    """
    net = build_initial_net()
    inputs, targets = load_committed_batch()
    trainer = LocalTrainer(net, _OneBatchLoader(inputs, targets), device="cpu")
    # config values flow through a protobuf map<string,string> in production — pass them as strings
    # so this exercises the same str->float coercion the wire path does.
    trainer.fit(None, {"learning_rate": str(lr), "local_epochs": str(local_epochs), "proximal_mu": "0"})
    return flat_params(net).detach().cpu().numpy().astype("<f4")


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
    }
    with open(os.path.join(HERE, "fedavg_local_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"lr={LR} local_epochs={LOCAL_EPOCHS} d={d} torch={torch.__version__}")
    print("final_flat[:5] =", final_flat[:5].tolist())
    print("final_flat_sha256 =", final_sha[:12], "| state_sha256 =", state_sha[:12])


if __name__ == "__main__":
    main()
