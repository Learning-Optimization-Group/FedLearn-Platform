#!/usr/bin/env python3
"""Export a run's RECIPE model to a per-run on-device training bundle (MO-15 — the real path).

`stage_model_bundle.py` only ever stages the fixed golden TinyNet fixture; this is the path its
docstring promised: instantiate the run's recipe model, export the two weight-free ExecuTorch graphs
(loss `forward(flat, x, y) -> cross_entropy`, infer `forward(flat, x) -> logits`) via the reusable
primitives in `mobile_client/scripts/pte_export.py`, then hand the result to `stage_model_bundle`'s
staging so the output `{out}/{run_id}/manifest.json` is byte-for-byte the same nested shape the Spring
backend serves (`GET /api/runs/{runId}/model-bundle`) and the mobile client provisions.

The exported flat-trainable layout is `named_parameters()` requires_grad order — identical to the
server's `ZerothOrderEstimator._get_flat_params` order — so a phone's DeComFL gradient scalars aggregate
against the server model. Frozen params (e.g. TinyNet's `fc2`) are baked into the graphs as constants.

The bundle carries the GRAPHS + param layout + an on-device data partition; the initial/global weights
arrive from the FL server over gRPC during training, so no weight tensor is staged.

Requires the ExecuTorch host toolchain (validated pin: torch 2.12.0 + executorch 1.3.1) for `.pte`
lowering. Recipe-model construction reuses the backend recipe catalog (`recipes.py`) as the single
source of truth, so a new recipe needs no change here.

Usage:
    python3 scripts/export_model.py <run_id> --recipe TINYNET_GOLDEN --out /var/models \
        [--samples 8] [--seed 0] [--init-state <safetensors|npz>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
FL_RUNTIME = REPO / "fl-runtime"
MOBILE_SCRIPTS = REPO / "mobile_client" / "scripts"
# recipes.py + execution_plan.py (the recipe catalog and its contract plans), pte_export.py (the ExecuTorch export
# primitives), stage_model_bundle.py (the staging that owns the served bundle shape) — reused, not duplicated.
for p in (FL_RUNTIME, MOBILE_SCRIPTS, REPO / "scripts"):
    sys.path.insert(0, str(p))

import recipes  # noqa: E402
import pte_export  # noqa: E402
import stage_model_bundle  # noqa: E402

# Per-recipe example-input shape (feature dims, no batch). recipes.py has no input-shape field
# (input_kind is only a UI hint), and the real data loaders need datasets present on the host — so the
# graph export uses a correctly-SHAPED example (the exported graph depends only on shape, never values).
# One entry per functional/mobile recipe; extend when a new mobile recipe lands.
EXAMPLE_SHAPES = {
    "TINYNET_GOLDEN": (4,),
    "MLP": (140,),
    "CNN": (3, 32, 32),
    "BLOOD_CNN": (3, 28, 28),
    "PNEUMONIA_CNN": (1, 224, 224),
}


# Declared resource budgets per recipe: what the contract's envelope states and what qualification holds a device to
# (probeMs). They are declarations, not measurements. A recipe with none declared gets no contract fields, so the
# backend publishes no contract for it rather than one with invented budgets.
ENVELOPES = {
    "CNN": {
        "peakMemoryBytes": 128 * 1024 * 1024,
        "probeMs": 5000,
        "trainMs": 120000,
        "basis": "Declared budgets, not measurements. The CNN has 62,006 parameters (248 KB as float32) and trains "
                 "batches of at most 32 images of 3 x 32 x 32, so these bound it generously; portable-CPU "
                 "qualification measures a device against probeMs. Storage is the staged model files' total size.",
    },
    "MLP": {
        "peakMemoryBytes": 64 * 1024 * 1024,
        "probeMs": 2000,
        "trainMs": 60000,
        "basis": "Declared budgets, not measurements. The MLP has 13,314 parameters (53 KB as float32) and trains "
                 "batches of at most 128 x 140 features, so these bound it generously; portable-CPU qualification "
                 "measures a device against probeMs. Storage is the staged model files' total size.",
    },
}

# Most examples a probe uses: the probe is two steps on a small synthetic batch, whatever the run's batch size.
PROBE_ROWS = 8


def _plan_batch_size(key: str) -> int | None:
    """The batch size the recipe's first-order contract plan trains with, or None when it has no plan."""
    import execution_plan
    try:
        return int(execution_plan.resolve_model_training(key, "FedAvg", "FULL").local_training.batch_size)
    except execution_plan.NotRepresentable:
        return None


def _operators(program: bytes) -> set[str]:
    """The runtime operators a program calls, read from the program itself."""
    from executorch.exir._serialize._program import deserialize_pte_binary
    parsed = deserialize_pte_binary(program)
    parsed = getattr(parsed, "program", parsed)
    return {f"{op.name}.{op.overload}" if op.overload else op.name
            for plan in parsed.execution_plan for op in plan.operators}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_recipe_model(recipe_key: str) -> tuple[torch.nn.Module, int]:
    """Return (eval model on CPU, num_classes) for a recipe key. Uses the backend recipe catalog as
    the single source of truth. Handles the off-catalog BLOOD_CNN and the non-functional CNN by name."""
    key = recipe_key.upper()
    if key == "CNN":
        # CNN's model lives in init_model.py (CnnNet), not recipes.py (it isn't is_functional).
        import init_model
        return init_model.CnnNet().eval(), 10
    if key == "BLOOD_CNN":
        # functional but deliberately off-catalog (recipes.py:364) -> get_recipe() raises; build directly.
        return recipes.build_blood_cnn().eval(), 8
    recipe = recipes.get_recipe(key)
    if not recipe.is_functional:
        raise SystemExit(f"recipe {key} is not functional (no build_model); cannot export")
    return recipe.build_model(device="cpu").eval(), len(recipe.classes)


def _load_init_state(model: torch.nn.Module, path: Path) -> None:
    """Load initial weights into the model (so frozen params baked into the graph match a reference).
    Accepts a .safetensors or an .npz whose keys use the '__DOT__' encoding (init_model.py convention)."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        raw = np.load(path)
        sd = {k.replace("__DOT__", "."): torch.from_numpy(raw[k]) for k in raw.files}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [init-state] not set (kept init): {list(missing)}")
    if unexpected:
        print(f"  [init-state] ignored unexpected: {list(unexpected)}")


def export_recipe_bundle(
    run_id: str,
    recipe_key: str,
    out_root: Path,
    num_samples: int = 8,
    seed: int = 0,
    init_state: Path | None = None,
) -> Path:
    key = recipe_key.upper()
    if key not in EXAMPLE_SHAPES:
        raise SystemExit(f"no example-input shape registered for recipe {key}; add it to EXAMPLE_SHAPES")

    # Seeded, so the weights a program embeds (and its probe reference is computed from) are the same on every export.
    torch.manual_seed(seed)
    model, num_classes = build_recipe_model(key)
    if init_state is not None:
        _load_init_state(model, init_state)

    # Deterministic example batch (the graph depends only on shapes; also doubles as the demo on-device
    # data partition so the phone has something to train on until real local data is wired).
    g = torch.Generator().manual_seed(seed)
    feat = EXAMPLE_SHAPES[key]
    x = torch.randn(num_samples, *feat, generator=g, dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_samples,), generator=g, dtype=torch.int64)

    # Every program takes 1..max_batch examples per call, so a device can train its own dataset (and, with
    # minibatching, a final partial batch) rather than only a batch of exactly the example's size, which a static
    # export refuses at runtime (ExecuTorch NotSupported). The bound is the batch size the recipe's contract plan
    # trains with (the backend refuses a contract whose batch exceeds it), else the example count.
    max_batch = _plan_batch_size(key) or num_samples
    if num_samples > max_batch:
        raise SystemExit(f"{num_samples} example rows exceed the programs' batch bound {max_batch}")

    # Weight-free graphs. pte_export bakes frozen params as constants; trainable params are the flat input.
    loss_pte = pte_export.export_functional_pte(model, (x, y), max_batch=max_batch)
    infer_pte = pte_export.export_functional_infer_pte(model, x, max_batch=max_batch)

    # First-order (FedAvg) trainable graph: forward(x, y) -> (cross_entropy, prediction) with a CAPTURED
    # backward pass, loadable by ET's TrainingModule on the phone (execute_forward_backward + SGD). This is
    # what unblocks real on-device backprop; the native TrainableExecutorchModel re-maps ET's alphabetical
    # named_parameters onto training_trainable_names() order. Feasible recipes only (small enough to ship +
    # fit device memory); a heavy recipe still ships loss/infer for DeComFL. Best-effort: if the ET training
    # export cannot capture this model's backward, we degrade to a DeComFL-only bundle rather than fail.
    # A model with dropout gets the masked program, forward(x, y, masks): its masks come from the contract's seeded
    # stream, so a device's step is reproducible, which ExecuTorch's own dropout RNG would not be.
    trainable_pte: bytes | None = None
    trainable_param_names: list[str] = []
    try:
        if pte_export.dropout_layers(model):
            trainable_pte = pte_export.export_masked_trainable_pte(
                model, (x, y), mask_shapes=pte_export.dropout_mask_shapes(model, x), max_batch=max_batch)
        else:
            trainable_pte = pte_export.export_trainable_pte(model, (x, y), max_batch=max_batch)
        trainable_param_names = pte_export.training_trainable_names(model)
    except Exception as e:  # noqa: BLE001 — export is experimental; never let it abort the (DeComFL) bundle
        print(f"  [trainable-pte] export failed for {key} ({type(e).__name__}: {e}); "
              f"bundle will be DeComFL-only (no on-device first-order)")

    names = pte_export.trainable_names(model)
    param_layout = [
        {"name": n, "shape": list(model.get_parameter(n).shape), "numel": model.get_parameter(n).numel()}
        for n in names
    ]
    trainable = int(pte_export.trainable_flat(model).numel())
    total = int(sum(p.numel() for p in model.parameters()))

    # Sanity: the flat layout the phone will perturb must equal the server's aggregation order.
    assert trainable == sum(pl["numel"] for pl in param_layout), "trainable flat != sum(param_layout)"

    # Reference metrics on the example batch (reporting only).
    with torch.no_grad():
        flat = pte_export.trainable_flat(model)
        logits = pte_export._FunctionalInfer(model).eval()(flat, x)
        loss = torch.nn.functional.cross_entropy(logits, y).item()
        acc = (logits.argmax(1) == y).float().mean().item()

    # Write a fixture-shaped intermediate, then reuse stage_model_bundle.stage_bundle so the SERVED
    # bundle shape is produced by exactly one code path (no divergence from the golden path).
    with tempfile.TemporaryDirectory() as tmp:
        fx = Path(tmp)
        (fx / "loss.pte").write_bytes(loss_pte)
        (fx / "infer.pte").write_bytes(infer_pte)
        x.numpy().astype("<f4").tofile(fx / "inputs.f32")
        y.numpy().astype("<i8").tofile(fx / "targets.i64")
        manifest = {
            "torch_version": torch.__version__.split("+")[0],
            "architecture": f"recipe:{key}",
            "total_params": total,
            "trainable_params": trainable,
            "pte_file": "loss.pte",
            "pte_sha256": _sha256(fx / "loss.pte"),
            "infer_file": "infer.pte",
            "infer_sha256": _sha256(fx / "infer.pte"),
            "inputs_file": "inputs.f32",
            "inputs_shape": [num_samples, *feat],
            "targets_file": "targets.i64",
            "targets_shape": [num_samples],
            "param_layout": param_layout,
            "golden_loss": loss,
            "golden_accuracy": acc,
        }
        # First-order trainable graph (optional): present iff export_trainable_pte succeeded above. Its
        # sha256 is verified during staging; trainable_param_names carry the canonical base.<name> order
        # the phone's TrainableExecutorchModel re-maps ET's alphabetical named_parameters onto.
        if trainable_pte is not None:
            (fx / "trainable.pte").write_bytes(trainable_pte)
            manifest["trainable_file"] = "trainable.pte"
            manifest["trainable_sha256"] = _sha256(fx / "trainable.pte")
            manifest["trainable_param_names"] = trainable_param_names
        (fx / "zo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        # What the execution contract needs from the bundle, for a recipe with declared budgets: the operators the
        # programs call, the envelope, the batch bound the programs were exported with, and the qualification probe.
        envelope = ENVELOPES.get(key)
        if envelope is not None:
            programs = [loss_pte, infer_pte] + ([trainable_pte] if trainable_pte is not None else [])
            (fx / "artifact_metadata.json").write_text(json.dumps({
                "requiredOperators": sorted(set().union(*(_operators(p) for p in programs))),
                "resourceEnvelope": envelope,
            }, indent=2) + "\n")
            dynbatch = {"max_batch": max_batch}
            # The probe's batch takes each example's shape (an image's [channels, height, width]).
            if trainable_pte is not None:
                dynbatch["probe"] = pte_export.probe_reference(
                    model, rows=min(PROBE_ROWS, max_batch), width=int(np.prod(feat)), classes=num_classes,
                    input_shape=feat)
            (fx / "fedavg_pte_manifest.json").write_text(json.dumps({"dynbatch": dynbatch}, indent=2) + "\n")
        dest = stage_model_bundle.stage_bundle(run_id, out_root, fixture=fx)

    # stamp the real recipe key into the staged manifest meta (stage_bundle hardcodes "tinynet-golden").
    # Re-write atomically so this second write of the commit-marker manifest can't leave a torn file.
    staged = json.loads((dest / "manifest.json").read_text())
    staged["meta"]["recipe"] = key.lower()
    stage_model_bundle.atomic_write_text(dest / "manifest.json", json.dumps(staged, indent=2) + "\n")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--recipe", required=True, help="recipe key (TINYNET_GOLDEN, PNEUMONIA_CNN, ...)")
    ap.add_argument("--out", default="/var/models", type=Path)
    ap.add_argument("--samples", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--init-state", default=None, type=Path,
                    help="optional initial weights (.safetensors/.npz); frozen params get baked into the graph")
    args = ap.parse_args()
    dest = export_recipe_bundle(args.run_id, args.recipe, args.out, args.samples, args.seed, args.init_state)
    print(f"exported recipe bundle -> {dest}")
    for p in sorted(dest.iterdir()):
        print(f"  {p.stat().st_size:>8} {p.name}")


if __name__ == "__main__":
    main()
