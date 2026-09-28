"""The staged on-device bundle carries what the execution contract needs from it.

scripts/stage_model_bundle.py stays stdlib-only (a fixture recipe needs no ExecuTorch toolchain on the
backend host), so the operators the programs require come from committed fixture metadata. These tests
read the staged programs with ExecuTorch and require the recorded operator set to be exactly theirs, and
require a complete declared resource envelope whose storage covers the model files.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os

from executorch.exir._serialize._program import deserialize_pte_binary

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_FILES = ("loss.pte", "infer.pte", "trainable.pte")


def _stage(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "stage_model_bundle", os.path.join(REPO, "scripts", "stage_model_bundle.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest = module.stage_bundle("4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17", tmp_path)
    with open(dest / "manifest.json") as fh:
        return dest, json.load(fh)


def _operators(path) -> set[str]:
    with open(path, "rb") as fh:
        program = deserialize_pte_binary(fh.read())
    program = getattr(program, "program", program)
    return {f"{op.name}.{op.overload}" if op.overload else op.name
            for plan in program.execution_plan for op in plan.operators}


def test_the_manifest_records_exactly_the_operators_the_staged_programs_use(tmp_path):
    dest, manifest = _stage(tmp_path)
    used = set().union(*(_operators(dest / name) for name in MODEL_FILES))
    assert manifest["requiredOperators"] == sorted(used)


def test_the_manifest_declares_a_complete_resource_envelope(tmp_path):
    dest, manifest = _stage(tmp_path)
    envelope = manifest["resourceEnvelope"]
    assert set(envelope) == {"peakMemoryBytes", "storageBytes", "probeMs", "trainMs", "basis"}
    for key in ("peakMemoryBytes", "storageBytes", "probeMs", "trainMs"):
        assert isinstance(envelope[key], int) and envelope[key] > 0
    assert envelope["storageBytes"] >= sum((dest / name).stat().st_size for name in MODEL_FILES)
    assert envelope["basis"]


def test_the_manifest_records_every_model_file_digest(tmp_path):
    dest, manifest = _stage(tmp_path)
    for name in MODEL_FILES:
        digest = hashlib.sha256((dest / name).read_bytes()).hexdigest()
        assert {"file": name, "sha256": digest, "byteSize": (dest / name).stat().st_size} \
            in manifest["modelFiles"]
    assert [f["file"] for f in manifest["modelFiles"]] == list(MODEL_FILES)


# Stage 3: a device trains its own dataset, so the staged programs must take any example count up to the run's batch,
# not only the 8 examples of the fixture batch (a static program refused 6 on a phone: ExecuTorch NotSupported).

def test_the_staged_programs_are_the_dynamic_batch_ones_and_the_manifest_names_their_digests(tmp_path):
    dest, manifest = _stage(tmp_path)
    with open(os.path.join(REPO, "framework", "tests", "fixtures", "decomfl_golden", "fedavg_pte_manifest.json")) as fh:
        dyn = json.load(fh)["dynbatch"]
    for staged, key in (("loss.pte", "loss"), ("infer.pte", "infer"), ("trainable.pte", "trainable")):
        assert hashlib.sha256((dest / staged).read_bytes()).hexdigest() == dyn[f"{key}_sha256"]
    assert manifest["lossPte"]["sha256"] == dyn["loss_sha256"]
    assert manifest["modelManifest"]["inferSha256"] == dyn["infer_sha256"]
    assert manifest["modelManifest"]["trainableSha256"] == dyn["trainable_sha256"]


def test_the_staged_loss_and_infer_programs_take_fewer_examples_than_the_fixture_batch(tmp_path):
    import numpy as np
    import torch
    from executorch.runtime import Runtime

    dest, _ = _stage(tmp_path)
    golden = os.path.join(REPO, "framework", "tests", "fixtures", "decomfl_golden")
    flat = torch.from_numpy(np.fromfile(os.path.join(golden, "zo_flat.f32"), dtype="<f4").copy())
    x = torch.from_numpy(np.fromfile(os.path.join(golden, "zo_inputs.f32"), dtype="<f4").reshape(8, 4).copy())
    y = torch.from_numpy(np.fromfile(os.path.join(golden, "zo_targets.i64"), dtype="<i8").reshape(8).copy())
    loss = Runtime.get().load_program(str(dest / "loss.pte")).load_method("forward")
    infer = Runtime.get().load_program(str(dest / "infer.pte")).load_method("forward")
    for n in (1, 6, 8):
        assert torch.isfinite(loss.execute([flat, x[:n], y[:n]])[0]).all()
        assert tuple(infer.execute([flat, x[:n]])[0].shape) == (n, 3)
