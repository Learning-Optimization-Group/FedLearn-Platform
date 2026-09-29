"""A recipe exported for devices carries what the execution contract needs from it (Stage 4 S1).

scripts/export_model.py builds a run's recipe model, exports its programs with ExecuTorch and stages them. Until S1 it
staged no contract fields, so no recipe other than the TinyNet fixture could ever get a contract. These tests export
the MLP and require: exactly the operators its programs use, every program's digest, a declared envelope, the most
examples a call takes (the contract's batch size), a trainable program that takes the dropout masks, and the
qualification probe recomputed from torch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

from executorch.exir._serialize._program import deserialize_pte_binary

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_ID = "0b6f3a52-2d1e-4c8a-9f47-3e5d7c1a9b20"
MODEL_FILES = ("loss.pte", "infer.pte", "trainable.pte")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter():
    return _module("export_model", os.path.join(REPO, "scripts", "export_model.py"))


@pytest.fixture(scope="module")
def mlp_bundle(exporter, tmp_path_factory):
    dest = exporter.export_recipe_bundle(RUN_ID, "MLP", tmp_path_factory.mktemp("bundles"))
    with open(dest / "manifest.json") as fh:
        return dest, json.load(fh)


def _operators(path) -> set[str]:
    with open(path, "rb") as fh:
        program = deserialize_pte_binary(fh.read())
    program = getattr(program, "program", program)
    return {f"{op.name}.{op.overload}" if op.overload else op.name
            for plan in program.execution_plan for op in plan.operators}


def _mlp_plan_batch_size():
    import execution_plan
    return execution_plan.resolve_model_training("MLP", "FedAvg", "FULL").local_training.batch_size


def test_the_exported_bundle_names_its_recipe(mlp_bundle):
    _, manifest = mlp_bundle
    assert manifest["meta"]["recipe"] == "mlp"


def test_the_exported_bundle_records_exactly_the_operators_its_programs_use(mlp_bundle):
    dest, manifest = mlp_bundle
    used = set().union(*(_operators(dest / name) for name in MODEL_FILES))
    assert manifest["requiredOperators"] == sorted(used)


def test_the_exported_bundle_records_every_program_digest(mlp_bundle):
    dest, manifest = mlp_bundle
    assert manifest["modelFiles"] == [
        {"file": name, "sha256": hashlib.sha256((dest / name).read_bytes()).hexdigest(),
         "byteSize": (dest / name).stat().st_size} for name in MODEL_FILES]


def test_the_exported_bundle_declares_a_complete_envelope(mlp_bundle):
    dest, manifest = mlp_bundle
    envelope = manifest["resourceEnvelope"]
    assert set(envelope) == {"peakMemoryBytes", "storageBytes", "probeMs", "trainMs", "basis"}
    for key in ("peakMemoryBytes", "probeMs", "trainMs"):
        assert isinstance(envelope[key], int) and envelope[key] > 0
    assert envelope["storageBytes"] == sum((dest / name).stat().st_size for name in MODEL_FILES)
    assert envelope["basis"]


def test_the_programs_take_the_contracts_batch_and_the_manifest_states_it(mlp_bundle):
    """The backend refuses a contract whose batch exceeds maxBatch, so the export bound is the plan's batch size."""
    from executorch.runtime import Runtime

    dest, manifest = mlp_bundle
    batch = _mlp_plan_batch_size()
    assert manifest["maxBatch"] == batch
    flat = torch.zeros(sum(int(np.prod(p["shape"])) for p in manifest["modelManifest"]["paramLayout"]))
    x = torch.randn(batch, 140)
    y = torch.zeros(batch, dtype=torch.int64)
    loss = Runtime.get().load_program(str(dest / "loss.pte")).load_method("forward")
    assert torch.isfinite(loss.execute([flat, x, y])[0]).all()
    assert torch.isfinite(loss.execute([flat, x[:5], y[:5]])[0]).all()


def test_the_trainable_program_takes_a_mask_per_dropout_layer(mlp_bundle):
    from executorch.runtime import Runtime

    dest, _ = mlp_bundle
    method = Runtime.get().load_program(str(dest / "trainable.pte")).load_method("forward")
    # (x, y, one mask per Dropout) for the MLP's two Dropout(0.3) layers.
    assert method.metadata.num_inputs() == 4


def test_the_trainable_parameter_names_are_the_canonical_order(mlp_bundle):
    _, manifest = mlp_bundle
    layout = [p["name"] for p in manifest["modelManifest"]["paramLayout"]]
    assert manifest["modelManifest"]["trainableParamNames"] == [f"base.{n}" for n in layout]


def test_the_exported_bundle_carries_the_probe_torch_computes(mlp_bundle):
    sys.path.insert(0, os.path.join(REPO, "mobile_client", "scripts"))
    import pte_export
    import recipes

    _, manifest = mlp_bundle
    torch.manual_seed(0)
    reference = pte_export.probe_reference(recipes.get_recipe("MLP").build_model("cpu"), rows=8, width=140,
                                           classes=2)
    probe = manifest["modelManifest"]["trainableProbe"]
    assert probe == {
        "rows": 8, "width": 140, "classes": 2, "learningRate": reference["learning_rate"],
        "lossStep1": reference["loss_step1"], "lossStep2": reference["loss_step2"],
        "lossTolerance": reference["loss_tolerance"], "maxProbeMs": manifest["resourceEnvelope"]["probeMs"],
    }


def test_the_probe_moves_every_trainable_parameter_at_both_steps():
    """A probe step that leaves a layer's gradient at zero cannot tell a working program from one that never updates
    that layer, and a device's STALE_WEIGHTS check would fail a correct program. The MLP starts with zero biases, so an
    all-zeros probe input gave no gradient at all."""
    sys.path.insert(0, os.path.join(REPO, "mobile_client", "scripts"))
    import pte_export
    import recipes

    torch.manual_seed(0)
    model = recipes.get_recipe("MLP").build_model("cpu")
    wrapper = pte_export._MaskedTrainingGraph(model)
    for step in (1, 2):
        x, y = pte_export.probe_batch(step, 8, 140, 2)
        assert len({tuple(row.tolist()) for row in x}) == 8, "probe rows must differ"
        wrapper.zero_grad()
        loss, _ = wrapper(x, y, (torch.ones(8, 64), torch.ones(8, 64)))
        loss.backward()
        for name, p in wrapper.named_parameters():
            assert p.grad is not None and p.grad.abs().max() > 0, f"step {step}: no gradient on {name}"


def test_a_recipe_without_a_declared_envelope_gets_no_contract_fields(exporter, tmp_path):
    """Budgets are declared per recipe; one nobody declared gets no contract rather than an invented one."""
    assert "TINYNET_GOLDEN" not in exporter.ENVELOPES
    dest = exporter.export_recipe_bundle(RUN_ID, "TINYNET_GOLDEN", tmp_path)
    manifest = json.loads((dest / "manifest.json").read_text())
    for field in ("modelFiles", "requiredOperators", "resourceEnvelope", "maxBatch"):
        assert field not in manifest
