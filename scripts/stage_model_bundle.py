#!/usr/bin/env python3
"""Stage a per-run on-device training bundle (end-to-end training, phase P3 / MVP).

Copies the weight-free ExecuTorch graphs + the on-device data partition + a manifest into
{out}/{run_id}/, in the shape the Spring backend serves (GET /api/runs/{runId}/model-bundle, P2) and
the mobile client stages + loads (provisionTrainingBundle, P4). Every staged file's sha256 is verified
against the source manifest so a corrupt bundle is caught here, not on the device.

MVP source is the committed golden TinyNet fixture (framework/tests/fixtures/decomfl_golden/):
Linear(4,5)->ReLU->Linear(5,3) with fc2 frozen (25 trainable / 43 total params). The real path
(post-MVP) regenerates per-run bundles via scripts/export_model.py from the run's recipe; the output
manifest shape is identical, so P2/P4 are agnostic to which produced it.

Usage:
    python3 scripts/stage_model_bundle.py <run_id> [--out /var/models] [--fixture <dir>]
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO / "framework" / "tests" / "fixtures" / "decomfl_golden"


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file in the same dir + os.replace). manifest.json is
    the bundle's COMMIT MARKER — the backend gates a served bundle on it existing and parsing (RunService
    .getModelBundle). A truncate-in-place write leaves a torn manifest visible to a concurrent reader or
    after a crash mid-write (-> a 500 on read); an atomic rename makes it appear whole-or-not-at-all."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_bundle(run_id: str, out_root: Path, fixture: Path = DEFAULT_FIXTURE) -> Path:
    """Stage the bundle for ``run_id`` under ``out_root/run_id`` and return that directory."""
    src = json.loads((fixture / "zo_manifest.json").read_text())
    dest = out_root / run_id
    dest.mkdir(parents=True, exist_ok=True)

    # The programs a run stages take a dynamic example count (1..the fixture batch), so a device can train its own
    # dataset; the static goldens refuse any count but 8 at runtime. Staged when the fixture ships them.
    dyn_path = fixture / "fedavg_pte_manifest.json"
    dyn = json.loads(dyn_path.read_text()).get("dynbatch") if dyn_path.exists() else None
    # A recipe export stages its programs under their own names and states only the bound and probe here.
    if dyn and "loss_file" in dyn:
        src = {**src,
               "pte_file": dyn["loss_file"], "pte_sha256": dyn["loss_sha256"],
               "infer_file": dyn["infer_file"], "infer_sha256": dyn["infer_sha256"],
               "trainable_file": dyn["trainable_file"], "trainable_sha256": dyn["trainable_sha256"]}

    # (source filename, canonical staged name, expected sha256 from the source manifest or None)
    copies = [
        (src["pte_file"], "loss.pte", src["pte_sha256"]),        # forward(flat,x,y) -> loss
        (src["infer_file"], "infer.pte", src["infer_sha256"]),   # forward(flat,x)  -> logits
        (src["inputs_file"], "inputs.f32", None),                # on-device features (row-major f32)
        (src["targets_file"], "targets.i64", None),              # on-device labels  (int64)
    ]
    for src_name, dest_name, expected in copies:
        shutil.copyfile(fixture / src_name, dest / dest_name)
        got = sha256(dest / dest_name)
        if expected is not None and got != expected:
            raise SystemExit(f"sha256 mismatch for {dest_name}: expected {expected}, staged {got}")

    # First-order (FedAvg) trainable graph — OPTIONAL. Present only when the exporter captured a backward
    # graph for this recipe (export_model.py) or the golden fixture ships one. When absent, the bundle is
    # DeComFL-only (the phone's fail-closed gate refuses FedAvg), exactly as before this change.
    trainable_src = src.get("trainable_file")
    trainable_meta = None
    if trainable_src:
        shutil.copyfile(fixture / trainable_src, dest / "trainable.pte")
        got = sha256(dest / "trainable.pte")
        expected = src.get("trainable_sha256")
        if expected is not None and got != expected:
            raise SystemExit(f"sha256 mismatch for trainable.pte: expected {expected}, staged {got}")
        trainable_meta = {
            "trainablePtePath": "trainable.pte",  # relative; the mobile client rewrites to the staged path
            "trainableSha256": got,
            "trainableParamNames": src.get("trainable_param_names", []),
        }

    # What the execution contract needs from the bundle: each model program's digest and size, the runtime
    # operators they require, and the declared resource envelope. Recorded only when the source carries the
    # committed operator metadata; without it the backend cannot publish a contract for the run.
    model_files = [name for name in ("loss.pte", "infer.pte", "trainable.pte") if (dest / name).exists()]
    contract_fields = {}
    metadata_path = fixture / "artifact_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        envelope = metadata["resourceEnvelope"]
        contract_fields = {
            "modelFiles": [{"file": name, "sha256": sha256(dest / name), "byteSize": (dest / name).stat().st_size}
                           for name in model_files],
            "requiredOperators": sorted(metadata["requiredOperators"]),
            "resourceEnvelope": {
                "peakMemoryBytes": envelope["peakMemoryBytes"],
                "storageBytes": sum((dest / name).stat().st_size for name in model_files),
                "probeMs": envelope["probeMs"],
                "trainMs": envelope["trainMs"],
                "basis": envelope["basis"],
            },
        }
        # The most examples one call of a program takes (they take 1..maxBatch). Stated only for the dynamic-batch
        # programs: a static program takes exactly its example count, so without them the backend, which refuses a
        # bundle that states no maxBatch, publishes no contract rather than one a device's data would fail.
        if dyn:
            contract_fields["maxBatch"] = dyn["max_batch"]
        # The qualification probe the device runs against the trainable program before its first round (Stage 3
        # D2): the exporter's reference for this exact program, with the declared probe time budget.
        probe = (dyn or {}).get("probe")
        if probe and trainable_meta:
            trainable_meta["trainableProbe"] = {
                "rows": probe["rows"], "width": probe["width"], "classes": probe["classes"],
                "learningRate": probe["learning_rate"], "lossStep1": probe["loss_step1"],
                "lossStep2": probe["loss_step2"], "lossTolerance": probe["loss_tolerance"],
                "maxProbeMs": envelope["probeMs"],
                # The shape of one example when it has more than one dimension; absent means [width].
                **({"inputShape": probe["input_shape"]} if "input_shape" in probe else {}),
            }

    manifest = {
        "runId": run_id,
        # Mirrors the mobile ModelManifest (bridge/specs/NativeFedLearnCore.ts): paramLayout order is the
        # trainable named_parameters() requires_grad order the native ModelManager loads against.
        "modelManifest": {
            "paramLayout": [{"name": p["name"], "shape": p["shape"]} for p in src["param_layout"]],
            "totalParamCount": src["total_params"],
            "inferPtePath": "infer.pte",  # relative; the mobile client rewrites to the staged local path
            "inferSha256": src["infer_sha256"],
            # First-order trainable graph fields (trainablePtePath/trainableSha256/trainableParamNames)
            # spliced in ONLY when a trainable.pte was staged — mirrors the native ModelManifest, which
            # treats a missing trainablePtePath as "DeComFL-only".
            **(trainable_meta or {}),
        },
        "lossPte": {"file": "loss.pte", "sha256": src["pte_sha256"]},
        "dataset": {
            "inputsFile": "inputs.f32",
            "inputsSha256": sha256(dest / "inputs.f32"),
            "inputShape": src["inputs_shape"],
            "targetsFile": "targets.i64",
            "targetsSha256": sha256(dest / "targets.i64"),
            "targetsShape": src["targets_shape"],
        },
        "meta": {
            "recipe": "tinynet-golden",
            "torchVersion": src["torch_version"],
            "trainableParamCount": src["trainable_params"],
            "goldenLoss": src["golden_loss"],
            "goldenAccuracy": src["golden_accuracy"],
        },
        **contract_fields,
    }
    atomic_write_text(dest / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--out", default="/var/models", type=Path)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE, type=Path)
    args = ap.parse_args()
    dest = stage_bundle(args.run_id, args.out, args.fixture)
    print(f"staged model bundle -> {dest}")
    for p in sorted(dest.iterdir()):
        print(f"  {p.stat().st_size:>7} {p.name}")


if __name__ == "__main__":
    main()
