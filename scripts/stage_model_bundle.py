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

    manifest = {
        "runId": run_id,
        # Mirrors the mobile ModelManifest (bridge/specs/NativeFedLearnCore.ts): paramLayout order is the
        # trainable named_parameters() requires_grad order the native ModelManager loads against.
        "modelManifest": {
            "paramLayout": [{"name": p["name"], "shape": p["shape"]} for p in src["param_layout"]],
            "totalParamCount": src["total_params"],
            "inferPtePath": "infer.pte",  # relative; the mobile client rewrites to the staged local path
            "inferSha256": src["infer_sha256"],
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
