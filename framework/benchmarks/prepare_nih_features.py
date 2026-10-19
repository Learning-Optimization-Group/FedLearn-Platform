"""Prepare frozen ResNet-18 features from an independently verified NIH cohort."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from numbers import Integral
import os
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from benchmarks.dp_on_head_xray import _build_backbone, _IMAGENET_MEAN, _IMAGENET_STD
from benchmarks.nih_cohort import verify_cohort
from fedlearn.simulation.nih_cache import (CACHE_VERSION, PAYLOADS, SPLITS, TRANSFORM,
                                           json_bytes, load_nih_cache, read_manifest, sha256_file)
from fedlearn.simulation.rng import torch_rng_scope


def _integer(value, name, upper):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer in 1..{upper}")
    return int(value)


def _model_state_sha256(model):
    digest = hashlib.sha256(b"fedlearn-frozen-state-v1\0")
    for name, tensor in sorted(model.state_dict().items()):
        metadata = json_bytes({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _read_tensor(root, row, identity, transform):
    path = root / row["image_path"]
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("image identity escaped source root")
    size = int(identity["file_bytes"])
    with path.open("rb") as file:
        raw = file.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != identity["file_sha256"]:
        raise ValueError(f"image raw identity hash changed: {row['image_index']}")
    with Image.open(io.BytesIO(raw)) as image:
        if (image.format != "PNG" or getattr(image, "n_frames", 1) != 1
                or image.size != (int(identity["width"]), int(identity["height"]))):
            raise ValueError("image decoded identity changed")
        rgb = image.convert("RGB")
        digest = hashlib.sha256(b"capstone-nih-rgb-v1\0")
        digest.update(f"{rgb.width}x{rgb.height}\0".encode("ascii"))
        digest.update(rgb.tobytes())
        if digest.hexdigest() != identity["rgb_sha256"]:
            raise ValueError("image RGB identity hash changed")
        return transform(rgb)


def prepare(root: Path, original_preparation: Path, image_audit: Path,
            curated_cohort: Path, output: Path, *, device: str = "cpu", batch_size: int = 64,
            pretrained: bool = True, backbone_seed: int = 1234, audit_workers: int = 1,
            image_workers: int = 4) -> dict:
    """Create a new immutable cache; final report is the completion marker."""
    batch_size = _integer(batch_size, "batch_size", 256)
    audit_workers = _integer(audit_workers, "audit_workers", 16)
    image_workers = _integer(image_workers, "image_workers", 16)
    if (isinstance(backbone_seed, bool) or not isinstance(backbone_seed, Integral)
            or not 0 <= backbone_seed < 2**63):
        raise ValueError("backbone_seed must be an integer in 0..2**63-1")
    backbone_seed = int(backbone_seed)
    if type(pretrained) is not bool:
        raise ValueError("pretrained must be boolean")
    if (device not in ("cpu", "mps", "cuda")
            or device == "cuda" and not torch.cuda.is_available()
            or device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("explicit device is unsupported or unavailable")
    root, original_preparation, image_audit, curated_cohort, output = map(
        Path, (root, original_preparation, image_audit, curated_cohort, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing feature output: {output}")
    parents = (root, original_preparation, image_audit, curated_cohort)
    if any(output.resolve().is_relative_to(parent.resolve()) for parent in parents):
        raise ValueError("feature output must be outside all source/evidence directories")
    cohort = verify_cohort(root, original_preparation, image_audit, curated_cohort,
                           audit_workers=audit_workers)
    cohort_raw = (curated_cohort / "report.json").read_bytes()
    if cohort_raw != json_bytes(cohort):
        raise ValueError("cohort report changed after verification")
    root = root.resolve(strict=True)
    # Retain the exact verified bytes; do not read mutable manifests again later.
    csv_payloads = {f"{s}.csv": (curated_cohort / f"{s}.csv").read_bytes() for s in SPLITS}
    if any(hashlib.sha256(raw).hexdigest() != cohort["payload_hashes"][name]
           for name, raw in csv_payloads.items()):
        raise ValueError("cohort manifest hash changed after verification")
    audit_report_raw = (image_audit / "report.json").read_bytes()
    if hashlib.sha256(audit_report_raw).hexdigest() != cohort["image_audit_sha256"]:
        raise ValueError("image audit report changed after verification")
    audit_report = json.loads(audit_report_raw)
    audit_raw = (image_audit / "images.csv").read_bytes()
    if hashlib.sha256(audit_raw).hexdigest() != audit_report["payload_hashes"]["images.csv"]:
        raise ValueError("image audit identities changed after verification")
    identities = {r["image_index"]: r for r in csv.DictReader(io.StringIO(audit_raw.decode()))}
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        with torch_rng_scope(backbone_seed):
            model, width = _build_backbone("resnet18", pretrained, backbone_seed)
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
    if width != 512:
        raise ValueError("unexpected ResNet-18 feature width")
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state_sha = _model_state_sha256(model)
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for name, raw in {**csv_payloads, "cohort-report.json": cohort_raw}.items():
        with (output / name).open("xb") as file:
            file.write(raw)
    with ThreadPoolExecutor(max_workers=image_workers) as pool, torch.inference_mode():
        for split in SPLITS:
            rows = read_manifest(output / f"{split}.csv")
            features = np.lib.format.open_memmap(output / f"{split}-x.npy", mode="w+",
                                                 dtype=np.float32, shape=(len(rows), 512))
            labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
            np.save(output / f"{split}-y.npy", labels, allow_pickle=False)
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                # map receives at most this batch, preserving order and bounding futures.
                tensors = list(pool.map(lambda row: _read_tensor(root, row, identities[row["image_index"]], transform), batch))
                result = model(torch.stack(tensors).to(device)).detach().cpu().numpy()
                if result.dtype != np.float32 or result.shape != (len(batch), 512) or not np.isfinite(result).all():
                    raise ValueError("invalid frozen backbone feature output")
                features[start:start + len(batch)] = result
            features.flush()
            del features
    if _model_state_sha256(model) != state_sha:
        raise ValueError("frozen model state changed during extraction")
    report = {
        "schema_version": 1, "cache_version": CACHE_VERSION, "backbone": "resnet18",
        "feat_dim": 512, "n_classes": 2, "feature_dtype": "float32", "label_dtype": "int64",
        "pretrained": pretrained, "backbone_seed": backbone_seed, "model_state_sha256": state_sha,
        "weight_identity": (f"ResNet18_Weights.{torchvision.models.ResNet18_Weights.DEFAULT.name}"
                            if pretrained else f"random-seed:{backbone_seed}"),
        "transform": TRANSFORM, "device": device, "batch_size": batch_size,
        "audit_workers": audit_workers, "image_workers": image_workers,
        "torch_num_threads": torch.get_num_threads(),
        "library_versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                             "numpy": np.__version__, "pillow": pillow_version},
        **{key: cohort[key] for key in ("dataset", "classes", "label_definition", "source_hashes",
                                        "original_preparation_sha256", "image_audit_sha256")},
        "curated_cohort_sha256": hashlib.sha256(cohort_raw).hexdigest(),
        "selected_manifest_hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in csv_payloads.items()},
        "splits": cohort["selected_splits"],
        "payload_hashes": {name: sha256_file(output / name) for name in sorted(PAYLOADS)},
    }
    # Durability precedes completion publication, including array data and directory entries.
    for name in PAYLOADS:
        with (output / name).open("rb") as file:
            os.fsync(file.fileno())
    with (output / ".report.json.tmp").open("xb") as file:
        file.write(json_bytes(report))
        file.flush()
        os.fsync(file.fileno())
    os.replace(output / ".report.json.tmp", output / "report.json")
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    load_nih_cache(output, expected_report_sha256=hashlib.sha256(json_bytes(report)).hexdigest())
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "original-preparation", "image-audit", "curated-cohort", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    for name, default in (("batch-size", 64), ("backbone-seed", 1234), ("audit-workers", 1), ("image-workers", 4)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--random-weights", action="store_true", help="non-pretrained control only")
    args = vars(parser.parse_args(argv))
    args["root"] = args.pop("data_root")
    args["pretrained"] = not args.pop("random_weights")
    try:
        report = prepare(**args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json_bytes(report).decode(), end="")


if __name__ == "__main__":
    main()
