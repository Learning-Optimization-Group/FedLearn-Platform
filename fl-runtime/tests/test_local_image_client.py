"""The canonical client trains the same image tensors and round order as Android."""
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest
import torch
from google.protobuf import json_format
from PIL import Image

import client
import execution_plan
import recipes
from fedlearn.communication.generated import execution_contract_pb2 as pb
from fedlearn.contract.image_package import write_package
from fedlearn.contract import parse_contract_binary


def training(recipe="CNN"):
    return execution_plan.resolve_model_training(recipe, "FedAvg", "FULL", data_source="LOCAL_SNAPSHOT")


def package(tmp_path, n=3):
    pixels = np.arange(n * 32 * 32 * 3, dtype=np.uint8).reshape(n, 32, 32, 3)
    classes = recipes.get_recipe("CNN").classes
    path = tmp_path / "images.zip"
    write_package(path, pixels, [classes[i % 10] for i in range(n)], class_names=classes)
    return path, pixels


def android_snapshot(tmp_path):
    path = tmp_path / "snapshot"; path.mkdir()
    # Literal normalization fingerprint matches Kotlin Float bits for 0.5.
    x = np.linspace(-1, 1, 3 * 3 * 32 * 32, dtype="<f4").reshape(3, 3, 32, 32)
    y = np.array([0, 1, 2], dtype="<i8")
    (path / "inputs.f32").write_bytes(x.tobytes())
    (path / "targets.i64").write_bytes(y.tobytes())
    meta = dict(schemaVersion=1, modality="image", inputShape=[3, 32, 32], inputDtype="f32",
                classNames=recipes.get_recipe("CNN").classes, labelSchemaId=training().data.label_schema_id,
                recordCount=3, inputsSha256=hashlib.sha256(x.tobytes()).hexdigest(),
                targetsSha256=hashlib.sha256(y.tobytes()).hexdigest(),
                transforms="image:32x32x3;mean=3f000000,3f000000,3f000000;std=3f000000,3f000000,3f000000")
    save_meta(path, meta)
    return path, x, y


def save_meta(path, meta):
    meta.pop("snapshotId", None)
    meta["snapshotId"] = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (path / "snapshot.json").write_text(json.dumps(meta))


def test_package_transforms_match_torchvision_bits_and_android_snapshot_identity(tmp_path):
    from local_snapshot import load_local_snapshot
    path, pixels = package(tmp_path)
    local = load_local_snapshot(path, training().data, recipes.get_recipe("CNN").classes)
    expected = torch.stack([recipes._cnn_transform()(Image.fromarray(p))
                            for p in pixels])
    assert torch.equal(local.dataset.tensors[0].view(torch.int32), expected.view(torch.int32))
    assert local.dataset.tensors[1].tolist() == [0, 1, 2]
    snapshot, _, _ = android_snapshot(tmp_path)
    meta = json.loads((snapshot / "snapshot.json").read_text())
    x = expected.numpy().astype("<f4").tobytes()
    (snapshot / "inputs.f32").write_bytes(x)
    meta["inputsSha256"] = hashlib.sha256(x).hexdigest()
    save_meta(snapshot, meta)
    restored = load_local_snapshot(snapshot, training().data, recipes.get_recipe("CNN").classes)
    assert local.snapshot_id == restored.snapshot_id
    assert torch.equal(local.dataset.tensors[0], restored.dataset.tensors[0])


@pytest.mark.parametrize("damage", ["transform", "digest", "size", "shape", "classes", "nonfinite", "labels"])
def test_corrupt_or_differently_prepared_android_images_are_refused(tmp_path, damage):
    from local_snapshot import load_local_snapshot
    path, x, y = android_snapshot(tmp_path)
    meta = json.loads((path / "snapshot.json").read_text())
    if damage == "transform":
        meta["transforms"] = "image:32x32x3"
    elif damage == "digest":
        (path / "inputs.f32").write_bytes(b"\0" * x.nbytes)
    elif damage == "size":
        meta["recordCount"] += 1
    elif damage == "shape":
        meta["inputShape"] = [32, 32, 3]
    elif damage == "classes":
        meta["classNames"].reverse()
    elif damage == "nonfinite":
        x[0, 0, 0, 0] = np.nan
        (path / "inputs.f32").write_bytes(x.tobytes())
        meta["inputsSha256"] = hashlib.sha256(x.tobytes()).hexdigest()
    else:
        y[0] = 10
        (path / "targets.i64").write_bytes(y.tobytes())
        meta["targetsSha256"] = hashlib.sha256(y.tobytes()).hexdigest()
    save_meta(path, meta)
    with pytest.raises(ValueError):
        load_local_snapshot(path, training().data, recipes.get_recipe("CNN").classes)


@pytest.mark.parametrize("damage", ["extra_entry", "duplicate_entry", "count", "large_count", "shape", "label",
                                   "pixel_length", "base64", "classes", "duplicate_json_field"])
def test_malformed_image_package_is_refused_before_training(tmp_path, damage):
    from local_snapshot import load_local_snapshot
    path, _ = package(tmp_path)
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("dataset.json"))
        records = [json.loads(s) for s in z.read("records.jsonl").decode().splitlines()]
    if damage == "count": meta["recordCount"] += 1
    elif damage == "large_count": meta["recordCount"] = 1_000_000
    elif damage == "shape": meta["width"] = 31
    elif damage == "label": records[0]["label"] = "unknown"
    elif damage == "pixel_length": records[0]["pixels"] = "AAAA"
    elif damage == "base64": records[0]["pixels"] = "***"
    elif damage == "classes": meta["classNames"].reverse()
    text = json.dumps(meta)
    if damage == "duplicate_json_field": text = text[:-1] + ',"recordCount":3}'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("dataset.json", text)
        z.writestr("records.jsonl", "\n".join(json.dumps(r) for r in records))
        if damage == "extra_entry": z.writestr("../inputs.f32", "unsafe")
        if damage == "duplicate_entry":
            with pytest.warns(UserWarning): z.writestr("dataset.json", text)
    with pytest.raises(ValueError):
        load_local_snapshot(path, training().data, recipes.get_recipe("CNN").classes)


def test_pneumonia_uses_explicit_local_loader_before_recipe_download_or_partition(monkeypatch):
    from local_snapshot import SeededSnapshotLoader
    local = SeededSnapshotLoader(torch.utils.data.TensorDataset(torch.zeros(2, 1, 224, 224),
                                 torch.tensor([0, 1])), batch_size=16, seed=42, drop_last=False)
    monkeypatch.setattr(client, "LOCAL_DATASET", local)
    monkeypatch.setattr(client, "USE_PNEUMONIA", True)
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit local images must not fall back to recipe data")
    monkeypatch.setattr(recipes, "get_recipe", forbidden)
    loader, _ = client.load_data(9, "pneumonia", "own.zip", 10)
    assert len(loader.dataset) == 2


@pytest.mark.parametrize("shape,classes,means,stds", [([1, 224, 224], ["NORMAL", "PNEUMONIA"], [.5], [.5]),
    ([3, 112, 112], recipes.get_recipe("CNN").classes, [.485, .456, .406], [.229, .224, .225])])
def test_grayscale_and_resized_rgb_packages_match_reference_normalization(tmp_path, shape, classes, means, stds):
    from local_snapshot import load_local_snapshot
    c, h, w = shape
    requirement = pb.DataRequirement(task=pb.TASK_IMAGE_CLASSIFICATION, input_shape=shape,
        input_dtype=pb.DTYPE_F32, class_count=len(classes), label_schema_id=execution_plan.label_schema_id(classes),
        transforms=[pb.Transform(image_to_unit_tensor=pb.ImageToUnitTensor(height=h, width=w, channels=c)),
                    pb.Transform(normalize_channels=pb.NormalizeChannels(mean=means, std=stds))])
    pixels = np.arange(h * w * c, dtype=np.uint8).reshape(1, h, w, c)
    path = tmp_path / "prepared.zip"
    write_package(path, pixels, [classes[1]], class_names=classes)
    local = load_local_snapshot(path, requirement, classes)
    expected = torch.from_numpy(pixels[0].copy()).permute(2, 0, 1).float() / 255
    expected = (expected - torch.tensor(means)[:, None, None]) / torch.tensor(stds)[:, None, None]
    assert torch.equal(local.dataset.tensors[0][0].view(torch.int32), expected.view(torch.int32))
    assert local.dataset.tensors[1].tolist() == [1]


def test_image_budget_is_enforced_before_tensor_allocation(tmp_path, monkeypatch):
    import local_images
    from local_snapshot import load_local_snapshot
    path, _ = package(tmp_path)
    monkeypatch.setattr(local_images, "MAX_BYTES", 20_000)
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized declaration allocated tensor storage")
    monkeypatch.setattr(local_images.np, "empty", forbidden)
    with pytest.raises(ValueError, match="size limit"):
        load_local_snapshot(path, training().data, recipes.get_recipe("CNN").classes)


@pytest.mark.parametrize("damage", ["many_entries", "huge_directory", "conflicting_zip64"])
def test_zip_metadata_budget_is_enforced_before_full_directory_parsing(tmp_path, monkeypatch, damage):
    from local_snapshot import load_local_snapshot
    path, _ = package(tmp_path)
    if damage in ("many_entries", "conflicting_zip64"):
        with zipfile.ZipFile(path, "w") as z:
            for i in range(2000): z.writestr(f"empty-{i}", b"")
        if damage == "conflicting_zip64":
            raw = path.read_bytes()
            fields = struct.unpack("<4s4H2IH", raw[-22:])
            zip64 = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                                2000, 2000, fields[5], fields[6])
            locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, len(raw) - 22, 1)
            ordinary = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 2, 2, 1, fields[6], 0)
            path.write_bytes(raw[:-22] + zip64 + locator + ordinary)
    else:
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, len(raw) - 22 + 12, 128 * 1024 * 1024)
        path.write_bytes(raw)
    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded ZIP central directory was parsed")
    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="directory"):
        load_local_snapshot(path, training().data, recipes.get_recipe("CNN").classes)


def test_cli_accepts_local_image_package_and_cnn_adam_fit_matches_manual_steps(tmp_path, monkeypatch):
    from local_snapshot import load_local_snapshot
    path, _ = package(tmp_path, n=65)
    plan = training()
    root = Path(__file__).resolve().parents[2]
    c = parse_contract_binary((root / "framework/tests/fixtures/execution_contract_v1/golden_tinynet_fedavg.binpb").read_bytes())
    plan.model_revision = c.model_training.model_revision
    plan.initial_state_sha256 = c.model_training.initial_state_sha256
    plan.artifacts.extend(c.model_training.artifacts)
    c.model_training.CopyFrom(plan)
    c.recipe = pb.RECIPE_CNN
    c.num_rounds = 2
    c.seed = 77
    contract = tmp_path / "contract.json"; contract.write_text(json_format.MessageToJson(c))
    for key in ("USE_LLM", "USE_MLP", "USE_PNEUMONIA", "USE_LLM_LORA", "USE_DERIVED", "USE_WIRE_SUBSET"):
        monkeypatch.setattr(client, key, False)
    monkeypatch.setattr(client, "MODEL_TYPE", "CNN")
    monkeypatch.setattr(client, "TRAINING_ARM", "FULL")
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    monkeypatch.setattr(client, "LOCAL_DATASET", None)
    args = client.parse_args(["--project-id", c.project_id, "--run-id", c.run_id,
        "--server-address", "localhost:50000", "--partition-id", "0", "--model-type", "CNN",
        "--execution-contract", str(contract), "--dataset-path", str(path)])
    client.enforce_execution_contract(args, "CNN", "FULL")
    participant = client.ZOSLClient(partition_id=0, dataset_name="cifar10")
    reference = recipes.get_recipe("CNN").build_model("cpu")
    reference.load_state_dict(participant.net.state_dict())
    local = load_local_snapshot(path, plan.data, recipes.get_recipe("CNN").classes)
    x, y = local.dataset.tensors
    # Explicit contracted indices; compare actual optimizer outputs, including the partial batch.
    from fedlearn.contract.batch_order import seeded_permutation
    for round_ in (1, 2):
        optimizer = torch.optim.Adam(reference.parameters(), lr=plan.local_training.adam.learning_rate)
        reference.train()
        order = seeded_permutation(65, 77, round_, 0)
        for offset in range(0, 65, plan.local_training.batch_size):
            indices = order[offset:offset + plan.local_training.batch_size]
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(reference(x[indices]), y[indices]).backward()
            optimizer.step()
        actual, count = participant.fit(participant.get_parameters(), {"server_round": round_})
        assert count == 65
        for name, value in actual.items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
