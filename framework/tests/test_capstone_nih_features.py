"""Source-bound frozen features, explicit splits and cache refusal contracts."""
import csv
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from tests.test_capstone_nih_data import nih_tree
from benchmarks.nih_data import prepare as prepare_manifests
from benchmarks.nih_image_audit import audit_images
from benchmarks.nih_cohort import prepare_cohort
from benchmarks.dp_on_head_xray import _build_backbone, _IMAGENET_MEAN, _IMAGENET_STD
from benchmarks import prepare_nih_features as preparation
from fedlearn.simulation.nih_cache import load_nih_cache


def _setup(root, base):
    original, audit, curated = (base / name for name in ("original", "audit", "curated"))
    prepare_manifests(root, original, num_clients=2)
    audit_images(root, original, audit)
    prepare_cohort(root, original, audit, curated)
    return root, original, audit, curated


@pytest.fixture
def prepared(nih_tree, tmp_path):
    return _setup(nih_tree, tmp_path)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    base = tmp_path_factory.mktemp("nih_features_template")
    root = nih_tree.__wrapped__(base)
    inputs = _setup(root, base)
    output = base / "features"
    preparation.prepare(*inputs, output, pretrained=False, batch_size=4, image_workers=2)
    return inputs, output


@pytest.fixture
def cached(template, tmp_path):
    inputs, source = template
    output = tmp_path / "features"
    shutil.copytree(source, output)
    return inputs, output


def _rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _rewrite_report(output, changed=None):
    report = json.loads((output / "report.json").read_text())
    if changed:
        report["payload_hashes"][changed] = hashlib.sha256((output / changed).read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def test_cache_preserves_every_row_and_matches_direct_frozen_reference(cached):
    (root, _, _, curated), output = cached
    loaded = load_nih_cache(output)
    report = loaded["provenance"]
    assert report["feat_dim"] == 512 and report["n_classes"] == 2
    assert report["pretrained"] is False and report["device"] == "cpu"
    assert report["weight_identity"] == "random-seed:1234"
    assert report["transform"] == {
        "color": "RGB", "resize": [224, 224], "interpolation": "bilinear", "antialias": True,
        "scale": "uint8/255", "mean": list(_IMAGENET_MEAN), "std": list(_IMAGENET_STD)}
    assert (output / "cohort-report.json").read_bytes() == (curated / "report.json").read_bytes()
    model, _ = _build_backbone("resnet18", False, 1234)
    assert report["model_state_sha256"] == preparation._model_state_sha256(model)
    tf = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
                             transforms.ToTensor(), transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    patient_sets = []
    for split, actual in loaded["splits"].items():
        rows = _rows(curated / f"{split}.csv")
        assert (output / f"{split}.csv").read_bytes() == (curated / f"{split}.csv").read_bytes()
        assert actual["image_ids"] == [r["image_index"] for r in rows]
        assert actual["patient_ids"] == [r["patient_id"] for r in rows]
        np.testing.assert_array_equal(actual["y"], [int(r["label"]) for r in rows])
        expected = []
        with torch.no_grad():
            for start in range(0, len(rows), 4):
                images = []
                for row in rows[start:start + 4]:
                    with Image.open(root / row["image_path"]) as image:
                        images.append(tf(image.convert("RGB")))
                expected.append(model(torch.stack(images)).numpy())
        np.testing.assert_allclose(actual["x"], np.concatenate(expected), rtol=1e-5, atol=1e-6)
        assert actual["x"].dtype == np.float32 and np.isfinite(actual["x"]).all()
        assert actual["y"].dtype == np.int64
        patient_sets.append(set(actual["patient_ids"]))
    assert sum(len(s["y"]) for s in loaded["splits"].values()) == 24
    assert all(not left & right for i, left in enumerate(patient_sets) for right in patient_sets[i + 1:])
    assert loaded["report_sha256"] == hashlib.sha256((output / "report.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("key,value", [("batch_size", True), ("batch_size", 0), ("batch_size", 257),
                                       ("batch_size", 1.5), ("backbone_seed", True), ("backbone_seed", -1),
                                       ("backbone_seed", 1.5), ("backbone_seed", 2**64), ("pretrained", "yes"),
                                       ("image_workers", False), ("image_workers", 0), ("image_workers", 17),
                                       ("image_workers", 2.5), ("device", "invalid"), ("device", "cuda")])
def test_invalid_preparation_arguments_refuse_before_publication(prepared, tmp_path, key, value):
    output = tmp_path / "features"
    with pytest.raises(ValueError):
        preparation.prepare(*prepared, output, **{key: value})
    assert not output.exists()


@pytest.mark.parametrize("where", ["existing", "symlink", "source", "original", "audit", "curated"])
def test_unsafe_output_never_overwrites_sources(prepared, tmp_path, where):
    output = tmp_path / "features"
    if where == "existing":
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    elif where == "symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        output = prepared[{"source": 0, "original": 1, "audit": 2, "curated": 3}[where]] / "features"
    with pytest.raises((ValueError, FileExistsError)):
        preparation.prepare(*prepared, output, pretrained=False)
    if where == "existing":
        assert (output / "sentinel").read_text() == "preserve"
    elif where != "symlink":
        assert not output.exists()


@pytest.mark.parametrize("mutation", ["source", "cohort", "audit", "missing_cohort"])
def test_changed_or_missing_source_evidence_refuses(prepared, tmp_path, mutation):
    root, _, audit, curated = prepared
    if mutation == "source":
        Image.new("RGB", (4, 4), "red").save(root / _rows(curated / "train.csv")[0]["image_path"])
    elif mutation == "cohort":
        (curated / "train.csv").write_text("wrong rows")
    elif mutation == "audit":
        (audit / "images.csv").write_text("wrong hashes")
    else:
        (curated / "report.json").unlink()
    with pytest.raises(ValueError):
        preparation.prepare(*prepared, tmp_path / "features", pretrained=False)


def test_source_mutation_after_verification_is_caught_during_read(prepared, tmp_path, monkeypatch):
    root, _, _, curated = prepared
    original_builder = preparation._build_backbone
    def mutate_after_verification(*args):
        result = original_builder(*args)
        Image.new("RGB", (4, 4), "blue").save(root / _rows(curated / "train.csv")[0]["image_path"])
        return result
    monkeypatch.setattr(preparation, "_build_backbone", mutate_after_verification)
    output = tmp_path / "features"
    with pytest.raises(ValueError, match="identity|hash"):
        preparation.prepare(*prepared, output, pretrained=False)
    assert not (output / "report.json").exists()


def test_all_forwards_are_bounded_frozen_and_rng_is_restored(prepared, tmp_path, monkeypatch):
    builder = preparation._build_backbone
    observations = []
    def observe(*args):
        model, width = builder(*args)
        def before(module, inputs):
            observations.append((len(inputs[0]), torch.is_grad_enabled(),
                                 any(m.training for m in module.modules()),
                                 any(p.requires_grad for p in module.parameters())))
        model.register_forward_pre_hook(before)
        return model, width
    monkeypatch.setattr(preparation, "_build_backbone", observe)
    state = torch.get_rng_state().clone()
    preparation.prepare(*prepared, tmp_path / "features", pretrained=False, batch_size=3, image_workers=2)
    assert observations and all(n <= 3 and not grad and not training and not requires for n, grad, training, requires in observations)
    assert torch.equal(state, torch.get_rng_state())


def test_model_state_identity_includes_buffers_and_named_parameters():
    model = torch.nn.BatchNorm1d(2).eval()
    before = preparation._model_state_sha256(model)
    model.running_mean.add_(1)
    after_buffer = preparation._model_state_sha256(model)
    assert before != after_buffer
    with torch.no_grad():
        model.weight.add_(1)
    assert preparation._model_state_sha256(model) != after_buffer


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_cpu_preparation_also_restores_available_mps_rng(prepared, tmp_path):
    before = torch.mps.get_rng_state().clone()
    preparation.prepare(*prepared, tmp_path / "features", pretrained=False, batch_size=4)
    assert torch.equal(before, torch.mps.get_rng_state())


@pytest.mark.parametrize("mutation", ["labels", "row_order", "dtype", "shape", "nan", "inf", "extra", "missing", "symlink", "report_format", "cohort_report"])
def test_loader_rejects_rehashed_structural_and_alignment_errors(cached, mutation):
    _, output = cached
    changed = None
    if mutation == "labels":
        changed = "train-y.npy"
        y = np.load(output / changed)
        y[0] = 1 - y[0]
        np.save(output / changed, y, allow_pickle=False)
    elif mutation == "row_order":
        changed = "train.csv"
        lines = (output / changed).read_text().splitlines()
        (output / changed).write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    elif mutation in {"dtype", "shape", "nan", "inf"}:
        changed = "train-x.npy"
        x = np.load(output / changed)
        if mutation == "dtype": x = x.astype(np.float64)
        elif mutation == "shape": x = x[:, :-1]
        else: x[0, 0] = np.nan if mutation == "nan" else np.inf
        np.save(output / changed, x, allow_pickle=False)
    elif mutation == "extra":
        (output / "unexpected").touch()
    elif mutation == "missing":
        (output / "validation-y.npy").unlink()
    elif mutation == "symlink":
        (output / "train-x.npy").unlink()
        (output / "train-x.npy").symlink_to(output / "test-x.npy")
    elif mutation == "report_format":
        (output / "report.json").write_text(json.dumps(json.loads((output / "report.json").read_text())))
    else:
        changed = "cohort-report.json"
        report = json.loads((output / changed).read_text())
        report["development_content_qualified"] = False
        (output / changed).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if mutation != "report_format": _rewrite_report(output, changed)
    with pytest.raises(ValueError, match="cache"):
        load_nih_cache(output)


def test_pinned_identity_rejects_rehashed_feature_permutation(cached):
    _, output = cached
    expected = hashlib.sha256((output / "report.json").read_bytes()).hexdigest()
    x = np.load(output / "train-x.npy")
    np.save(output / "train-x.npy", x[::-1], allow_pickle=False)
    _rewrite_report(output, "train-x.npy")
    with pytest.raises(ValueError, match="identity|digest"):
        load_nih_cache(output, expected_report_sha256=expected)


@pytest.mark.parametrize("expected", ["", "invalid", "0" * 64, True])
def test_bad_expected_cache_identity_refuses(cached, expected):
    with pytest.raises(ValueError):
        load_nih_cache(cached[1], expected_report_sha256=expected)


def test_archive_disguised_as_npy_refuses_with_cache_error(cached):
    _, output = cached
    with (output / "train-x.npy").open("wb") as file:
        np.savez(file, x=np.zeros((2, 512), dtype=np.float32))
    _rewrite_report(output, "train-x.npy")
    with pytest.raises(ValueError, match="cache"):
        load_nih_cache(output)


def test_interrupted_cache_never_has_completion_marker(prepared, tmp_path, monkeypatch):
    def interrupt(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(preparation.os, "replace", interrupt)
    output = tmp_path / "features"
    with pytest.raises(OSError, match="interrupted"):
        preparation.prepare(*prepared, output, pretrained=False, batch_size=4)
    assert not (output / "report.json").exists()
    with pytest.raises(ValueError, match="cache"):
        load_nih_cache(output)


@pytest.mark.parametrize("bad", ["nan", "width"])
def test_invalid_backbone_output_refuses_completion(prepared, tmp_path, monkeypatch, bad):
    builder = preparation._build_backbone
    def broken(*args):
        model, width = builder(*args)
        model.register_forward_hook(lambda module, inputs, result: result * float("nan") if bad == "nan" else result[:, :-1])
        return model, width
    monkeypatch.setattr(preparation, "_build_backbone", broken)
    output = tmp_path / "features"
    with pytest.raises(ValueError):
        preparation.prepare(*prepared, output, pretrained=False)
    assert not (output / "report.json").exists()


def test_quarantined_patients_never_reenter_feature_rows(nih_tree, tmp_path):
    images = nih_tree / "images_001" / "images"
    shutil.copyfile(images / "00000002_000.png", images / "00000011_000.png")
    inputs = _setup(nih_tree, tmp_path)
    output = tmp_path / "features"
    preparation.prepare(*inputs, output, pretrained=False, batch_size=4)
    loaded = load_nih_cache(output)
    assert all("2" not in split["patient_ids"] for split in loaded["splits"].values())
    assert sum(len(split["y"]) for split in loaded["splits"].values()) == 22
    assert (output / "test.csv").read_bytes() == (inputs[1] / "test.csv").read_bytes()


def test_cli_runs_in_fresh_process_and_refuses_existing_output(prepared, tmp_path):
    output = tmp_path / "features"
    command = [sys.executable, "-m", "benchmarks.prepare_nih_features",
               "--data-root", str(prepared[0]), "--original-preparation", str(prepared[1]),
               "--image-audit", str(prepared[2]), "--curated-cohort", str(prepared[3]),
               "--output", str(output), "--random-weights", "--batch-size", "4"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["pretrained"] is False
    assert load_nih_cache(output)["provenance"]["device"] == "cpu"
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeated.returncode == 2 and "existing" in repeated.stderr
