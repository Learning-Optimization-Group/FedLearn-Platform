"""prepare_features writes a features file the engine can read, with row-aligned triggered copies."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from benchmarks.prepare_features import prepare  # noqa: E402


def _imagefolder(root, per_class=4, size=32):
    import numpy as np
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(per_class):
                arr = (rng.random((size, size)) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{cls}_{i}.png")   # 2-D uint8 -> grayscale
    return root


def test_prepare_writes_aligned_features_and_triggered_copies(tmp_path):
    root = _imagefolder(tmp_path / "xray")
    out = tmp_path / "feats.pt"
    prepare(str(root), str(out), backbone="resnet18", pretrained=False, img_size=32,
            trigger_patch=8, cache_dir=str(tmp_path / "cache"))
    data = torch.load(out, weights_only=True)
    assert data["train_x"].shape == (8, 512) and data["test_x"].shape == (8, 512)
    assert data["trigger_train_x"].shape == data["train_x"].shape
    assert data["trigger_test_x"].shape == data["test_x"].shape
    assert data["n_classes"] == 2 and data["feat_dim"] == 512
    # the patch changes the features of every image
    assert not torch.allclose(data["trigger_train_x"], data["train_x"])


def test_prepare_without_trigger_has_no_triggered_keys(tmp_path):
    root = _imagefolder(tmp_path / "xray")
    out = tmp_path / "feats.pt"
    prepare(str(root), str(out), backbone="resnet18", pretrained=False, img_size=32,
            cache_dir=str(tmp_path / "cache"))
    data = torch.load(out, weights_only=True)
    assert "trigger_train_x" not in data and "trigger_test_x" not in data
