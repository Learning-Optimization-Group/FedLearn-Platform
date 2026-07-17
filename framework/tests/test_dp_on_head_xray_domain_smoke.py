"""Smoke test for the DA-11 domain-backbone comparison (`dp_on_head_xray_domain`): does a DOMAIN-adapted
backbone (ImageNet-init fine-tuned on the X-ray train split, then frozen) beat a frozen ImageNet backbone
— raising the no-DP ceiling and, at the SAME head d, improving the DP escape? This pins the machinery on
the tiny synthetic ImageFolder fixture (fine-tune a random-init backbone 1 epoch, extract, run a tiny DP
sweep) — fast, deterministic, never skips.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from benchmarks.dp_on_head_xray import extract_domain_features, run_sweep  # noqa: E402
from benchmarks.dp_on_head_xray_domain import tightest_escaping_epsilon  # noqa: E402


def _make_imagefolder(root, *, per_class_train=10, per_class_test=4, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    for split, per in (("train", per_class_train), ("test", per_class_test)):
        for cls, base in (("NORMAL", 60), ("PNEUMONIA", 195)):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(per):
                noise = (torch.rand(size, size, generator=g) * 40).to(torch.uint8)
                arr = (torch.clamp(torch.full((size, size), base, dtype=torch.int16)
                                   + noise.to(torch.int16) - 20, 0, 255)).to(torch.uint8)
                Image.fromarray(arr.numpy(), mode="L").save(os.path.join(d, f"{cls}_{i}.png"))
    return root


def test_domain_extract_shape_and_determinism(tmp_path):
    root = _make_imagefolder(str(tmp_path / "xray"))
    kw = dict(backbone="resnet18", epochs=1, lr=1e-3, img_size=32, device="cpu",
              seed=1234, pretrained=False)
    f1 = extract_domain_features(root, cache_dir=str(tmp_path / "c1"), **kw)
    assert f1["feat_dim"] == 512
    assert "domain" in f1["variant"]
    # deterministic fine-tune + extract on CPU with a fixed seed (fresh cache dir re-runs it).
    f2 = extract_domain_features(root, cache_dir=str(tmp_path / "c2"), **kw)
    assert torch.allclose(f1["train_x"], f2["train_x"], atol=1e-5)
    # the DP sweep runs over domain features unchanged; head d is set by the (same) feat_dim.
    out = run_sweep(features=f1, epsilons=[8.0, 1.0], rounds=3, clients=4,
                    clip=0.4, delta=1e-5, seed=1234, dp_seed=777)
    assert out["meta"]["head_d"] == 512 * 2 + 2


def test_tightest_escaping_epsilon():
    sweep = {"results": [
        {"target_epsilon": None, "escapes_collapse": None},
        {"target_epsilon": 8.0, "escapes_collapse": True},
        {"target_epsilon": 4.0, "escapes_collapse": True},
        {"target_epsilon": 1.0, "escapes_collapse": False},
    ]}
    assert tightest_escaping_epsilon(sweep) == 4.0                       # tightest ε that still escapes
    none_escape = {"results": [{"target_epsilon": 1.0, "escapes_collapse": False}]}
    assert tightest_escaping_epsilon(none_escape) is None
