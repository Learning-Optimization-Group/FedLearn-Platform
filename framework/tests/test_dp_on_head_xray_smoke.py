"""Smoke test for the DA-11 x FR-13 `dp_on_head_xray` benchmark — central-DP on a small trainable HEAD
over a FROZEN backbone, measured on REAL image features instead of the synthetic Gaussian-blob task of
`dp_on_head`. This pins the machinery, not the dataset: it runs the whole pipeline (PIL images on disk ->
frozen torchvision backbone -> cached features -> head-only FedAvg with the REAL DP mechanism + RDP
accountant) on a tiny, seeded, synthetic ImageFolder fixture so it is fast and never skips.

Assertions:
1. feature extraction is deterministic (same fixture + seed -> identical features), and the head
   dimension d equals feat_dim * n_classes + n_classes (a REAL, small head — the DP-friendly d);
2. the accountant-solved z is finite/positive and the accounted ε round-trips to the target;
3. tighter ε -> more noise -> lower utility SNR (the FR-13 ordering), and the DP sweep is deterministic;
4. the frozen backbone never rides the wire and the DP mechanism only touches the head keys.
"""
import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from benchmarks.dp_on_head_xray import (  # noqa: E402
    extract_features,
    run_sweep,
    solve_noise_multiplier,
)


def _make_imagefolder(root, *, per_class_train=6, per_class_test=4, size=32, seed=0):
    """Write a tiny deterministic 2-class ImageFolder (NORMAL/PNEUMONIA) with a real class signal:
    NORMAL images are darker, PNEUMONIA brighter (+ seeded noise), so a linear head over frozen
    features can separate them above chance without depending on the real dataset."""
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


def _features(tmp_path):
    root = _make_imagefolder(str(tmp_path / "xray"))
    return extract_features(
        root, backbone="resnet18", pretrained=False, img_size=32,
        device="cpu", backbone_seed=1234, cache_dir=str(tmp_path / "cache"),
    )


def test_accountant_solves_finite_positive_z():
    z = solve_noise_multiplier(target_epsilon=1.0, q=1.0, rounds=3, delta=1e-5)
    assert math.isfinite(z) and z > 0.0


def test_extraction_is_deterministic_and_head_d_is_real_and_small(tmp_path):
    f1 = _features(tmp_path)
    # (1) Deterministic extraction: a second extraction (fresh cache) reproduces the features bit-for-bit.
    f2 = extract_features(
        str(tmp_path / "xray"), backbone="resnet18", pretrained=False, img_size=32,
        device="cpu", backbone_seed=1234, cache_dir=str(tmp_path / "cache2"),
    )
    assert torch.equal(f1["train_x"], f2["train_x"])
    assert torch.equal(f1["test_x"], f2["test_x"])
    # resnet18 penultimate features are 512-dim; head d = 512*2 + 2 for the 2-class linear probe.
    assert f1["feat_dim"] == 512
    out = run_sweep(features=f1, epsilons=[8.0, 1.0], rounds=3, clients=4,
                    clip=0.4, delta=1e-5, seed=1234, dp_seed=777)
    d = out["results"][0]["aggregatable_coords_d"]
    assert d == f1["feat_dim"] * 2 + 2
    from benchmarks.dp_on_head import FEDLORA_REFERENCE_D
    assert 0 < d < FEDLORA_REFERENCE_D            # a real, smaller-than-FedLoRA head


def test_snr_ordering_accountant_roundtrip_and_wire_invariants(tmp_path):
    f = _features(tmp_path)
    out = run_sweep(features=f, epsilons=[8.0, 1.0], rounds=3, clients=4,
                    clip=0.4, delta=1e-5, seed=1234, dp_seed=777)
    results = out["results"]
    loose = next(r for r in results if r["target_epsilon"] == 8.0)
    tight = next(r for r in results if r["target_epsilon"] == 1.0)

    # (2) accountant: finite positive z, accounted ε round-trips to the requested budget.
    assert math.isfinite(tight["noise_multiplier_z"]) and tight["noise_multiplier_z"] > 0.0
    assert tight["accounted_epsilon"] == pytest.approx(1.0, abs=1e-3)
    # (3) tighter ε -> lower utility SNR.
    assert tight["utility_snr"] < loose["utility_snr"]
    # (4) frozen-backbone invariants survive the real-feature DP path.
    assert tight["wire_is_head_only"] is True
    assert tight["backbone_federated"] is False


def test_sweep_is_deterministic(tmp_path):
    f = _features(tmp_path)
    kw = dict(features=f, epsilons=[1.0], rounds=3, clients=4, clip=0.4, delta=1e-5,
              seed=1234, dp_seed=777)
    a = run_sweep(**kw)["results"]
    b = run_sweep(**kw)["results"]
    assert [r["per_round_accuracy"] for r in a] == [r["per_round_accuracy"] for r in b]
    assert [r["noise_multiplier_z"] for r in a] == [r["noise_multiplier_z"] for r in b]
