"""Smoke test for the DA-11 × FR-13 `dp_on_head_xray_cohort` benchmark — the COHORT-SIZE (N) lever of
central-DP on a small head, measured on REAL image features. Fixes ε (hence z) and sweeps N with the
per-client shard HELD CONSTANT (bootstrap from the real feature pool), so the sweep isolates the
noise-averaging SNR = N/(z·√d) effect from a "less data per client" confound. Runs on the same tiny,
seeded synthetic ImageFolder fixture as the ε-sweep smoke test — fast, deterministic, never skips.

Assertions:
1. at q=1 the accountant-solved z is CONSTANT across N, and SNR = N/(z·√d) grows exactly linearly in N;
2. the predicted SNR=1 crossing equals z·√d, and the head d is constant across the sweep;
3. the wire stays head-only at every N, and the sweep is deterministic.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from benchmarks.dp_on_head_xray import extract_features  # noqa: E402
from benchmarks.dp_on_head_xray_cohort import crossing_cohort, run_cohort_sweep  # noqa: E402


def _make_imagefolder(root, *, per_class_train=8, per_class_test=4, size=32, seed=0):
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
    return extract_features(root, backbone="resnet18", pretrained=False, img_size=32,
                            device="cpu", backbone_seed=1234, cache_dir=str(tmp_path / "cache"))


def test_z_constant_and_snr_linear_in_n(tmp_path):
    f = _features(tmp_path)
    out = run_cohort_sweep(features=f, target_epsilon=4.0, n_values=[4, 8, 16], per_client=8,
                           rounds=3, clip=0.4, delta=1e-5, seed=1234, dp_seeds=[777, 778])
    meta, results = out["meta"], out["results"]
    # (1) q=1 -> z depends only on (ε, rounds, δ), not N: constant across the sweep.
    assert meta["z_constant_across_n"] is True
    # SNR = N/(z·√d): each row's stored SNR matches the formula (abs tol covers 4-dp rounding).
    z = results[0]["noise_multiplier_z"]
    d = results[0]["aggregatable_coords_d"]
    for r in results:
        assert r["utility_snr"] == pytest.approx(r["n"] / (z * d ** 0.5), abs=1e-4)
    # (2) predicted SNR=1 crossing == z·√d; head d fixed across N.
    assert meta["predicted_crossing_n"] == pytest.approx(meta["noise_multiplier_z"] * (d ** 0.5), rel=1e-3)
    assert all(r["aggregatable_coords_d"] == d for r in results)
    assert meta["head_d"] == d
    # (3) head-only wire at every N.
    assert all(r["wire_is_head_only"] for r in results)
    # crossing_cohort returns the first N reaching SNR>=1 (or None) — must be consistent with the rows.
    cn = crossing_cohort(results)
    assert cn is None or any(r["n"] == cn and r["snr_ge_one"] for r in results)


def test_cohort_sweep_is_deterministic(tmp_path):
    f = _features(tmp_path)
    kw = dict(features=f, target_epsilon=4.0, n_values=[4, 8], per_client=8, rounds=3,
              clip=0.4, delta=1e-5, seed=1234, dp_seeds=[777, 778])
    a = run_cohort_sweep(**kw)["results"]
    b = run_cohort_sweep(**kw)["results"]
    assert [r["dp_final_accuracy"] for r in a] == [r["dp_final_accuracy"] for r in b]
    assert [r["noise_multiplier_z"] for r in a] == [r["noise_multiplier_z"] for r in b]
